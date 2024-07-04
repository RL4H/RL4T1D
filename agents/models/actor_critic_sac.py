import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.models.q import QvalModel
from agents.models.value import ValueModule
from agents.models.feature_extracter import LSTMFeatureExtractor


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class CriticValueNetwork(nn.Module):
    def __init__(self, args):
        super(CriticValueNetwork, self).__init__()
        self.FeatureExtractor = LSTMFeatureExtractor(args)
        self.ValueModule = ValueModule(args)

    def forward(self, s):
        extract_states = self.FeatureExtractor.forward(s)
        value = self.ValueModule.forward(extract_states)
        return value


class CriticQNetwork(nn.Module):
    def __init__(self, args):
        super(CriticQNetwork, self).__init__()
        self.FeatureExtractor = LSTMFeatureExtractor(args)
        self.QvalModel = QvalModel(args)

    def forward(self, s, action, mode='batch'):
        extract_states = self.FeatureExtractor.forward(s)
        qvalue = self.QvalModel.forward(extract_states, action, mode)
        return qvalue


class ActionModule(nn.Module):
    def __init__(self, args, device):
        super(ActionModule, self).__init__()
        self.device = device
        self.args = args
        self.output = args.n_action
        self.feature_extractor = (args.n_rnn_hidden * args.n_rnn_layers * args.rnn_directions)

        self.last_hidden = self.feature_extractor * 2
        self.fc_layer1 = nn.Linear(self.feature_extractor, self.last_hidden)
        self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)
        self.mu = nn.Linear(self.last_hidden, self.output)
        self.sigma = nn.Linear(self.last_hidden, self.output)
        self.normalDistribution = torch.distributions.Normal

    def forward(self, extract_states, worker_mode='training'):
        fc_output1 = F.relu(self.fc_layer1(extract_states))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))
        mu = self.mu(fc_output)
        sigma = self.sigma(fc_output)  # * 0.66, + 1e-5
        log_std = torch.clamp(sigma, LOG_STD_MIN, LOG_STD_MAX)
        action_std = torch.exp(log_std)

        dst = self.normalDistribution(mu, action_std)
        if worker_mode == 'training':
            gaussian_action = dst.rsample()
        else:
            gaussian_action = mu

        action = torch.tanh(gaussian_action)

        # calc log_prob
        # openai implementation
        logp_pi = dst.log_prob(gaussian_action[0])  #.sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - gaussian_action[0] - F.softplus(-2 * gaussian_action[0])))  #.sum(axis=1)
        # SAC paper implementation
        # log_prob = dst.log_prob(gaussian_action[0]) - torch.log(1 - action[0] ** 2 + 1e-6)

        return mu, action_std, action, logp_pi


class ActorNetwork(nn.Module):
    def __init__(self, args):
        super(ActorNetwork, self).__init__()
        self.device = args.device
        self.args = args
        self.FeatureExtractor = LSTMFeatureExtractor(args)
        self.ActionModule = ActionModule(args, self.device)
        self.distribution = torch.distributions.Normal

    def forward(self, s):
        extract_states = self.FeatureExtractor.forward(s)
        mu, sigma, action, log_prob = self.ActionModule.forward(extract_states, worker_mode='training')
        return mu, sigma, action, log_prob


class ActorCritic(nn.Module):
    def __init__(self, args, load, actor_path, critic_path, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.experiment_dir = args.experiment_dir

        self.sac_v2 = args.sac_v2
        self.policy_net = ActorNetwork(args)
        self.soft_q_net1 = CriticQNetwork(args)
        self.soft_q_net2 = CriticQNetwork(args)
        if self.sac_v2:
            self.target_q_net1 = CriticQNetwork(args)
            self.target_q_net2 = CriticQNetwork(args)
        else:
            self.value_net = CriticValueNetwork(args)
            self.value_net_target = CriticValueNetwork(args)

    def get_action(self, s):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, sigma, action, log_prob = self.policy_net.forward(s)
        data = dict(mu=mu[0], std=sigma[0], action=action[0])
        return {k: v.detach().cpu().numpy() for k, v in data.items()}

    def evaluate_policy(self, state):  # evaluate policy <batch>
        mu, sigma, action, log_prob = self.policy_net.forward(state)
        return action, log_prob

    def save(self, episode):  # save checkpoints
        if self.sac_v2:
            policy_net_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_policy_net.pth'
            soft_q_net1_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_soft_q_net1.pth'
            soft_q_net2_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_soft_q_net2.pth'
            torch.save(self.policy_net, policy_net_path)
            torch.save(self.soft_q_net1, soft_q_net1_path)
            torch.save(self.soft_q_net2, soft_q_net2_path)
        else:
            policy_net_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_policy_net.pth'
            soft_q_net1_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_soft_q_net1.pth'
            soft_q_net2_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_soft_q_net2.pth'
            value_net_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_value_net.pth'
            torch.save(self.policy_net, policy_net_path)
            torch.save(self.soft_q_net1, soft_q_net1_path)
            torch.save(self.soft_q_net2, soft_q_net2_path)
            torch.save(self.value_net, value_net_path)
