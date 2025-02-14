import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import mlflow
from mlflow.models import infer_signature

from agents.models.feature_extracter import LSTMFeatureExtractor
from agents.models.policy import PolicyModule
from agents.models.value import ValueModule
from agents.models.normed_linear import NormedLinear
from utils import core
from environment.reward_func import composite_reward


class GlucoseModel(nn.Module):
    def __init__(self, args, device):
        super(GlucoseModel, self).__init__()
        self.n_features = args.n_features
        self.device = device
        self.output = args.n_action

        self.n_hidden = args.n_rnn_hidden
        self.n_layers = args.n_rnn_layers
        self.bidirectional = args.bidirectional
        self.directions = args.rnn_directions
        self.feature_extractor = self.n_hidden * self.n_layers * self.directions
        self.last_hidden = self.feature_extractor #* 2
        self.fc_layer1 = nn.Linear(self.feature_extractor + self.output, self.last_hidden)
        self.cgm_mu = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.cgm_sigma = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.normal = torch.distributions.Normal(0, 1)

    def forward(self, extract_state, action, mode):
        #concat_dim = 1 if (mode == 'batch') else 0
        concat_dim = 1

        concat_state_action = torch.cat((extract_state, action), dim=concat_dim)
        fc_output1 = F.relu(self.fc_layer1(concat_state_action))
        fc_output = fc_output1
        # fc_output2 = F.relu(self.fc_layer2(fc_output1))
        # fc_output = F.relu(self.fc_layer3(fc_output2))
        cgm_mu = F.tanh(self.cgm_mu(fc_output))
        # deterministic
        # cgm_sigma = torch.zeros(1, device=self.device, dtype=torch.float32)
        # cgm = cgm_mu
        # probabilistic
        cgm_sigma = F.softplus(self.cgm_sigma(fc_output) + 1e-5)
        z = self.normal.sample()
        cgm = cgm_mu + cgm_sigma * z
        cgm = torch.clamp(cgm, -1, 1)
        return cgm_mu, cgm_sigma, cgm


class ActorNetwork(nn.Module):
    def __init__(self, args, device):
        super(ActorNetwork, self).__init__()
        self.device = device
        self.args = args
        self.FeatureExtractor = LSTMFeatureExtractor(args)
        self.GlucoseModel = GlucoseModel(args, self.device)
        self.ActionModule = PolicyModule(args)
        self.distribution = torch.distributions.Normal
        self.planning_n_step = args.planning_n_step
        self.n_planning_simulations = args.n_planning_simulations
        self.t_to_meal = core.linear_scaling(x=0, x_min=0, x_max=self.args.t_meal)

    def forward(self, s, old_action=None, mode="forward"):
        extract_states = self.FeatureExtractor.forward(s)
        mu, sigma, action, log_prob = self.ActionModule.forward(extract_states)
        if mode == 'forward':
            cgm_mu, cgm_sigma, cgm = self.GlucoseModel.forward(extract_states, action.detach(), mode)
        else:
            cgm_mu, cgm_sigma, cgm = self.GlucoseModel.forward(extract_states, old_action.detach(), mode)
        return mu, sigma, action, log_prob, cgm_mu, cgm_sigma, cgm

    def update_state(self, s, cgm_pred, action, batch_size):
        if batch_size == 1:
            if self.args.n_features == 2:
                s_new = torch.cat((cgm_pred, action), dim=0)
            if self.args.n_features == 3:
                s_new = torch.cat((cgm_pred, action, self.t_to_meal * torch.ones(1, device=self.device)), dim=0)
            s_new = s_new.unsqueeze(0)
            s = torch.cat((s[1:self.args.feature_history, :], s_new), dim=0)
        else:
            if self.args.n_features == 3:
                s_new = torch.cat((cgm_pred, action, self.t_to_meal * torch.ones(batch_size, 1, device=self.device)), dim=1)
            if self.args.n_features == 2:
                s_new = torch.cat((cgm_pred, action), dim=1)
            s_new = s_new.unsqueeze(1)
            s = torch.cat((s[:, 1:self.args.feature_history, :], s_new), dim=1)
        return s

    def expert_search(self, s, rew_norm_var, mode):
        pi, mu, sigma, s_e, r = self.expert_MCTS_rollout(s, mode, rew_norm_var)
        return pi, mu, sigma, s_e, r

    def expert_MCTS_rollout(self, s, mode, rew_norm_var=1):
        batch_size = s.shape[0]
        first_action, first_mu, first_sigma, cum_reward, mu, sigma = 0, 0, 0, 0, 0, 0
        for i in range(self.planning_n_step):
            extract_states = self.FeatureExtractor.forward(s) #.detach()# todo: fix handcraft features
            extract_states = extract_states.detach()
            mu, sigma, action, log_prob = self.ActionModule.forward(extract_states)
            if i == 0:
                first_action = action
                first_mu = mu
                first_sigma = sigma
            _, _, cgm_pred = self.GlucoseModel.forward(extract_states, action, mode)
            bg = core.inverse_linear_scaling(y=cgm_pred.detach().cpu().numpy(), x_min=self.args.glucose_min, x_max=self.args.glucose_max)
            reward = np.array([[composite_reward(self.args, state=xi, reward=None)] for xi in bg])
            reward = reward / (math.sqrt(rew_norm_var + 1e-8))
            reward = np.clip(reward, 10, 10)
            discount = (self.args.gamma ** i)
            cum_reward += (reward * discount)

            # todo: fix - this is a hardcoded to pump action exponential!!!
            action = action.detach()
            cgm_pred = cgm_pred.detach()
            pump_action = self.args.action_scale * (torch.exp((action - 1) * 4))
            action = core.linear_scaling(x=pump_action, x_min=self.args.insulin_min, x_max=self.args.insulin_max)
            ### #todo

            s = self.update_state(s, cgm_pred, action, batch_size)
        cum_reward = torch.as_tensor(cum_reward, dtype=torch.float32, device=self.device)
        return first_action, first_mu, first_sigma, s, cum_reward

    def horizon_error(self, s, feat, actions, real_glucose, mode):
        horizon_error = 0
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        feat = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
        for i in range(0, len(actions)):
            cur_action = torch.as_tensor(actions[i], dtype=torch.float32, device=self.device).reshape(1)
            extract_states, lstmOut = self.FeatureExtractor.forward(s) #.detach()
            extract_states, lstmOut = extract_states.detach(), lstmOut.detach()

            cgm_mu, cgm_sigma, cgm_pred = self.GlucoseModel.forward(lstmOut, cur_action, mode)
            pred = core.inverse_linear_scaling(y=cgm_pred.detach().cpu().numpy(), x_min=self.args.glucose_min,
                                               x_max=self.args.glucose_max)
            horizon_error += ((pred - real_glucose[i])**2)
            s = self.update_state(s, cgm_pred, cur_action, batch_size=1)
        return horizon_error / len(actions)


class CriticNetwork(nn.Module):
    def __init__(self, args):
        super(CriticNetwork, self).__init__()
        self.FeatureExtractor = LSTMFeatureExtractor(args)
        self.ValueModule = ValueModule(args)
        self.aux_mode = args.aux_mode
        self.GlucoseModel = GlucoseModel(args, args.device)

    def forward(self, s, action, cgm_pred=True, mode='forward'):
        extract_states = self.FeatureExtractor.forward(s)
        value = self.ValueModule.forward(extract_states)
        if cgm_pred:
            cgm_mu, cgm_sigma, cgm = self.GlucoseModel.forward(extract_states, action.detach(), mode)
        else:
            cgm_mu, cgm_sigma, cgm = None, None, None
        return value, cgm_mu, cgm_sigma, cgm


class ActorCritic(nn.Module):
    def __init__(self, args, load, actor_path, critic_path):
        super(ActorCritic, self).__init__()
        self.device = args.device
        self.experiment_dir = args.experiment_dir
        self.Actor = ActorNetwork(args, args.device)
        self.Critic = CriticNetwork(args)
        if load:
            self.Actor = torch.load(actor_path, map_location=self.device)
            self.Critic = torch.load(critic_path, map_location=self.device)
        self.distribution = torch.distributions.Normal

    def get_action(self, s):  # pass values to worker for simulation on cpu.
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)  # add batch dimension
        mu, std, act, log_prob, cgm_mu, cgm_std, cgm = self.Actor(s, mode='forward')
        s_val, _, _, _ = self.Critic(s, action=None, cgm_pred=False)
        data = dict(mu=mu[0], std=std[0], action=act[0], log_prob=log_prob[0], state_value=s_val[0],
                    cgm_mu=cgm_mu[0], cgm_std=cgm_std[0], cgm=cgm[0])
        return {k: v.detach().cpu().numpy() for k, v in data.items()}

    def get_final_value(self, s):  # terminating V(s) of traj
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)  # add batch dimension
        state_value, _, _, _ = self.Critic(s, action=None, cgm_pred=False)
        return state_value[0].detach().cpu().numpy()

    def evaluate_actor(self, state, action, mode="forward"):  # evaluate actor <batch>
        if mode=="aux":
            action_mean, action_std, action, log_prob, cgm_mu, cgm_sigma, cgm = self.Actor(state, action)
            dist = self.distribution(action_mean, action_std)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            return action_logprobs, dist_entropy, cgm_mu, cgm_sigma, cgm
        else:
            action_mean, action_std, _, _, _, _, _ = self.Actor(state)
            dist = self.distribution(action_mean, action_std)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            return action_logprobs, dist_entropy


    def evaluate_critic(self, state, action=None, cgm_pred=False):  # evaluate critic <batch>
        if cgm_pred:
            state_value, cgm_mu, cgm_sigma, cgm = self.Critic(state, action, cgm_pred)
            return state_value, cgm_mu, cgm_sigma, cgm
        else:
            state_value, _, _, _ = self.Critic(state, action, cgm_pred)
            return torch.squeeze(state_value)

    def save(self, episode):  # save checkpoints for networks.
        actor_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_Actor.pth'
        critic_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_Critic.pth'

        torch.save(self.Actor, actor_path)
        mlflow.pytorch.log_model(self.Actor, 'actor_model')

        torch.save(self.Critic, critic_path)
        mlflow.pytorch.log_model(self.Critic, 'critic_model')

