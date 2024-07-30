import torch
import torch.nn as nn


from agents.models.feature_extracter import LSTMFeatureExtractor
from agents.models.policy import PolicyModule
from agents.models.value import ValueModule


class ActorNetwork(nn.Module):
    def __init__(self, args):
        super(ActorNetwork, self).__init__()
        self.FeatureExtractor = LSTMFeatureExtractor(args)
        self.PolicyModule = PolicyModule(args)

    def forward(self, s):
        lstmOut = self.FeatureExtractor.forward(s)
        mu, sigma, action, log_prob = self.PolicyModule.forward(lstmOut)
        return mu, sigma, action, log_prob


class CriticNetwork(nn.Module):
    def __init__(self, args):
        super(CriticNetwork, self).__init__()
        self.FeatureExtractor = LSTMFeatureExtractor(args)
        self.ValueModule = ValueModule(args)

    def forward(self, s):
        lstmOut = self.FeatureExtractor.forward(s)
        value = self.ValueModule.forward(lstmOut)
        return value


class ActorCritic(nn.Module):
    def __init__(self, args, load, actor_path, critic_path):
        super(ActorCritic, self).__init__()
        self.device = args.device
        self.experiment_dir = args.experiment_dir
        self.Actor = ActorNetwork(args)
        self.Critic = CriticNetwork(args)
        self.Critic = CriticNetwork(args)
        if load:
            self.Actor = torch.load(actor_path, map_location=self.device)
            self.Critic = torch.load(critic_path, map_location=self.device)
        self.distribution = torch.distributions.Normal

    def get_action(self, s):  # pass values to worker for simulation on cpu.
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)  # add batch dimension
        mu, std, act, log_prob = self.Actor(s)
        s_val = self.Critic(s)
        data = dict(mu=mu[0], std=std[0], action=act[0], log_prob=log_prob[0], state_value=s_val[0])
        return {k: v.detach().cpu().numpy() for k, v in data.items()}

    def get_final_value(self, s):  # terminating V(s) of traj
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)  # add batch dimension
        state_value = self.Critic(s)
        return state_value[0].detach().cpu().numpy()

    def evaluate_actor(self, state, action):  # evaluate actor <batch>
        action_mean, action_std, _, _ = self.Actor(state)
        dist = self.distribution(action_mean, action_std)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

    def evaluate_critic(self, state):  # evaluate critic <batch>
        state_value = self.Critic(state)
        return torch.squeeze(state_value)

    def save(self, episode):  # save checkpoints for networks.
        actor_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_Actor.pth'
        critic_path = self.experiment_dir + '/checkpoints/episode_' + str(episode) + '_Critic.pth'
        torch.save(self.Actor, actor_path)
        torch.save(self.Critic, critic_path)

    def get_fim(self, x): 
        # calculate fisher information matrix and return matrix, mu, info
        #pdb.set_trace()
        mean, _, _ = self.forward(x)
        #vec of len = No. of states*size of action e.g. cov_inv.shape = 2085*6
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0)) 
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        #pdb.set_trace()
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}


