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
    
    def get_fim(self, x):
        mu, sigma, _, _ = self.forward(x)

        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(0)

        cov_inv = sigma.pow(-2).repeat(x.size(0), 1)

        param_count = 0
        std_index = 0
        id = 0
        std_id = id
        for name, param in self.named_parameters():
            if name == "sigma.weight":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1

        return cov_inv.detach(), mu, {'std_id': std_id, 'std_index': std_index}
    
    def get_kl(self, x):
        mu1, sigma1, _, _ = self.forward(x)

        mu0 = mu1.detach()
        sigma0 = sigma1.detach()

        kl = sigma1.log() - sigma0.log() + (sigma0.pow(2) + (mu0 - mu1).pow(2)) / (2.0 * sigma1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)


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

