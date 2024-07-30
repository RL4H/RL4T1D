import torch
import torch.nn as nn

from agents.algorithm.agent import Agent
from agents.models.actor_critic import ActorCritic
from utils.onpolicy_buffers import RolloutBuffer
from utils.logger import LogExperiment


class PPO(Agent):
    def __init__(self, args, env_args, load_model, actor_path, critic_path):
        super(PPO, self).__init__(args, env_args=env_args)
        self.args = args
        self.env_args = env_args
        self.device = args.device
        self.completed_interactions = 0

        # training params
        self.train_v_iters = args.n_vf_epochs
        self.train_pi_iters = args.n_pi_epochs
        self.batch_size = args.batch_size
        self.pi_lr = args.pi_lr
        self.vf_lr = args.vf_lr

        # load models and setup optimiser.
        self.policy = ActorCritic(args, load_model, actor_path, critic_path).to(self.device)
        if args.verbose:
            print('PolicyNet Params: {}'.format(sum(p.numel() for p in self.policy.Actor.parameters() if p.requires_grad)))
            print('ValueNet Params: {}'.format(sum(p.numel() for p in self.policy.Critic.parameters() if p.requires_grad)))
        self.optimizer_Actor = torch.optim.Adam(self.policy.Actor.parameters(), lr=self.pi_lr)
        self.optimizer_Critic = torch.optim.Adam(self.policy.Critic.parameters(), lr=self.vf_lr)
        self.value_criterion = nn.MSELoss()

        self.RolloutBuffer = RolloutBuffer(args)
        self.rollout_buffer = {}

        # ppo params
        self.grad_clip = args.grad_clip
        self.entropy_coef = args.entropy_coef
        self.eps_clip = args.eps_clip
        self.target_kl = args.target_kl

        # logging
        self.model_logs = torch.zeros(7, device=self.args.device)
        self.LogExperiment = LogExperiment(args)

    def train_pi(self):
        print('Running Policy Update...')
        temp_loss_log = torch.zeros(1, device=self.device)
        policy_grad, pol_count = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        continue_pi_training, buffer_len = True, self.rollout_buffer['len']
        for i in range(self.train_pi_iters):
            start_idx, n_batch = 0, 0
            while start_idx < buffer_len:
                n_batch += 1
                end_idx = min(start_idx + self.batch_size, buffer_len)

                states_batch = self.rollout_buffer['states'][start_idx:end_idx, :, :]
                actions_batch = self.rollout_buffer['action'][start_idx:end_idx, :]
                logprobs_batch = self.rollout_buffer['log_prob_action'][start_idx:end_idx, :]
                advantages_batch = self.rollout_buffer['advantage'][start_idx:end_idx]
                
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-5)

                self.optimizer_Actor.zero_grad()
                logprobs_prediction, dist_entropy = self.policy.evaluate_actor(states_batch, actions_batch)
                ratios = torch.exp(logprobs_prediction - logprobs_batch)
                ratios = ratios.squeeze()
                r_theta = ratios * advantages_batch
                r_theta_clip = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch
                policy_loss = -torch.min(r_theta, r_theta_clip).mean() - self.entropy_coef * dist_entropy.mean()
                # constraint_loss  = 
                # early stop: approx kl calculation
                log_ratio = logprobs_prediction - logprobs_batch
                approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).detach().cpu().numpy()
                if approx_kl > 1.5 * self.target_kl:
                    if self.args.verbose:
                        print('Early stop => Epoch {}, Batch {}, Approximate KL: {}.'.format(i, n_batch, approx_kl))
                    continue_pi_training = False
                    break
                if torch.isnan(policy_loss):  # for debugging only!
                    print('policy loss: {}'.format(policy_loss))
                    exit()
                temp_loss_log += policy_loss.detach()
                policy_loss.backward()
                policy_grad += torch.nn.utils.clip_grad_norm_(self.policy.Actor.parameters(), self.grad_clip)  # clip gradients before optimising
                pol_count += 1
                self.optimizer_Actor.step()
                start_idx += self.batch_size

            if not continue_pi_training:
                break
        mean_pi_grad = policy_grad / pol_count if pol_count != 0 else 0
        print('The policy loss is: {}'.format(temp_loss_log))
        return mean_pi_grad, temp_loss_log

    def train_vf(self):
        print('Running Value Function Update...')

        # variables to be logged for debugging purposes.
        val_loss_log, value_grad = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        true_var, explained_var = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        val_count = torch.zeros(1, device=self.device)

        for i in range(self.train_v_iters):
            start_idx = 0
            while start_idx < self.rollout_buffer['len']:
                end_idx = min(start_idx + self.batch_size, self.rollout_buffer['len'])

                states_batch = self.rollout_buffer['states'][start_idx:end_idx, :, :]
                value_target = self.rollout_buffer['value_target'][start_idx:end_idx]

                self.optimizer_Critic.zero_grad()
                value_prediction = self.policy.evaluate_critic(states_batch)
                value_loss = self.value_criterion(value_prediction, value_target)
                value_loss.backward()
                value_grad += torch.nn.utils.clip_grad_norm_(self.policy.Critic.parameters(), self.grad_clip)  # clip gradients before optimising
                self.optimizer_Critic.step()
                val_count += 1
                start_idx += self.batch_size

                # logging.
                val_loss_log += value_loss.detach()
                y_pred = value_prediction.detach().flatten()
                y_true = value_target.flatten()
                var_y = torch.var(y_true)
                true_var += var_y
                explained_var += 1 - torch.var(y_true - y_pred) / (var_y + 1e-5)

        return value_grad / val_count, val_loss_log, explained_var / val_count, true_var / val_count

    def update(self):
        self.rollout_buffer = self.RolloutBuffer.prepare_rollout_buffer()
        self.model_logs[0], self.model_logs[5] = self.train_pi()
        self.model_logs[1], self.model_logs[2], self.model_logs[3], self.model_logs[4] = self.train_vf()
        self.LogExperiment.save(log_name='/model_log', data=[self.model_logs.detach().cpu().flatten().numpy()])


