import torch
import torch.nn as nn

from agents.agent import Agent
from agents.models.actor_critic import ActorCritic
from utils.onpolicy_buffers import RolloutBuffer
from utils.logger import LogExperiment


class A2C(Agent):
    def __init__(self, args, load_model, actor_path, critic_path):
        super(A2C, self).__init__(args)
        self.args = args
        self.device = args.device
        self.completed_interactions = 0

        # training params
        self.train_v_iters = args.n_vf_epochs
        self.train_pi_iters = args.n_pi_epochs
        self.batch_size = args.batch_size
        self.pi_lr = args.pi_lr
        self.vf_lr = args.vf_lr
        self.grad_clip = args.grad_clip
        self.entropy_coef = args.entropy_coef

        # load models and setup optimiser.
        self.policy = ActorCritic(args, load_model, actor_path, critic_path).to(self.device)

        self.optimizer_Actor = torch.optim.Adam(self.policy.Actor.parameters(), lr=self.pi_lr)
        self.optimizer_Critic = torch.optim.Adam(self.policy.Critic.parameters(), lr=self.vf_lr)
        self.value_criterion = nn.MSELoss()

        self.RolloutBuffer = RolloutBuffer(args)
        self.rollout_buffer = {}

        # logging
        self.model_logs = torch.zeros(7, device=self.args.device)
        self.LogExperiment = LogExperiment(args)

    def train_pi(self):
        print('Running pi update...')
        temp_loss_log = torch.zeros(1, device=self.device)
        policy_grad, pol_count = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        continue_pi_training, buffer_len = True, self.rollout_buffer['len']

        start_idx, end_idx= 0, buffer_len
        old_states_batch = self.rollout_buffer['states'][start_idx:end_idx, :, :]
        old_actions_batch = self.rollout_buffer['action'][start_idx:end_idx, :]
        advantages_batch = self.rollout_buffer['advantage'][start_idx:end_idx]
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-5)

        self.optimizer_Actor.zero_grad()
        logprobs, dist_entropy = self.policy.evaluate_actor(old_states_batch, old_actions_batch)
        policy_loss = -(logprobs.squeeze() * advantages_batch).mean() - self.entropy_coef * dist_entropy.mean()

        temp_loss_log += policy_loss.detach()
        policy_loss.backward()
        policy_grad += torch.nn.utils.clip_grad_norm_(self.policy.Actor.parameters(), self.grad_clip)
        pol_count += 1
        self.optimizer_Actor.step()
        start_idx += self.batch_size

        mean_pi_grad = policy_grad / pol_count if pol_count != 0 else 0
        print('The policy loss is: {}'.format(temp_loss_log))
        return mean_pi_grad, temp_loss_log

    def train_vf(self):
        print('Running vf update...')
        explained_var = torch.zeros(1, device=self.device)
        val_loss_log = torch.zeros(1, device=self.device)
        val_count = torch.zeros(1, device=self.device)
        value_grad = torch.zeros(1, device=self.device)
        true_var = torch.zeros(1, device=self.device)
        buffer_len = self.rollout_buffer['len']
        start_idx, end_idx = 0, buffer_len

        old_states_batch = self.rollout_buffer['states'][start_idx:end_idx, :, :]
        returns_batch = self.rollout_buffer['value_target'][start_idx:end_idx]

        self.optimizer_Critic.zero_grad()
        state_values = self.policy.evaluate_critic(old_states_batch)
        value_loss = self.value_criterion(state_values, returns_batch)
        value_loss.backward()
        value_grad += torch.nn.utils.clip_grad_norm_(self.policy.Critic.parameters(), self.grad_clip)
        self.optimizer_Critic.step()
        val_count += 1
        # logging.
        val_loss_log += value_loss.detach()
        y_pred = state_values.detach().flatten()
        y_true = returns_batch.flatten()
        var_y = torch.var(y_true)
        true_var += var_y
        explained_var += 1 - torch.var(y_true - y_pred) / (var_y + 1e-5)

        return value_grad / val_count, val_loss_log, explained_var / val_count, true_var / val_count

    def update(self):
        self.rollout_buffer = self.RolloutBuffer.prepare_rollout_buffer()
        self.model_logs[0], self.model_logs[5] = self.train_pi()
        self.model_logs[1], self.model_logs[2], self.model_logs[3], self.model_logs[4]  = self.train_vf()
        self.LogExperiment.save(log_name='/model_log', data=[self.model_logs.detach().cpu().flatten().numpy()])


