import torch
import torch.nn as nn

from agents.algorithm.agent import Agent
from utils.offpolicy_buffers import ReplayMemory, Transition
from agents.models.actor_critic_sac import ActorCritic

from utils.logger import LogExperiment


class SAC(Agent):
    def __init__(self, args, env_args, load_model, actor_path, critic_path):
        super(SAC, self).__init__(args, env_args=env_args, type="OffPolicy")

        self.args = args
        self.gamma = args.gamma
        self.device = args.device
        self.n_step = args.n_step

        self.grad_clip = args.grad_clip

        self.replay_buffer_size = args.replay_buffer_size if not args.debug else 1024
        self.batch_size = args.batch_size if not args.debug else 32
        self.sample_size = args.sample_size if not args.debug else 64

        self.target_update_interval = 1  # 100
        self.n_updates = 0

        self.soft_tau = args.soft_tau
        self.train_pi_iters = args.n_pi_epochs

        self.vf_lr = args.vf_lr
        self.pi_lr = args.pi_lr
        self.grad_clip = args.grad_clip

        self.shuffle_rollout = args.shuffle_rollout

        self.entropy_coef = args.entropy_coef
        self.target_entropy = args.target_entropy
        self.entropy_lr = args.entropy_lr
        self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * self.entropy_coef).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.entropy_lr)
        self.sac_v2 = args.sac_v2
        self.weight_decay = args.weight_decay

        self.policy = ActorCritic(args, load_model, actor_path, critic_path, args.device).to(self.device)
        self.buffer = ReplayMemory(self.replay_buffer_size)

        print('Policy Parameters: {}'.format(sum(p.numel() for p in self.policy.policy_net.parameters() if p.requires_grad)))
        print('Q1 Parameters: {}'.format(sum(p.numel() for p in self.policy.soft_q_net1.parameters() if p.requires_grad)))
        print('Q2 Parameters: {}'.format(sum(p.numel() for p in self.policy.soft_q_net2.parameters() if p.requires_grad)))
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        self.soft_q_optimizer1 = torch.optim.Adam(self.policy.soft_q_net1.parameters(), lr=self.vf_lr, weight_decay=self.weight_decay)
        self.soft_q_optimizer2 = torch.optim.Adam(self.policy.soft_q_net2.parameters(), lr=self.vf_lr, weight_decay=self.weight_decay)
        self.policy_optimizer = torch.optim.Adam(self.policy.policy_net.parameters(), lr=self.pi_lr, weight_decay=self.weight_decay)

        if self.sac_v2:
            for target_param, param in zip(self.policy.target_q_net1.parameters(), self.policy.soft_q_net1.parameters()):
                target_param.data.copy_(param.data)
            for target_param, param in zip(self.policy.target_q_net2.parameters(), self.policy.soft_q_net2.parameters()):
                target_param.data.copy_(param.data)
            for p in self.policy.target_q_net1.parameters():
                p.requires_grad = False
            for p in self.policy.target_q_net2.parameters():
                p.requires_grad = False
        else:
            self.value_criterion = nn.MSELoss()
            self.value_optimizer = torch.optim.Adam(self.policy.value_net.parameters(), lr=self.vf_lr, weight_decay=self.weight_decay)
            for target_param, param in zip(self.policy.value_net_target.parameters(), self.policy.value_net.parameters()):
                target_param.data.copy_(param.data)
            for p in self.policy.value_net_target.parameters():
                p.requires_grad = False

        # logging
        self.model_logs = torch.zeros(9, device=self.args.device)
        self.LogExperiment = LogExperiment(self.args)


    def update(self):
        if len(self.buffer) < self.sample_size * 10:
            return

        print('Running network update...')
        cl, pl, ql1, ql2, count = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), \
                                  torch.zeros(1,device=self.device), \
                                  torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        pi_grad, q1_grad, q2_grad, coeff_grad = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), \
                                                torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)

        for i in range(self.train_pi_iters):
            # sample from buffer

            transitions = self.buffer.sample(self.sample_size)
            batch = Transition(*zip(*transitions))

            cur_state_batch = torch.cat(batch.state)
            actions_batch = torch.cat(batch.action).unsqueeze(1)
            reward_batch = torch.cat(batch.reward).unsqueeze(1)
            next_state_batch = torch.cat(batch.next_state)
            done_batch = torch.cat(batch.done).unsqueeze(1)

            actions_pi, log_prob = self.policy.evaluate_policy(cur_state_batch)
            self.entropy_coef = torch.exp(self.log_ent_coef.detach()) if self.sac_v2 else 0.001

            # value network update
            if not self.sac_v2:
                self.value_optimizer.zero_grad()
                with torch.no_grad():
                    min_qf_val = torch.min(self.policy.soft_q_net1(cur_state_batch, actions_pi),
                                           self.policy.soft_q_net2(cur_state_batch, actions_pi))
                predicted_value = self.policy.value_net(cur_state_batch)
                value_func_estimate = min_qf_val - (self.entropy_coef * log_prob)  # todo the temperature paramter
                value_loss = 0.5 * self.value_criterion(predicted_value, value_func_estimate.detach())
                value_loss.backward()
                coeff_grad += torch.nn.utils.clip_grad_norm_(self.policy.value_net.parameters(), self.grad_clip)
                self.value_optimizer.step()
                cl += value_loss.detach()

            # q network update
            self.soft_q_optimizer1.zero_grad()
            self.soft_q_optimizer2.zero_grad()
            with torch.no_grad():  # calculate the target q vals here.
                if self.sac_v2:
                    new_action, next_log_prob = self.policy.evaluate_policy(next_state_batch)
                    next_q_values = torch.min(self.policy.target_q_net1(next_state_batch, new_action),
                                              self.policy.target_q_net2(next_state_batch, new_action))
                    next_q_values = next_q_values - self.entropy_coef * next_log_prob
                    target_q_values = (reward_batch + (self.gamma * (1 - done_batch) * next_q_values))
                else:
                    target_value = self.policy.value_net_target(next_state_batch)
                    target_q_values = (reward_batch + self.gamma * (1 - done_batch) * target_value)

            predicted_q_value1 = self.policy.soft_q_net1(cur_state_batch, actions_batch)
            predicted_q_value2 = self.policy.soft_q_net2(cur_state_batch, actions_batch)
            q_value_loss1 = 0.5 * self.soft_q_criterion1(predicted_q_value1, target_q_values)
            q_value_loss2 = 0.5 * self.soft_q_criterion2(predicted_q_value2, target_q_values)
            q_value_loss1.backward()
            q1_grad += torch.nn.utils.clip_grad_norm_(self.policy.soft_q_net1.parameters(), self.grad_clip)
            self.soft_q_optimizer1.step()
            q_value_loss2.backward()
            q2_grad += torch.nn.utils.clip_grad_norm_(self.policy.soft_q_net2.parameters(), self.grad_clip)
            self.soft_q_optimizer2.step()

            # actor update : next q values
            # freeze q networks save compute: ref: openai:
            for p in self.policy.soft_q_net1.parameters():
                p.requires_grad = False
            for p in self.policy.soft_q_net2.parameters():
                p.requires_grad = False

            self.policy_optimizer.zero_grad()
            min_qf_pi = torch.min(self.policy.soft_q_net1(cur_state_batch, actions_pi),
                                  self.policy.soft_q_net2(cur_state_batch, actions_pi))

            policy_loss = (self.entropy_coef * log_prob - min_qf_pi).mean()
            policy_loss.backward()
            pi_grad += torch.nn.utils.clip_grad_norm_(self.policy.policy_net.parameters(), 10)
            self.policy_optimizer.step()

            # save compute: ref: openai:
            for p in self.policy.soft_q_net1.parameters():
                p.requires_grad = True
            for p in self.policy.soft_q_net2.parameters():
                p.requires_grad = True

            # entropy coeff update
            if self.sac_v2:
                self.ent_coef_optimizer.zero_grad()
                _, log_prob = self.policy.evaluate_policy(cur_state_batch)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_loss.backward()
                coeff_grad += torch.nn.utils.clip_grad_norm_([self.log_ent_coef], self.grad_clip)
                self.ent_coef_optimizer.step()
                cl += ent_coef_loss.detach()

            self.n_updates += 1

            if self.n_updates % self.target_update_interval == 0:
                with torch.no_grad():
                    print("################updated target")
                    if self.sac_v2:
                        for param, target_param in zip(self.policy.soft_q_net1.parameters(), self.policy.target_q_net1.parameters()):
                            target_param.data.mul_((1 - self.soft_tau))
                            target_param.data.add_(self.soft_tau * param.data)
                        for param, target_param in zip(self.policy.soft_q_net2.parameters(), self.policy.target_q_net2.parameters()):
                            target_param.data.mul_((1 - self.soft_tau))
                            target_param.data.add_(self.soft_tau * param.data)
                    else:
                        for param, target_param in zip(self.policy.value_net.parameters(), self.policy.value_net_target.parameters()):
                            target_param.data.mul_((1 - self.soft_tau))
                            target_param.data.add_(self.soft_tau * param.data)

            pl += policy_loss.detach()
            ql1 += q_value_loss1.detach()
            ql2 += q_value_loss2.detach()

        # logging
        self.model_logs[0] = cl  # value loss or coeff loss
        self.model_logs[1] = pl
        self.model_logs[2] = ql1
        self.model_logs[3] = ql2
        self.model_logs[4] = self.entropy_coef
        self.model_logs[5] = pi_grad
        self.model_logs[6] = q1_grad
        self.model_logs[7] = q2_grad
        self.model_logs[8] = coeff_grad  # value loss grad or coeff loss grad
        print('success')


