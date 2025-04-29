import torch
import torch.nn as nn

import numpy as np

from agents.algorithm.agent import Agent
from agents.models.actor_critic_td3_bc import ActorCritic

from decouple import config
import sys

import csv
from collections import namedtuple, deque


MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

Transition = namedtuple('Transition', ('state', 'feat', 'action', 'reward', 'next_state', 'next_feat', 'done'))


DEFAULT_FEAT = 0

class TD3_BC(Agent):
    def __init__(self, args, env_args, logger, load_model, actor_path, critic_path):
        print("features:",args.n_features,args.custom_arg)
        super(TD3_BC, self).__init__(args, env_args=env_args, logger=logger, type="Offline")
        self.device = args.device
        self.completed_interactions = 0

        # training params
        self.train_v_iters = args.n_vf_epochs
        self.train_pi_iters = args.n_pi_epochs
        self.batch_size = args.batch_size
        self.pi_lr = args.pi_lr
        self.vf_lr = args.vf_lr
        self.alpha = 2.5 #FIXME make arg for this

        ### TD3 Params
        self.n_step = args.n_step
        self.feature_history = args.feature_history
        self.n_handcrafted_features = args.n_handcrafted_features
        self.grad_clip = args.grad_clip

        self.gamma = args.gamma
        self.n_training_workers = args.n_training_workers
        self.n_testing_workers = args.n_testing_workers
        self.device = self.device

        self.replay_buffer_size = args.replay_buffer_size
        self.batch_size = args.batch_size
        self.sample_size = args.sample_size

        self.target_update_interval = 2  # 100
        self.n_updates = 0

        self.soft_tau = args.soft_tau
        self.train_pi_iters = args.n_pi_epochs
        self.shuffle_rollout = args.shuffle_rollout
        # self.soft_q_lr = args.vf_lr
        self.value_lr = args.vf_lr
        self.policy_lr = args.pi_lr
        self.grad_clip = args.grad_clip

        # self.mu_penalty = args.mu_penalty
        # self.action_penalty_limit = args.action_penalty_limit
        # self.action_penalty_coef = args.action_penalty_coef

        # self.replay_buffer_type = args.replay_buffer_type
        # self.replay_buffer_alpha = args.replay_buffer_alpha
        # self.replay_buffer_beta = args.replay_buffer_beta
        # self.replay_buffer_temporal_decay = args.replay_buffer_temporal_decay


        self.weight_decay = 0


        ### TD3 networks:
        self.policy = ActorCritic(args, load_model, actor_path, critic_path, self.device).to(self.device)

        self.value_criterion1 = nn.MSELoss()
        self.value_criterion2 = nn.MSELoss()
        self.value_optimizer1 = torch.optim.Adam(self.policy.value_net1.parameters(), lr=self.value_lr, weight_decay=self.weight_decay)
        self.value_optimizer2 = torch.optim.Adam(self.policy.value_net2.parameters(), lr=self.value_lr, weight_decay=self.weight_decay)

        self.policy_optimizer = torch.optim.Adam(self.policy.policy_net.parameters(), lr=self.policy_lr, weight_decay=self.weight_decay)
        for target_param, param in zip(self.policy.policy_net.parameters(), self.policy.policy_net_target.parameters()):
            target_param.data.copy_(param.data)

        for p in self.policy.policy_net_target.parameters():
            p.requires_grad = False

        for target_param, param in zip(self.policy.value_net1.parameters(), self.policy.value_net_target1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.policy.value_net2.parameters(), self.policy.value_net_target2.parameters()):
            target_param.data.copy_(param.data)

        for p in self.policy.value_net_target1.parameters():
            p.requires_grad = False

        for p in self.policy.value_net_target2.parameters():
            p.requires_grad = False

        print('Policy Parameters: {}'.format(sum(p.numel() for p in self.policy.policy_net.parameters() if p.requires_grad)))
        print('Value network 1 Parameters: {}'.format(sum(p.numel() for p in self.policy.value_net1.parameters() if p.requires_grad)))
        print('Value network 2 Parameters: {}'.format(sum(p.numel() for p in self.policy.value_net2.parameters() if p.requires_grad)))
        

        # readout
        print("Setting up offline Agent")
        print(f"Using {args.data_type} data.")

    def update(self):
        print("Running network update...")

        cl, pl = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        pi_grad, val_grad = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)

        for i in range(self.train_pi_iters):
            # sample from buffer
            transitions = self.buffer.sample(self.sample_size)
            

            batch = Transition(*zip(*transitions))
            cur_state_batch = torch.cat(batch.state)
            cur_feat_batch = torch.cat(batch.feat)
            actions_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward).unsqueeze(1)
            next_state_batch = torch.cat(batch.next_state)
            next_feat_batch = torch.cat(batch.next_feat)
            done_batch = torch.cat(batch.done).unsqueeze(1)

            # value network update
            new_action, next_log_prob = self.policy.evaluate_target_policy_noise(next_state_batch, next_feat_batch)
            next_values = torch.min(self.policy.value_net_target1(next_state_batch, next_feat_batch, new_action),
                                    self.policy.value_net_target2(next_state_batch, next_feat_batch, new_action))
            target_value = (reward_batch + (self.gamma * (1 - done_batch) * next_values))

            predicted_value1 = self.policy.value_net1(cur_state_batch, cur_feat_batch, actions_batch)
            predicted_value2 = self.policy.value_net2(cur_state_batch, cur_feat_batch, actions_batch)

            value_loss1 = self.value_criterion1(target_value.detach(), predicted_value1)
            value_loss2 = self.value_criterion2(target_value.detach(), predicted_value2)
            
            self.value_optimizer1.zero_grad()
            self.value_optimizer2.zero_grad()

            value_loss1.backward()
            value_loss2.backward()

            self.value_optimizer1.step()
            self.value_optimizer2.step()

            cl += value_loss1.detach()

            for param in self.policy.value_net1.parameters():
                if param.grad is not None:
                    val_grad += param.grad.sum()

            for param in self.policy.value_net2.parameters():
                if param.grad is not None:
                    val_grad += param.grad.sum()

            self.n_updates += 1

            # actor update
            if self.n_updates % self.target_update_interval == 0:
                # freeze value networks save compute: ref: openai:
                for p in self.policy.value_net1.parameters():
                    p.requires_grad = False
                for p in self.policy.value_net2.parameters():
                    p.requires_grad = False

                for p in self.policy.value_net_target1.parameters():
                    p.requires_grad = False
                for p in self.policy.value_net_target2.parameters():
                    p.requires_grad = False

                # evaluate action taken by policy, in a batch
                policy_action, _ = self.policy.evaluate_policy_no_noise(cur_state_batch, cur_feat_batch)

                # take minimum evaluation by critics
                policy_loss = torch.min(
                    self.policy.value_net1(cur_state_batch, cur_feat_batch, policy_action), 
                    self.policy.value_net2(cur_state_batch, cur_feat_batch, policy_action)
                )

                # evaluate mean of q values
                q_mean = policy_loss.mean()

                # assign lambda constant to scale correctly
                lmbda = self.alpha / ( policy_loss.abs().mean() )

                # calculate policy loss, ref: Fujimoto and Gu (2021)
                policy_loss = -lmbda * q_mean + nn.functional.mse_loss(policy_action,actions_batch)

                # perform optimisation
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                pl += policy_loss.detach()
                pi_grad += torch.nn.utils.clip_grad_norm_(self.policy.policy_net.parameters(), 10)

                # save compute: ref: openai:
                for p in self.policy.value_net1.parameters():
                    p.requires_grad = True
                for p in self.policy.value_net2.parameters():
                    p.requires_grad = True

                for p in self.policy.value_net_target1.parameters():
                    p.requires_grad = True
                for p in self.policy.value_net_target2.parameters():
                    p.requires_grad = True

                # Update target networks
                with torch.no_grad():
                    print("################updated target networks")
                    for param, target_param in zip(self.policy.value_net1.parameters(), self.policy.value_net_target1.parameters()):
                        target_param.data.mul_((1 - self.soft_tau))
                        target_param.data.add_(self.soft_tau * param.data)
                    for param, target_param in zip(self.policy.value_net2.parameters(), self.policy.value_net_target2.parameters()):
                        target_param.data.mul_((1 - self.soft_tau))
                        target_param.data.add_(self.soft_tau * param.data)

                    for param, target_param in zip(self.policy.policy_net.parameters(), self.policy.policy_net_target.parameters()):
                        target_param.data.mul_((1 - self.soft_tau))
                        target_param.data.add_(self.soft_tau * param.data)

        # logging
        data = dict(policy_grad=pi_grad, policy_loss=pl, coeff_loss=cl, value_gradient=val_grad)
        return {k: v.detach().cpu().flatten().numpy()[0] for k, v in data.items()}

