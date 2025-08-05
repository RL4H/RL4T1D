import torch
import torch.nn as nn

import numpy as np
import math

from agents.algorithm.agent import Agent
from agents.models.actor_critic_td3_bc import ActorCritic

from decouple import config
import sys

import csv
from collections import namedtuple, deque


MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


DEFAULT_FEAT = 0

def depack(*args): return args

class TD3_BC(Agent):
    def __init__(self, args, env_args, logger, load_model, actor_path, critic_path):
        super(TD3_BC, self).__init__(args, env_args=env_args, logger=logger, type="Offline")
        self.device = args.device
        self.completed_interactions = 0

        # training params
        self.train_pi_iters = args.n_pi_epochs
        self.pi_lr = args.pi_lr
        self.vf_lr = args.vf_lr
        self.alpha = args.alpha
        self.beta = args.beta
        self.pi_lambda = args.pi_lambda

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
        self.mini_batch_size = args.mini_batch_size
        self.mini_batch_num = args.batch_size // args.mini_batch_size

        self.target_update_interval = args.target_update_interval  # 100
        self.n_updates = 0

        self.soft_tau = args.soft_tau
        self.shuffle_rollout = args.shuffle_rollout
        self.preserve_trajectories = args.preserve_trajectories
        self.value_lr = args.vf_lr
        self.policy_lr = args.pi_lr
        self.grad_clip = args.grad_clip


        self.weight_decay = args.vf_lambda


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

    def direct_sample_buffer(self, n):
        transitions_cpu = self.buffer_queue.pop_batch(n) #import data
        # transitions = [Transition( *(torch.as_tensor([arg], dtype=torch.float32, device=self.args.device) for arg in depack(*transition)) ) for transition in transitions_cpu]#move data to gpu
        # del transitions_cpu
        batch = Transition(*zip(*transitions_cpu))

        cur_state_batch = torch.tensor(batch.state, dtype=torch.float32, device=self.args.device)
        actions_batch = torch.tensor(batch.action, dtype=torch.float32, device=self.args.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.args.device).unsqueeze(1)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32, device=self.args.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.args.device).unsqueeze(1)

        return cur_state_batch, actions_batch, reward_batch, next_state_batch, done_batch
    
    def update(self):
        print("Running network update...")

        cl, pl = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        pi_grad, val_grad = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        vf_loss = torch.zeros(1, device=self.device)
        q_abs_list = []
        bc_loss_list = []
        td3_loss_list = []
        vf_completed_iters = pi_completed_iters = 0
        for pi_train_iter in range(self.train_pi_iters):
            vf_completed_iters += 1
            # cur_state_batch, actions_batch, reward_batch, next_state_batch, done_batch = self.buffer.sample(self.mini_batch_size)
            transitions = self.buffer.sample(self.mini_batch_size)

            batch_size = self.mini_batch_size

            batch = Transition(*zip(*transitions))
            cur_state_batch = torch.cat(batch.state)
            actions_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward).unsqueeze(1)
            next_state_batch = torch.cat(batch.next_state)
            done_batch = torch.cat(batch.done).unsqueeze(1)

            # value network update
            with torch.no_grad():
                new_action, next_log_prob = self.policy.evaluate_target_policy_noise(next_state_batch)

                next_values = torch.min(self.policy.value_net_target1(next_state_batch, new_action),
                                        self.policy.value_net_target2(next_state_batch, new_action))

                target_value = (reward_batch + (self.gamma * (1 - done_batch) * next_values))

            print(reward_batch[:5], target_value[:5], done_batch[:5],"\n")

            # critic 1 optimisation
            predicted_value1 = self.policy.value_net1(cur_state_batch, actions_batch)
            value_loss1 = self.value_criterion1(predicted_value1, target_value)
            self.value_optimizer1.zero_grad()
            value_loss1.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.value_net1.parameters(), 1) #clip value gradients
            self.value_optimizer1.step()

            # critic 2 optimisation
            predicted_value2 = self.policy.value_net2(cur_state_batch, actions_batch)
            value_loss2 = self.value_criterion2(predicted_value2, target_value)
            self.value_optimizer2.zero_grad()
            value_loss2.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.value_net2.parameters(), 1)
            self.value_optimizer2.step()

            vf_loss += (value_loss1).detach() / self.train_pi_iters

            for param in self.policy.value_net1.parameters():
                if param.grad is not None:
                    val_grad += param.grad.sum() / self.train_pi_iters

            for param in self.policy.value_net2.parameters():
                if param.grad is not None:
                    val_grad += param.grad.sum() / self.train_pi_iters


            # actor update

            if pi_train_iter % self.target_update_interval == 0 and self.completed_interactions >= self.args.vf_pretrain_iters:
                pi_completed_iters += 1
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
                policy_action, _ = self.policy.evaluate_policy_no_noise(cur_state_batch)

                # take minimum evaluation by critics
                critic_eval = torch.min(
                    self.policy.value_net1(cur_state_batch, policy_action), 
                    self.policy.value_net2(cur_state_batch, policy_action)
                )

                # evaluate mean of q values
                q_mean = critic_eval.mean() 

                # assign lambda constant to scale correctly

                # calculate policy loss, ref: Fujimoto and Gu (2021)
                reg_term = sum(torch.norm(param, p=2)**2 for param in self.policy.policy_net.parameters() if param.requires_grad)

                # alpha_adj = (self.alpha / critic_eval.abs().mean().clamp(min=0.1, max=10.0)).detach()
                alpha, beta = (self.alpha, self.beta) if self.completed_interactions >= self.args.bc_pretrain_iters else (0, 1)
                lmbda = (beta / critic_eval.abs().mean().clamp(min=0.1, max=10.0)).detach()

                policy_loss = -alpha * q_mean + lmbda * nn.functional.mse_loss(policy_action, actions_batch.detach())

                self.policy_optimizer.zero_grad()
                policy_loss.backward() 
                torch.nn.utils.clip_grad_norm_(self.policy.policy_net.parameters(), self.args.pi_clip) #clip policy gradient #TODO: decide if 20 or 10

                pi_grad += torch.norm(torch.stack([
                    p.grad.norm(2) for p in self.policy.policy_net.parameters() if p.grad is not None
                ]))

                self.policy_optimizer.step()

                # perform optimisation for actor
                pl += policy_loss.item() 
                q_abs_list.append( critic_eval.abs().mean().item() )
                bc_loss_list.append((lmbda * nn.functional.mse_loss(policy_action,actions_batch.detach())).item())
                td3_loss_list.append((-alpha * q_mean ).item())

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
                    # print("################updated target networks")
                    for param, target_param in zip(self.policy.value_net1.parameters(), self.policy.value_net_target1.parameters()):
                        target_param.data.mul_((1 - self.soft_tau))
                        target_param.data.add_(self.soft_tau * param.data)
                    for param, target_param in zip(self.policy.value_net2.parameters(), self.policy.value_net_target2.parameters()):
                        target_param.data.mul_((1 - self.soft_tau))
                        target_param.data.add_(self.soft_tau * param.data)

                    for param, target_param in zip(self.policy.policy_net.parameters(), self.policy.policy_net_target.parameters()):
                        target_param.data.mul_((1 - self.soft_tau))
                        target_param.data.add_(self.soft_tau * param.data)
            #     print("\t############ Policy Network Updated")
            # print("################ updated target networks")
        print(f"################ updated value networks {vf_completed_iters} times and policy network {pi_completed_iters} times. Average abs q: {np.mean(q_abs_list)}, BC loss: {np.mean(bc_loss_list)}, TD3 loss: {np.mean(td3_loss_list)}" + (". Used BC default" if self.completed_interactions < self.args.bc_pretrain_iters else ""))
        # logging
        data = dict(policy_grad=pi_grad, policy_loss=pl, coeff_loss=cl, value_grad=val_grad, val_loss=vf_loss)
        return {k: v.detach().cpu().flatten().numpy()[0] for k, v in data.items()}

    def minibatch_update(self):
        pass

    def finetune_critics(self, fqe_epochs=20):

        for _ in range(fqe_epochs):

            cur_state_batch, actions_batch, reward_batch, next_state_batch, done_batch = self.sample_buffer(self.mini_batch_size)

            # value network update
            new_action, next_log_prob = self.policy.evaluate_policy_no_noise(next_state_batch)

            next_values = torch.min(self.policy.value_net_target1(next_state_batch, new_action),
                                    self.policy.value_net_target2(next_state_batch, new_action))

            target_value = (reward_batch + (self.gamma * (1 - done_batch) * next_values))

            predicted_value1 = self.policy.value_net1(cur_state_batch, actions_batch)
            predicted_value2 = self.policy.value_net2(cur_state_batch, actions_batch)

            value_loss1 = self.value_criterion1(reward_batch, predicted_value1)
            value_loss2 = self.value_criterion2(reward_batch, predicted_value2)

            self.value_optimizer1.zero_grad()
            self.value_optimizer2.zero_grad()

            value_loss1.backward()
            value_loss2.backward()

            self.value_optimizer1.step()
            self.value_optimizer2.step()

            vf_loss += (value_loss1).detach() 

            print("################ updated critic networks")
    
    def evaluate_fqe(self, fqe_states=100000):
        v_data = []
        for _ in range(fqe_states):
            cur_state_batch, actions_batch, reward_batch, next_state_batch, done_batch = self.sample_buffer(self.mini_batch_size)

            min_critic_value = list(torch.min( self.policy.value_net1(cur_state_batch, actions_batch), self.policy.value_net2(cur_state_batch, actions_batch) ) .detach().cpu().numpy())

            v_data += min_critic_value

        return np.mean(min_critic_value)
            
            