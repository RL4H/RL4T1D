import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from agents.algorithm.agent import Agent
from agents.models.actor_critic_iql import QNetwork, ValueNetwork, PolicyNetwork, ActorCritic

from decouple import config
import sys
from collections import namedtuple, deque
from copy import deepcopy

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class IQL(Agent):
    def __init__(self, args, env_args, logger, load_model, actor_path, critic_path, value_path):
        super(IQL, self).__init__(args, env_args=env_args, logger=logger, type="Offline")
        self.device = args.device
        self.completed_interactions = 0
        self.batch_size = args.batch_size

        # training params
        self.train_pi_iters = args.n_pi_epochs
        self.pi_lr = args.pi_lr
        self.vf_lr = args.vf_lr

        # IQL params
        self.discount = args.gamma
        self.soft_tau = args.soft_tau # Soft target update rate
        self.beta = args.beta # Advantage weighting exponent
        self.value_lr = args.vf_lr
        self.critic_lr = args.cr_lr
        self.actor_lr = args.pi_lr

        # component networks
        self.policy = ActorCritic(args, load_model, actor_path, critic_path, value_path, self.device).to(self.device)


        self.critic_optim_1 = torch.optim.Adam(self.policy.critic_net1.parameters() , lr=self.critic_lr, weight_decay=0)
        self.critic_optim_2 = torch.optim.Adam(self.policy.critic_net2.parameters() , lr=self.critic_lr, weight_decay=0)
        self.value_optim = torch.optim.Adam(self.policy.value_net.parameters() , lr=self.value_lr, weight_decay=0)
        self.policy_optim = torch.optim.Adam(self.policy.parameters() , lr=self.actor_lr, weight_decay=0)


        # readout
        print("Setting up offline Agent")
        print(f"Using {args.data_type} data.")

    def update(self):
        print("Running network update...")

        cl, pl = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        pi_grad, val_grad = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        vf_loss = torch.zeros(1, device=self.device)

        for _ in range(self.train_pi_iters):
            transitions = self.buffer.sample(self.batch_size)

            batch = Transition(*zip(*transitions))
            cur_state_batch = torch.cat(batch.state)
            actions_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward).unsqueeze(1)
            next_state_batch = torch.cat(batch.next_state)
            done_batch = torch.cat(batch.done).unsqueeze(1)

            # update critic networks
            with torch.no_grad():
                next_value_batch = self.policy.value_net(next_state_batch) #use stabilised value network instead
                target_q_batch = reward_batch + self.discount * (1 - done_batch) * next_value_batch #FIXME check elementwise

            q1_batch = self.policy.critic_net1(cur_state_batch, actions_batch)
            critic_loss_1 = F.mse_loss(q1_batch, target_q_batch)
            self.critic_optim_1.zero_grad()
            critic_loss_1.backward()
            self.critic_optim_1.step()

            q2_batch = self.policy.critic_net2(cur_state_batch, actions_batch)
            critic_loss_2 = F.mse_loss(q2_batch, target_q_batch)
            self.critic_optim_2.zero_grad()
            critic_loss_2.backward()
            self.critic_optim_2.step()


            # update value function network
            with torch.no_grad():
                q_min = torch.min(q1_batch.detach(), q2_batch.detach())
            
            value_batch = self.policy.value_net(cur_state_batch)
            value_loss = F.mse_loss(value_batch, torch.clamp_max(q_min, value_batch))


            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()

            vf_loss += (value_loss).detach() / self.train_pi_iters
            for param in self.policy.value_net.parameters():
                if param.grad is not None:
                    val_grad += param.grad.sum() / self.train_pi_iters

            # update actor/policy network
            with torch.no_grad():
                advantage = q_min - self.policy.value_net(cur_state_batch)
                weights = torch.exp(self.beta * advantage).clamp(max=1e2)
            
            log_prob_batch = self.policy.policy_net.log_prob(cur_state_batch, actions_batch)
            actor_loss = -(weights * log_prob_batch).mean()

            self.policy_optim.zero_grad()
            actor_loss.backward()
            self.policy_optim.step()

            pi_grad += torch.norm(torch.stack([
                p.grad.norm(2) for p in self.policy.policy_net.parameters() if p.grad is not None
            ]))

        print(f"################ updated target networks {self.train_pi_iters} times")




        # logging
        data = dict(policy_grad=pi_grad, policy_loss=pl, coeff_loss=cl, value_grad=val_grad, val_loss=vf_loss)
        return {k: v.detach().cpu().flatten().numpy()[0] for k, v in data.items()}


