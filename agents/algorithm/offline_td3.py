import torch
import torch.nn as nn

import numpy as np

from agents.algorithm.agent import Agent
from agents.models.actor_critic import ActorCritic

from decouple import config
import sys

import csv
from collections import namedtuple, deque


MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

Transition = namedtuple('Transition',
                        ('state', 'feat', 'action', 'reward', 'next_state', 'next_feat', 'done'))

def convert_trial_into_transitions(data_obj, window_size=16, default_starting_window=True, default_starting_value=0):
    #data_obj is a 2D numpy array , rows x columns. Columns are :  cgm, meal, ins, t, meta_data
    rows, _ = data_obj.shape
    ins_column = data_obj[:, 2]
    cgm_column = data_obj[:, 0]

    assert rows > window_size
    
    states = np.zeros((rows, 2, window_size)) if default_starting_window else np.zeros((rows-window_size, 2, window_size))

    for row in range(rows):
        if row < window_size and default_starting_window:
            ins_window = np.append(np.array([default_starting_value]*(window_size-row)), ins_column[0: row])
            cgm_window = np.append(np.array([default_starting_value]*(window_size-row)), cgm_column[0: row])
        else:
            ins_window = ins_column[row-window_size: row]
            cgm_window = cgm_column[row-window_size: row]

        states[row] = np.array([ins_window, cgm_window])
    
    transitions = []
    for row_n in range(rows-1):
        state = states[row_n]
        feat = None #unused
        action = state[0][-1]
        reward = -1 #filled in by algo
        next_state = states[row_n+1]
        next_feat = None #unused
        done = (row_n == rows - 2)
        transitions.append(Transition(state, feat, action, reward, next_state, next_feat, done))

    return transitions

# ### Saving state
# policy_step, mu, sigma = ddpg.get_action(self.cur_state, self.feat, worker_mode=self.worker_mode)
# selected_action = policy_step[0]
# rl_action, pump_action = self.pump.action(agent_action=selected_action, prev_state=self.init_state, prev_info=None)
# state, reward, is_done, info = self.env.step(pump_action)
# reward = composite_reward(self.args, state=state.CGM, reward=reward)
# this_state = deepcopy(self.cur_state)
# this_feat = deepcopy(self.feat)
# done_flag = 1 if state.CGM <= 40 or state.CGM >= 600 else 0
# self.cur_state, self.feat = self.state_space.update(cgm=state.CGM, ins=pump_action,
#                                                     meal=info['remaining_time'], hour=(self.counter+1),
#                                                     meal_type=info['meal_type'], carbs=info['future_carb'])

# if self.worker_mode == 'training':
#     replay_memory.push(torch.as_tensor(this_state, dtype=torch.float32, device=self.device).unsqueeze(0),
#                         torch.as_tensor(this_feat, dtype=torch.float32, device=self.device).unsqueeze(0),
#                         torch.as_tensor([selected_action], dtype=torch.float32, device=self.device),
#                         torch.as_tensor([reward], dtype=torch.float32, device=self.device),
#                         torch.as_tensor(self.cur_state, dtype=torch.float32, device=self.device).unsqueeze(0),
#                         torch.as_tensor(self.feat, dtype=torch.float32, device=self.device).unsqueeze(0),
#                         torch.as_tensor([done_flag], dtype=torch.float32, device=self.device))
                
class TD3_BC(Agent):
    def __init__(self, args, env_args, logger, load_model, actor_path, critic_path):
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

        ### TD3 networks:
        self.td3 = ActorCritic(args, load_model, actor_path, critic_path, self.device).to(self.device)
        self.value_criterion1 = nn.MSELoss()
        self.value_criterion2 = nn.MSELoss()
        self.value_optimizer1 = torch.optim.Adam(self.td3.value_net1.parameters(), lr=self.value_lr,
                                                 weight_decay=self.weight_decay)
        self.value_optimizer2 = torch.optim.Adam(self.td3.value_net2.parameters(), lr=self.value_lr,
                                                 weight_decay=self.weight_decay)
        self.policy_optimizer = torch.optim.Adam(self.td3.policy_net.parameters(), lr=self.policy_lr,
                                                 weight_decay=self.weight_decay)
        for target_param, param in zip(self.td3.policy_net.parameters(), self.td3.policy_net_target.parameters()):
            target_param.data.copy_(param.data)

        for p in self.td3.policy_net_target.parameters():
            p.requires_grad = False

        for target_param, param in zip(self.td3.value_net1.parameters(), self.td3.value_net_target1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.td3.value_net2.parameters(), self.td3.value_net_target2.parameters()):
            target_param.data.copy_(param.data)

        for p in self.td3.value_net_target1.parameters():
            p.requires_grad = False
        for p in self.td3.value_net_target2.parameters():
            p.requires_grad = False

        print('Policy Parameters: {}'.format(
            sum(p.numel() for p in self.td3.policy_net.parameters() if p.requires_grad)))
        print(
            'Value network 1 Parameters: {}'.format(
                sum(p.numel() for p in self.td3.value_net1.parameters() if p.requires_grad)))
        print(
            'Value network 2 Parameters: {}'.format(
                sum(p.numel() for p in self.td3.value_net2.parameters() if p.requires_grad)))

        self.save_log([['policy_loss', 'value_loss', 'pi_grad', 'val_grad']], '/model_log')
        self.model_logs = torch.zeros(4, device=self.device)
        self.save_log([['ri', 'alive_steps', 'normo', 'hypo', 'sev_hypo', 'hyper', 'lgbi', 'hgbi',
                        'sev_hyper', 'rollout', 'trial']], '/evaluation_log')
        self.save_log([['status', 'rollout', 't_rollout', 't_update', 't_test']], '/experiment_summary')
        self.save_log([[1, 0, 0, 0, 0]], '/experiment_summary')
        self.completed_interactions = 0
        

        # readout
        print("Setting up offline Agent")
        print(f"Using {args.data_type} data.")


    def save_log(self, log_name, file_name):
        with open(self.args.experiment_dir + file_name + '.csv', 'a+') as f:
            csvWriter = csv.writer(f, delimiter=',')
            csvWriter.writerows(log_name)
            f.close()

    def update(self):
        print("Running network update...")

        cl, pl, ql1, ql2, count = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), \
            torch.zeros(1, device=self.device), \
            torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        pi_grad, val_grad, q2_grad, coeff_grad = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), \
            torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)

        for i in range(self.train_pi_iters):
            # sample from buffer

            # if self.replay_buffer_type == "random":
            #     transitions = self.replay_memory.sample(self.sample_size)
            # elif self.replay_buffer_type == "per_proportional" or self.replay_buffer_type == "per_rank":
            #     transitions, indices, weights = self.replay_memory.sample(self.sample_size,
            #                                                               beta=self.replay_buffer_beta,
            #                                                               buffer_type=self.replay_buffer_type)
            #     weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

            #FIXME add transitions = read_buffer()
            transitions = [self.buffer.pop() for _ in range(self.sample_size)]
            

            batch = Transition(*zip(*transitions))
            cur_state_batch = torch.cat(batch.state)
            cur_feat_batch = torch.cat(batch.feat)
            actions_batch = torch.cat(batch.action).unsqueeze(1)
            reward_batch = torch.cat(batch.reward).unsqueeze(1)
            next_state_batch = torch.cat(batch.next_state)
            next_feat_batch = torch.cat(batch.next_feat)
            done_batch = torch.cat(batch.done).unsqueeze(1)

            # value network update
            new_action, next_log_prob = self.td3.evaluate_target_policy_noise(next_state_batch, next_feat_batch)
            next_values = torch.min(self.td3.value_net_target1(next_state_batch, next_feat_batch, new_action),
                                    self.td3.value_net_target2(next_state_batch, next_feat_batch, new_action))
            target_value = (reward_batch + (self.gamma * (1 - done_batch) * next_values))

            predicted_value1 = self.td3.value_net1(cur_state_batch, cur_feat_batch, actions_batch)
            predicted_value2 = self.td3.value_net2(cur_state_batch, cur_feat_batch, actions_batch)

            value_loss1 = self.value_criterion1(target_value.detach(), predicted_value1)
            value_loss2 = self.value_criterion2(target_value.detach(), predicted_value2)
            # td_error = predicted_value - target_value
            # value_loss = (td_error.pow(2) * weights).mean()
            # self.replay_memory.update_priorities(indices, np.abs(td_error.cpu().detach().numpy()))

            self.value_optimizer1.zero_grad()
            self.value_optimizer2.zero_grad()

            value_loss1.backward()
            value_loss2.backward()

            self.value_optimizer1.step()
            self.value_optimizer2.step()

            cl += value_loss1.detach()

            for param in self.td3.value_net1.parameters():
                if param.grad is not None:
                    val_grad += param.grad.sum()

            for param in self.td3.value_net2.parameters():
                if param.grad is not None:
                    val_grad += param.grad.sum()

            self.n_updates += 1

            # actor update
            if self.n_updates % self.target_update_interval == 0:
                # freeze value networks save compute: ref: openai:
                for p in self.td3.value_net1.parameters():
                    p.requires_grad = False
                for p in self.td3.value_net2.parameters():
                    p.requires_grad = False

                for p in self.td3.value_net_target1.parameters():
                    p.requires_grad = False
                for p in self.td3.value_net_target2.parameters():
                    p.requires_grad = False

                policy_action, _ = self.td3.evaluate_policy_no_noise(cur_state_batch, cur_feat_batch)
                policy_loss = torch.min(self.td3.value_net1(cur_state_batch, cur_feat_batch, policy_action),
                                        self.td3.value_net2(cur_state_batch, cur_feat_batch, policy_action))

                if self.replay_buffer_type == "random":
                    q_mean = (1 * policy_loss).mean()
                elif self.replay_buffer_type == "per_proportional" or self.replay_buffer_type == "per_rank":
                    q_mean = (1 * policy_loss * weights).mean()

                # policy_loss += self.mu_penalty * action_penalty(policy_action, lower_bound=-self.action_penalty_limit,
                #                                                 upper_bound=self.action_penalty_limit,
                #                                                 penalty_factor=self.action_penalty_coef)


                
                lmbda = self.alpha / ( policy_loss.abs().mean() )
                pi = cur_state_batch
                action = policy_action
                policy_loss = -lmbda * q_mean + nn.functional.mse_loss(pi,action)

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                pl += policy_loss.detach()
                pi_grad += torch.nn.utils.clip_grad_norm_(self.td3.policy_net.parameters(), 10)

                # save compute: ref: openai:
                for p in self.td3.value_net1.parameters():
                    p.requires_grad = True
                for p in self.td3.value_net2.parameters():
                    p.requires_grad = True

                for p in self.td3.value_net_target1.parameters():
                    p.requires_grad = True
                for p in self.td3.value_net_target2.parameters():
                    p.requires_grad = True

                # Update target networks
                with torch.no_grad():
                    print("################updated target networks")
                    for param, target_param in zip(self.td3.value_net1.parameters(),
                                                   self.td3.value_net_target1.parameters()):
                        target_param.data.mul_((1 - self.soft_tau))
                        target_param.data.add_(self.soft_tau * param.data)
                    for param, target_param in zip(self.td3.value_net2.parameters(),
                                                   self.td3.value_net_target2.parameters()):
                        target_param.data.mul_((1 - self.soft_tau))
                        target_param.data.add_(self.soft_tau * param.data)

                    for param, target_param in zip(self.td3.policy_net.parameters(),
                                                   self.td3.policy_net_target.parameters()):
                        target_param.data.mul_((1 - self.soft_tau))
                        target_param.data.add_(self.soft_tau * param.data)

        self.model_logs[0] = cl  # value loss or coeff loss
        self.model_logs[1] = pl
        self.model_logs[2] = pi_grad
        self.model_logs[3] = val_grad

        self.save_log([self.model_logs.detach().cpu().flatten().numpy()], '/model_log')
        print('success')

