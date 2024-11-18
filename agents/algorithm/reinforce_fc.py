import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import sys
from decouple import config
from collections import deque

from clinical.carb_counting import carb_estimate
from utils import core

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)


#from utils.time_in_range import time_in_range
#from utils.plot import plot_simulation_results
#from utils.stateSpace import StateSpace


class PolicyNetwork(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(n_input, n_hidden)
        self.mu = nn.Linear(n_hidden, n_output)
        self.sigma = nn.Linear(n_hidden, n_output)
        self.distribution = torch.distributions.Normal
        self.log_std_min = 0
        self.log_std_max = 3
        self.mu_min = 0
        self.mu_max = 10

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = F.softplus(self.mu(x))
        #mu = torch.sigmoid(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 1e-5
        # dist = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        sigma = torch.clamp(sigma, self.log_std_min, self.log_std_max)
        mu = torch.clamp(mu, self.mu_min, self.mu_max)
        return mu, sigma


class ValueNetwork(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Linear(n_input, n_hidden)
        self.value = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        value = self.value(x)
        return value


class ActorCritic:  #changed n_obs <- n_state
    def __init__(self, n_obs, n_action, n_hidden, device='cpu', lr=0.001, load=False):
        self.device = device
        self.Actor = PolicyNetwork(n_obs, n_action, n_hidden).to(self.device)
        self.Critic = ValueNetwork(n_obs, n_action, n_hidden).to(self.device)
        self.actor_path = '../saved_models/A2C_MC_FC_Actor.pth'
        self.critic_path = '../saved_models/A2C_MC_FC_Critic.pth'
        if load:
            self.Actor = torch.load(self.actor_path).to(self.device)
            self.Critic = torch.load(self.critic_path).to(self.device)
        self.optimizer_Actor = torch.optim.Adam(self.Actor.parameters(), lr)
        self.optimizer_Critic = torch.optim.Adam(self.Critic.parameters(), lr)
        self.distribution = torch.distributions.Normal
        self.w = 1  #Just a placeholder value currently

    #kept as is, should be fine
    def update(self, returns, log_probs, state_values, trajectories):
        """
        Update the weights of the Actor Critic network given the training samples
        @param returns: return (cumulative rewards) for each step in an episode
        @param log_probs: log probability for each step
        @param state_values: state-value for each step
        """
        loss_Actor = 0
        loss_Critic = 0
        for log_prob, value, Gt in zip(log_probs, state_values, returns):
            advantage = Gt - value.item()
            policy_loss = - log_prob * advantage
            value_loss = F.smooth_l1_loss(value[0], Gt)
            loss_Actor += policy_loss
            loss_Critic += value_loss

        loss_Actor = loss_Actor / trajectories  # average across trajectories
        loss_Critic = loss_Critic / trajectories  # average across trajectories

        self.optimizer_Actor.zero_grad()
        #print(loss_Actor)
        loss_Actor.backward()
        self.optimizer_Actor.step()

        self.optimizer_Critic.zero_grad()
        loss_Critic.backward()
        self.optimizer_Critic.step()

    #Next two defined in terms of states, in these places being treated as observations
    #and just hope that is correct in this context
    def predict(self, s):
        """
        Compute the output using the continuous Actor Critic model
        @param s: input state
        @return: Gaussian distribution, state_value
        """
        s = s.flatten()
        self.Actor.training = False
        self.Critic.training = False
        return self.Actor(torch.Tensor(s).to(self.device)), self.Critic(torch.Tensor(s).to(self.device))

    def get_action(self, s):
        """
        Estimate the policy and sample an action, compute its log probability
        @param s: input state
        @return: the selected action, log probability, predicted state-value
        """
        (mean, sigma), state_value = self.predict(s)
        std = sigma.exp()

        normal = self.distribution(0, 1)
        z = normal.sample()
        action = mean + std * z
        action = action.detach().numpy()
        action[0] = np.clip(action[0], 0, 10)
        log_prob = self.distribution(mean, std).log_prob(mean + std * z)  # - torch.log(1 - action.pow(2) + epsilon)
        return action[0], log_prob, state_value

    def save(self):
        torch.save(self.Actor, self.actor_path)
        torch.save(self.Critic, self.critic_path)

    def update_reward(self, w):
        self.w = w

    def get_reward(self):
        return self.w


#changed, left original code commented out for reference

def train_actor_critic(args=None, env=None, estimator=None, controlspace=None, n_episode=5, episode_length=1000,
                       gamma=1.0, trajectories=1, device='cpu',
                       feature_history=40, calibration=1, STD_BASAL=0, action_stop_horizon=1, folder_id='None',
                       penalty=10):
    """
    continuous Actor Critic algorithm
    @param env: Gym environment
    @param estimator: policy network
    @param n_episode: number of episodes
    @param episode_length: The length of an episode, usually setup based on sim_days
    @param gamma: the discount factor
    @param STD_BASAL: Used when there is calibration period.
    @param trajectories: # of trajectories used for learning
    @param feature_history: # the feature space
    @param calibration: The length of the calibration period. Should be >= feature history
    @param action_stop_horizon: to be used when actions are not needed at each sample. if 1, action always taken

    Pseudocode of the flow
    -- Iterate over episodes
        -- Iterate of Trajectories
            -- Calibration period using std_basal
            -- Control using RL agent
                -- Check for stop period or action period.
            -- Check of the trajectory termination conditions are met (e.g. episode length)
        -- Update the RL agent.
        -- Calculate the metrics
    """
    # total_reward_episode = [0] * n_episode
    # for episode in range(n_episode):
    #     logging.info({'Episode': episode})
    #
    #     # for agent updating / learning.
    #     log_probs = []
    #     state_values = []
    #     returns = []
    #
    #     for trajectory in range(0, trajectories):
    #         state_space = StateSpace(feature_history=feature_history)
    #         state = env.reset()
    #         cur_state = state_space.update(cgm=state.CGM, ins=0)
    #
    #         rewards = []  # rewards of the trajectory
    #
    #         glucose_info = []
    #         meal_info = []
    #         insulin_info = []
    #
    #         action_flag = 1  # incremented during action_stop periods.
    #         temp_reward = 0  # rewards accumulated during no actions, only relavent when there is action_stop_horizon.
    #         counter = 0
    #         while True:
    #             counter = counter + 1
    #             # The normal controller is used to obtain the past history required.
    #             if counter < calibration:
    #                 next_state, reward, is_done, info = env.step(STD_BASAL)
    #                 action = STD_BASAL
    #                 logging.warning({'Phase': 'Calibration Phase', 'Glucose state': state.CGM, 'Action': action,
    #                                  'Reward': reward, 'info': info})
    #
    #             # control using the target agent.
    #             else:
    #                 if action_flag / action_stop_horizon == 1.0:
    #                     action, log_prob, state_value = estimator.get_action(cur_state)
    #                     action = action / 20
    #                     # print(action[0])
    #                     state, reward, is_done, info = env.step(action)
    #                     logging.info({'Phase': 'RL Agent Action Phase', 'Glucose state': state.CGM, 'Action': action,
    #                                   'Reward': reward, 'info': info})
    #
    #                     reward = reward + temp_reward  # this_reward + reward_without_action
    #                     if state.CGM < 40:
    #                         reward = reward * (episode_length - counter) * penalty  # large penalty for death
    #
    #                     total_reward_episode[episode] += reward
    #
    #                     log_probs.append(log_prob)  # update episode wise
    #                     state_values.append(state_value)  # # update episode wise
    #                     rewards.append(reward)  # rewards per trajectory appended
    #
    #                     glucose_info.append(state.CGM)
    #                     meal_info.append(info['meal'] * info['sample_time'])
    #                     insulin_info.append(action)
    #
    #                     temp_reward = 0
    #                     action_flag = 1
    #                 else:
    #                     action_flag = action_flag + 1
    #                     state, reward, is_done, info = env.step(0)
    #                     action = 0
    #                     logging.info({'Phase': 'Action Stop Phase', 'Glucose state': state.CGM, 'Action': action,
    #                                   'Reward': reward, 'info': info})
    #                     temp_reward = temp_reward + reward
    #
    #                 if counter > episode_length or state.CGM < 40:  # or next_state.CGM < 40 or next_state.CGM >= 600
    #                     trajectory_returns = []
    #                     Gt = 0
    #                     pw = 0
    #                     for reward in rewards[::-1]:
    #                         Gt = reward + Gt * (gamma**pw)
    #                         pw += 1
    #                         trajectory_returns.append(Gt)
    #                     trajectory_returns = trajectory_returns[::-1]
    #                     trajectory_returns = [float(i) for i in trajectory_returns]
    #                     returns = returns + trajectory_returns
    #                     break
    #
    #             next_state = state_space.update(cgm=state.CGM, ins=action)
    #             cur_state = next_state
    #
    #     # update agent
    #     returns = torch.tensor(returns)
    #     returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    #     estimator.update(returns, log_probs, state_values, trajectories)
    #
    #     # metrics
    #     logging.info('Episode: {}, total reward: {}'.format(episode, total_reward_episode[episode]))
    #     logging.info('The simulation ran for {} samples'.format(counter))
    #     print('Episode: {}, total reward: {}'.format(episode, total_reward_episode[episode]))
    #     print('The final trajectory of simulation ran for {} samples'.format(counter))
    #     print("Average metrics for the total episode, considering all trajectories.")
    #
    #     #time_in_range(glucose_info)
    #     #plot_simulation_results(glucose_info, meal_info, insulin_info,
    #                             #episode, trajectory, folder=folder_id, show=False)
    #
    # estimator.save()
    #
    # return total_reward_episode
    w = estimator.w
    for _ in range(n_episode):  #how many times we update the network
        #print('rl_ep')
        returns = []
        log_probs = []
        state_values = []
        for i in range(trajectories):  #how many trajectories used to update
            rewards = []  #immediate rewards for each step of traj
            observation = env.reset()
            #print(i, observation)
            #Probably should be renamed but length of each trajectory
            for _ in range(episode_length):
                rl_action, log_prob, state_val = estimator.get_action(torch.tensor(observation).to(device))  # get RL action
                pump_action = controlspace.map(agent_action=rl_action)
                log_probs.append(log_prob)
                state_values.append(state_val)
                observation, _, is_done, info = env.step(pump_action)
                observation = torch.tensor(observation)
                feature = torch.tensor([x[0] for x in observation]).to(device)
                new_glucose = round(info["cgm"].CGM, 2)
                # print(w)
                reward = np.matmul(w, feature)
                rewards.append(reward)
                #print(rl_action, log_prob, state_val, new_glucose, reward)
                if is_done == 1:  #i.e patient dies
                    break
            #Now convert trajectory rewards into returns
            #code copied from old method
            trajectory_returns = []
            Gt = 0
            pw = 0
            for reward in rewards[::-1]:
                Gt = reward + Gt * (gamma ** pw)
                pw += 1
                trajectory_returns.append(Gt)
            trajectory_returns = trajectory_returns[::-1]
            trajectory_returns = [float(i) for i in trajectory_returns]
            returns = returns + trajectory_returns

        # Now have the info, use that to update the policy
        #print("returns: ", returns)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        estimator.update(returns, log_probs, state_values, trajectories)
        #print('rl_ep_fin')
        #estimator.save()


# class StateSpace:
#     def __init__(self, args):
#         self.feature_history = args.feature_history
#
#         self.glucose = deque(self.feature_history*[0], self.feature_history)
#         self.insulin = deque(self.feature_history*[0], self.feature_history)
#         self.glucose_max = 600
#         self.glucose_min = 40
#         self.insulin_max = args.action_scale
#         self.insulin_min = 0
#
#         self.mealAnnounce = args.use_meal_announcement  # if non-zero meal announcement
#         self.meal_announce_time = args.meal_announce_time
#
#         if not self.mealAnnounce:
#             self.state = np.stack((self.glucose, self.insulin), axis=-1).astype(float)
#         else:
#             self.meal_announcement_arr = deque(self.feature_history * [0], self.feature_history)
#             self.state = np.stack((self.glucose, self.insulin, self.meal_announcement_arr), axis=-1).astype(float)
#
#
#     def update(self, cgm=0, ins=0, meal=0):
#         """
#         normalise and update statespace
#         :param cgm: next_CGM reading
#         :param ins: Infused insulin
#         :return: state
#         """
#         # [0, 1] range
#         # cgm = (cgm - self.glucose_min) / (self.glucose_max - self.glucose_min)
#         # ins = (ins - self.insulin_min) / (self.insulin_max - self.insulin_min)
#
#         # [-1, 1] range
#         cgm = ((cgm - self.glucose_min) * 2 / (self.glucose_max - self.glucose_min)) - 1
#         ins = ((ins - self.insulin_min) * 2 / (self.insulin_max - self.insulin_min)) - 1
#
#
#
#         self.glucose.appendleft(cgm)
#         self.insulin.appendleft(ins)
#
#         if not self.mealAnnounce:
#             self.state = np.stack((self.glucose, self.insulin), axis=-1).astype(float)
#         else:
#             # [0, 1]
#             # meal = (meal - 0) / (self.meal_announce_time - 0)
#
#             # [-1, 1]
#             meal = ((meal - 0) * 2 / (self.meal_announce_time - 0)) - 1
#             self.meal_announcement_arr.appendleft(meal)
#             self.state = np.stack((self.glucose, self.insulin, self.meal_announcement_arr), axis=-1).astype(float)
#
#         return self.state
