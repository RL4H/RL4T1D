import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import numpy as np

import sys
from decouple import config

from environment.obs_space import ObservationSpace
from environment.t1denv import T1DEnv
from utils.control_space import ControlSpace

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

#from utils.time_in_range import time_in_range
#from utils.plot import plot_simulation_results
#from utils.stateSpace import StateSpace
from agents.models.actor_critic import ActorCritic


class ReinforceBaseline:
    #Actor critic now joined into ActorCritic instance (self.policy)
    def __init__(self, args, lr=0.001, load=False):
        self.args = args
        self.actor_path = '../saved_models/A2C_MC_LSTM_Actor.pth'
        self.critic_path = '../saved_models/A2C_MC_LSTM_Critic.pth'
        self.policy = ActorCritic(self.args, load, self.actor_path, self.critic_path)  #changed
        # if load:
        # self.Actor = torch.load(self.actor_path)
        # self.Critic = torch.load(self.critic_path)
        self.optimizer_Actor = torch.optim.Adam(self.policy.Actor.parameters(), lr=lr)  #changed
        self.optimizer_Critic = torch.optim.Adam(self.policy.Critic.parameters(), lr=lr)  #changed
        self.distribution = torch.distributions.Normal
        self.w = 0
        #self.env = T1DEnv(args=args.env, mode='testing', worker_id=1)

    #havent changed anything in this method,. hope it works
    def update(self, returns, log_probs, state_values, trajectories):
        """
        Update the weights of the Actor Critic network given the training samples
        @param returns: return (cumulative rewards) for each step in an episode
        @param log_probs: log probability for each step
        @param trajectories: # of trajectories.
        @param state_values: state-value for each step
        """
        loss_Actor = 0
        loss_Critic = 0
        for log_prob, value, Gt in zip(log_probs, state_values, returns):
            advantage = Gt - value.item()
            policy_loss = -log_prob * advantage
            value_loss = F.smooth_l1_loss(value[0], Gt)
            loss_Actor += policy_loss
            loss_Critic += value_loss

        loss_Actor = loss_Actor / trajectories  # average across trajectories
        loss_Critic = loss_Critic / trajectories  # average across trajectories

        self.optimizer_Actor.zero_grad()
        loss_Actor.backward()
        self.optimizer_Actor.step()

        self.optimizer_Critic.zero_grad()
        loss_Critic.backward()
        self.optimizer_Critic.step()

    #Am I fine keeping as is, or use the built in functions in actor critic
    def predict(self, s):
        """
        Compute the output using the continuous Actor Critic model
        @param s: features (state_space)
        @return: Gaussian distribution, state_value
        """
        self.policy.Actor.training = False
        self.policy.Critic.training = False
        return self.policy.Actor(torch.Tensor(s)), self.policy.Critic(torch.Tensor(s))

    #currently keeping as is, but actor critic has its own get action method
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

    #keeping as is
    def save(self):
        torch.save(self.policy.Actor, self.actor_path)
        torch.save(self.policy.Critic, self.critic_path)

    #Added: since reward is going to keep changing need to update
    def update_reward(self, w):
        self.w = w

    #Changed
    #most instances of state -> observation (just renaming no changed functionality)
    def train(self, env=None, controlspace=None, n_episode=20, traj_length=1000, gamma=1.0, trajectories=1,
              feature_history=40, calibration=1, STD_BASAL=0, action_stop_horizon=1, folder_id="folder", penalty=10):
        """
        @param env: Gym environment
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
        #         self.policy.Actor.init_hidden()
        #         self.policy.Critic.init_hidden()
        #         #observation_space = ObservationSpace(feature_history=feature_history) #changed
        #         observation = env.reset() #changed
        #         #cur_state = state_space.update(cgm=state.CGM, ins=0)

        # rewards = []  # rewards of the trajectory
        #
        # glucose_info = []
        # meal_info = []
        # insulin_info = []
        #
        # action_flag = 1  # incremented during action_stop periods.
        # temp_reward = 0  # rewards accumulated during no actions, only relavent when there is action_stop_horizon.
        # counter = 0
        # while True:
        #     counter = counter + 1
        #     # The normal controller is used to obtain the past history required.
        #     if counter < calibration:
        #         next_observation, reward, is_done, info = env.step(STD_BASAL)
        #         action = STD_BASAL
        #         logging.warning({'Phase': 'Calibration Phase', 'Glucose state': observation.CGM, 'Action': action,
        #                          'Reward': reward, 'info': info})
        #
        #     # control using the target agent.
        #     else:
        #         if action_flag / action_stop_horizon == 1.0:
        #             action, log_prob, state_value = self.get_action(observation) #changed
        #             action = action / 20
        #             observation, reward, is_done, info = env.step(action)
        #             logging.info({'Phase': 'RL Agent Action Phase', 'Glucose state': observation.CGM, 'Action': action,
        #                           'Reward': reward, 'info': info})
        #
        #     reward = reward + temp_reward  # this_reward + reward_without_action
        #     reward = reward + 30
        #     if observation.CGM < 40:
        #         reward = reward * (episode_length - counter) * penalty  # large penalty for death
        #
        #     total_reward_episode[episode] += reward
        #
        #     log_probs.append(log_prob)  # update episode wise
        #     state_values.append(state_value)  # # update episode wise
        #     rewards.append(reward)  # rewards per trajectory appended
        #
        #     glucose_info.append(observation.CGM)
        #     meal_info.append(info['meal'] * info['sample_time'])
        #     insulin_info.append(action)
        #
        #     temp_reward = 0
        #     action_flag = 1
        # else:
        #     action_flag = action_flag + 1
        #     observation, reward, is_done, info = env.step(0)
        #         action = 0
        #         logging.info({'Phase': 'Action Stop Phase', 'Glucose state': observation.CGM, 'Action': action,
        #                       'Reward': reward, 'info': info})
        #         temp_reward = temp_reward + reward
        #
        #     if counter > episode_length or observation.CGM < 40:
        #         trajectory_returns = []
        #         Gt = 0
        #         pw = 0
        #         for reward in rewards[::-1]:
        #             Gt = reward + Gt * (gamma ** pw)
        #             pw += 1
        #             trajectory_returns.append(Gt)
        #         trajectory_returns = trajectory_returns[::-1]
        #         trajectory_returns = [float(i) for i in trajectory_returns]
        #         returns = returns + trajectory_returns
        #         break
        #
        # next_state = state_space.update(cgm=state.CGM, ins=action)
        # cur_state = next_state

        #
        # # update agent
        # returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        # self.update(returns, log_probs, state_values, trajectories)
        #
        # # metrics
        # logging.info('\n Episode: {}, total reward: {}'.format(episode, total_reward_episode[episode]))
        # logging.info('The simulation ran for {} samples'.format(counter))
        # print('Episode: {}, total reward: {}'.format(episode, total_reward_episode[episode]))
        # print('The final trajectory of simulation ran for {} samples'.format(counter))
        # print("Average metrics for the total episode, considering all trajectories.")

        #     time_in_range(glucose_info)
        # plot_simulation_results(glucose_info, meal_info, insulin_info,
        #                         episode, trajectory, folder=folder_id, show=False)
        #total discounted rewqrds for each trajectory

        #Generate the samples
        #controlspace = ControlSpace(control_space_type=self.args.agent.control_space_type,
        #                             insulin_min=env.action_space.low[0],
        #                          insulin_max=env.action_space.high[0])
        returns = []
        log_probs = []
        state_values = []

        for _ in range(trajectories):
            observation = env.reset()
            discount = 1
            for _ in range(traj_length):
                rl_action = self.policy.get_action(observation)  # get RL action
                pump_action = controlspace.map(agent_action=rl_action['action'][0])
                log_probs.append(rl_action['log_probs'])
                state_values.append(rl_action['state_values'])
                observation, _, _, info = env.step(pump_action)
                returns.append(discount * self.w * info["cgm"].CGM)
                discount = discount * gamma

        #Now have the info, use that to update the policy
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        self.update(returns, log_probs, state_values, trajectories)
        self.save()

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


class ActorCritic:
    def __init__(self, n_state, n_action, n_hidden, lr=0.001, load=False):
        self.Actor = PolicyNetwork(n_state, n_action, n_hidden)
        self.Critic = ValueNetwork(n_state, n_action, n_hidden)
        self.actor_path = '../saved_models/A2C_MC_FC_Actor.pth'
        self.critic_path = '../saved_models/A2C_MC_FC_Critic.pth'
        if load:
            self.Actor = torch.load(self.actor_path)
            self.Critic = torch.load(self.critic_path)
        self.optimizer_Actor = torch.optim.Adam(self.Actor.parameters(), lr)
        self.optimizer_Critic = torch.optim.Adam(self.Critic.parameters(), lr)
        self.distribution = torch.distributions.Normal

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
        loss_Actor.backward()
        self.optimizer_Actor.step()

        self.optimizer_Critic.zero_grad()
        loss_Critic.backward()
        self.optimizer_Critic.step()

    def predict(self, s):
        """
        Compute the output using the continuous Actor Critic model
        @param s: input state
        @return: Gaussian distribution, state_value
        """
        s = s.flatten()
        self.Actor.training = False
        self.Critic.training = False
        return self.Actor(torch.Tensor(s)), self.Critic(torch.Tensor(s))

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
