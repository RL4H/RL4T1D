#MMP projection for IRL
import numpy as np
import torch

from environment.t1denv import T1DEnv

from utils.control_space import ControlSpace
from agents.algorithm.reinforce_fc import ActorCritic
from agents.algorithm.reinforce_fc import actor_critic

from clinical.carb_counting import carb_estimate
from utils import core

#Get demonstrations

#demo: Array shape (m,n,k), m = number of demonstrations, n = length of trajectory
#k = number of features
#Each entry is feature vector for each state in the trajectory
#Gamma: float, discount factor


#Initialise a policy

#approximate feature expectation using MC (above)
#also to do that run throuh the poicy on a variety of trajectories


#termination condition
#if t_new < tol:
#terminate

#Use reinforcement learning to compute optimal policy given this new reward function
#according to the w found in thr projection

#then repeat

#arguemnts, currently have no idea what the values should be
n_obs = 12
n_action = 1
n_hidden = 10


#Class that does the main irl
class MaxMarginProjection:
    def __init__(self, args, exp_samples, n_traj, traj_len, clin_agent, patients, discount_factor=0.9, tol=1e-10,
                 env=None, k=2):
        self.rl_agent = ActorCritic(n_obs, n_action, n_hidden)
        self.expert = exp_samples
        self.discount_factor = discount_factor
        self.tol = tol
        self.n_traj = n_traj
        self.traj_len = traj_len
        #self.env = T1DEnv(args=args.env, mode='testing', worker_id=1)
        self.env = env  #I think want the sam environment
        self.w = 1
        self.k = k
        self.args = args
        self.clinical_agent = clin_agent
        self.patients = patients
        self.controlspace = ControlSpace(control_space_type=self.args.agent.control_space_type,
                                         insulin_min=self.env.action_space.low[0],
                                         insulin_max=self.env.action_space.high[0])

    # calculate projection
    def projection(self, feat_exp_expert, feat_exp, proj_prev):
        diff = feat_exp - proj_prev

        scalar = (diff.T @ (feat_exp_expert - proj_prev)) / (diff.T @ diff)

        new_proj = proj_prev + scalar * diff

        w_new = feat_exp_expert - new_proj
        t_new = np.linalg.norm(w_new, 2)
        return new_proj, w_new, t_new

    def mc_exp(self, demo):
        # demo[i][j] is jth state in the ith episode (demonstration)
        gamma = self.discount_factor
        m = len(demo)
        n = len(demo[0])
        #m, n, k = demo.shape()
        #k = 1  #currently only using glucose as feature
        res = np.zeros(self.k)
        discount = 1
        for t in range(n):
            for i in range(m):
                print(demo[i][t], i)
                res += discount * np.array(demo[i][t])  #need to convert to numpy to do discounting
            discount = discount * gamma
        return res / m

    #Given a policy, use MC to calculate feature expectations
    def policy_expectations(self):
        #sampling from the policy
        samples = []

        for _ in range(self.n_traj):
            observation = self.env.reset()
            glucose, meal = core.inverse_linear_scaling(y=observation[-1][0], x_min=self.args.env.glucose_min,
                                                        x_max=self.args.env.glucose_max), 0
            history = []
            #Use the clinical agent to get the initial history (i.e prior to any rl action)

            for _ in range(self.k - 1):  #have to get history for the states
                action = self.clinical_agent.get_action(meal=meal, glucose=glucose)  # insulin action of BB treatment
                observation, reward, is_done, info = self.env.step(action[0])  # take an env step

                # clinical algorithms require "manual meal announcement and carbohydrate estimation."
                if self.args.env.t_meal == 0:  # no meal announcement: take action for the meal as it happens.
                    meal = info['meal'] * info['sample_time']
                elif self.args.env.t_meal == info[
                    'remaining_time_to_meal']:  # meal announcement: take action "t_meal" minutes before the actual meal.
                    meal = info['future_carb']
                else:
                    meal = 0
                if meal != 0:  # simulate the human carbohydrate estimation error or ideal scenario.
                    meal = carb_estimate(meal, info['day_hour'], self.patients[id],
                                         type=self.args.agent.carb_estimation_method)
                glucose = info['cgm'].CGM
                history.append(glucose)

            observation = torch.tensor([x[0] for x in observation])
            traj = []
            for _ in range(self.traj_len):
                rl_action, _, _ = self.rl_agent.get_action(observation)  # get RL action
                pump_action = self.controlspace.map(agent_action=rl_action)
                observation, _, _, info = self.env.step(pump_action)
                glucose = info['cgm'].CGM  # the actual glucose value
                traj.append(history + [glucose])  #saving the feature vector to the traj
                history = history[1:] + [glucose]  #updating the history
                observation = torch.tensor([x[0] for x in observation])
            samples.append(traj)

        #Now we have our trajectories, can approximate feature using mc
        feature_exp = self.mc_exp(samples)
        return samples, feature_exp

    def train(self, max_iters=5):
        iters = 0
        data = []  #used for plotting
        #get expert feature expectation
        expert_exp = self.mc_exp(self.expert)
        samples, pol_exp = self.policy_expectations()  #with a randomly initialised policy
        converged = False
        #First iteration (i = 1)
        self.proj = pol_exp
        self.w = expert_exp - self.proj
        self.rl_agent.update_reward(self.w)  # update reward function
        # Now use RL algorithm to find a new policy
        actor_critic(args=self.args, env=self.env, estimator=self.rl_agent, controlspace=self.controlspace,
                     episode_length=self.traj_len,
                     gamma=self.discount_factor,
                     trajectories=self.n_traj, k=self.k,
                     clinical_agent=self.clinical_agent, patients=self.patients)  # i think currently only does one pass
        samples, pol_exp = self.policy_expectations()
        while not converged:
            #perform projection
            p, w, t = self.projection(expert_exp, pol_exp, self.proj)
            data.append(t)
            iters += 1
            self.proj = p
            self.w = w
            converged = t <= self.tol or iters == max_iters
            if converged:
                break
            #print("w in mmp: ", self.w)
            self.rl_agent.update_reward(self.w)  #update reward function
            #Now use RL algorithm to find a new policy
            actor_critic(args=self.args, env=self.env, estimator=self.rl_agent, controlspace=self.controlspace,
                         episode_length=self.traj_len,
                         gamma=self.discount_factor,
                         trajectories=self.n_traj, k=self.k,
                         clinical_agent=self.clinical_agent,
                         patients=self.patients)  #i think currently only does one pass
            samples, pol_exp = self.policy_expectations()  #expectations of new policy

        return iters, data

    def get_rwd_param(self):
        return self.w
