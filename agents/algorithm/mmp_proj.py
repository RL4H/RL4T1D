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
n_obs = 24 #incorporate entirety of observation
n_action = 1
n_hidden = 25


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
        #m, n, k = demo.shape()
        #k = 1  #currently only using glucose as feature
        res = np.zeros(self.k)
        for i in range(m):
            n = len(demo[i])
            discount = 1
            for j in range(n):
                res += discount * np.array(demo[i][j])
                discount = discount * gamma

        return res / m

    #Given a policy, use MC to calculate feature expectations
    def policy_expectations(self):
        #sampling from the policy
        samples = []

        for _ in range(self.n_traj):
            observation = self.env.reset()

            traj = [[x[0] for x in observation]]
            for _ in range(self.traj_len):
                rl_action, _, _ = self.rl_agent.get_action(observation)  # get RL action
                pump_action = self.controlspace.map(agent_action=rl_action)
                observation, _, is_done, info = self.env.step(pump_action)
                traj.append([x[0] for x in observation])  # saving the feature vector to the traj
                if is_done ==1: #i.e the patient dies
                    break
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
                     trajectories=self.n_traj)  # i think currently only does one pass
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
                         trajectories=self.n_traj)  #i think currently only does one pass
            samples, pol_exp = self.policy_expectations()  #expectations of new policy

        return iters, data

    def get_rwd_param(self):
        return self.w
