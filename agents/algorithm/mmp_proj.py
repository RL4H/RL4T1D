#MMP projection for IRL
import numpy as np
import torch

from environment.t1denv import T1DEnv

from utils.control_space import ControlSpace
from agents.algorithm.reinforce_fc import ActorCritic
from agents.algorithm.reinforce_fc import actor_critic

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
    def __init__(self, args, exp_samples, n_traj, traj_len, discount_factor=0.9, tol=1e-10, env=None):
        self.rl_agent = ActorCritic(n_obs, n_action, n_hidden)
        self.expert = exp_samples
        self.discount_factor = discount_factor
        self.tol = tol
        self.n_traj = n_traj
        self.traj_len = traj_len
        #self.env = T1DEnv(args=args.env, mode='testing', worker_id=1)
        self.env = env  #I think want the sam environment
        self.w = 1
        self.controlspace = ControlSpace(control_space_type=args.agent.control_space_type,
                                         insulin_min=self.env.action_space.low[0],
                                         insulin_max=self.env.action_space.high[0])

    # calculate projection
    def projection(self, feat_exp_expert, feat_exp, proj_prev):
        diff = feat_exp - proj_prev
        print("expert: ", feat_exp_expert)
        print("sampled: ", feat_exp)
        print("proj, prev", proj_prev)
        print("diff:", diff)
        scalar = (diff.T @ (feat_exp_expert - proj_prev)) / (diff.T @ diff)
        print("scalar: ", scalar)
        new_proj = proj_prev + scalar * diff
        print("new_proj", new_proj)
        w_new = feat_exp_expert - new_proj
        print("w_new: ", w_new)
        t_new = np.linalg.norm(w_new, 2)
        return new_proj, w_new, t_new

    #TODO: check that numpy works with rest of formatting
    def mc_exp(self, demo):
        # demo[i][j] is jth state in the ith episode (demonstration)
        gamma = self.discount_factor
        m = len(demo)
        n = len(demo[0])
        #m, n, k = demo.shape()
        k = 1  #currently only using glucose as feature
        res = np.zeros(k)
        discount = 1
        for t in range(n):
            for i in range(m):
                res += discount * demo[i][t]
            discount = discount * gamma
        return res / m

    #Given a policy, use MC to calculate feature expectations
    def policy_expectations(self):
        samples = []
        for _ in range(self.n_traj):
            observation = self.env.reset()
            observation = torch.tensor([x[0] for x in observation])
            traj = []
            for _ in range(self.traj_len):
                rl_action, _, _ = self.rl_agent.get_action(observation)  # get RL action
                pump_action = self.controlspace.map(agent_action=rl_action)
                observation, _, _, info = self.env.step(pump_action)
                traj.append(info["cgm"].CGM)  #the actual glucose value
                observation = torch.tensor([x[0] for x in observation])
            samples.append(traj)

        #Now we have our trajectories, can approximate feature using mc
        feature_exp = self.mc_exp(samples)
        return samples, feature_exp

    def train(self):
        #get expert feature expectation
        expert_exp = self.mc_exp(self.expert)
        samples, pol_exp = self.policy_expectations()  #with a randomly initialised policy
        print("have sampled policy and got expectation")
        converged = False
        #First iteration (i = 1)
        self.proj = pol_exp
        self.w = expert_exp - self.proj
       # print("init w:", self.w)
        print("performing first iteration RL test")
        self.rl_agent.update_reward(self.w)  # update reward function
        # Now use RL algorithm to find a new policy
        actor_critic(env=self.env, estimator=self.rl_agent, controlspace=self.controlspace,
                     episode_length=self.traj_len,
                     gamma=self.discount_factor,
                     trajectories=self.n_traj)  # i think currently only does one pass
        samples, pol_exp = self.policy_expectations()
        print("completed first iteration RL test")
        while not converged:
            #perform projection
            p, w, t = self.projection(expert_exp, pol_exp, self.proj)
            print("in inner loop")
            print("t: ", t)
            self.proj = p
            self.w = w
            converged = t <= self.tol
            if converged:
                print("I have converged")
                break
            #print("w in mmp: ", self.w)
            self.rl_agent.update_reward(self.w)  #update reward function
            #Now use RL algorithm to find a new policy
            actor_critic(env=self.env,estimator=self.rl_agent, controlspace=self.controlspace, episode_length=self.traj_len,
                         gamma=self.discount_factor,
                         trajectories=self.n_traj)  #i think currently only does one pass
            samples, pol_exp = self.policy_expectations()  #expectations of new policy
            break #TODO change when done testing

    def get_rwd_param(self):
        return self.w
