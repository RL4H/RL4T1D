#MMP projection for IRL
import numpy as np
import torch
import os

from environment.t1denv import T1DEnv

from utils.control_space import ControlSpace
from agents.algorithm.reinforce_fc import ActorCritic
from agents.algorithm.reinforce_fc import train_actor_critic

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
    def __init__(self, args, exp_samples, n_traj, traj_len, rl_updates,device='cpu', discount_factor=0.9, tol=1e-10,
                 env=None, k=2):
        self.device = device
        self.rl_agent = ActorCritic(n_obs, n_action, n_hidden, device=self.device)
        self.expert = exp_samples
        self.discount_factor = discount_factor
        self.tol = tol
        self.n_traj = n_traj
        self.traj_len = traj_len
        self.rl_updates = rl_updates
        #self.env = T1DEnv(args=args.env, mode='testing', worker_id=1)
        self.env = env  #I think want the sam environment
        self.w = 1
        self.k = k
        self.args = args
        self.iters = 0
        self.irl_path = os.path.abspath('../results/mmp_proj_test/irl')
        self.rl_path = os.path.abspath('../results/mmp_proj_test/rl')
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
        res = torch.zeros(self.k)
        for i in range(m):
            n = len(demo[i])
            discount = 1
            for j in range(n):
                res += discount * torch.tensor(demo[i][j])
                discount = discount * gamma

        return res / m

    #Given a policy, use MC to calculate feature expectations
    def policy_expectations(self):
        #sampling from the policy
        samples = []

        for i in range(self.n_traj):
            timestep = 0
            observation = self.env.reset()
            rl_f = open(self.rl_path, 'a')
            for x in observation:
                rl_f.write(", ".join([str(self.iters), str(i), str(timestep), str(x[0]), str(x[1]), 'S'])+"\n")
                timestep += 1
            rl_f.close()
            traj = [np.array([x[0] for x in observation])]
            for _ in range(self.traj_len):
                rl_action, _, _ = self.rl_agent.get_action(torch.tensor(observation).to(self.device))  # get RL action
                pump_action = self.controlspace.map(agent_action=rl_action)
                observation, _, is_done, info = self.env.step(pump_action)
                scaled_feature = np.array([x[0] for x in observation])#scaled
                rl_f = open(self.rl_path, 'a')
                rl_f.write(", ".join([str(self.iters), str(i), str(timestep), str(observation[-1][0]), str(observation[-1][1]), str(rl_action)])+"\n")
                rl_f.close()
                traj.append(scaled_feature)  # saving the feature vector to the traj
                timestep +=1
                if is_done == 1: #i.e the patient dies
                    break
            samples.append(traj)

        #Now we have our trajectories, can approximate feature using mc
        feature_exp = self.mc_exp(samples)
        self.iters += 1
        return feature_exp

    def train(self, max_iters=5):
        #clear both result files
        # irl_path = os.path.abspath('../..results/mmp_proj_test/irl')
        #rl_path = os.path.abspath('../..results/mmp_proj_test/rl')
        # open(irl_path, 'w').close()
        open(self.rl_path, 'w').close()
        iters = 0
        data = []  #used for plotting
        #get expert feature expectation
        #print("expert_exp")
        expert_exp = self.mc_exp(self.expert)
        #print(expert_exp)
        print("expert_exp_fin")
        pol_exp = self.policy_expectations()  #with a randomly initialised policy
        converged = False
        #First iteration (i = 1)
        self.proj = pol_exp
        self.w = expert_exp - self.proj
        #wrting to results
        f1 = open(self.irl_path, 'w')
        f1.write(", ".join([str(iters), str(self.proj), str(self.w)])+"\n")
        f1.close()
        #print('irl: ', iters, self.proj, self.w)
        self.rl_agent.update_reward(self.w.to(self.device))  # update reward function
        # Now use RL algorithm to find a new policy
        #print('rl_train')
        train_actor_critic(args=self.args, env=self.env, estimator=self.rl_agent, controlspace=self.controlspace,
                     episode_length=self.traj_len, n_episode=self.rl_updates,
                     gamma=self.discount_factor, device=self.device,
                     trajectories=self.n_traj)
        #print('rl_train_fin')
        pol_exp = self.policy_expectations() #torch tensor of scalars
        while not converged:
            #perform projection
            p, w, t = self.projection(expert_exp, pol_exp, self.proj)
            data.append(t)
            iters += 1
            #print("irl: ",iters, p, w, t)
            self.proj = p
            self.w = w
            file = open(self.irl_path, 'a')
            file.write(", ".join([str(iters), str(self.proj), str(self.w),str(t) ])+"\n")
            file.close()
            converged = t <= self.tol or iters == max_iters
            if converged:
                break
            #print("w in mmp: ", self.w)
            self.rl_agent.update_reward(self.w)  #update reward function
            #Now use RL algorithm to find a new policy
            #print("rl_train")
            train_actor_critic(args=self.args, env=self.env, estimator=self.rl_agent, controlspace=self.controlspace,
                         episode_length=self.traj_len, n_episode=self.rl_updates,
                         gamma=self.discount_factor,
                         trajectories=self.n_traj)  #i think currently only does one pass
            #print("rl_train_fin")
            #print("pol_exp")
            pol_exp = self.policy_expectations()  #expectations of new policy
            #print(pol_exp)
            #print("pol_exp_fin")

        return iters, data

    def get_rwd_param(self):
        return self.w




