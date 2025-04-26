#MMP projection for IRL
import numpy as np
import torch
import os
import time

from environment.t1denv import T1DEnv

from utils.control_space import ControlSpace
from agents.algorithm.reinforce_fc import ActorCritic
from agents.algorithm.reinforce_fc import train_actor_critic
from agents.algorithm.ppo import PPO

from clinical.carb_counting import carb_estimate
from utils import core
from utils.load_args import load_arguments
from utils.logger import Logger


#TODO: change around arguments so that can have PPO train using different arguments within same experiment
#TODO: ACtually test to see if works

#arguemnts, currently have no idea what the values should be
n_obs = 24 #incorporate entirety of observation
n_action = 1
n_hidden = 25


#Class that does the main irl
class ProjectionPPO:
    def __init__(self, exp_samples, agent,cfg, env):
        self.cfg = cfg
        self.device = cfg.experiment.device
        self.rl_agent = agent
        self.expert = exp_samples
        self.discount_factor = cfg.agent.gamma
        self.tol = cfg.agent.tol
        self.n_traj = cfg.agent.n_sim
        self.traj_len = cfg.agent.l_sim
        # self.rl_u_init = rl_u_init
        # self.rl_updates = rl_updates
        #self.env = T1DEnv(args=args.env, mode='testing', worker_id=1)
        self.env = env #I think want the sam environment
        self.w = 1
        self.k = 12
        #self.args = args
        self.iters = 0
        self.irl_path = os.path.abspath('../results/mmp_proj_test/'+cfg.agent.irl_file+'.txt')
        #self.rl_path = os.path.abspath('results/mmp_proj_test/rl.txt')
        #might need to change this to account for PPO -> doesnt look like it
        self.controlspace = ControlSpace(control_space_type=self.cfg.agent.control_space_type,
                                         insulin_min=self.env.action_space.low[0],
                                         insulin_max=self.env.action_space.high[0])

    # calculate projection: seem valid
    def projection(self, feat_exp_expert, feat_exp, proj_prev):
        diff = feat_exp - proj_prev

        scalar = (diff.T @ (feat_exp_expert - proj_prev)) / (diff.T @ diff)

        new_proj = proj_prev + scalar * diff

        w_new = feat_exp_expert - new_proj
        t_new = np.linalg.norm(w_new, 2)
        return new_proj, w_new, t_new

    #should be the same
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
            # rl_f = open(self.rl_path, 'a')
            for x in observation:
                # rl_f.write(", ".join([str(self.iters), str(i), str(timestep), str(x[0]), str(x[1]), 'S'])+"\n")
                timestep += 1
            # rl_f.close()
            traj = [np.array([x[0] for x in observation])]
            for _ in range(self.traj_len):
                pol_ret = self.rl_agent.policy.get_action(torch.tensor(observation).to(self.device))  # get RL action
                
                pump_action = self.controlspace.map(agent_action=pol_ret['action'])
                observation, _, is_done, info = self.env.step(pump_action)
                scaled_feature = np.array([x[0] for x in observation])#scaled
                # rl_f = open(self.rl_path, 'a')
                # rl_f.write(", ".join([str(self.iters), str(i), str(timestep), str(observation[-1][0]), str(observation[-1][1]), str(pol_ret['action'])])+"\n")
                # rl_f.close()
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
        
        #randomly initialise w
        self.w = torch.rand(12)
        self.rl_agent.update_worker_rwd(self.w) 

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
        f1 = open(self.irl_path, 'w+')
        f1.write(", ".join(['_'+str(iters), str(self.proj), str(self.w)])+"\n")
        f1.close()
        self.rl_agent.update_worker_rwd(self.w.to(self.device))  # update reward function
        self.rl_agent.increment_irl_iter()
        # Now use RL algorithm to find a new policy
        #print('rl_train')

        t_0 = time.perf_counter()
        #self.rl_agent.run() #training the agent
        t_1 = time.perf_counter()
        print('RL: ', (t_1 - t_0)/60)
        
        pol_exp = self.policy_expectations() #torch tensor of scalars
        print("into loop")
        while not converged:
            #perform projection
            t_0 = time.perf_counter()
            p, w, t = self.projection(expert_exp, pol_exp, self.proj)
        
            p.to(self.device)
            w.to(self.device)
            data.append(t)
            iters += 1
            #print("irl: ",iters, p, w, t)
            self.proj = p
            self.w = w
            file = open(self.irl_path, 'a')
            file.write(", ".join(['_'+str(iters), str(self.proj), str(self.w),str(t) ])+"\n")
            file.close()
            converged = t <= self.tol or iters == max_iters
            if converged:
                break
            self.rl_agent.update_worker_rwd(self.w)  #update reward function
            self.rl_agent.increment_irl_iter()
            t_1= time.perf_counter()
            print("IRL: ", (t_1 - t_0)/60)
            #Now use RL algorithm to find a new policy
            print("rl_train second time")
            t_0 = time.perf_counter()
            if iters == max_iters - 1: #final time to be trained, so train for longer
                self.rl_agent.args.total_interactions = self.cfg.agent.final_iter_interactions
            self.rl_agent.run()  #traiing rl_agent
            t_1= time.perf_counter()
            print("RL: ", (t_1 - t_0)/60)
            #print("rl_train_fin")
            #print("pol_exp")
            t_0 = time.perf_counter()
            pol_exp = self.policy_expectations()  #expectations of new policy
            t_1= time.perf_counter()
            print("Expectation: ", (t_1 - t_0)/60)
            #print(pol_exp)
            #print("pol_exp_fin")

        return iters, data

    def get_rwd_param(self):
        return self.w




