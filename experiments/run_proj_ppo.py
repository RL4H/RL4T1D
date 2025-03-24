import sys
import argparse
import os

import torch.cuda
from decouple import config
import numpy as np
import time

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

# Running a Clinical Treatment Algorithm
from environment.t1denv import T1DEnv
from clinical.basal_bolus_treatment import BasalBolusController
from clinical.carb_counting import carb_estimate
from utils.load_args import load_arguments
from environment.utils import get_env, get_patient_env
from utils import core
from agents.algorithm.mmp_proj import MaxMarginProjection
from agents.algorithm.proj_ppo import ProjectionPPO
import matplotlib.pyplot as plt

# import numpy as np
#arguments (hyperparameters)
parser = argparse.ArgumentParser()
parser.add_argument("--patient_id", type=int, default = 0)#patient id
parser.add_argument("--n_expert", type = int, default=2) #Number of expert trajs
parser.add_argument("--l_expert", type=int, default=3) #Max length of expert traj
parser.add_argument("--i_irl", type=int,default=5) #iterations irl
parser.add_argument("--i_update_init", type=int, default = 3)#updates used to initally train rl agent
parser.add_argument("--i_update", type=int,default=3)#updates per rl train
parser.add_argument("--n_sim", type=int, default=2) #Number of sim traj
parser.add_argument("--l_sim", type=int, default=5)#max length of sim traj
parser.add_argument("--total_inters", type=int, default=1)#total number of interactions
parser.add_argument("--dvc", default = 'cuda', type=str, choices=['cpu','cuda'] )#device for pytorch

input_args = parser.parse_args()

#Hyperparameters of experiment
patient_id = input_args.patient_id
traj_len = input_args.l_expert
n_samples = input_args.n_expert
irl_max_iters = input_args.i_irl
rl_u_init = input_args.i_update_init
rl_updates = input_args.i_update
sim_samples = input_args.n_sim #might wan to split in rl update sim and mc sim
sim_length = input_args.l_sim
total_inters = input_args.total_inters
device = input_args.dvc

k = 12  #feature size -> observation already has past incorporated

if torch.cuda.is_available():
#     device = 'cuda'
    print('cuda is available')
else:
    print("CUDA not available")
#     device = 'cpu'

#expert_samples = np.zeros((traj_len, n_samples))
print("Gathering expert samples")
start = time.time()
expert_samples = []

args = load_arguments(
    overrides=["experiment.name=test2", "agent=clinical_treatment", "env.patient_id="+str(patient_id), "agent.debug=True",
               "hydra/job_logging=disabled"])
patients, env_ids = get_patient_env()

env_clin = T1DEnv(args=args.env, mode='testing', worker_id=1)
clinical_agent = BasalBolusController(args.agent, patient_name=patients[args.env.patient_id],
                                      sampling_rate=env_clin.env.sampling_time)

observation = env_clin.reset()  # observation is the state-space (features x history) of the RL algorithm.
glucose, meal = core.inverse_linear_scaling(y=observation[-1][0], x_min=args.env.glucose_min,
                                            x_max=args.env.glucose_max), 0
# clinical algorithms uses the glucose value, rather than the observation-space of RL algorithms, which is normalised for training stability.
#print("expert")
for _ in range(n_samples):  #samples, each of length traj_len
    observation = env_clin.reset()  # observation is the state-space (features x history) of the RL algorithm.
    glucose, meal = core.inverse_linear_scaling(y=observation[-1][0], x_min=args.env.glucose_min,
                                                x_max=args.env.glucose_max), 0
    traj = [np.array([x[0] for x in observation])]
    for _ in range(traj_len):

        action = clinical_agent.get_action(meal=meal, glucose=glucose)  # insulin action of BB treatment
        observation, reward, is_done, info = env_clin.step(action[0])  # take an env step

        # clinical algorithms require "manual meal announcement and carbohydrate estimation."
        if args.env.t_meal == 0:  # no meal announcement: take action for the meal as it happens.
            meal = info['meal'] * info['sample_time']
        elif args.env.t_meal == info[
            'remaining_time_to_meal']:  # meal announcement: take action "t_meal" minutes before the actual meal.
            meal = info['future_carb']
        else:
            meal = 0
        if meal != 0:  # simulate the human carbohydrate estimation error or ideal scenario.
            meal = carb_estimate(meal, info['day_hour'], patients[args.env.patient_id], type=args.agent.carb_estimation_method)
        glucose = info['cgm'].CGM

        traj.append(np.array([x[0] for x in observation]))
        #print(f'Latest glucose level: {info["cgm"].CGM:.2f} mg/dL, administered insulin: {action[0]:.2f} U.')
    expert_samples.append(traj)

#expert_samples.to(device)
#print(expert_samples)
print('expert_fin')
# finish = time.time()
# print('Gathered expert samples in ', finish - start, "seconds")
# start = time.time()

#Now we have the expert samples, try to get the RL environment working
print("Trying to initialise irl agent")

irl_agent = ProjectionPPO( exp_samples=expert_samples, n_traj=sim_samples,
                                traj_len=sim_length, rl_u_init=rl_u_init,
                                rl_updates=rl_updates, env=env_clin, k=k,total_inters=total_inters, device=device)
print("Begin training irl agent")
iters, data = irl_agent.train(max_iters=irl_max_iters)  #train the irl agent and gain data for plotting
print("Concluded training")
finish = time.time()
#print("Finished training irl agent in ", finish - start, 'seconds')
# ax = plt.subplot()
# ax.scatter([i for i in range(iters)], data)
# plt.show()
rwd_param = irl_agent.get_rwd_param()  #final output
#TODO look at other visualisations

# #plotting (normalised) glucose vs reward
# x = np.linspace(0,1,1000)
# w1 = rwd_param[-1]
# w2 = rwd_param[-2]
# w3 = rwd_param[-3]
# ax2 = plt.subplot()
# plt.plot(x, w1*x, label='current glucose')
# plt.plot(x, w2*x, label = 'previous')
# plt.plot(x, w3*x, label = 'two steps ago')
# plt.legend()
# plt.show()
