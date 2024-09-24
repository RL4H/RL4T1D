import sys
from decouple import config

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
import matplotlib.pyplot as plt

# import numpy as np


traj_len = 3
n_samples = 2
k = 2  #feature size

#expert_samples = np.zeros((traj_len, n_samples))
print("Gathering expert samples")
expert_samples = []
#TODO check everything with arguments
args = load_arguments(
    overrides=["experiment.name=test2", "agent=clinical_treatment", "env.patient_id=0", "agent.debug=True",
               "hydra/job_logging=disabled"])
patients, env_ids = get_patient_env()

env_clin = T1DEnv(args=args.env, mode='testing', worker_id=1)
clinical_agent = BasalBolusController(args.agent, patient_name=patients[args.env.patient_id],
                                      sampling_rate=env_clin.env.sampling_time)

observation = env_clin.reset()  # observation is the state-space (features x history) of the RL algorithm.
glucose, meal = core.inverse_linear_scaling(y=observation[-1][0], x_min=args.env.glucose_min,
                                            x_max=args.env.glucose_max), 0
# clinical algorithms uses the glucose value, rather than the observation-space of RL algorithms, which is normalised for training stability.

for _ in range(n_samples):  #samples, each of length traj_len
    previous = []  #the previous k - 1 glucose values
    observation = env_clin.reset()  # observation is the state-space (features x history) of the RL algorithm.
    glucose, meal = core.inverse_linear_scaling(y=observation[-1][0], x_min=args.env.glucose_min,
                                                x_max=args.env.glucose_max), 0
    #getting the first k - 1 values to be used as history
    for _ in range(k - 1):
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
            meal = carb_estimate(meal, info['day_hour'], patients[id], type=args.agent.carb_estimation_method)
        glucose = info['cgm'].CGM
        previous.append(glucose)
        print("previous init: ", previous)

    traj = []
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
            meal = carb_estimate(meal, info['day_hour'], patients[id], type=args.agent.carb_estimation_method)
        glucose = info['cgm'].CGM
        #print("Testing prev: ", previous, previous[1:] + [glucose])

        #print("previous: ", previous)
        traj.append(previous + [glucose])
        previous = previous[1:] + [glucose]  # updating the history
        print(f'Latest glucose level: {info["cgm"].CGM:.2f} mg/dL, administered insulin: {action[0]:.2f} U.')
    expert_samples.append(traj)
print('Gathered expert samples')


#Now we have the expert samples, try to get the RL environment working
print("Trying to initialise irl agent")
irl_agent = MaxMarginProjection(args=args, exp_samples=expert_samples, n_traj=n_samples,
                                traj_len=traj_len, env=env_clin, k=k, clin_agent=clinical_agent,
                                patients=patients)  #create the irl agent
print("Begin training irl agent")
iters, data = irl_agent.train(max_iters = 5)  #train the irl agent and gain data for plotting
print("Finished training irl agent")
plt.scatter([i for i in range(iters)], data)
plt.show()
rwd_param = irl_agent.get_rwd_param()  #final output
