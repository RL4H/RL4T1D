import sys
from decouple import config
import numpy as np

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
from agents.algorithm.reinforce_fc import min_max_norm
import matplotlib.pyplot as plt

# import numpy as np


traj_len = 3
n_samples = 3
k = 12  #feature size -> observation already has past incorporated

#expert_samples = np.zeros((traj_len, n_samples))
print("Gathering expert samples")
expert_samples = []

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
    observation = env_clin.reset()  # observation is the state-space (features x history) of the RL algorithm.
    glucose, meal = core.inverse_linear_scaling(y=observation[-1][0], x_min=args.env.glucose_min,
                                                x_max=args.env.glucose_max), 0
    traj = [min_max_norm(np.array([x[0] for x in observation]))]
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

        traj.append(min_max_norm(np.array([x[0] for x in observation])))
        print(f'Latest glucose level: {info["cgm"].CGM:.2f} mg/dL, administered insulin: {action[0]:.2f} U.')
    expert_samples.append(traj)
print('Gathered expert samples')


#Now we have the expert samples, try to get the RL environment working
print("Trying to initialise irl agent")
irl_agent = MaxMarginProjection(args=args, exp_samples=expert_samples, n_traj=n_samples,
                                traj_len=traj_len, env=env_clin, k=k, clin_agent=clinical_agent,
                                patients=patients)  #create the irl agent
print("Begin training irl agent")
iters, data = irl_agent.train(max_iters = 10)  #train the irl agent and gain data for plotting
print("Finished training irl agent")
ax = plt.subplot()
ax.scatter([i for i in range(iters)], data)
plt.show()
rwd_param = irl_agent.get_rwd_param()  #final output
#TODO look at other visualisations

#plotting (normalised) glucose vs reward
x = np.linspace(0,1,1000)
w1 = rwd_param[-1]
w2 = rwd_param[-2]
w3 = rwd_param[-3]
ax2 = plt.subplot()
plt.plot(x, w1*x, label='current glucose')
plt.plot(x, w2*x, label = 'previous')
plt.plot(x, w3*x, label = 'two steps ago')
plt.legend()
plt.show()

