import sys
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

# Running a RL Algorithm
from environment.t1denv import T1DEnv
from agents.algorithm.ppo import PPO
from utils.control_space import ControlSpace
from utils.logger import Logger
from utils.load_args import load_arguments

args = load_arguments(overrides=["experiment.name=test2", "agent=ppo", "env.patient_id=0", "agent.debug=True",
                                 "hydra/job_logging=disabled"])
# print(vars(args))

agent = PPO(args=args.agent, env_args=args.env, logger=Logger(args), load_model=False, actor_path='', critic_path='')
env = T1DEnv(args=args.env, mode='testing', worker_id=1)

controlspace = ControlSpace(control_space_type=args.agent.control_space_type, insulin_min=env.action_space.low[0],
                            insulin_max=env.action_space.high[0])

observation = env.reset()
for _ in range(10):
    rl_action = agent.policy.get_action(observation)  # get RL action
    pump_action = controlspace.map(agent_action=rl_action['action'][0])  # map RL action => control space (pump)

    observation, reward, is_done, info = env.step(pump_action)  # take an env step
    print(f'Latest glucose level: {info["cgm"].CGM:.2f} mg/dL, administered insulin: {pump_action:.2f} U.')

# OR simply use agent.run() to train the agent.

print('Done')


# Running a Clinical Treatment Algorithm
# from environment.t1denv import T1DEnv
# from clinical.basal_bolus_treatment import BasalBolusController
# from clinical.carb_counting import carb_estimate
# from utils.load_args import load_arguments
# from environment.utils import get_env, get_patient_env
# from utils import core
#
# args = load_arguments(overrides=["experiment.name=test2", "agent=clinical_treatment", "env.patient_id=0", "agent.debug=True", "hydra/job_logging=disabled"])
# patients, env_ids = get_patient_env()
#
# env = T1DEnv(args=args.env, mode='testing', worker_id=1)
# clinical_agent = BasalBolusController(args.agent, patient_name=patients[args.env.patient_id], sampling_rate=env.env.sampling_time)
#
# observation = env.reset()  # observation is the state-space (features x history) of the RL algorithm.
# glucose, meal = core.inverse_linear_scaling(y=observation[-1][0], x_min=args.env.glucose_min, x_max=args.env.glucose_max), 0
# # clinical algorithms uses the glucose value, rather than the observation-space of RL algorithms, which is normalised for training stability.
#
# for _ in range(10):
#
#     action = clinical_agent.get_action(meal=meal, glucose=glucose)  # insulin action of BB treatment
#     observation, reward, is_done, info = env.step(action[0])  # take an env step
#
#     # clinical algorithms require "manual meal announcement and carbohydrate estimation."
#     if args.env.t_meal == 0:  # no meal announcement: take action for the meal as it happens.
#         meal = info['meal'] * info['sample_time']
#     elif args.env.t_meal == info['remaining_time_to_meal']:  # meal announcement: take action "t_meal" minutes before the actual meal.
#         meal = info['future_carb']
#     else:
#         meal = 0
#     if meal != 0:  # simulate the human carbohydrate estimation error or ideal scenario.
#         meal = carb_estimate(meal, info['day_hour'], patients[id], type=args.agent.carb_estimation_method)
#     glucose = info['cgm'].CGM
#
#     print(f'Latest glucose level: {info["cgm"].CGM:.2f} mg/dL, administered insulin: {action[0]:.2f} U.')
#
# print('Done')
