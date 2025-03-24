import sys
import torch
import random
import warnings
import numpy as np
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf
from utils.logger import setup_folders, Logger
from utils import core
from environment.t1denv import T1DEnv
from environment.utils import get_env, get_patient_env
from clinical.carb_counting import carb_estimate
from clinical.basal_bolus_treatment import BasalBolusController
from agents.algorithm.mmp_proj import MaxMarginProjection
from agents.algorithm.proj_ppo import ProjectionPPO

warnings.simplefilter('ignore', Warning)


def set_agent_parameters(cfg):
    agent = None

    setup_folders(cfg)
    logger = Logger(cfg)

    if cfg.agent.agent == 'ppo':
        from agents.algorithm.ppo import PPO
        agent = PPO(args=cfg.agent, env_args=cfg.env, logger=logger, load_model=False, actor_path='', critic_path='')

    elif cfg.agent.agent == 'a2c':
        from agents.algorithm.a2c import A2C
        agent = A2C(args=cfg.agent, env_args=cfg.env, logger=logger, load_model=False, actor_path='', critic_path='')

    elif cfg.agent.agent == 'sac':
        from agents.algorithm.sac import SAC
        agent = SAC(args=cfg.agent, env_args=cfg.env, logger=logger, load_model=False, actor_path='', critic_path='')

    elif cfg.agent.agent == 'g2p2c':
        from agents.algorithm.g2p2c import G2P2C
        agent = G2P2C(args=cfg.agent, env_args=cfg.env, logger=logger, load_model=False, actor_path='', critic_path='')

    elif cfg.agent.agent == 'irl':
        from agents.algorithm.ppo import PPO
        cfg.agent.agent='ppo'
        agent = PPO(args=cfg.agent, env_args=cfg.env, logger=logger, load_model=False, actor_path='', critic_path='')

    else:
        print('Please select an agent for the experiment. Hint: a2c, sac, ppo, g2p2c')
    return agent


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:

    if cfg.agent.debug:  # if debug override params to run experiment in a smaller scale.
        for key in cfg.agent.debug_params:
            if key in cfg.agent:
                cfg.agent[key] = cfg.agent.debug_params[key]

    if cfg.experiment.verbose:
        print('\nExperiment Starting...')
        print("\nOptions =================>")
        print(vars(cfg))
        print('\nDevice which the program run on:', cfg.experiment.device)

    agent = set_agent_parameters(cfg)  # load agent
    torch.manual_seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    # add irl
    

    print("Gathering expert trajectories")
    patients, env_ids = get_patient_env()
    env_clin = T1DEnv(args=cfg.env, mode='testing', worker_id=1)
    clinical_agent = BasalBolusController(cfg.agent, patient_name=patients[cfg.env.patient_id],
                                           sampling_rate=env_clin.env.sampling_time)
    
    observation = env_clin.reset()  # observation is the state-space (features x history) of the RL algorithm.
    glucose, meal = core.inverse_linear_scaling(y=observation[-1][0], x_min=cfg.env.glucose_min,
                                            x_max=cfg.env.glucose_max), 0
    expert_samples = []
    for _ in range(cfg.agent.n_samples_expert):  #samples, each of length traj_len
        observation = env_clin.reset()  # observation is the state-space (features x history) of the RL algorithm.
        glucose, meal = core.inverse_linear_scaling(y=observation[-1][0], x_min=cfg.env.glucose_min,
                                                x_max=cfg.env.glucose_max), 0
        traj = [np.array([x[0] for x in observation])]
        for _ in range(cfg.agent.l_expert):

            action = clinical_agent.get_action(meal=meal, glucose=glucose)  # insulin action of BB treatment
            observation, reward, is_done, info = env_clin.step(action[0])  # take an env step

            # clinical algorithms require "manual meal announcement and carbohydrate estimation."
            if cfg.env.t_meal == 0:  # no meal announcement: take action for the meal as it happens.
                meal = info['meal'] * info['sample_time']
            elif cfg.env.t_meal == info['remaining_time_to_meal']:  # meal announcement: take action "t_meal" minutes before the actual meal.
                meal = info['future_carb']
            else:
                meal = 0
            if meal != 0:  # simulate the human carbohydrate estimation error or ideal scenario.
                meal = carb_estimate(meal, info['day_hour'], patients[cfg.env.patient_id], type=cfg.agent.carb_estimation_method)
            glucose = info['cgm'].CGM

            traj.append(np.array([x[0] for x in observation]))
            #print(f'Latest glucose level: {info["cgm"].CGM:.2f} mg/dL, administered insulin: {action[0]:.2f} U.')
        expert_samples.append(traj)

    print("Expert Samples gathered")
    print("initialing IRL agent")
    print("Trying to initialise irl agent")

    irl_agent = ProjectionPPO( exp_samples=expert_samples,cfg=cfg,env=env_clin, agent=agent)
    print("Begin training irl agent")
    iters, data = irl_agent.train(max_iters=cfg.agent.i_irl)  #train the irl agent and gain data for plotting
    print("Concluded training")
    

if __name__ == '__main__':
    main()