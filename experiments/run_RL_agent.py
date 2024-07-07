import sys
import torch
import random
import warnings
import numpy as np
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

warnings.simplefilter('ignore', Warning)


from utils.logger import setup_folders


def set_agent_parameters(cfg):
    agent = None
    if cfg.agent.agent == 'ppo':
        from agents.algorithm.ppo import PPO
        setup_folders(cfg)
        agent = PPO(args=cfg.agent, env_args=cfg.env, load_model=False, actor_path='', critic_path='')

    # elif args.agent.agent == 'a2c':
    #     from agents.algorithm.a2c import A2C
    #     agent = A2C(args=args, load_model=False, actor_path='', critic_path='')
    #
    # elif args.agent.agent == 'sac':
    #     from agents.algorithm.sac import SAC
    #     agent = SAC(args=args, load_model=False, actor_path='', critic_path='')
    #
    # elif args.agent.agent == 'g2p2c':
    #     from agents.g2p2c.g2p2c import G2P2C
    #     agent = G2P2C(args=args, load_model=False, actor_path='', critic_path='')

    else:
        print('Please select an agent for the experiment. Hint: a2c, sac, ppo, g2p2c')
    return agent


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    agent = set_agent_parameters(cfg)  # load agent
    #run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    # wandb.config = OmegaConf.to_container(
    #     cfg, resolve=True, throw_on_missing=True
    # )
    #wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    if cfg.experiment.verbose:
        print('\nExperiment Starting...')
        print("\nOptions =================>")
        print(vars(cfg))
        print('\nDevice which the program run on:', cfg.experiment.device)

    #exit()

    torch.manual_seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    agent.run()


if __name__ == '__main__':
    main()


#python run_RL_agent.py experiment.folder=test4 agent.debug=True hydra/job_logging=disabled
