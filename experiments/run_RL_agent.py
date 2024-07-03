import sys
import torch
import random
import warnings
import numpy as np
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from utils.options import Options
warnings.simplefilter('ignore', Warning)


def set_agent_parameters(args):
    agent = None
    if args.agent == 'ppo':
        from agents.algorithm.ppo import PPO
        agent = PPO(args=args, load_model=False, actor_path='', critic_path='')

    elif args.agent == 'a2c':
        from agents.algorithm.a2c import A2C
        agent = A2C(args=args, load_model=False, actor_path='', critic_path='')

    elif args.agent == 'sac':
        from agents.sac.sac import SAC
        agent = SAC(args=args, load_model=False, actor_path='', critic_path='')

    elif args.agent == 'g2p2c':
        from agents.g2p2c.g2p2c import G2P2C
        agent = G2P2C(args=args, load_model=False, actor_path='', critic_path='')

    else:
        print('Please select an agent for the experiment. Hint: a2c, sac, ppo, g2p2c')
    return agent


def main():
    args = Options().parse()  # load arguments
    agent = set_agent_parameters(args)  # load agent

    if args.verbose:
        print('\nExperiment Starting...')
        print("\nOptions =================>")
        print(vars(args))
        print('\nDevice which the program run on:', args.device)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    agent.run()


if __name__ == '__main__':
    main()
