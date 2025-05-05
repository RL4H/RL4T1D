import numpy as np
from gym import spaces
from environment.utils import get_env


class T1DEnv:
    def __init__(self, args, mode, worker_id):
        self.args = args
        self.worker_mode = mode
        self.worker_id = worker_id
        self.state = 0
        self.env = get_env(args, worker_id=worker_id, env_type=mode)  # setup environment
        self.reset()

    def reset(self):
        self.state, self.info = self.env.reset()
        return self.state, self.info

    def step(self, action):
        state, reward, is_done, info = self.env.step(action)
        if is_done:
            self.reset()
        return state, reward, is_done, info

    @property
    def action_space(self):
        return spaces.Box(low=self.args.insulin_min, high=self.args.insulin_max, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,))
