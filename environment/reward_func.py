import math
import torch
import numpy as np
from collections import deque

from environment.utils import custom_reward


def composite_reward(args, state=None, reward=None):
    MAX_GLUCOSE = 600
    if reward == None:
        reward = custom_reward([state])
    x_max, x_min = 0, custom_reward([MAX_GLUCOSE]) #get_IS_Rew(MAX_GLUCOSE, 4) # custom_reward([MAX_GLUCOSE])
    reward = ((reward - x_min) / (x_max - x_min))
    if state <= 40:
        reward = -15
    elif state >= MAX_GLUCOSE:
        reward = 0
    else:
        reward = reward
    return reward
