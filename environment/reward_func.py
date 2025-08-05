import math
import torch
import numpy as np
from collections import deque
from random import randrange

from environment.utils import custom_reward, custom_reward_2, custom_reward_3


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


def composite_reward_2(args, state=None, reward=None):
    MAX_GLUCOSE = 600
    if reward == None:
        reward = custom_reward_2([state])
    x_max, x_min = 0, custom_reward_2([MAX_GLUCOSE]) #get_IS_Rew(MAX_GLUCOSE, 4) # custom_reward([MAX_GLUCOSE])
    reward = ((reward - x_min) / (x_max - x_min)) ** 3
    if state <= 40:
        reward = -6
    elif state >= MAX_GLUCOSE:
        reward = -5
    else:
        reward = reward
    return reward

def composite_reward_3(args, state=None, reward=None):
    MAX_GLUCOSE = 600
    if reward == None:
        reward = custom_reward_3(state)
    x_max, x_min = 0, custom_reward([MAX_GLUCOSE]) #get_IS_Rew(MAX_GLUCOSE, 4) # custom_reward([MAX_GLUCOSE])
    reward = ((reward - x_min) / (x_max - x_min))
    if state[-1] <= 40:
        reward = -6
    elif state[-1] >= MAX_GLUCOSE:
        reward = -5
    else:
        reward = reward
    return reward
