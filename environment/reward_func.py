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


EXP_SCALING_BASE = 100
def scaling_func(reward):
    return (math.exp(EXP_SCALING_BASE * reward) - 1) / (math.exp(EXP_SCALING_BASE) - 1)

def composite_reward_2(args, state=None, reward=None):
    MAX_GLUCOSE = 600
    if reward == None:
        reward = custom_reward_2([state])
    x_max, x_min = 0, custom_reward_2([MAX_GLUCOSE]) #get_IS_Rew(MAX_GLUCOSE, 4) # custom_reward([MAX_GLUCOSE])
    reward = ((reward - x_min) / (x_max - x_min))
    reward = scaling_func(reward)
    if state <= 40:
        reward = -2
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

TARGET_GLUC = 125
MAX_GLUCOSE = 600
MIN_GLUCOSE = 40

def custom_reward_4(state):
    b,m,M = TARGET_GLUC, MIN_GLUCOSE, MAX_GLUCOSE
    if state <= TARGET_GLUC: return -1/((b-m)**2) * ((state-m)**2) + 2/(b-m)*(state-m)
    else: return -1/((b-M)**2) * ((state-M)**2) + 2/(b-M)*(state-M)

def composite_reward_4(args, state=None, reward=None):
    if reward == None: reward = custom_reward_4(state)
    if state <= 40:  reward = -20
    elif state >= MAX_GLUCOSE: reward = -20
    return reward

