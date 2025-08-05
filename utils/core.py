import scipy.signal
import numpy as np
import torch
import math


# from environment.utils import risk_index
# def custom_reward_traj(bg_hist, k, **kwargs):
#     return -risk_index([bg_hist], k)[-1]
#
#
# def custom_reward2(bg_hist, **kwargs):
#     return risk_index([bg_hist[-1]], 1)


def get_exp_avg(arr, scale):
    pow, ema = 0, 0
    for t in reversed(range(len(arr))):
        ema += (scale**pow)*arr[t]
        pow += 1
    return ema


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """

    # pw, Gt = 0, 0
    # returns = []
    # for r in x[::-1]:
    #     Gt = r + Gt * (discount ** pw)
    #     pw += 1
    #     returns.append(Gt)
    # returns = returns[::-1]

    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def linear_scaling(x, x_min, x_max): # scale to [-1, 1] range
    y = ((x - x_min) * 2 / (x_max - x_min)) - 1
    return y


def inverse_linear_scaling(y, x_min, x_max):  # scale back to original
    x = (y+1) * (x_max - x_min) * (1/2) + x_min
    return x


# KL Divergence implementations.
def reverse_kl_approx(p, q):
    # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
    # https://github.com/DLR-RM/stable-baselines3/issues/417
    # https://dibyaghosh.com/blog/probability/kldivergence.html
    # KL (q||p) = (r-1) - log(r)
    # x~q, r = p(x)/q(x) = new/old
    log_ratio = p - q
    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
    return approx_kl


def forward_kl_approx(p, q):
    # KL (p||q) = rlog(r)-(r-1)
    # x~q, r = p(x)/q(x)
    log_ratio = p - q
    approx_kl = torch.mean((torch.exp(log_ratio)*log_ratio) - (torch.exp(log_ratio)-1))
    return approx_kl


def f_kl(log_p, log_q):
    # KL[q,p] = (r-1) - log(r) ;forward KL
    log_ratio = log_p - log_q
    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
    return approx_kl


def r_kl(log_p, log_q):
    # KL[p, q] = rlog(r) -(r-1)
    log_ratio = log_p - log_q
    approx_kl = torch.mean(torch.exp(log_ratio)*log_ratio - (torch.exp(log_ratio) - 1))
    return approx_kl
    
EXP_SCALING_FACT = 4
MEAL_MAX = 100 #FIXME paramaterise
EXP_FACT_PS_SUM = math.exp(EXP_SCALING_FACT) - math.exp(-EXP_SCALING_FACT)
EXP_S_FACT = math.exp(-EXP_SCALING_FACT)

def calculate_features(data_row, args, env_args):
    cgm, meal, ins, t, meta_data = tuple(data_row)
    days,hours,mins = tuple([int(i) for i in t.split(':')])

    info = dict()

    if ins > args.insulin_max: #FIXME remove
        print(data_row, ins)
        raise ValueError
    # info["insulin"] = linear_scaling(x=ins, x_min=args.insulin_min, x_max=args.insulin_max)
    info["insulin"] = pump_to_rl_action(ins, args, env_args) #TODO decide if to use this or not


    info["cgm"] = linear_scaling(x=cgm, x_min=args.glucose_min, x_max=args.glucose_max)
    
    info['future_carb'] = 0 #FIXME implement
    info['remaining_time'] = 0 #FIXME implement
    info['day_hour'] = linear_scaling(x=hours, x_min=0, x_max=23)
    info['day_min'] = linear_scaling(x=mins, x_min=0, x_max=59)
    info['meal_type'] = 0 #FIXME implement
    info['meal'] = linear_scaling(meal, 0, MEAL_MAX)

    return [ info[feat] for feat in env_args.obs_features]


def limit(n, t, b): return max(min(n, t), b)
def pump_to_rl_action(pump_action, args, env_args):
    control_space_type = args.control_space_type
    pump_max = args.insulin_max

    if control_space_type == 'normal':
        # pump_action = ((rl_action + 1) / 2) * pump_max
        rl_action = (2 * pump_action / pump_max) - 1


    elif control_space_type == 'sparse':
        # if agent_action <= 0: agent_action = 0
        # else: agent_action = agent_action * pump_max
        #not reversible

        raise NotImplementedError

    elif control_space_type == 'exponential':
        # pump_action = pump_max * (math.exp((rl_action - 1) * 4))
        rl_action = math.log((pump_action / pump_max)) / 4 + 1
    
    elif control_space_type == 'exponential_alt':
        rl_action = 1/EXP_SCALING_FACT * math.log((pump_action / pump_max) * EXP_FACT_PS_SUM + EXP_S_FACT ) #maps to [0,1]
        # rl_action =limit ( 1/EXP_SCALING_FACT * math.log((pump_action / pump_max) * (MATH_EXP_FACT - 1) + 1 ), 1, 0)



    elif control_space_type == 'quadratic':
        # if agent_action < 0:
        #     agent_action = (agent_action**2) * 0.05
        #     agent_action = min(0.05, agent_action)
        # elif agent_action == 0: agent_action = 0
        # else: agent_action = (agent_action**2) * pump_max

        raise NotImplementedError

    elif control_space_type == 'proportional_quadratic':
        # if agent_action <= 0.5:
        #     agent_action = ((agent_action-0.5)**2) * (0.5/(1.5**2))
        #     agent_action = min(0.5, agent_action)
        # else:
        #     agent_action = ((agent_action-0.5)**2) * (pump_max/(0.5**2))

        raise NotImplementedError

    return rl_action