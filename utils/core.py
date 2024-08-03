import scipy.signal
import numpy as np
import torch


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

def get_flat_params_from(model):
    #pdb.set_trace()
    params = []
    for param in model.parameters():
        params.append(param.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def compute_flat_grad(output, inputs, filter_input_ids=set(), retain_graph=True, create_graph=False):
    if create_graph:
        retain_graph = True

    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)

    grads = torch.autograd.grad(output, params, retain_graph=retain_graph, create_graph=create_graph, allow_unused=True)

    j = 0
    out_grads = []
    for i, param in enumerate(inputs):
        if (i in filter_input_ids):
            out_grads.append(torch.zeros(param.view(-1).shape, device=param.device, dtype=param.dtype))
        else:
            if (grads[j] == None):
                out_grads.append(torch.zeros(param.view(-1).shape, device=param.device, dtype=param.dtype))
            else:
                out_grads.append(grads[j].view(-1))
            j += 1
    grads = torch.cat(out_grads)

    for param in params:
        param.grad = None
    return grads
