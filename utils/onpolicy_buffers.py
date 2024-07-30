import torch
import numpy as np
from utils import core
from utils.reward_normalizer import RewardNormalizer
from utils.core import linear_scaling


class RolloutBuffer:
    def __init__(self, args):
        self.size = args.n_step
        self.device = args.device

        # discounted vs average reward RL
        self.return_type = args.return_type
        self.gamma = args.gamma if args.return_type == 'discount' else 1
        self.lambda_ = args.lambda_ if args.return_type == 'discount' else 1

        self.n_training_workers = args.n_training_workers
        self.n_step = args.n_step
        self.feature_history = args.feature_history
        self.n_features = args.n_features

        self.RolloutWorker = RolloutWorker(args)
        self.shuffle_rollout = args.shuffle_rollout
        self.normalize_reward = args.normalize_reward
        self.reward_normaliser = RewardNormalizer(num_envs=self.n_training_workers, cliprew=10.0,
                                                  gamma=self.gamma, epsilon=1e-8, per_env=False)

        self.rollout_buffer = {}

        self.old_states = torch.rand(self.n_training_workers, self.n_step, self.feature_history, self.n_features, device=self.device)

        self.old_actions = torch.rand(self.n_training_workers, self.n_step, device=self.device)
        self.old_logprobs = torch.rand(self.n_training_workers, self.n_step, device=self.device)
        self.reward = torch.rand(self.n_training_workers, self.n_step, device=self.device)
        self.v_targ = torch.rand(self.n_training_workers, self.n_step, device=self.device)
        self.adv = torch.rand(self.n_training_workers, self.n_step, device=self.device)
        self.v_pred = torch.rand(self.n_training_workers, self.n_step + 1, device=self.device)
        self.first_flag = torch.rand(self.n_training_workers, self.n_step + 1, device=self.device)

        self.cost = torch.rand(self.n_training_workers, self.n_step+1, device=self.device) # CPO changes

    def save_rollout(self, training_agent_index):
        data = self.RolloutWorker.get()
        self.old_states[training_agent_index] = data['obs']
        self.old_actions[training_agent_index] = data['act']
        self.old_logprobs[training_agent_index] = data['logp']
        self.v_pred[training_agent_index] = data['v_pred']
        self.reward[training_agent_index] = data['reward']
        self.first_flag[training_agent_index] = data['first_flag']

    def compute_gae(self):  # TODO: move to different script, optimise/resolve moving across devices back and forth.
        orig_device = self.v_pred.device
        assert orig_device == self.reward.device == self.first_flag.device
        vpred, reward, first = (x.cpu() for x in (self.v_pred, self.reward, self.first_flag))
        first = first.to(dtype=torch.float32)
        assert first.dim() == 2
        nenv, nstep = reward.shape
        assert vpred.shape == first.shape == (nenv, nstep + 1)
        adv = torch.zeros(nenv, nstep, dtype=torch.float32)
        lastgaelam = 0
        for t in reversed(range(nstep)):
            notlast = 1.0 - first[:, t + 1]
            nextvalue = vpred[:, t + 1]
            # notlast: whether next timestep is from the same episode
            delta = reward[:, t] + notlast * self.gamma * nextvalue - vpred[:, t]
            adv[:, t] = lastgaelam = delta + notlast * self.gamma * self.lambda_ * lastgaelam
        vtarg = vpred[:, :-1] + adv
        return adv.to(device=orig_device), vtarg.to(device=orig_device)
    
    def compute_constraint_adv(self):
        orig_device = self.v_pred.device
        assert orig_device == self.reward.device == self.first_flag.device
        vpred, cost, first = (x.cpu() for x in (self.v_pred, self.cost, self.first_flag))
        first = first.to(dtype=torch.float32)
        assert first.dim() == 2
        nenv, nstep = cost.shape
        assert vpred.shape == first.shape == (nenv, nstep + 1)
        adv = torch.zeros(nenv, nstep, dtype=torch.float32)
        lastgaelam = 0
        for t in reversed(range(nstep)):
            notlast = 1.0 - first[:, t + 1]
            nextvalue = vpred[:, t + 1]
            # notlast: whether next timestep is from the same episode
            delta = cost[:, t] + notlast * self.gamma * nextvalue - vpred[:, t]
            adv[:, t] = lastgaelam = delta + notlast * self.gamma * self.lambda_ * lastgaelam
        return adv.to(device=orig_device)
          
# define the compute cost function
    def estimate_constraint_value(self):
        orig_device = self.cost.device
        assert orig_device == self.first_flag.device
        first, cost = (x.cpu() for x in (self.first_flag, self.cost))
        first = first.to(dtype=torch.float32)
        assert first.dim() == 2
        nenv, nstep = cost.shape
        assert first.shape == (nenv, nstep+1)
        constraint_value = torch.tensor(nenv)
        
        j = 1
        traj_num = 1
        for i in range(self.cost.size(0)):
            constraint_value = constraint_value + self.cost[i] * self.gamma**(j-1)
            notlast = 1.0 - first[:, i + 1]
            if notlast == 0:
                j = 1 #reset
                traj_num = traj_num + 1
            else: 
                j = j+1
                
        constraint_value = constraint_value/traj_num
        constraint_value = constraint_value.to(device = orig_device)
        return constraint_value

    def prepare_rollout_buffer(self):

        if self.return_type == 'discount':
            if self.normalize_reward:  # reward normalisation
                self.reward = self.reward_normaliser(self.reward, self.first_flag)
            self.adv, self.v_targ = self.compute_gae()  # # calc returns
            self.cost_adv = self.compute_constraint_adv()
            self.constraint_value = self.estimate_constraint_value()

        if self.return_type == 'average':
            self.reward = self.reward_normaliser(self.reward, self.first_flag, type='average')
            self.adv, self.v_targ = self.compute_gae()
            self.cost_adv = self.compute_constraint_adv()
            self.constraint_value = self.estimate_constraint_value()


        '''concat data from different workers'''
        s_hist = self.old_states.view(-1, self.feature_history, self.n_features)
        act = self.old_actions.view(-1, 1)
        logp = self.old_logprobs.view(-1, 1)
        v_targ = self.v_targ.view(-1)
        adv = self.adv.view(-1)
        cost_adv = self.cost_adv.view(-1)
        first_flag = self.first_flag.view(-1)
        buffer_len = s_hist.shape[0]

        if self.shuffle_rollout:
            rand_perm = torch.randperm(buffer_len)
            s_hist = s_hist[rand_perm, :, :]  # torch.Size([batch, n_steps, features])
            act = act[rand_perm, :]  # torch.Size([batch, 1])
            logp = logp[rand_perm, :]  # torch.Size([batch, 1])
            v_targ = v_targ[rand_perm]  # torch.Size([batch])
            adv = adv[rand_perm]  # torch.Size([batch])
            cost_adv = cost_adv[rand_perm]

        self.rollout_buffer = dict(states=s_hist, action=act, log_prob_action=logp, value_target=v_targ, advantage=adv, cost_advantage = cost_adv, len=buffer_len)
        return self.rollout_buffer


class RolloutWorker:
    def __init__(self, args):
        self.size = args.n_step
        self.device = args.device
        self.args = args

        self.feature_hist = args.feature_history
        self.features = args.n_features

        self.state = np.zeros(core.combined_shape(self.size, (self.feature_hist, self.features)), dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.float32)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.state_values = np.zeros(self.size + 1, dtype=np.float32)
        self.logprobs = np.zeros(self.size, dtype=np.float32)
        self.first_flag = np.zeros(self.size + 1, dtype=np.bool_)
        self.cgm_target = np.zeros(self.size, dtype=np.float32)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, self.size

    def store(self, obs, act, rew, val, logp, cgm_target, is_first):
        assert self.ptr < self.max_size
        scaled_cgm = linear_scaling(x=cgm_target, x_min=self.args.glucose_min, x_max=self.args.glucose_max)
        self.state[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.state_values[self.ptr] = val
        self.logprobs[self.ptr] = logp
        self.first_flag[self.ptr] = is_first
        self.cgm_target[self.ptr] = scaled_cgm
        self.ptr += 1

    def finish_path(self, final_v):
        self.state_values[self.ptr] = final_v
        self.first_flag[self.ptr] = False

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        data = dict(obs=self.state, act=self.actions, v_pred=self.state_values,
                    logp=self.logprobs, first_flag=self.first_flag, reward=self.rewards, cgm_target=self.cgm_target)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in data.items()}

