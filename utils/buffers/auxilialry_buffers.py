import torch
import numpy as np
from utils import core


class AuxiliaryBuffer:
    def __init__(self, args):
        self.size = args.aux_buffer_max
        self.n_bgp_steps = args.n_bgp_steps
        self.bgp_pred_mode = args.bgp_pred_mode

        self.n_training_workers = args.n_training_workers
        self.n_step = args.n_step
        self.feature_history = args.obs_window
        self.n_features = args.n_features

        self.old_states = torch.zeros(self.size, self.feature_history, self.n_features, device=args.device, dtype=torch.float32)
        self.actions = torch.zeros(self.size, 1, device=args.device, dtype=torch.float32)
        self.logprob = torch.zeros(self.size, 1, device=args.device, dtype=torch.float32)
        self.value_target = torch.zeros(self.size, device=args.device, dtype=torch.float32)
        self.aux_batch_size = args.aux_batch_size

        self.device = args.device
        self.buffer_filled = False
        self.buffer_level = 0

        if self.bgp_pred_mode:
            self.cgm_target = torch.zeros(self.size, 1, self.n_bgp_steps, device=args.device ,dtype=torch.float32)
        else:
            self.cgm_target = torch.zeros(self.size, 1, device=args.device, dtype=torch.float32)

    def update(self, s, cgm_target, actions, first_flag):
        if self.bgp_pred_mode:
            s, actions, cgm_target = self.prepare_bgp_prediction(s, cgm_target, actions, first_flag)
            update_size = actions.shape[0]  # this size is lesser then rollout samples.
            self.old_states = torch.cat((self.old_states[update_size:, :, :], s), dim=0)
            self.cgm_target = torch.cat((self.cgm_target[update_size:, :, :], cgm_target), dim=0)
            self.actions = torch.cat((self.actions[update_size:], actions), dim=0)
        else:  # normal buffer updating approach.
            cgm_target = cgm_target.view(-1, 1)
            update_size = actions.shape[0]
            self.old_states = torch.cat((self.old_states[update_size:, :, :], s), dim=0)
            self.cgm_target = torch.cat((self.cgm_target[update_size:, :], cgm_target), dim=0)
            self.actions = torch.cat((self.actions[update_size:], actions), dim=0)

        if not self.buffer_filled:
            self.buffer_level += update_size
            if self.buffer_level >= self.size:
                self.buffer_filled = True

        if update_size > self.size:
            print('The auxilliary update at rollout is larger than MAX buffer size!')
            exit()

        assert self.old_states.shape[0] == self.size
        assert self.cgm_target.shape[0] == self.size
        assert self.actions.shape[0] == self.size

    def prepare_bgp_prediction(self, s_hist, cgm_target, act, first_flag):
        buffer_len = s_hist.shape[0]
        bgp_first_flag = first_flag.view(-1).cpu().numpy()
        bgp_cgm_target = cgm_target.cpu().numpy()
        bgp_s_hist = s_hist.cpu().numpy()
        # bgp_s_handcraft = s_handcraft.cpu().numpy()
        bgp_act = act.cpu().numpy()
        new_cgm_target = np.zeros(core.combined_shape(buffer_len, (1, self.n_bgp_steps)), dtype=np.float32)
        delete_arr = list(range((buffer_len - self.n_bgp_steps + 1), buffer_len))
        for ii in range(0, buffer_len - (self.n_bgp_steps - 1)):
            flag_status = np.sum(bgp_first_flag[ii + 1:ii + self.n_bgp_steps])  # future steps cant have flag = 1
            new_cgm_target[ii] = bgp_cgm_target[ii:ii + self.n_bgp_steps]
            if flag_status >= 1:
                delete_arr.append(ii)
        bgp_s_hist = torch.from_numpy(np.delete(bgp_s_hist, delete_arr, axis=0)).to(self.device)
        # bgp_s_handcraft = torch.from_numpy(np.delete(bgp_s_handcraft, delete_arr, axis=0)).to(self.device)
        bgp_act = torch.from_numpy(np.delete(bgp_act, delete_arr, axis=0)).to(self.device)
        new_cgm_target = torch.from_numpy(np.delete(new_cgm_target, delete_arr, axis=0)).to(self.device)
        return bgp_s_hist, bgp_act, new_cgm_target

    def update_targets(self, policy):
        # calculate the new targets for value and log prob.
        # done batch wise to reduce memory, aux batch size is used.
        start_idx = 0
        while start_idx < self.size:
            end_idx = min(start_idx + self.aux_batch_size, self.size)
            state_batch = self.old_states[start_idx:end_idx, :, :]
            # handcraft_feat_batch = self.handcraft_feat[start_idx:end_idx, :, :]
            actions_old_batch = self.actions[start_idx:end_idx]
            value_predict = policy.evaluate_critic(state_batch, action=None, cgm_pred=False)
            logprobs, _ = policy.evaluate_actor(state_batch, actions_old_batch)
            self.logprob[start_idx:end_idx, :] = logprobs.detach()
            self.value_target[start_idx:end_idx] = value_predict.detach()
            start_idx += self.aux_batch_size
        assert self.value_target.shape[0] == self.size
        assert self.logprob.shape[0] == self.size
