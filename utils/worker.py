from environment.t1denv import T1DEnv
from utils.logger import LogWorker
from utils.control_space import ControlSpace


class Worker(T1DEnv):
    def __init__(self, args, env_args, mode, worker_id):
        T1DEnv.__init__(self, env_args, mode, worker_id)
        self.episode, self.counter = 0, 0
        self.rollout_steps = args.n_step if self.worker_mode == 'training' else args.max_test_epi_len
        self.stop_factor = (args.max_epi_length - 1) if self.worker_mode == 'training' else (args.max_test_epi_len - 1)
        self.controlspace = ControlSpace(args)
        self.logger = LogWorker(args, mode, worker_id)

    def _reset(self):
        self.episode += 1
        self.counter = 0
        self.state = self.reset()


class OnPolicyWorker(Worker):
    def __init__(self, args, env_args, mode, worker_id):
        Worker.__init__(self, args, env_args, mode, worker_id)

    def rollout(self, policy=None, buffer=None):
        if self.worker_mode != 'training':  # always a fresh env for testing.
            self._reset()

        for steps in range(0, self.rollout_steps):

            rl_action = policy.get_action(self.state)  # get RL action
            pump_action = self.controlspace.map(agent_action=rl_action['action'][0])  # map RL action => control space (pump)

            state, reward, is_done, info = self.env.step(pump_action)

            # store -> rollout data for training.
            if self.worker_mode == 'training':
                is_first = True if self.counter == 0 else False
                print(rl_action)
                buffer.store(self.state, rl_action['action'][0], reward, rl_action['state_value'], rl_action['log_prob'][0], info['cgm'].CGM, is_first)

            self.logger.update(self.counter, self.episode, info['cgm'], rl_action, pump_action, 0, reward, info)

            self.state = state  # update -> state.
            self.counter += 1

            if is_done or self.counter > self.stop_factor:  # episode termination criteria.
                self.logger.save(self.episode, self.counter)
                if self.worker_mode == 'training':
                    final_val = policy.get_final_value(self.state)
                    buffer.finish_path(final_val)
                # stop rollout if this is a testing worker; else reset an env and continue.
                if self.worker_mode != 'training': break
                self._reset()
        return


import torch
from copy import deepcopy


class OffPolicyWorker(Worker):
    def __init__(self, args, env_args, mode, worker_id):
        Worker.__init__(self, args, env_args, mode, worker_id)

    def rollout(self, policy=None, buffer=None):

        if self.worker_mode != 'training':  # always a fresh env for testing.
            self._reset()

        for steps in range(0, self.rollout_steps):

            rl_action = policy.get_action(self.state)
            pump_action = self.controlspace.map(agent_action=rl_action['action'][0])  # map RL action => control space (pump)

            state, reward, is_done, info = self.env.step(pump_action)

            this_state = deepcopy(self.state)

            self.state = state  # update -> state.

            # store -> rollout data for training. TODO: generalise among different buffers.
            if self.worker_mode == 'training':
                buffer.push(torch.as_tensor(this_state, dtype=torch.float32, device=self.args.device).unsqueeze(0),
                                   torch.as_tensor([rl_action['action'][0]], dtype=torch.float32, device=self.args.device),
                                   torch.as_tensor([reward], dtype=torch.float32, device=self.args.device),
                                   torch.as_tensor(self.state, dtype=torch.float32, device=self.args.device).unsqueeze(0),
                                   torch.as_tensor([is_done], dtype=torch.float32, device=self.args.device))

            rl_action['log_prob']=[0] # todo fix the logger
            rl_action['state_value'] =[0]  # todo fix the logger
            self.logger.update(self.counter, self.episode, info['cgm'], rl_action, pump_action, 0, reward, info)

            self.counter += 1

            if is_done or self.counter > self.stop_factor:  # episode termination criteria.
                self.logger.save(self.episode, self.counter)
                # stop rollout if this is a testing worker; else reset an env and continue.
                if self.worker_mode != 'training': break
                self._reset()
        return
