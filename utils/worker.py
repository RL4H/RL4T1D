from environment.t1denv import T1DEnv
from utils.control_space import ControlSpace
import torch
from copy import deepcopy
import numpy as np
from utils.core import linear_scaling


class Worker(T1DEnv):
    def __init__(self, args, env_args, mode, worker_id):
        T1DEnv.__init__(self, env_args, mode, worker_id)
        self.env_args = env_args
        self.worker_id = worker_id
        self.episode, self.counter = 0, 0
        self.rollout_steps = args.n_step if self.worker_mode == 'training' else args.max_test_epi_len
        self.stop_factor = (args.max_epi_length - 1) if self.worker_mode == 'training' else (args.max_test_epi_len - 1)
        self.controlspace = ControlSpace(control_space_type=args.control_space_type,
                                         insulin_min=self.action_space.low[0],
                                         insulin_max=self.action_space.high[0])
    def _reset(self):
        self.episode += 1
        self.counter = 0
        self.state, self.info = self.reset()


class OnPolicyWorker(Worker):
    def __init__(self, args, env_args, mode, worker_id):
        Worker.__init__(self, args, env_args, mode, worker_id)

    def rollout(self, policy=None, buffer=None, logger=None):
        logger = logger[self.worker_id]
        if self.worker_mode != 'training':  # always a fresh env for testing.
            self._reset()

        self.ins_history = [0] * self.args.obs_window #FIXME make consistent with params

        for _ in range(0, self.rollout_steps):

            rl_action = policy.get_action(self.state)


            self.ins_history.append(rl_action['action'][0])
            pump_action = self.controlspace.map(agent_action=rl_action['action'][0])  # map RL action => control space (pump)

            state, reward, is_done, info = self.env.step(pump_action)

            if self.worker_mode == 'training': # store -> rollout data for training.
                is_first = True if self.counter == 0 else False
                buffer.store(self.state, rl_action['action'][0], reward, rl_action['state_value'], rl_action['log_prob'], info['cgm'].CGM, is_first)

            logger.update(self.counter, self.episode, info['cgm'], rl_action, pump_action, 0, reward, info)

            self.state = state  # update -> state.
            self.info = info

            self.counter += 1
            if is_done or self.counter > self.stop_factor:  # episode termination criteria.
                logger.save(self.episode, self.counter)
                if self.worker_mode == 'training':
                    final_val = policy.get_final_value(self.state)
                    buffer.finish_path(final_val)
                # stop rollout if this is a testing worker; else reset an env and continue.
                if self.worker_mode != 'training': break
                self._reset()
                self.ins_history = [0] * self.args.obs_window #FIXME make consistent with params
        return


class OffPolicyWorker(Worker):
    def __init__(self, args, env_args, mode, worker_id):
        Worker.__init__(self, args, env_args, mode, worker_id)

    def rollout(self, policy=None, buffer=None, logger=None):
        logger = logger[self.worker_id]
        if self.worker_mode != 'training':  # always a fresh env for testing.
            self._reset()

        for _ in range(0, self.rollout_steps):

            if len(self.args.obs_features) == 0:
                rl_action = policy.get_action(self.state)
            else:
                features = [ ((self.rollout_steps * 5) // 60) % 24 ] #FIXME undo hard coding of features
                rl_action = policy.get_action(self.state, features)

            pump_action = self.controlspace.map(agent_action=rl_action['action'][0])  # map RL action => control space (pump)

            state, reward, is_done, info = self.env.step(pump_action)

            this_state = deepcopy(self.state)
            self.state = state  # update -> state.

            if self.worker_mode == 'training':  # store -> rollout data for training.
                buffer.store(this_state, rl_action['action'][0], reward, self.state, is_done)

            rl_action['log_prob']=[0] # todo fix the logger
            rl_action['state_value'] =[0]  # todo fix the logger
            logger.update(self.counter, self.episode, info['cgm'], rl_action, pump_action, 0, reward, info)

            self.counter += 1
            if is_done or self.counter > self.stop_factor:  # episode termination criteria.
                logger.save(self.episode, self.counter)
                # stop rollout if this is a testing worker; else reset an env and continue...
                if self.worker_mode != 'training': break
                self._reset()
        return

class OfflineSampler(Worker):
    def __init__(self, args, env_args, mode, worker_id, importer_queue):
        # Worker.__init__(self, args, env_args, mode, worker_id)
        self.importer_queue = importer_queue
        self.args, self.env_args = args, env_args
        self.worker_mode, self.worker_id = mode, worker_id
        
        self.rollout_steps = args.n_step if self.worker_mode == 'training' else args.max_test_epi_len
        self.stop_factor = (args.max_epi_length - 1) if self.worker_mode == 'training' else (args.max_test_epi_len - 1)
        
        self.episode, self.counter = 0, 0

    def rollout(self, policy=None, buffer=None,logger=None):
        logger = logger[self.worker_id]

        for _ in range(0, self.rollout_steps):
            item = self.importer_queue.pop()
            # print("Saving transition, ",item)
            buffer.store(*item)

        return

        




