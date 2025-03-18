from environment.t1denv import T1DEnv
from utils.control_space import ControlSpace
import torch
from copy import deepcopy


class Worker(T1DEnv):
    def __init__(self, args, env_args, mode, worker_id, rwd_params = None):
        T1DEnv.__init__(self, env_args, mode, worker_id)
        self.worker_id = worker_id
        self.episode, self.counter = 0, 0
        self.rwd_params = rwd_params
        self.rollout_steps = args.n_step if self.worker_mode == 'training' else args.max_test_epi_len
        self.stop_factor = (args.max_epi_length - 1) if self.worker_mode == 'training' else (args.max_test_epi_len - 1)
        self.controlspace = ControlSpace(control_space_type=args.control_space_type,
                                         insulin_min=self.action_space.low[0],
                                         insulin_max=self.action_space.high[0])

    def _reset(self):
        self.episode += 1
        self.counter = 0
        self.state = self.reset()


class OnPolicyWorker(Worker):
    def __init__(self, args, env_args, mode, worker_id, rwd_params = None):
        Worker.__init__(self, args, env_args, mode, worker_id, rwd_params)

    def rollout(self, policy=None, buffer=None, logger=None):
        logger = logger[self.worker_id]
        if self.worker_mode != 'training':  # always a fresh env for testing.
            self._reset()

        for _ in range(0, self.rollout_steps):

            rl_action = policy.get_action(self.state)  # get RL action
            pump_action = self.controlspace.map(agent_action=rl_action['action'][0])  # map RL action => control space (pump)

            state, reward, is_done, info = self.env.step(pump_action)
            if self.rwd_params is not None:
                obs = torch.tensor([x[0] for x in state]) #need to check this is correct for PPO
                print("obs: ", obs.get_device())
                print("w: ", self.rwd_params.get_device())
                reward = torch.matmul(self.rwd_params, obs)
                

            if self.worker_mode == 'training': # store -> rollout data for training.
                is_first = True if self.counter == 0 else False
                buffer.store(self.state, rl_action['action'][0], reward, rl_action['state_value'], rl_action['log_prob'], info['cgm'].CGM, is_first)

            logger.update(self.counter, self.episode, info['cgm'], rl_action, pump_action, 0, reward, info)

            self.state = state  # update -> state.
            self.counter += 1
            if is_done or self.counter > self.stop_factor:  # episode termination criteria.
                logger.save(self.episode, self.counter)
                if self.worker_mode == 'training':
                    final_val = policy.get_final_value(self.state)
                    buffer.finish_path(final_val)
                # stop rollout if this is a testing worker; else reset an env and continue.
                if self.worker_mode != 'training': break
                self._reset()
        return
    
    def update_rwd_params(self, w):
        self.rwd_params = w.to(self.args.device)



class OffPolicyWorker(Worker):
    def __init__(self, args, env_args, mode, worker_id):
        Worker.__init__(self, args, env_args, mode, worker_id)

    def rollout(self, policy=None, buffer=None, logger=None):
        logger = logger[self.worker_id]
        if self.worker_mode != 'training':  # always a fresh env for testing.
            self._reset()

        for _ in range(0, self.rollout_steps):

            rl_action = policy.get_action(self.state)
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
