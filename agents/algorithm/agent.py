import gc
import abc
import time
import torch
from utils.worker import OnPolicyWorker as Worker


class Agent:
    def __init__(self, args):
        self.args = None
        self.policy = None
        self.RolloutBuffer = None

        # workers run the simulations. For each worker an env is created, and the worker ID should be unique.
        self.n_training_workers = args.n_training_workers
        self.n_testing_workers = args.n_testing_workers

        # The offset params above are for visual convenience of raw logs when going through worker logs which are saved as:
        # e.g., worker_10.csv, worker_5000.csv, workers with 5000+ are testing; workers with 6000+ are validation
        self.training_agent_id_offset = 5  # 5, 6, 7, ... (5+n_training_workers)
        self.testing_agent_id_offset = 5000  # 5000, 5001, 5002, ... (5000+n_testing_workers)
        self.validation_agent_id_offset = 6000  # 6000, 6001, 6002, ... (6000+n_val_trials)

    @abc.abstractmethod
    def update(self):
        """
        Implement the update rule.
        """

    def run(self):
        # initialise workers for training
        training_agents = [Worker(args=self.args, mode='training', worker_id=i+self.training_agent_id_offset)
                           for i in range(self.n_training_workers)]

        # initialise workers for testing after each update step
        testing_agents = [Worker(args=self.args, mode='testing', worker_id=i+self.testing_agent_id_offset)
                          for i in range(self.n_testing_workers)]

        # start ppo learning
        rollout, completed_interactions = 0, 0
        while completed_interactions < self.args.total_interactions:  # steps * n_workers * epochs. 3000 is just a large number
            tstart = time.perf_counter()
            # run training workers to collect data
            for i in range(self.n_training_workers):
                training_agents[i].rollout(policy=self.policy, buffer=self.RolloutBuffer.RolloutWorker)
                self.RolloutBuffer.save_rollout(training_agent_index=i)
            self.update()  # update the models
            self.policy.save(rollout)  # save model weights as checkpoints

            # testing: run testing workers on the validation scenario
            with torch.no_grad():
                for i in range(self.n_testing_workers):
                    testing_agents[i].rollout(policy=self.policy, buffer=None)  # these logs will be saved by the worker.

            # update the total number of completed interactions.
            completed_interactions += (self.args.n_step * self.n_training_workers)
            rollout += 1
            gc.collect()  # garbage collector to clean unused objects.

            # decay lr and set entropy coeff to zero to stabilise the policy towards the end.
            if completed_interactions > self.args.n_interactions_lr_decay:
                self.decay_lr()

            experiment_done = True if completed_interactions > self.args.total_interactions else False

            # logging
            print('\n---------------------------------------------------------')
            print('Training Progress: {:.2f}%, Elapsed time: {:.4f} minutes.'.format(min(100.00, (completed_interactions/self.args.total_interactions)*100),
                                                                                     (time.perf_counter() - tstart)/60))
            print('---------------------------------------------------------')

            # print('Rollout Time (seconds): {}, update: {}, testing: {}'.format((t2 - t1), (t4 - t2), (t6 - t4))) if self.args.verbose else None
            # self.LogExperiment.save(log_name='/experiment_summary', data=[[experiment_done, rollout, (t2 - t1), (t4 - t2), (t6 - t4)]])

            # when training complete conduct final validation: typically n=500.
            if experiment_done:
                self.evaluate()

    def evaluate(self):
        print('\n################## Starting Validation Trials #######################')
        validation_agents = [Worker(args=self.args, mode='testing', worker_id=i + self.validation_agent_id_offset)
                             for i in range(self.args.n_val_trials)]
        with torch.no_grad():
            for i in range(self.args.n_val_trials):
                validation_agents[i].rollout(policy=self.policy, buffer=None)
            print('Algorithm Training/Validation Completed Successfully.')
            exit()

    def decay_lr(self):
        self.entropy_coef = 0  # self.entropy_coef / 100
        self.pi_lr = self.pi_lr / 10
        self.vf_lr = self.vf_lr / 10
        for param_group in self.optimizer_Actor.param_groups:
            param_group['lr'] = self.pi_lr
        for param_group in self.optimizer_Critic.param_groups:
            param_group['lr'] = self.vf_lr
