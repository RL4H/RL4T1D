import gc
import abc
import time
import torch

from utils.worker import OnPolicyWorker, OffPolicyWorker, OfflineSampler
from utils.buffers import onpolicy_buffers, offpolicy_buffers
from metrics.metrics import time_in_range
from metrics.statistics import calc_stats

import pandas as pd
from omegaconf import OmegaConf, open_dict

def patient_id_to_label(patient_id): #FIXME move to utils file
    if patient_id < 0 or patient_id >= 30: raise ValueError("Invalid patient id")
    return ["adolescent","adult","child"][patient_id//10] + str(patient_id % 10)


DEBUG_SHOW = True #FIXME remove
class Agent:
    def __init__(self, args, env_args, logger, type="None"):
        self.args = args
        self.env_args = env_args
        self.agent_type = type
        self.policy = None

        with open_dict(self.args):  # TODO: the interface between env - agent, improve?
            self.args.n_features = len(env_args.obs_features)
            self.args.obs_window = env_args.obs_window
            self.args.glucose_max = env_args.glucose_max  # Note the selected sensors range would affect this
            self.args.glucose_min = env_args.glucose_min
            self.args.n_action = env_args.n_actions
            self.args.insulin_min = env_args.insulin_min
            self.args.insulin_max = env_args.insulin_max
            self.args.action_scale = env_args.insulin_max
            self.args.patient_id = env_args.patient_id

            self.args.feature_history = env_args.obs_window  # TODO: refactor G2P2C to use obs_window

        # initialise workers and buffers
        if type == "OnPolicy":
            self.training_agents = [OnPolicyWorker(args=self.args, env_args=self.env_args, mode='training',
                                           worker_id=i+args.training_agent_id_offset) for i in range(self.args.n_training_workers)]
            self.testing_agents = [OnPolicyWorker(args=self.args, env_args=self.env_args, mode='testing',
                                          worker_id=i+args.testing_agent_id_offset) for i in range(self.args.n_testing_workers)]
            self.validation_agents = [OnPolicyWorker(args=self.args, env_args=self.env_args, mode='testing',
                                             worker_id=i + args.validation_agent_id_offset) for i in range(self.args.n_val_trials)]
            self.buffer = onpolicy_buffers.RolloutBuffer(self.args)

        elif type == "OffPolicy":
            self.training_agents = [OffPolicyWorker(args=self.args, env_args=self.env_args, mode='training',
                                           worker_id=i+args.training_agent_id_offset) for i in range(self.args.n_training_workers)]
            self.testing_agents = [OffPolicyWorker(args=self.args, env_args=self.env_args, mode='testing',
                                          worker_id=i+args.testing_agent_id_offset) for i in range(self.args.n_testing_workers)]
            self.validation_agents = [OffPolicyWorker(args=self.args, env_args=self.env_args, mode='testing',
                                             worker_id=i + args.validation_agent_id_offset) for i in range(self.args.n_val_trials)]
            self.buffer = offpolicy_buffers.ReplayMemory(self.args)

        elif type == "Offline":
            if args.data_type == "simulated":
                from utils.sim_data import DataImporter
                
                # setup imported data buffer
                importer = DataImporter(subjects=[patient_id_to_label(self.args.patient_id)],args=args)
                importer.create_queue(minimum_length=args.batch_size*2, maximum_length=args.batch_size*20) #FIXME determine minimum and maximum buffer size, and maybe add args as param?
                importer.queue.start()
                if DEBUG_SHOW: print("Queue Started!")
                # self.buffer = importer.queue

                self.training_agents = [OfflineSampler(args=self.args, env_args=self.env_args, mode='training', worker_id=i+args.training_agent_id_offset,importer_queue=importer.queue) for i in range(self.args.n_training_workers)]
                if DEBUG_SHOW: print("Training Agents Initialised")
                self.testing_agents = [OnPolicyWorker(args=self.args, env_args=self.env_args, mode='testing', worker_id=i+args.testing_agent_id_offset) for i in range(self.args.n_testing_workers)]
                if DEBUG_SHOW: print("Testing Agents Initialised")
                self.validation_agents = [OnPolicyWorker(args=self.args, env_args=self.env_args, mode='testing', worker_id=i + args.validation_agent_id_offset) for i in range(self.args.n_val_trials)]
                if DEBUG_SHOW: print("Validation Agents Initialised")

                self.buffer = offpolicy_buffers.ReplayMemory(self.args)
                if DEBUG_SHOW: print("Agent setup completed")
                
            elif args.data_type == "clinical":
                import utils.cln_data as cln_data
                raise NotImplementedError("Clinical data importing not implemented.")
            
            else:
                raise KeyError("Invlid data_type parameter.")

        self.logger = logger

    @abc.abstractmethod
    def update(self):
        """
        Implement the update rule.
        """

    def run(self):
        # learning
        rollout, completed_interactions, logs = 0, 0, {}
        while completed_interactions < self.args.total_interactions:  # steps * n_workers * epochs.
            tstart = time.perf_counter()
            for i in range(self.args.n_training_workers):  # run training workers to collect data

                # TODO: handle buffers better
                if self.agent_type == "OnPolicy":
                    self.training_agents[i].rollout(policy=self.policy, buffer=self.buffer.Rollout, logger=self.logger.logWorker)
                    self.buffer.save_rollout(training_agent_index=i)
                elif self.agent_type == "OffPolicy":
                    self.training_agents[i].rollout(policy=self.policy, buffer=self.buffer, logger=self.logger.logWorker)
                elif self.agent_type == "Offline": #FIXME
                    self.training_agents[i].rollout(policy=self.policy, buffer=self.buffer, logger=self.logger.logWorker) 


            logs = self.update()  # update the models
            self.logger.save_rollout(logs)  # logging
            self.policy.save(rollout)  # save model weights as checkpoints

            # testing: run testing workers on the validation scenario
            with torch.no_grad():
                for i in range(self.args.n_testing_workers):
                    self.testing_agents[i].rollout(policy=self.policy, buffer=None, logger=self.logger.logWorker)  # these logs will be saved by the worker.

            # update the total number of completed interactions.
            completed_interactions += (self.args.n_step * self.args.n_training_workers)
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

            # when training complete conduct final validation: typically n=500.
            if experiment_done:
                self.evaluate()

    def evaluate(self):  # TODO: refactor below
        print('\n---------------------------------------------------------')
        print('===> Starting Validation Trials ....')
        
        with torch.no_grad():
            for i in range(self.args.n_val_trials):
                self.validation_agents[i].rollout(policy=self.policy, buffer=None, logger=self.logger.logWorker)

            # calculate the final metrics.
            cohort_res, summary_stats = [], []
            secondary_columns = ['epi', 't', 'reward', 'normo', 'hypo', 'sev_hypo', 'hyper', 'lgbi',
                             'hgbi', 'ri', 'sev_hyper', 'aBGP_rmse', 'cBGP_rmse']
            data = []
            FOLDER_PATH = self.args.experiment_folder+'/testing/'
            for i in range(0, self.args.n_val_trials):
                test_i = 'worker_episode_'+str(self.args.validation_agent_id_offset+i)+'.csv'
                df = pd.read_csv(FOLDER_PATH+ '/'+test_i)
                normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper = time_in_range(df['cgm'])
                reward_val = df['rew'].sum()*(100/288)
                e = [[i, df.shape[0], reward_val, normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper, 0, 0]]
                dataframe = pd.DataFrame(e, columns=secondary_columns)
                data.append(dataframe)
            res = pd.concat(data)
            res['PatientID'] = self.args.patient_id
            res.rename(columns={'sev_hypo':'S_hypo', 'sev_hyper':'S_hyper'}, inplace=True)
            summary_stats.append(res)
            metric=['mean', 'std', 'min', 'max']
            print(calc_stats(res, metric=metric, sim_len=288))

            print('\nAlgorithm Training/Validation Completed Successfully.')
            print('---------------------------------------------------------')
            exit()

    def decay_lr(self):
        return
        # self.entropy_coef = 0  # self.entropy_coef / 100
        # self.pi_lr = self.pi_lr / 10
        # self.vf_lr = self.vf_lr / 10
        # for param_group in self.optimizer_Actor.param_groups:
        #     param_group['lr'] = self.pi_lr
        # for param_group in self.optimizer_Critic.param_groups:
        #     param_group['lr'] = self.vf_lr
