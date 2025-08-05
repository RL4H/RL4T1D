import gc
import abc
import time
import torch

from utils.worker import OnPolicyWorker, OffPolicyWorker, OfflineSampler
from utils.buffers import onpolicy_buffers, offpolicy_buffers
from metrics.metrics import time_in_range
from metrics.statistics import calc_stats

import pandas as pd
from omegaconf import OmegaConf, DictConfig, open_dict


from decouple import config
MAIN_PATH = config('MAIN_PATH')
SIM_DATA_PATH = config('SIM_DATA_PATH')
CLN_DATA_SAVE_DEST = "/home/users/u7482502/data/cln_pickled_data" #FIXME make into env variable 

from experiments.glucose_prediction.portable_loader import CompactLoader, load_compact_loader_object
from utils.sim_data import patient_id_to_label

from copy import deepcopy

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

        self.using_OPE = self.agent_type == "Offline" and self.args.data_type == "clinical"
        if self.using_OPE: self.args.n_val_trials = 0

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

                if args.data_preload:
                    folder = SIM_DATA_PATH + "/object_save/"
                    data_save_path = folder + f"temp_data_patient_{args.patient_id}_{args.seed}.pkl"
                    data_save_path_args = folder + f"temp_args_{args.patient_id}_{args.seed}.pkl"
                    print("Loading prebuilt data from",data_save_path_args)
                    
                    queue = load_compact_loader_object(data_save_path_args)
                    queue.start()
                    gc.collect()
                else:
                    from utils.sim_data import DataImporter, calculate_augmented_features
                    from utils.core import inverse_linear_scaling, MEAL_MAX, calculate_features

                    importer = DataImporter(args=args,env_args=env_args)

                    handler = importer.get_trials()
                    handler.flatten()
                    flat_trials = handler.flat_trials
                    del handler
                    del importer
                    queue = CompactLoader(
                        args, args.batch_size*10, args.batch_size*101, 
                        flat_trials,
                        lambda trial : calculate_augmented_features(trial, args, env_args),
                        1,
                        lambda trial : max(0, len(trial) - args.obs_window - 1),
                        args.seed,
                        0,
                        folder=SIM_DATA_PATH + "/object_save/"
                    )
                    gc.collect()
                    queue.start()
                    gc.collect()


                if args.use_all_interactions: #override total_interactions, use 98% of total transitions to avoid spilling over
                    print("overriding total interactions from",args.total_interactions,"to",queue.total_transitions)
                    self.args.total_interactions = int(queue.total_transitions*0.98)
                    args.total_interactions = int(queue.total_transitions*0.98)
                elif args.total_interactions > queue.total_transitions:
                    print("WARNING: total interactions set (",args.total_interactions,") is greater than available data (",queue.total_transitions,"). ")

                # self.training_agents = []
                # self.testing_agents = [OnPolicyWorker(args=self.args, env_args=self.env_args, mode='testing', worker_id=i+args.testing_agent_id_offset) for i in range(self.args.n_testing_workers)]
                # if DEBUG_SHOW: print("Testing Agents Initialised")
                # self.validation_agents = [OnPolicyWorker(args=self.args, env_args=self.env_args, mode='testing', worker_id=i + args.validation_agent_id_offset) for i in range(self.args.n_val_trials)]
                # if DEBUG_SHOW: print("Validation Agents Initialised")
                
            elif args.data_type == "clinical":

                if args.data_preload:

                    folder = CLN_DATA_SAVE_DEST + '/'
                    data_save_path = folder + f"temp_data_patient_{args.patient_id}_{args.seed}.pkl"
                    data_save_path_args = folder + f"temp_args_{args.patient_id}_{args.seed}.pkl"
                    
                    print("Loading prebuilt data from", data_save_path_args)
                    queue = load_compact_loader_object(data_save_path_args)
                    queue.start()
                    gc.collect()
                
                else:
                    from utils.cln_data import ClnDataImporter, get_patient_attrs, convert_df_to_arr

                    gc.collect()
                    print("Importing for patient id",args.patient_id,"index",get_patient_attrs("clinical" + str(args.patient_id))['subj_id'])
                    args = OmegaConf.create({
                        "patient_ind" : args.patient_id,
                        "patient_id" : args.patient_id,
                        "batch_size" : 8192,
                        "data_type" : "simulated", #simulated | clinical,
                        "data_protocols" : ["evaluation","training"], #None defaults to all,
                        "data_algorithms" : ["G2P2C","AUXML", "PPO","TD3"], #None defaults to all,
                        "obs_window" : 12,
                        "control_space_type" : 'exponential_alt',
                        "insulin_min" : 0,
                        "insulin_max" : 20,
                        "glucose_min" : 39,
                        "glucose_max" : 600,
                        "obs_features" : ['cgm','insulin','day_hour']
                    })

                    importer = ClnDataImporter(args=args,env_args=args)
                    
                    flat_trials = importer.load()
                    del importer

                    queue = CompactLoader(
                        args, args.batch_size*10, args.batch_size*101, 
                        flat_trials,
                        lambda trial : calculate_augmented_features(convert_df_to_arr(trial), args, args),
                        1,
                        lambda trial : max(0, len(trial) - args.obs_window - 1) if trial['meta'].loc[0].split('_')[-1] == 'Pump' else 0, #exclude non pump data
                        0,
                        0,
                        folder= CLN_DATA_SAVE_DEST + '/current_run/'
                    )

                    gc.collect()
                    queue.start()
                    gc.collect()

                    if len(queue) == 0:
                        raise ValueError("Queue length is 0.")
                    

            else:
                raise KeyError("Invlid data_type parameter.")
            
            if DEBUG_SHOW: print("Queue Started!")

            self.training_agents = []#[OfflineSampler(args=self.args, env_args=self.env_args, mode='training', worker_id=i+args.training_agent_id_offset,importer_queue=queue) for i in range(self.args.n_training_workers)]
            if DEBUG_SHOW: print("Training Agents Initialised")

            if args.data_type == "simulated": self.testing_agents = [OnPolicyWorker(args=self.args, env_args=self.env_args, mode='testing', worker_id=i+args.testing_agent_id_offset) for i in range(self.args.n_testing_workers)]
            else: self.testing_agents = [] #FIXME
            if DEBUG_SHOW: print("Testing Agents Initialised")

            if args.data_type == "simulated": self.validation_agents = [OnPolicyWorker(args=self.args, env_args=self.env_args, mode='testing', worker_id=i + args.validation_agent_id_offset) for i in range(self.args.n_val_trials)]
            else: self.validation_agents = [] #FIXME
            if DEBUG_SHOW: print("Validation Agents Initialised")

            self.buffer = offpolicy_buffers.ReplayMemory(self.args)
            self.buffer_queue = queue
            if DEBUG_SHOW: print("Agent setup completed")

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
                # elif self.agent_type == "Offline": 
                #     self.training_agents[i].rollout(policy=self.policy, buffer=self.buffer, logger=self.logger.logWorker)

            if self.agent_type == "Offline":
                #store batch directly to avoid bottleneck
                self.buffer.store_batch(self.buffer_queue.pop_batch(self.args.replay_buffer_step))

            # self.alpha = self.args.alpha * (1 - completed_interactions / self.args.total_interactions)
            logs = self.update()  # update the models
            self.logger.save_rollout(logs)  # logging
            self.policy.save(rollout)  # save model weights as checkpoints

            # testing: run testing workers on the validation scenario
            with torch.no_grad():
                if self.using_OPE:
                    pass #TODO 
                else:
                    for i in range(self.args.n_testing_workers):
                        self.testing_agents[i].rollout(policy=self.policy, buffer=None, logger=self.logger.logWorker)  # these logs will be saved by the worker.

            # update the total number of completed interactions.
            completed_interactions += (self.args.n_step * self.args.n_training_workers)
            self.completed_interactions = completed_interactions
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
            
            if self.using_OPE:
                print("Conducting Offline Evaluation")

                del self.buffer

                res = self.evaluate_fqe()
                for k in res:
                    print(k, '\t', res[k])


                #TODO: implement OPE
                print("OPE Completed")
                exit()
            else:

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
