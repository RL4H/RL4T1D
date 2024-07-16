import os
import csv
import sys
import shutil
import pandas as pd
import numpy as np
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

import logging
import torch
from utils.core import combined_shape
from metrics.metrics import time_in_range
import json

import mlflow

def set_logger(LOG_DIR):
    log_filename = LOG_DIR + '/debug.log'
    #logging.basicConfig(filename=log_filename, filemode='a', format='%(levelname)s - %(message)s', level=logging.INFO)


def setup_folders(args: dict) -> None:  # create the folder which will save experiment data.
    LOG_DIR = args.experiment.experiment_dir
    CHECK_FOLDER = os.path.isdir(LOG_DIR)
    if CHECK_FOLDER:
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR + '/checkpoints')
    os.makedirs(LOG_DIR + '/training/data')
    os.makedirs(LOG_DIR + '/testing/data')
    set_logger(LOG_DIR)

#copy_folder(src=MAIN_PATH + '/agents/'+ self.opt.agent, dst=MAIN_PATH + '/results/' + self.opt.experiment_folder + '/code')  # copy running agent code to outputs

# with open(args.experiment_dir + '/args.json', 'w') as fp:  # save the experiments args.
#     json.dump(vars(args), fp, indent=4)
#     fp.close()



def copy_folder(src, dst):
    for folders, subfolders, filenames in os.walk(src):
        for filename in filenames:
            shutil.copy(os.path.join(folders, filename), dst)


def save_log(experiment_dir, log_name, file_name):
    with open(experiment_dir + file_name + '.csv', 'a+') as f:
        csvWriter = csv.writer(f, delimiter=',')
        csvWriter.writerows(log_name)
        f.close()


class LogExperiment:
    def __init__(self, args):
        self.args = args
        self.model_logs = torch.zeros(7, device=self.args.device)
        save_log(self.args.experiment_dir, [['policy_grad', 'value_grad', 'val_loss', 'exp_var', 'true_var', 'pi_loss', 'avg_rew']], '/model_log')
        save_log(self.args.experiment_dir, [['status', 'rollout', 't_rollout', 't_update', 't_test']], '/experiment_summary')

    def save(self, log_name, data):
        save_log(self.args.experiment_dir, data, log_name)
        data = data[0]
        mlflow.log_metrics({'policy_grad': data[0], 'value_grad': data[1], 'val_loss': data[2], 'exp_var':data[3],
                            'true_var': data[4], 'pi_loss': data[5], 'avg_rew': data[6]})


class LogWorker:
    def __init__(self, args, mode, worker_id):
        self.args = args
        self.worker_mode = mode
        self.worker_id = worker_id

        self.episode_logs = ['epi', 't', 'cgm', 'meal', 'ins', 'rew', 'rl_ins', 'mu', 'sigma', 'prob', 'state_val', 'day_hour', 'day_min']
        self.episode_summary = ['epi', 't', 'reward', 'normo', 'hypo', 'sev_hypo', 'hyper', 'lgbi', 'hgbi', 'ri', 'sev_hyper', 'aBGP_rmse', 'cBGP_rmse']

        save_log(self.args.experiment_dir, [self.episode_logs], '/' + self.worker_mode + '/data/logs_worker_' + str(self.worker_id))
        save_log(self.args.experiment_dir, [self.episode_summary], '/' + self.worker_mode + '/data/' + self.worker_mode + '_episode_summary_' + str(self.worker_id))

        self.episode_history = np.zeros(combined_shape(args.max_epi_length, 13), dtype=np.float32)

    def update(self, counter, episode, state, policy_step, pump_action, rl_action, reward, info):
        self.episode_history[counter] = [episode, counter, state.CGM,
                                                  info['meal'] * info['sample_time'],
                                                  pump_action, reward, rl_action, policy_step['mu'][0],
                                                  policy_step['std'][0],
                                                  policy_step['log_prob'][0], policy_step['state_value'][0],
                                                  info['day_hour'],
                                                  info['day_min']]

    def save(self, episode, counter):
        # log raw data of the episode
        df = pd.DataFrame(self.episode_history[0:counter], columns=self.episode_logs)
        df.to_csv(self.args.experiment_dir + '/' + self.worker_mode + '/data/logs_worker_' + str(self.worker_id) + '.csv',
                  mode='a', header=False, index=False)
        mlflow.log_table(data=df, artifact_file='logs_worker_' + str(self.worker_id) + '.json')

        # log the summary stats for the episode (rollout)
        normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper = time_in_range(df['cgm'])
        save_log(self.args.experiment_dir,
                [[episode, counter, df['rew'].sum(), normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper, 0, 0]],
                '/' + self.worker_mode + '/data/' + self.worker_mode + '_episode_summary_' + str(self.worker_id))
