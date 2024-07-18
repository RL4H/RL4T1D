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
from omegaconf import OmegaConf
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
    os.makedirs(LOG_DIR + '/training/')
    os.makedirs(LOG_DIR + '/testing/')
    set_logger(LOG_DIR)

    with open(args.experiment.experiment_dir + '/args.json', 'w') as fp:  # save the experiments args.
        json.dump(OmegaConf.to_container(args, resolve=True), fp, indent=4)
        fp.close()

    # copy running agent code to outputs
    #copy_folder(src=MAIN_PATH + '/agents/algorithm/'+ self.opt.agent, dst=MAIN_PATH + '/results/' + self.opt.experiment_folder + '/code')


def copy_folder(src, dst):
    for folders, subfolders, filenames in os.walk(src):
        for filename in filenames:
            shutil.copy(os.path.join(folders, filename), dst)


def save_log(directory, file, data):
    with open(directory + '/' + file + '.csv', 'a+') as f:
        csvWriter = csv.writer(f, delimiter=',')
        csvWriter.writerows(data)
        f.close()


class Logger:
    def __init__(self, cfg):
        self.mlflow_track = cfg.mlflow.track
        self.experiment_dir = cfg.experiment.experiment_dir

        # setup experiment logs
        self.experiment_logs = cfg.logger.experiment_logs
        self.experiment_logs_keys = {}
        for log in self.experiment_logs:
            save_log(directory=self.experiment_dir, file=log, data=[cfg.logger[log]])
            self.experiment_logs_keys[log] = cfg.logger[log]

        # setup worker logs
        self.worker_logs = cfg.logger.worker_logs
        self.worker_logs_keys = {}
        for log in self.worker_logs:
            self.worker_logs_keys[log] = cfg.logger[log]
        self.logWorker = {}
        worker_ids = [[i+cfg.agent.training_agent_id_offset, 'training'] for i in range(cfg.agent.n_training_workers)] + \
                     [[i+cfg.agent.testing_agent_id_offset, 'testing'] for i in range(cfg.agent.n_testing_workers)] + \
                     [[i+cfg.agent.validation_agent_id_offset, 'testing'] for i in range(cfg.agent.n_val_trials)]
        for ids in range(0, len(worker_ids)):
            for log in self.worker_logs:
                save_log(directory=self.experiment_dir, file=worker_ids[ids][1] + '/'+log+'_' + str(worker_ids[ids][0]),
                         data=[cfg.logger[log]])
            self.logWorker[worker_ids[ids][0]] = LogWorker(cfg.agent, worker_ids[ids][1], worker_ids[ids][0], self.worker_logs_keys)

    def save_rollout(self, data):
        for log in self.experiment_logs:
            arr = []
            for key in self.experiment_logs_keys[log]:
                arr.append(data.get(key, 0))
            save_log(directory=self.experiment_dir, file=log, data=[arr])
        if self.mlflow_track:
            mlflow.log_metrics(data)


class LogWorker:  # TODO: improve handling the logs.
    def __init__(self, args, mode, worker_id, keys):
        self.args = args
        self.keys = keys
        self.worker_mode = mode
        self.worker_id = worker_id
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
        df = pd.DataFrame(self.episode_history[0:counter], columns=self.keys['worker_episode'])

        df.to_csv(self.args.experiment_dir + '/' + self.worker_mode + '/worker_episode_' + str(self.worker_id) + '.csv',
                  mode='a', header=False, index=False)

        if self.args.mlflow_track:
            mlflow.log_table(data=df, artifact_file='logs_worker_' + str(self.worker_id) + '.json')

        # log the summary stats for the episode (rollout)
        normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper = time_in_range(df['cgm'])
        save_log(self.args.experiment_dir,
                self.worker_mode + '/worker_episode_summary_' + str(self.worker_id),
                 [[episode, counter, df['rew'].sum(), normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper, 0, 0]])
