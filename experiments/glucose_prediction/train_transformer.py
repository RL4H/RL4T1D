import numpy as np
import pandas as pd
import os
import sys
from decouple import config
from torch.utils.data import random_split
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from utils.sim_data import convert_trial_into_windows
from experiments.glucose_prediction.transformer_decoder import MultiBranchAutoregressiveDecoder


"""
Transformer setup:

Given an array of the last 24 hours (24 * (60/5) = 288 data points) of glucose, meal, and insulin values. Predicts the next glucose value, and provided the next meal and insulin values.
Mapping: 3 x 288 -> 1x1

evaluated by RMSE to target values, validated with dataset


"""

class SimpleLogger:
    def __init__(self, args, title, keys):
        self.args = args
        self.keys, self.title = keys, title
        self.save_dest = MAIN_PATH + args.save_path + args.run_name + '/' + title + ".csv"
        self.di = dict()
        for k in self.keys:
            self.di[k] = []
        self.log_len = 0
    def add(self, log):
        for k in self.keys:
            if k in log: self.di[k].append(log[k])
            else: self.di[k].append(None)
        self.log_len += 1
    def save_logs(self):
        with open(self.save_dest, 'w') as f:
            f.write('\n'.join(
                [','.join(self.keys)] + 
                [','.join([str(self.di[k][row]) for k in self.keys]) for row in range(self.log_len)]
            ))
        print("Logs saved to",self.save_dest)
    def graph(self,key, ignore_empty):
        y_vals = self.di[key]
        if ignore_empty: y_vals = [i for i in y_vals if i != None]
        x_vals = list(range(len(y_vals)))

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, marker='o', linestyle='-')
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.title(f"Value of {key} over time.")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def pretty_seconds(seconds):
    s = int(seconds % 60)
    m = int((seconds // 60) % 60)
    h = int(seconds // (60 * 60))
    return f"{h}:{m:02d}:{s:02d}"
def pop_subset(subset):
    try:
        item = next(iter(subset))
    except StopIteration:
        raise NotImplementedError("#FIXME implement looping")
    return item
def pop_subset_batch(subset,n):
    return [pop_subset(subset) for _ in range(n)]



@hydra.main(version_base=None, config_path=MAIN_PATH + "/experiments/glucose_prediction/config/", config_name="prediction_config.yaml")
def main(args: DictConfig):
    device = args.device
    batch_size = args.batch_size
    assert args.input_window + args.t_future == args.obs_window

    if args.policy_type == "offline":
        print(f"Using Offline RL with a {args.data_type} data soure on patient id {args.patient_id}.")
        if args.data_type == "simulated":
            from utils.sim_data import DataImporter

            importer = DataImporter(args=args,env_args=args) #FIXME probably don't handle the args this way
            dataset = importer.create_torch_dataset(mapping = convert_trial_into_windows)

        elif args.dat_type == "clinical":
            raise NotImplementedError
        
        dataset_len = len(dataset)

        vld_len = batch_size * 4
        eval_len = batch_size * 12
        train_len = dataset_len - vld_len - eval_len

        train_set, vld_set, eval_set = tuple(random_split(dataset, [train_len, vld_len, eval_len], generator=torch.Generator().manual_seed(args.split_seed)))
        vld_loader = DataLoader(vld_set, batch_size=batch_size, shuffle=False)
    elif args.policy_type == "online":
        raise NotImplementedError("Online data collection not yet implemented.")
    else:
        raise ValueError("Invalid value ({args.policy_type}) of argument `policy type`.")

    print("Dataset initialised with length",len(dataset),"split into lengths of",len(train_set),"and",len(eval_set))

    # train_set = dataset #FIXME remove and fix error dealing with subsets
    decoder = MultiBranchAutoregressiveDecoder(args)
    try:
        os.mkdir(MAIN_PATH + args.save_path + args.run_name)
    except FileExistsError:
        print("Experiment has previously been run, data will be overwritten.")
    
    with open(MAIN_PATH + args.save_path + args.run_name + '/args.json', 'w') as fp:  # save the experiments args.
        json.dump(OmegaConf.to_container(args, resolve=True), fp, indent=4)
        fp.close()

    trn_logs = SimpleLogger(args, args.train_log_name, args.train_log_features)
    vld_logs = SimpleLogger(args, args.vld_log_name, args.vld_log_features)

    logging_interval = 0.5 #show every 0.5 %
    next_interval = logging_interval
    overall_start_time = datetime.now()
    interval_start_time = overall_start_time

    durations = [] #durations in seconds

    # Training
    print("Beginning Training.")
    interactions = 0
    iteration = 0
    while interactions < args.total_interactions:
        data = torch.as_tensor(np.array(pop_subset_batch(train_set, batch_size)), dtype=torch.float32, device=device) #(B, T, D)
        
        data_ctx = data[:, :decoder.Tc, :]
        data_fut = data[:, decoder.Tc:, :]

        new_log = decoder.update(data_ctx, data_fut)
        trn_logs.add(new_log)
        
        interactions += batch_size

        percent_complete = (interactions / args.total_interactions) * 100
        if percent_complete > next_interval:
            next_time = datetime.now()
            dur = (next_time - interval_start_time).total_seconds()
            interval_start_time = next_time
            durations.append(dur)
            time_remaining = max(0, ((100 - percent_complete) / logging_interval) * np.mean(durations)) #calculate expected time remaining

            print(f"================= Training {percent_complete:.2f}% complete, interval took {pretty_seconds(dur)}. Expected time remaining for training is {pretty_seconds(time_remaining)}. Loss={new_log["loss"]:.2f}")
            next_interval += logging_interval

        if iteration % args.vld_freq == 0:
            print("\t\t==== Performing Validation")
            loss_list = []
            vld_iter = iter(vld_loader) #reset iterations

            for _, batch in enumerate(vld_iter):
    
                data = torch.as_tensor(np.array(batch), dtype=torch.float32, device=device) #(B, T, D)

                data_ctx = data[:, :decoder.Tc, :]
                data_fut = data[:, decoder.Tc:, :]
        
                new_log = decoder.eval_update(data_ctx, data_fut)
                loss_list.append(new_log['loss'])
            
            mean_loss = np.mean(loss_list)
            vld_logs.add( {'loss' : mean_loss})

            print(f"\t\t==== Validation Complete: Loss of {mean_loss:.2f}")
        iteration += 1

    print("Training complete.")
    trn_logs.save_logs()
    vld_logs.save_logs()

    torch.save(decoder, MAIN_PATH + args.save_path + args.run_name + '/policy.pth')

    # Evaluation
    evl_logs = SimpleLogger(args, args.eval_log_name, args.eval_log_features)

    next_interval = logging_interval
    overall_start_time = datetime.now()
    interval_start_time = overall_start_time
    durations = [] #durations in seconds
    interactions = 0

    print("Beginning evaluation.")
    while interactions < len(eval_set):

        data = torch.as_tensor(np.array(pop_subset_batch(eval_set, batch_size)), dtype=torch.float32, device=device) #(B, T, D)

        data_ctx = data[:, :decoder.Tc, :]
        data_fut = data[:, decoder.Tc:, :]
        
        new_log = decoder.eval_update(data_ctx, data_fut)
        evl_logs.add(new_log)

        interactions += batch_size

        percent_complete = (interactions / len(eval_set)) * 100
        if percent_complete > next_interval:
            next_time = datetime.now()
            dur = (next_time - interval_start_time).total_seconds()
            interval_start_time = next_time
            durations.append(dur)
            time_remaining = max(0, ((100 - percent_complete) / logging_interval) * np.mean(durations))
            print(f"================= Evaluation {percent_complete:.2f}% complete, interval took {pretty_seconds(dur)}. Expected time remaining for training is {pretty_seconds(time_remaining)}.")
            next_interval += logging_interval

    print("Evaluation complete.")
    evl_logs.save_logs()
    print("Logs saved to",evl_logs.save_dest)

    trn_logs.graph(key="loss",ignore_empty=True)
    vld_logs.graph(key="loss",ignore_empty=True)
    evl_logs.graph(key="loss",ignore_empty=True)

if __name__ == '__main__':
    main()
