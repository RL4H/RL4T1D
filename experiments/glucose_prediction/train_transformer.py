import numpy as np
import pandas as pd
import os
import sys
from decouple import config
from torch.utils.data import random_split
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime, timedelta
import gc
import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
torch.cuda.empty_cache()

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from utils.sim_data import convert_trial_into_windows
from utils.core import inverse_linear_scaling, MEAL_MAX, calculate_features
from experiments.glucose_prediction.transformer_decoder import AutoregressiveDecoder
from experiments.glucose_prediction.portable_loader import CompactLoader, load_compact_loader_object


"""
Transformer setup:

Given an array of the last 24 hours (24 * (60/5) = 288 data points) of glucose, meal, and insulin values. Predicts the next glucose value, and provided the next meal and insulin values.
Mapping: 3 x 288 -> 1x1

evaluated by RMSE to target values, validated with dataset


"""

PRELOAD = True

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

EPISODE_HEADERS = ["epi", "cgm", "cgm_pred", "ins", "meal", "day_hour", "day_min", 't']
def save_episode_list(epi_list, file_dest):
    txt_lines = [','.join(EPISODE_HEADERS)]
    for epi_n, epi in enumerate(epi_list):
        for row in range(len(epi)):
            txt_lines.append( ','.join([str(epi_n)] + [str(epi[col][row]) for col in EPISODE_HEADERS[1:]]) )
    
    with open(file_dest, 'w') as f:
        f.write('\n'.join(txt_lines))
def compose_episode_data(y_pred, data_fut, args, env_args):
    """ 
    y_pred : (B, Tf)
    data_fut : (B, Tf, D)
    args
    env_args
    """
    B, Tf, D = data_fut.shape

    feature_list = env_args.obs_features

    t_list = list(range(Tf))

    epi_list = []
    for batch in range(B):
        glucose_predicted = [inverse_linear_scaling(y, args.glucose_min, args.glucose_max) for y in y_pred[batch, :]]
        glucose_actual = [inverse_linear_scaling(y, args.glucose_min, args.glucose_max) for y in data_fut[batch, :, feature_list.index("cgm")]]
        insulin = [inverse_linear_scaling(ins, args.insulin_min, args.insulin_max) for ins in data_fut[batch, :, feature_list.index("insulin")]]
        hours = [int(round(inverse_linear_scaling(hour, 0, 23))) for hour in data_fut[batch, :, feature_list.index("day_hour")]]
        # mins = [inverse_linear_scaling(hour, 0, 23) for hour in data_fut[batch, :, feature_list.index("day_min")]]
        meals = [round(inverse_linear_scaling(meal, 0, MEAL_MAX),3) for meal in data_fut[batch, :, feature_list.index("meal")]]
        
        epi_df = pd.DataFrame({
            'epi' : [0] * Tf,
            'cgm' : glucose_actual,
            'cgm_pred' : glucose_predicted,
            'ins' : insulin,
            'day_hour' : hours,
            'day_min' : [0] * Tf, #FIXME
            't' : t_list,
            'meal' : meals
        })
        epi_list.append(epi_df)
    
    return epi_list

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
def inverse_cgm(cgm_scaled, args):
    return inverse_linear_scaling(cgm_scaled, args.glucose_min, args.glucose_max)
def inverse_cgm_RMSE(loss, args): #converts RMSE as if unscaled glucose values were passed
    return 0.5 * (args.glucose_max - args.glucose_min) * loss


class DataSampler:
    def __init__(self, base_set, batch_size):
        self.batch_size = batch_size
        self.data_loader = DataLoader(base_set, 1, True)
        self.len = sum([len(i) for i in self.data_loader])
        self.reset()
    def reset(self):
        self.data_iter = iter(self.data_loader)
        # print(len(self.data_iter))
        self.in_trial_ind = self.trial_ind = 0
        self.current_trial = next(self.data_iter) #FIXME check if this is the 0th entry or the 1st
    def pop(self):
        data = self.current_trial[self.in_trial_ind]

        self.in_trial_ind += 1
        if self.in_trial_ind >= len(self.current_trial):
            self.in_trial_ind = 0
            self.trial_ind += 1
            try:
                self.current_trial = next(self.data_iter)
            except StopIteration:
                print("\tIteration Resetting.")
                self.reset()

        return data[0] #remove (empty) first axis
    def pop_batch(self): #takes own batch size param
        output = np.array([self.pop() for _ in range(self.batch_size)])
        return output

@hydra.main(version_base=None, config_path=MAIN_PATH + "/experiments/glucose_prediction/config/", config_name="prediction_config.yaml")
def main(args: DictConfig):
    
    device = args.device
    batch_size = args.batch_size
    mini_batch_n = args.mini_batch_n
    assert args.input_window + args.t_future == args.obs_window



    # inverse_cgm_func = lambda cgm : inverse_cgm(cgm, args)
    inverse_rmse_func = lambda loss : inverse_cgm_RMSE(loss, args)

    if args.policy_type == "offline":
        print(f"Using Offline RL with a {args.data_type} data soure on patient id {args.patient_id}.")
        if args.data_type == "simulated":

            if PRELOAD:
                print("Preloading data")
                folder = MAIN_PATH + f"/experiments/glucose_prediction/"
                data_save_path = folder + f"temp_data_patient_{args.patient_id}.pkl"
                data_save_path_args = folder + f"temp_args_{args.patient_id}.pkl"
                
                dataset_queue = load_compact_loader_object(data_save_path_args)
                dataset_queue.start()
            else:
                from utils.sim_data import DataImporter

                importer = DataImporter(args=args,env_args=args) #FIXME probably don't handle the args this way
                # dataset_queue = importer.create_queue(minimum_length=batch_size*10, maximum_length=batch_size*101, mapping=convert_trial_into_windows, reserve_validation=args.vld_interactions)
                # dataset_queue.start()

                handler = importer.get_trials()
                handler.flatten()
                flat_trials = handler.flat_trials
                del handler
                del importer
                dataset_queue = CompactLoader(
                    args, batch_size*10, batch_size*101, 
                    flat_trials,
                    lambda trial : [calculate_features(row, args, args) for row in trial],
                    0,
                    lambda trial : max(0, len(trial) - args.obs_window - 1),
                    1,
                    batch_size
                )
                gc.collect()
                dataset_queue.start()
                gc.collect()


            

        elif args.data_type == "clinical":
            from utils.cln_data import ClnDataImporter

            importer = ClnDataImporter(args=args,env_args=args) #FIXME probably don't handle the args this way
            dataset_queue = importer.create_queue(minimum_length=batch_size*10, maximum_length=batch_size*51, mapping=convert_trial_into_windows, reserve_validation=args.vld_interactions*2)
            dataset_queue.start()

    elif args.policy_type == "online":
        raise NotImplementedError("Online data collection not yet implemented.")
    else:
    
        raise ValueError("Invalid value ({args.policy_type}) of argument `policy type`.")

    vld_saves_folder = MAIN_PATH + args.save_path + args.run_name + "/vld_saves"
    decoder = AutoregressiveDecoder(args)
    try:
        os.mkdir(MAIN_PATH + args.save_path + args.run_name)
        os.mkdir(vld_saves_folder)
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
    vld_iteration = 0
    while interactions < args.total_interactions:
        decoder.update_lr(interactions)
        
        data = [torch.as_tensor(np.array(dataset_queue.pop_batch(batch_size)), dtype=torch.float32, device=device) for _ in range(mini_batch_n)] #(B, T, D)
        
        new_log = decoder.mini_batch_update(data, loss_map=inverse_rmse_func)
    
        new_log['lr'] = decoder.lr
        trn_logs.add(new_log)
        
        interactions += batch_size * mini_batch_n

        percent_complete = (interactions / args.total_interactions) * 100
        if percent_complete > next_interval:
            next_time = datetime.now()
            dur = (next_time - interval_start_time).total_seconds()
            interval_start_time = next_time

            previous_dur_per = (durations[-1][1]) if durations != [] else 0
            durations.append( (dur, percent_complete - previous_dur_per) )
            time_remaining = max(0, ((100 - percent_complete) * np.mean([ i_dur / i_per for i_dur,i_per in durations]))) #calculate expected time remaining #FIXME account for batches bigger than the logging interval

            print(f"================= Training {percent_complete:.2f}% complete, interval took {pretty_seconds(dur)}. Expected time remaining for training is {pretty_seconds(time_remaining)}. Loss={new_log['loss']:.2f}")
            while next_interval < percent_complete: next_interval += logging_interval

        if iteration % args.vld_freq == 0 or interactions >= args.total_interactions:
            print("\t\t==== Performing Validation")
            
            decoder.eval()
            loss_list = []
            episode_list = []
            dataset_queue.start_validation()
            vld_interactions = 0
            while vld_interactions <  args.vld_interactions:
                
                data = torch.as_tensor(np.array(dataset_queue.pop_validation_queue(batch_size)), dtype=torch.float32, device=device) #(B, T, D)

                data_ctx = data[:, :decoder.Tc, :]
                data_fut = data[:, decoder.Tc:, :]
        
                new_log = decoder.eval_update(data_ctx, data_fut, loss_map=inverse_rmse_func)
                loss_list.append(new_log['loss'])

                episodes = compose_episode_data(new_log["y_pred"], data_fut.detach().cpu().numpy(), args, args)
                episode_list += episodes

                vld_interactions += batch_size
            
            mean_loss = np.mean(loss_list)
            vld_logs.add( {'loss' : mean_loss})

            if episode_list != []: save_episode_list(episode_list, vld_saves_folder + "/vld_trials_" + str(vld_iteration) + ".txt")
            torch.save(decoder, vld_saves_folder + "/policy_" + str(vld_iteration) + ".pth")

            print(f"\t\t==== Validation {vld_iteration} Complete: Loss of {mean_loss:.2f}")
            dataset_queue.reset_validation()

            vld_iteration += 1
            decoder.train()
            gc.collect()

        iteration += 1

    print("Training complete.")
    trn_logs.save_logs()
    vld_logs.save_logs()

    torch.save(decoder, MAIN_PATH + args.save_path + args.run_name + '/policy_final.pth')
    torch.cuda.empty_cache()

    trn_logs.graph(key="loss",ignore_empty=True)
    vld_logs.graph(key="loss",ignore_empty=True)

if __name__ == '__main__':
    main()
