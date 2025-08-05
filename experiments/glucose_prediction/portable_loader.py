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
import pickle
torch.cuda.empty_cache()
from random import seed, shuffle

from bisect import bisect_right

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from utils.sim_data import retrieval_augmented_feature_trial

SIM_DATA_PATH = config("SIM_DATA_PATH")

SAVE_PATH = SIM_DATA_PATH + "/temp_data.pkl"
SAVE_PATH_ARGS = SIM_DATA_PATH + "/temp_args.pkl"

def load_obj(file_dest):
    with open(file_dest, 'rb') as f:
        data = pickle.load(f)
    return data

def save_obj(data, file_dest):
    with open(file_dest, 'wb') as f:
        data = pickle.dump(data, f)
    return data

def load_compact_loader_object(file_dest):
    args, min_length, max_length, _1, _2, retrieval_func_ind, _3, shuffle_seed, validation_length, training_indicies, validation_indicies, length, save_dest = load_obj(file_dest)
    cl = CompactLoader(args, min_length, max_length, [], None, retrieval_func_ind, None, shuffle_seed, validation_length, prebuilt=True)
    cl.prebuilt_init(args, min_length, max_length, _1, _2, retrieval_func_ind, _3, shuffle_seed, validation_length, training_indicies, validation_indicies, length, save_dest, file_dest)
    return cl

retrieval_funcs = [
    lambda conv_trial, trial_ind, args : conv_trial[trial_ind: trial_ind+args.obs_window],
    lambda conv_trial, trial_ind, args : retrieval_augmented_feature_trial(np.array(conv_trial), trial_ind, args.obs_window)
]

class CompactLoader:
    def __init__(self, args, min_length, max_length, trials_list, compact_conversion, retrieval_func_ind, calculate_trial_n, shuffle_seed=1, validation_items=1024, prebuilt=False, folder=MAIN_PATH + f"/experiments/glucose_prediction/saves/"):
        self.retrieval_func = retrieval_funcs[retrieval_func_ind]
        self.retrieval_func_ind = retrieval_func_ind
        self.args = args
        self.minimum_length, self.maximum_length = min_length, max_length

        self.validation_items = validation_items


        self.loaded = True
        self.shuffle_seed = shuffle_seed

        self.folder = folder
        
        if not prebuilt:

            self.save_path = folder + f"temp_data_patient_{args.patient_id}_{shuffle_seed}.pkl"
            self.save_path_args = folder + f"temp_args_{args.patient_id}_{shuffle_seed}.pkl"

            print("Converting")
            n_list = []
            to_remove = []
            for c,trial in enumerate(trials_list):
                trial_n = calculate_trial_n(trial)
                if trial_n > 0:
                    trials_list[c] = compact_conversion(trial)
                    n_list.append(trial_n)
                else:
                    to_remove.append(c)
            
            for c in to_remove[::-1]:
                del trials_list[c]
            
            print("Calculating cumulative length.")
            #cum_n = [sum(n_list[:c+1]) for c in range(len(n_list))] #stores cumulative length of transitions, for each searching
            
            cum_n = []
            running_total = 0
            for n in n_list:
                running_total += n
                cum_n.append(running_total)
            self.length = running_total

            if self.validation_items > self.length / 2:
                self.validation_items = 0

            self.n_list = n_list #FIXME remove
            
            self.cum_n = cum_n
            print("Length!",self.length)

            self.trials_list = trials_list
            self.reset_shuffle(self.shuffle_seed, n_list)
            print("Shuffle initialised.")


            print("temp object saved with length",self.length,"to",self.save_path)

            print("clearing excess memory")
            del trials_list
            gc.collect()
            print("memory cleared")
            
            del self.cum_n
            del cum_n

            self.queue = []
            self.vld_queue = []
            
            self.save_compact_loader_object()
            self.total_transitions = self.length
        

        self.loaded = False
        self.validation_ind = 0
        self.training_ind = 0

    def prebuilt_init(self, args, min_length, max_length, _1, _2, retrieval_func_ind, _3, shuffle_seed, validation_length, training_indicies, validation_indicies, length, save_dest, args_save_dest):
        self.args, self.minimum_length, self.maximum_length, self.retrieval_func_ind, self.shuffle_seed, self.validation_length, self.training_indicies, self.validation_indicies, self.length = args, min_length, max_length, retrieval_func_ind, shuffle_seed, validation_length, training_indicies, validation_indicies, length
        self.retrieval_func = retrieval_funcs[self.retrieval_func_ind]
        self.queue = []
        self.vld_queue = []
        
        indicies = list(range(self.length))
        self.shuffle_seed = shuffle_seed
        seed(shuffle_seed)
        shuffle(indicies)

        self.validation_indicies = indicies[:validation_length]
        self.training_indicies = indicies[validation_length:]

        self.training_length = self.length - validation_length
        self.validation_length = validation_length
        self.total_transitions = self.length

        self.save_path = save_dest
        self.save_path_args = args_save_dest

        print(args, min_length, max_length, _1, _2, retrieval_func_ind, _3, shuffle_seed, validation_length, len(training_indicies), len(validation_indicies), length, save_dest)
    def recalculate_nlist(self):
        if not self.loaded:
            print("Loading cum_n")
            _, self.cum_n = load_obj(self.save_path) #load cum_n into memory
            print("cum_n loaded")

        cum_n_len = len(self.cum_n)
        c = 0
        n_list = []
        prev_item = 0
        while c < cum_n_len:
            cum_item = self.cum_n[c]

            new_n_item = cum_item - prev_item

            n_list.append(new_n_item)
            prev_item = cum_item
            c += 1
        print("Cum_n recalculated")
        
        if not self.loaded:
            del self.cum_n
        
        return n_list

    def reset_shuffle(self,new_seed, n_list):
        clear_load = False
        if not self.loaded: 
            self.load()

        self.shuffle_seed = new_seed
        self.save_path = self.folder + f"temp_data_patient_{self.args.patient_id}_{self.shuffle_seed}.pkl"
        self.save_path_args = self.folder + f"temp_args_{self.args.patient_id}_{self.shuffle_seed}.pkl"

        indicies = list(range(self.length))
        self.shuffle_seed = self.shuffle_seed
        seed(self.shuffle_seed)
        shuffle(indicies)

        #allocate validation indicies to not overlap
        VALIDATION_IN_TRIAL_REPS = self.args.obs_window * 4

        replacement_validation_indicies = []
        removed_indicies = []
        base_validation_inds = indicies[:self.validation_items]
        self.training_indicies = indicies[self.validation_items:]

        print("Allocating initial replacement validation indicies.")
        while len(replacement_validation_indicies) < self.validation_items:
            ind = base_validation_inds.pop(0)
            trial_ind, in_trial_ind = self.get_trial_inds(ind)
            for n in range(self.args.obs_window + VALIDATION_IN_TRIAL_REPS):
                trial_len = n_list[trial_ind]
                try_in_trial_ind = in_trial_ind + n
                if (try_in_trial_ind >= trial_len) or (len(replacement_validation_indicies) >= self.validation_items): break # exit loop if trial index out of range

                try_ind = ind + n
                if try_ind in self.training_indicies:
                    self.training_indicies.remove(try_ind)
                elif try_ind in base_validation_inds:
                    base_validation_inds.remove(try_ind)

                if n < VALIDATION_IN_TRIAL_REPS: #add tried index
                    replacement_validation_indicies.append(try_ind)
                else: #remove overlapping index
                    removed_indicies.append(try_ind)
        
        print("Allocated",len(replacement_validation_indicies),'/',self.validation_items)

        
        self.validation_indicies = replacement_validation_indicies

        print("Re-adding spare validation inds")

        for ind in base_validation_inds: #add spare validation inds back to training
            self.training_indicies.append(ind)
        
        print(f"Validation indicies applied with length {len(self.validation_indicies)}/{self.validation_items}, removing {len(removed_indicies)} items from training for validity. Remaining length: {len(self.training_indicies)}")
        self.validation_length = len(self.validation_indicies)
        self.training_length = self.length - self.validation_length
        

        print("Saving object.")
        save_obj((self.trials_list, self.cum_n), self.save_path)

        if clear_load:
            self.clear_load()



    def start(self):
        self.sync_queue()
    def load(self):
        print("\tLoading object.")
        self.trials_list, self.cum_n = load_obj(self.save_path)
        print("\tObject loaded.")
        self.loaded = True
    def clear_load(self):
        del self.trials_list
        self.loaded = False
        gc.collect()
    def get_trial_inds(self, ind): #also assumes loaded, for self.cum_n
        trial_ind = bisect_right(self.cum_n, ind)
        before_ind = int(trial_ind != 0) * self.cum_n[trial_ind-1] #default to starting at index 0 if this is first trial, avoiding if statment for faster retrieval
        in_trial_ind = ind - before_ind
        return trial_ind, in_trial_ind
    def __getitem__(self, ind):
        assert self.loaded
        trial_ind, in_trial_ind = self.get_trial_inds(ind)
        retrieved = self.retrieval_func(self.trials_list[trial_ind], in_trial_ind, self.args)
        return retrieved
    def __len__(self):
        return self.length
    def sync_queue(self):
        if len(self.queue) < self.minimum_length:
            self.load()
            remaining_length = self.maximum_length - len(self.queue)
            for _ in range(remaining_length):
                self.queue.append(self[self.training_indicies[self.training_ind]])
                self.training_ind = (self.training_ind + 1) % self.training_length
            self.clear_load()
            gc.collect()
    def pop(self):
        new_item = self.queue.pop(0)
        self.sync_queue()
        return new_item
    def pop_batch(self,n):
        data = [self.pop() for _ in range(n)]
        return data

    def reset_validation(self):
        self.validation_ind = 0
        self.vld_queue = []
    def start_validation(self):
        self.reset_validation()
        self.sync_validation()
    def sync_validation(self):
        if len(self.vld_queue) <= 0:
            self.load()
            remaining_length = self.validation_length - len(self.vld_queue)
            for _ in range(remaining_length):
                self.vld_queue.append(self[self.validation_indicies[self.validation_ind]])
                self.validation_ind = (self.validation_ind + 1) % self.validation_length
            self.clear_load()
            gc.collect()
    def pop_validation(self):
        out = self.vld_queue.pop(0)
        self.sync_validation()
        return out
    def pop_validation_batch(self, n):
        return [self.pop_validation() for _ in range(n)]
    
    def save_compact_loader_object(self,overwrite_dest=None):
        args = (self.args, self.minimum_length, self.maximum_length, [], None, self.retrieval_func_ind, None, self.shuffle_seed, self.validation_length, self.training_indicies, self.validation_indicies, self.length, self.save_path)
        print(self.args, self.minimum_length, self.maximum_length, [], None, self.retrieval_func_ind, None, self.shuffle_seed, self.validation_length, len(self.training_indicies), len(self.validation_indicies), self.length, self.save_path)
        save_obj(args, self.save_path_args if overwrite_dest == None else overwrite_dest)
        print("Object Args saved to",self.save_path_args)

        

