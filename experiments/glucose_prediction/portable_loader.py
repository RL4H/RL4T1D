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

SAVE_PATH = MAIN_PATH + "/experiments/glucose_prediction/temp_data.pkl"

def load_obj(file_dest):
    with open(file_dest, 'rb') as f:
        data = pickle.load(f)
    return data

def save_obj(data, file_dest):
    with open(file_dest, 'wb') as f:
        data = pickle.dump(data, f)
    return data

class CompactLoader:
    def __init__(self, args, min_length, max_length, trials_list, compact_conversion, retrieval_func, calculate_trial_n, shuffle_seed=1, validation_items=1024):
        self.retrieval_func = retrieval_func
        self.args = args
        self.minimum_length, self.maximum_length = min_length, max_length

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
                
        
        self.loaded = False

        save_obj(trials_list, SAVE_PATH)
        del trials_list
        
        self.cum_n = [sum(n_list[:c+1]) for c in range(len(n_list))] #stores cumulative length of transitions, for each searching
        self.length = self.cum_n[-1]
        print("temp object saved with length",self.length)

        indicies = list(range(self.length))
        seed(shuffle_seed)
        shuffle(indicies)

        self.validation_indicies = indicies[:validation_items]
        self.training_indicies = indicies[validation_items:]

        self.validation_ind = 0
        self.training_ind = 0

        self.training_length = self.length - validation_items
        self.validation_length = validation_items

        self.queue = []
        self.vld_queue = []
    def start(self):
        self.sync_queue()
    def load(self):
        print("\tLoading object.")
        self.trials_list = load_obj(SAVE_PATH)
        print("\tObject loaded.")
        self.loaded = True
    def clear_load(self):
        del self.trials_list
        self.loaded = False
    def __getitem__(self, ind):
        assert self.loaded
        trial_ind = bisect_right(self.cum_n, ind)
        before_ind = int(trial_ind != 0) * self.cum_n[trial_ind-1] #default to starting at index 0 if this is first trial, avoiding if statment for faster retrieval
        in_trial_ind = ind - before_ind
        retrieved = self.retrieval_func(self.trials_list[trial_ind], in_trial_ind)
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
    def pop_validation_queue(self, n):
        return [self.pop_validation() for _ in range(n)]




