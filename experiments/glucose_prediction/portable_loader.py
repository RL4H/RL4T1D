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
    def __init__(self, trials_list, compact_conversion, retrieval_func, calculate_trial_n):
        self.retrieval_func = retrieval_func

        n_list = []
        for c,trial in enumerate(trials_list):
            trial_n = calculate_trial_n(trial)
            if trial_n > 0:
                trials_list[c] = compact_conversion(trial)
                n_list.append(trial_n)
        
        self.loaded = False

        save_obj(SAVE_PATH, trials_list)
        print("temp object saved")
        del trials_list
        
        self.cum_n = [sum(n_list[:c+1]) for c in range(len(n_list))] #stores cumulative length of transitions, for each searching
        self.length = self.cum_n[-1]
    def load(self):
        print("Loading object.")
        self.trials_list = load_obj(SAVE_PATH)
        print("Object loaded.")
        self.loaded = True
    def clear_load(self):
        del self.trials_list
        self.loaded = False
    def __getitem__(self, ind):
        # assert self.loaded
        trial_ind = bisect_right(self.cum_n, ind)
        before_ind = int(trial_ind != 0) * self.cum_n[trial_ind-1] #default to starting at index 0 if this is first trial, avoiding if statment for faster retrieval
        in_trial_ind = ind - before_ind
        return self.retrieval_func(self.trial_list[trial_ind], in_trial_ind)
    def __len__(self):
        return self.length





