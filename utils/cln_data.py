import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import pickle
import json
from omegaconf import OmegaConf
import gc
import xport
import xport.v56

from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

SIM_DATA_PATH = config('SIM_DATA_PATH')
if SIM_DATA_PATH == '':
    raise ImportError("Environment variable 'SIM_DATA_PATH' not defined.")

CLN_DATA_PATH = config('CLN_DATA_PATH')

AGE_VALUES = ["adolescent", "adult"] 



## For Clinical Data

### Helpers

## For Simulated Data
def import_from_obj(file_dest):
    with open(file_dest, 'rb') as f:
        data_dict = pickle.load(f)
        f.close()
    return data_dict

def save_to_obj_file(data_dict, file_dest):
    with open(file_dest, 'wb') as f:
        data_dict = pickle.dump(data_dict, f)
        f.close()
    return data_dict



## For clinical Data
def import_xpt_file(file_dest):
    
    with open(file_dest, 'rb') as f:
        library = xport.v56.load(f)
    print("Library Loaded")
    
    df = library["DX"]
    print("Dataset parsed")
    print(df)



### Classes


### Main Loop

if __name__ == "__main__":
    file_dest = CLN_DATA_PATH + "/DX.xpt"
    import_xpt_file(CLN_DATA_PATH)
    

