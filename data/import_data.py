import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import pickle
import json
from omegaconf import OmegaConf

AGE_VALUES = ["adolescent", "adult"]
MODEL_TYPES = ["A2C", "AUXML", "BBHE", "BBI", "G2P2C", "PPO", "SAC"]

FOLDER_TYPE_MODELS = ["A2C", "AUXML", "G2P2C", "PPO", "SAC"]
CSV_TYPE_MODELS = ["BBHE", "BBI"]

OBJECT_SAVE_FILE = "../data" + "/object_save/data_dictionary.pkl"


"""
Ideas:
- Might want to add a column for patient id

"""
EXCLUDE_FILES = ["quadratic.csv", "real.csv"]

EXCLUDE_IN_FILES = "summary"

def import_all_data(
        dest="../data", 
        age_range = AGE_VALUES,
        model_range = MODEL_TYPES,
        csv_type_list = CSV_TYPE_MODELS,
        folder_type_list = FOLDER_TYPE_MODELS
        ):
    
    data_dict = dict() #consider file type for this! will be very slow

    for age in age_range:
        data_dict[age] = dict()
        for model in model_range:
            model_folder = dest + '/' + age + '/' + model + '/'
            print(model_folder)

            if model in csv_type_list:
                data_dict[age][model] = []

                available_files = os.listdir(model_folder)

                for excl_file in EXCLUDE_FILES:
                    if excl_file in available_files: 
                        available_files.remove(excl_file)
                        print("\t>>Excluded",excl_file,"from",model_folder)

                for file in available_files:
                    run_individual = age.capitalize() + file.split('_')[2]
                    expert_type = "clinical"
                    file_dest = model_folder + file
                    print('\t' + file_dest)
                    data_dict[age][model].append(import_from_csv(file_dest))

                # data_dict[age][model] = np.array(data_dict[age])


            elif model in folder_type_list:
                available_folders = os.listdir(model_folder)
                data_dict[age][model] = []
                for folder in available_folders:
                    folder_dest = model_folder + folder
                    print('\t' + folder_dest)
                    args = open_arg_file(folder_dest + '/args.json')
                    run_seed = folder[-1]
                    run_individual = age.capitalize() + folder[-3]


                    #taking validation and testing together in one list
                    for trial_folder in ['/testing/data/', '/training/data/']:
                        trial_folder_dest = folder_dest + trial_folder
                        available_files = os.listdir(trial_folder_dest)
                        for file in available_files:
                            if trial_folder == "/testing/data/": expert_type = "testing"
                            elif 6000 > worker_number >= 5000: expert_type = "training"
                            elif worker_number >= 6000: expert_type = "eval"
                            if not EXCLUDE_IN_FILES in file:
                                worker_number = int(file.split('_')[2][:-4])
                                file_dest = trial_folder_dest + file
                                new_data_dicts = import_from_big_csv(file_dest)
                                for new_data_dict in new_data_dicts:
                                    data_dict[age][model].append(new_data_dict)


                # data_dict[age][model] = np.array(data_dict)

    return data_dict

CSV_HEADERS = ["cgm", "carbs", "ins", "t"]
def import_from_csv(file_dest, headers=CSV_HEADERS): #imports data from a csv file
    df = pd.read_csv(file_dest, header=None, names=headers)
    data_dict = {col: df[col].to_numpy(dtype=float) for col in headers}
    return data_dict

def import_from_big_csv(file_dest, columns=["cgm","meal","rl_ins","t"]):
    df = pd.read_csv(file_dest, usecols=columns + ["epi"])

    end_episodes = max([int(float(i)) for i in df["epi"]])
    start_episode = min([int(float(i)) for i in df["epi"]])
    n_episodes = end_episodes - start_episode + 1
    data_dicts = [dict() for n in range(n_episodes)]

    copy_dict = {col: df[col].to_numpy(dtype=float) for col in columns + ["epi"]}

    copy_dict["ins"] = copy_dict["rl_ins"]
    copy_dict["carbs"] = copy_dict["meal"]
    del copy_dict["meal"]
    del copy_dict["rl_ins"]

     #obtain index boundaries for each episode
    episode_indices = [0]
    current_episode = start_episode
    for c,row_episode in enumerate(copy_dict["epi"]):
        if int(float(row_episode)) != current_episode:
            current_episode += 1
            episode_indices.append(c)
    episode_indices.append(None)

    #assign rows for each episode
    for n in range(n_episodes):
        current_slice = slice(episode_indices[n], episode_indices[n+1])
        for k in copy_dict: data_dicts[n][k] = copy_dict[k][current_slice]


    return data_dicts



def import_from_obj(file_dest=OBJECT_SAVE_FILE):
    with open(file_dest, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

def save_to_obj_file(data_dict, file_dest=OBJECT_SAVE_FILE):
    with open(file_dest, 'wb') as f:
        data_dict = pickle.dump(data_dict, f)
    return data_dict

def open_arg_file(file_dest):
    with open(file_dest,'r') as fp:
        args_dict = json.load(fp)
        fp.close()
    return OmegaConf.create(args_dict)
    
if __name__ == "__main__":


    SAVE_TO_PICKLE = True
    READ_FROM_PICKLE = False

    if READ_FROM_PICKLE:
        start_time = datetime.now()


        file_dest="../data/object_save/data_dictionary.pkl"
        print("Starting read from file",file_dest)
        data = import_from_obj(file_dest)

        end_time = datetime.now()
        duration = end_time - start_time
        print("Executed in",duration.total_seconds(), "seconds")
        file_size = os.path.getsize(file_dest)
        print(f"Read file has size {file_size / (1024 * 1024):.2f}MB")

    else:
        start_time = datetime.now()
        file_dest="../data/object_save/data_dictionary.pkl"
        
        data = import_all_data("../data")
        print("\nSuccesfully imported.")

        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Executed in {int(duration.total_seconds() // 60)}m {duration.total_seconds() % 60 :.1f}s")

        obj_size = sys.getsizeof(data)
        obj_size_mb = obj_size / (1024 ** 2)
        print("Returned object is",round(obj_size_mb,10),"MB")
        for age_k in data:
            print("Age:",age_k)
            for model_k in data[age_k]:
                print("\tModel:", model_k, "with length", len(data[age_k][model_k]))
        
        if SAVE_TO_PICKLE:
            save_to_obj_file(data, file_dest)
            file_size = os.path.getsize(file_dest)
            print(f"Object saved to {file_dest} with size {file_size / (1024 * 1024):.2f}MB")
        

