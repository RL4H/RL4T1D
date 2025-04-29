import os 
import sys 
from decouple import config 
import json
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from metrics.statistics_alt import get_summary_stats, read_file

RESULT_TITLE = "CustomTest"
EXPERIMENTS_DIR = "../results/"
EXCEPTION_NAMES = ["mlflow"]



exp_names = os.listdir(EXPERIMENTS_DIR)
for excp in EXCEPTION_NAMES: 
    if excp in exp_names: exp_names.remove(excp)
    
print("Examinable Experiments:")
print('\t'.join(exp_names))

#TODO: choose target experiment
chosen_exp = exp_names[0]

experiment_dir = EXPERIMENTS_DIR + chosen_exp + '/'
with open(experiment_dir+'args.json','r') as f:
    args = json.load(f)

## Display Args

def display_args(args_dict):
    print("======================================== ARGS for '" + chosen_exp + "'")
    for k in args_dict:
        print('>',k,':')
        for sub_k in args_dict[k]:
            print('\t',k,'.',sub_k,': ', args_dict[k][sub_k],sep='')
    print("========================================")

display_args(args)

## Display Test Data
print("### Test Data Trials ###")
exp_test_data = read_file(experiment_name=chosen_exp, algorithm=args["agent"]["agent"], n_trials=args["agent"]["n_testing_workers"], base_num=args["agent"]["testing_agent_id_offset"])
for nums in exp_test_data:
    print("worker_episode_",nums,sep='')
    for c,episode_data in enumerate(exp_test_data[nums]):
        print("\tEpisode",c+1)
        for k in episode_data:
            print("\t\t",k,':',','.join([str(round(float(i),2)) for i in episode_data[k]]))
        print()
print("========================================")

## Display Validation Data
print("### Validation Data Trials ###")
exp_vald_data = read_file(experiment_name=chosen_exp, algorithm=args["agent"]["agent"], n_trials=args["agent"]["debug_params"]["n_val_trials"], base_num=args["agent"]["validation_agent_id_offset"])
for nums in exp_vald_data:
    print("worker_episode_",nums,sep='')
    for c,episode_data in enumerate(exp_vald_data[nums]):
        print("\tEpisode",c+1)
        for k in episode_data:
            print("\t\t",k,':',','.join([str(round(float(i),2)) for i in episode_data[k]]))
        print()
print("========================================")