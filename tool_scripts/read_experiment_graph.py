import os 
import sys 
from decouple import config 
import json
import matplotlib.pyplot as plt
import numpy as np

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from metrics.statistics_alt import get_summary_stats, read_file

# RESULT_TITLE = "offline_td3_ad1"
EXPERIMENTS_DIR = "/results/"

EXCEPTION_NAMES = ["mlflow","g2p2c_debug","mushroom1","quick_run"]

def average_list(li,cast=float):
    if cast != None: return sum([cast(i) for i in li])/len(li)
    else: return sum(li)/len(li)



exp_names = os.listdir(MAIN_PATH + EXPERIMENTS_DIR)
for excp in EXCEPTION_NAMES: 
    if excp in exp_names: exp_names.remove(excp)

while 1:
    done_flag = False
    while not done_flag:
        for c,name in enumerate(exp_names):
            print(c,'|',name)
        exp_number = input("Enter experiment number: ")

        if exp_number.lower() in ['q','quit','exit','e']:
            quit()
        try:
            chosen_exp = exp_names[int(exp_number)]
            print("Chose '",chosen_exp,"'.",sep='')
            done_flag = True
        except:
            print("Invalid entry. Enter an integer in the appropiate range.")


    experiment_dir = MAIN_PATH + EXPERIMENTS_DIR +chosen_exp + '/'

    with open(experiment_dir +'args.json','r') as f:
        args = json.load(f)

    exp_test_data = read_file(experiment_name=chosen_exp, algorithm=args["agent"]["agent"], n_trials=args["agent"]["n_testing_workers"], base_num=args["agent"]["testing_agent_id_offset"])

    rew_data = []
    for nums in exp_test_data:
        temp_rew_data = []
        for c,episode_data in enumerate(exp_test_data[nums]):
            temp_rew_data += list(episode_data['rew'])

        rew_data.append(average_list(temp_rew_data))
        # print(nums, rew_data[-1])



    plt.figure(figsize=(10, 5))
    plt.plot(list(range(len(rew_data))), rew_data, linestyle='-', marker=None, color='blue')
    plt.title('Training Rewards Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()





