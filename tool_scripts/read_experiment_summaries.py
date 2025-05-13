import os 
import sys 
from decouple import config 
import json
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from metrics.statistics_alt import get_summary_stats, read_file

RESULT_TITLE = "offline_td3_ad0"
EXPERIMENTS_DIR = "/results/"

EXCEPTION_NAMES = ["mlflow","g2p2c_debug","mushroom1","quick_run"]



exp_names = os.listdir(MAIN_PATH + EXPERIMENTS_DIR)
for excp in EXCEPTION_NAMES: 
    if excp in exp_names: exp_names.remove(excp)


def average_list(li,cast=float):
    if cast != None: return sum([cast(i) for i in li])/len(li)
    else: return sum(li)/len(li)


def tir_metrics(cgm_li,min_glucose=70,max_glucose=180,cast=float):
    if cast != None: cgm_li = [cast(i) for i in cgm_li]
    tbr, tir, tar = 0,0,0
    for cgm in cgm_li:
        if cgm < min_glucose: tbr += 1
        elif cgm > max_glucose: tar += 1
        else: tir += 1
    
    le = len(cgm_li)

    tbr /= le
    tir /= le
    tar /= le

    return tbr, tir, tar

def tir_string(metrs):
    metr_words = ["TBR","TIR","TAR"]
    return ', '.join([str(round(metrs[n]*100,2)) + "% " + metr_words[n] for n in range(3)])

        


for chosen_exp in exp_names:
    experiment_dir = EXPERIMENTS_DIR + chosen_exp + '/'
    try:
        with open(MAIN_PATH + experiment_dir+'args.json','r') as f:
            args = json.load(f)

        ## Display Validation Data
        length_list = []
        cgm_data = []
        rew_data = []

        exp_vald_data = read_file(experiment_name=chosen_exp, algorithm=args["agent"]["agent"], n_trials=args["agent"]["debug_params"]["n_val_trials"], base_num=args["agent"]["validation_agent_id_offset"])
        for nums in exp_vald_data:
            for c,episode_data in enumerate(exp_vald_data[nums]):
                
                length_list.append(len(episode_data['t']))
                cgm_data += list(episode_data['cgm'])
                rew_data += list(episode_data['rew'])


        print("=====",chosen_exp)
        print("\tAverage Length:",  round(average_list(length_list),2))
        print("\tAverage Glucose:", round(average_list(cgm_data),2))
        print("\tAverage Reward:",  round(average_list(rew_data),2))
        print("\tTIR Metrics:",tir_string(tir_metrics(cgm_data)))
        print()
    except:
        print("'",chosen_exp,"' Import Failed.\n",sep='')



