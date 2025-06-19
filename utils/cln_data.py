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
from datetime import datetime, timezone
from visualiser.core import plot_episode

from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

SIM_DATA_PATH = config('SIM_DATA_PATH')
if SIM_DATA_PATH == '':
    raise ImportError("Environment variable 'SIM_DATA_PATH' not defined.")

CLN_DATA_DEST = config('CLN_DATA_PATH')

AGE_VALUES = ["adolescent", "adult"] 
 
CLN_DATA_SAVE_DEST = "/home/users/u7482502/data/cln_pickled_data" #FIXME make into env variable 


def import_xpt_file(file_dest,file_name,show=False):
    
    with open(file_dest + '/' + file_name + '.xpt', 'rb') as f:
        library = xport.v56.load(f)
    df = library[file_name]
    if show: print("Dataset parsed"); print(df)

    return df

def filter_for_subject(df, USUBJID):
    new_df = df[df['USUBJID'] == USUBJID]
    return new_df.reset_index(drop=True)

def import_for_subject(file_dest,file_name, USUBJID):
    df = import_xpt_file(file_dest, file_name,False)
    return filter_for_subject(df, USUBJID)

def read_time(epoch_num):
    return datetime.fromtimestamp(epoch_num, tz=timezone.utc)

def convert_mins_to_string(raw_mins):
    days = raw_mins//(60*24)
    hours = (raw_mins % (60*24)) // 60
    mins = raw_mins % (60)
    seconds = 0
    return f"{days}:{hours:02}:{mins:02}"

def convert_string_to_mins(time_string):
    days,hours,mins = tuple([int(i) for i in time_string.split(':')])
    return days*60*24 + hours*60 + mins

CGM_TOLERANCE = 140
MLS_TOLERANCE = 140
INS_TOLERANCE = 140

MINIMUM_EPI_LEN = 1000

BLANK_RESULT = {
    "data" : np.array([]),
    "meta" : {
        "blank" : True
    }
}

def read_individual(subj_id,debug_show=True):
    # read relevant data
    subj_LB = filter_for_subject(LB, subj_id) #cgm data
    subj_LB_len = len(subj_LB)
    if subj_LB_len == 0: return BLANK_RESULT
    subj_LB = subj_LB.drop(subj_LB_len-1)
    
    subj_FACM = filter_for_subject(FACM,subj_id) #insulin data
    subj_ML = filter_for_subject(ML,subj_id) #meals data

    # check type of treatment
    treatment_types = list(set(subj_FACM["INSDVSRC"]))
    if '' in treatment_types: treatment_types.remove('')
    if ["Pump"] == treatment_types: treatment_type = "Pump"
    elif ["Injections"] == treatment_types: treatment_type = "Injections"
    else: return BLANK_RESULT #treatment_type = "Blank"

    if debug_show: print("Treatment Type:",treatment_type)

    # read time information
    cgm_time = subj_LB["LBDTC"]
    ins_time = subj_FACM["FADTC"]
    mls_time = subj_ML["MLDTC"]
    
    subj_start_time = cgm_time.loc[0]
    subj_end_time = cgm_time.loc[ len(cgm_time) - 1]
    subj_duration = subj_end_time - subj_start_time

    # sample readings in 5 minute intervals, sepearting by episodes when longer gaps appear
    cur_episode = 0
    cur_episode_len = 0
    cur_time = subj_start_time

    cgm_ind = 0
    ins_ind = 0
    mls_ind = 0
    while mls_time[mls_ind] < subj_start_time: mls_ind += 1 #skip meals set before cgm data is recorded.

    if treatment_type == "Pump":
        current_insulin_rate = subj_FACM["FASTRESN"].loc[ins_ind]
        while ins_time[ins_ind] < subj_start_time: 
            current_insulin_rate = subj_FACM["FASTRESN"].loc[ins_ind]
            ins_ind += 1 #skip insulin readings set before cgm data is recorded.
    else: #Injections
        while ins_time[ins_ind] < subj_start_time: ins_ind += 1 #skip insulin readings set before cgm data is recorded.
        
    episodes = [] # [ [ (epi, cgm, meal, ins, t, meta), ... ] ]
    cur_episode_start_time = subj_start_time
    rows = [] 

    while cur_time < subj_end_time:
        cur_cgm_time = cgm_time[cgm_ind]
        
        if cur_cgm_time - cur_time > CGM_TOLERANCE: #check if timeskip in cgm monitor
            if debug_show: print("After episode of",cur_episode_len*5,"m, skip of", (cur_cgm_time - cur_time)/60,"m detected")
            episodes.append(rows)
            rows = []
            cur_episode_len = 0
            cur_episode += 1

            cur_time = cur_cgm_time
            cur_episode_start_time = cur_time
            while mls_ind < len(mls_time) and mls_time[mls_ind] < cur_time - MLS_TOLERANCE: mls_ind += 1 #skip meals between episodes
            while ins_ind < len(ins_time) and ins_time[ins_ind] < cur_time - INS_TOLERANCE: ins_ind += 1 #skip insulin between episodes

        if treatment_type == "Pump":
            while ins_ind < len(ins_time) and ins_time[ins_ind] < cur_time + INS_TOLERANCE: #find latest pump rate reading
                current_insulin_rate += subj_FACM["FASTRESN"].loc[ins_ind]
                ins_ind += 1
            ins_reading = current_insulin_rate
        else:
            ins_reading = 0
            while ins_ind < len(ins_time) and ins_time[ins_ind] < cur_time + INS_TOLERANCE:
                ins_reading += subj_FACM["FASTRESN"].loc[ins_ind]
                ins_ind += 1

        mls_total = 0
        while mls_ind < len(mls_time) and mls_time[mls_ind] < cur_time + MLS_TOLERANCE:
            mls_total += subj_ML["MLDOSE"].loc[mls_ind]
            mls_ind += 1
         
        rows.append([
            cur_episode,
            subj_LB["LBORRES"][cgm_ind],
            mls_total,
            ins_reading,
            convert_mins_to_string(int(cur_time - cur_episode_start_time)),
            "meta_TODO"
        ])

        print((cur_time - cur_episode_start_time)//60, convert_mins_to_string(int((cur_time - cur_episode_start_time)/60)))

        cur_time += 300 # increment by 5 minutes
        cgm_ind += 1 #index cgm index

        cur_episode_len += 1

    if debug_show: print("Final episode of",cur_episode_len*5,"m\n")
    episodes.append(rows)

    # filter out short episodes
    total_rows = []
    current_epi = 0
    for n_epi, epi_rows in enumerate(episodes):
        epi_len = len(epi_rows)*5
        if epi_len < MINIMUM_EPI_LEN:
            if debug_show: print("Episode",n_epi,"removed for being too short (",epi_len,"m)")
        else:
            for row in epi_rows: row[0] = current_epi #change episode to reconsider removed episodes
            total_rows += epi_rows
            current_epi += 1
    
    if debug_show: print("Total rows after filtration are",len(total_rows)*5,"m, removed",(sum([len(epi_rows) for epi_rows in episodes]) - len(total_rows))*5, 'm' )
        
    # convert trial data to numpy
    trial_data = np.array(total_rows)

    # generate meta data
    


        
    meta_data = {
        "treatment type": treatment_type,
        "subject id" : subj_id,
        "blank" : False
    }

    return {
        "data": trial_data,
        "meta": meta_data
    }
    
COLUMN_NAMES = ["epi", "cgm", "carbs", "ins", 't', "meta"]
CSV_COLUMN_TYPES = {
    "epi"       : "int32",
    "cgm"       : "float32",
    "carbs"     : "float32",
    "ins"       : "float64",
    "t"         : "int32",
    "meta"      : "string"
}

def save_subj_file(subj_info, filepath, filename):
    meta_content = "subject_id_" + str(subj_info["meta"]["subject_id"]) + "_" + str(subj_info["meta"]["treatment type"])
    txt = '\n'.join([','.join(line[:-1] + [meta_content]) for line in (COLUMN_NAMES + subj_info['data'])])
    with open(filepath + '/' + filename, 'w') as f:
        f.write(txt)
    return 1


def read_subj_file(file_num):
    file_name = "subj_ind_" + str(n) + ".pkl"
    file_dest = CLN_DATA_SAVE_DEST + '/' +  file_name

    df = pd.read_csv(file_dest, header="infer", dtype=CSV_COLUMN_TYPES)

    data_array = df.to_numpy()
    return data_array


def display_subj_epi_graph(file_num,epi=0):
    file_name = "subj_ind_" + str(n) + ".pkl"
    file_dest = CLN_DATA_SAVE_DEST + '/' +  file_name

    df = pd.read_csv(file_dest, header="infer", dtype=CSV_COLUMN_TYPES)

    df.rename(columns={"epi" : "episode"})
    # if not epi in df["episode"]: raise IndexError(f"Episode {epi} does not exist.")
    # df = df[df['episode'] == epi]

    df["time"] = df["t"].apply(lambda t : "2000-01-01 " + t)
    df["day_hour"] = df["t"].apply(lambda t : int(t.split(':')[1]))
    df["day_min"] = df["t"].apply(lambda t : int(t.split(':')[2]))
    df["t"] = df["t"].apply(lambda t : convert_string_to_mins(t) // 5)

    for col_name in ["rew","rl_ins","mu","sigma","prob","state_val"]: df[col_name] = 0.0

    tester = 0

    plot_episode(df, tester, episode=epi)




if __name__ == "__main__":

    LB = import_xpt_file(CLN_DATA_DEST,'LB') # glucose levels
    ML = import_xpt_file(CLN_DATA_DEST,'ML') # meal data
    FACM = import_xpt_file(CLN_DATA_DEST,'FACM') # insulin data


    subject_ids = list(set(LB['USUBJID']))
    subject_ids = sorted([int(i) for i in subject_ids])
    subject_len = len(subject_ids)
    print(subject_len," subjects found.")

    print("Reading from",CLN_DATA_DEST)
    print("Writing to",CLN_DATA_SAVE_DEST)

    for n in range(subject_len):
        subject_id = subject_ids[n]
        
        subject_data = read_individual(subject_id)

        file_name = "subj_ind_" + str(n) + ".pkl"

        save_subj_file(subject_data, CLN_DATA_SAVE_DEST, file_name)

        print(f"Subject {subject_id} ({n}/{subject_len-1}) info saved")

    

    
