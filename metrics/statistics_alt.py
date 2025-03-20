from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
from datetime import timedelta, datetime
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import math
import warnings
from decouple import config

MAIN_PATH = config('MAIN_PATH')

from environment.utils import risk_index

def time_in_range(CGM):
    '''
    slightly modified/latest way to calc the medical metrics
    :param CGM: an array with the cgm readings
    :return: time in ranges hypo, hyper, normo
    '''
    hypo, normo, hyper, severe_hypo, severe_hyper = 0, 0, 0, 0, 0

    if len(CGM) == 0:
        CGM.append(0)  # to avoid division by zero

    for reading in CGM:
        if reading <= 54:
            severe_hypo += 1
        elif reading <= 70:
            hypo += 1
        elif reading <= 180:
            normo += 1
        elif reading <= 250:
            hyper += 1
        else:
            severe_hyper += 1

    LBGI, HBGI, RI = risk_index(CGM, len(CGM))

    # todo add other useful metrics, capability to save metrics to a file.
    return (normo * 100 / len(CGM)), (hypo * 100 / len(CGM)), (severe_hypo * 100 / len(CGM)), \
           (hyper * 100 / len(CGM)), LBGI, HBGI, RI, (severe_hyper * 100 / len(CGM))

def calc_stats(res, metric, sim_len=288):
    target_metrics = ['normo', 'hypo', 'hyper', 'S_hypo', 'S_hyper', 'lgbi', 'hgbi', 'ri', 'reward']
    failures = res[res['t'] < sim_len].count()['t']
    tot = res.shape[0]
    #res = res[res['t'] == sim_len]  # only the completed simulations for calc
    res = res[target_metrics].describe().loc[metric]
    res['fail'] = (failures / tot) * 100
    target_metrics.append('fail')
    res = res[target_metrics].round(2)
    return res


def get_summary_stats(cohort, algo_type, algorithm, algoAbbreviation, metric=['mean', 'std','min', 'max'],
                      verbose=False, show_res=True, sort=[False, 'hgbi'],
                      subjects = ['0', '1', '2','3', '4', '5', '6', '7', '8', '9'],
                      seeds = ['1', '2', '3'], n_trials=500):

    if not algo_type == 'rl':
        n_trials = 1500

    if show_res:
        print("\nSummary statistics for {} cohort, {} Algorithm".format(cohort, algorithm))
    cohort_res, summary_stats = [], []
    secondary_columns = ['epi', 't', 'reward', 'normo', 'hypo', 'sev_hypo', 'hyper', 'lgbi',
                             'hgbi', 'ri', 'sev_hyper', 'aBGP_rmse', 'cBGP_rmse']
    for sub in subjects:
        data = []
        for s in seeds:
            if algo_type == 'rl':
                FOLDER_PATH='/results/'+cohort+'/'+algorithm+'/'+algoAbbreviation+sub+'_'+s+'/testing/data'
            else:
                FOLDER_PATH='/results/'+cohort+'/'+algorithm
            for i in range(0, n_trials):
                if algo_type == 'rl':
                    test_i = 'logs_worker_'+str(6000+i)+'.csv'
                    df = pd.read_csv(MAIN_PATH +FOLDER_PATH+ '/'+test_i)
                else:
                    if cohort == 'adolescent':
                        test_i = 'logs_worker_'+sub+'_'+str(i)+'.csv'
                    elif cohort == 'adult':
                        test_i = 'logs_worker_2'+sub+'_'+str(i)+'.csv'
                    elif cohort == 'child':
                        test_i = 'logs_worker_1'+sub+'_'+str(i)+'.csv'
                    df = pd.read_csv(MAIN_PATH +FOLDER_PATH+ '/'+test_i, names=['cgm', 'meal', 'ins', 't'])

                normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper = time_in_range(df['cgm'], df['meal'], df['ins'],
                                                                             i, df.shape[0], display=False)
                reward_val = df['rew'].sum() if algo_type == 'rl' else 0
                # calc rew as a % => df['rew'].sum()*(100/288)
                e = [[i, df.shape[0], reward_val, normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper, 0, 0]]
                dataframe=pd.DataFrame(e, columns=secondary_columns)
                data.append(dataframe)

        res = pd.concat(data)
        res['PatientID'] = sub
        res.rename(columns={'sev_hypo':'S_hypo', 'sev_hyper':'S_hyper'}, inplace=True)
        summary_stats.append(res)

        if verbose:
            print("\nT1D subject: ", sub)
            print(calc_stats(res, metric=metric, sim_len=288))

        res = calc_stats(res, metric=['mean'], sim_len=288)
        res['id'] = sub
        cohort_res.append(res)

    full = pd.concat(cohort_res)
    full.set_index('id', inplace=True)
    sum_stats = pd.concat(summary_stats)
    if show_res:
        print("\nSummarised cohort statistics (mean):")
        print(full)

    if sort[0]:
        print('\nSorted by {}'.format(sort[1]))
        print(full.sort_values('hgbi'))

    r = calc_stats(sum_stats, metric=metric, sim_len=288)
    if show_res:
        print("\nAveraged cohort statistics:")
        print(r)
    return r


def compare_algorithms(cohort, algo_types, algos, abbreviations):
    dfs = []
    for i in range(0, len(algos)):
        df = get_summary_stats(cohort, algo_types[i], algorithm=algos[i], algoAbbreviation=abbreviations[i],
                      metric=['mean'], verbose=False, show_res=False, sort=[False, 'hgbi'])
        df['Algo'] = algos[i]
        dfs.append(df)
    res = pd.concat(dfs)
    res.set_index('Algo', inplace=True)
    print("\nCompare algorithm performance for the {} cohort".format(cohort))
    print(res)
    return res

COL_NAMES = ["episode","t","cgm","meal","ins","rew","rl_ins","mu","sigma","prob","state_val","day_hour","day_min"]

def read_file(experiment_name, algorithm, n_trials, base_num=5000):
    FOLDER_PATH='/results/'+experiment_name+'/testing'
    trials_dict = dict()
    for i in range(0, n_trials):
        test_i = 'worker_episode_'+str(base_num+i)+'.csv'
        df = pd.read_csv(MAIN_PATH + FOLDER_PATH + '/' + test_i, names=COL_NAMES)

        copy_dict = dict()
        for k in df: copy_dict[k] = [i for i in df[k]][1:] #remove title entry from each column

        n_episodes = max([int(float(i)) for i in copy_dict["episode"]])
        trial_list = [dict() for n in range(n_episodes)]
        row_n = len(copy_dict["episode"])

        #obtain index boundaries for each episode
        episode_indices = [0]
        current_episode = 1
        for c,row_episode in enumerate(copy_dict["episode"]):
            if int(float(row_episode)) != current_episode:
                current_episode += 1
                episode_indices.append(c)
        episode_indices.append(None)

        #assign rows for each episode
        for n in range(n_episodes):
            current_slice = slice(episode_indices[n], episode_indices[n+1])
            for k in copy_dict: trial_list[n][k] = copy_dict[k][current_slice]
        
        trials_dict[base_num+i] = trial_list
    return trials_dict
