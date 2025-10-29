import scipy.signal
import numpy as np
import pandas as pd
import logging
import gym
from gym.envs.registration import register
import warnings
import math
import torch
import pkg_resources


CONTROL_QUEST = pkg_resources.resource_filename('simglucose', 'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename('simglucose', 'params/vpatient_params.csv')


def get_env(args, worker_id=None, env_type=None):

    patients, env_ids = get_patient_env()
    patient_name = patients[args.patient_id]
    env_id = str(worker_id) + '_' + env_ids[args.patient_id]
    seed = worker_id + 100

    register(
        id=env_id,
        entry_point='environment.extended_T1DSimEnv:T1DSimEnv',  # simglucose.envs:T1DSimEnv
        kwargs={'patient_name': patient_name,
                'reward_fun': custom_reward,
                'seed': seed,
                'args': args,
                'env_type': env_type}
    )
    env = gym.make(env_id)
    env_conditions = {'insulin_min': env.action_space.low, 'insulin_max': env.action_space.high,
                      'cgm_low': env.observation_space.low, 'cgm_high': env.observation_space.high}
    logging.info(env_conditions)
    # print("Experiment running for {}, creating env {}.".format(patient_name, env_id))
    # print(env.observation_space.shape[0], env.observation_space.shape[1])
    return env


def get_patient_env():
    patients = (['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
                ['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
                ['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
    env_ids = (['simglucose-adolescent{}-v0'.format(str(i)) for i in range(1, 11)] +
               ['simglucose-child{}-v0'.format(str(i)) for i in range(1, 11)] +
               ['simglucose-adult{}-v0'.format(str(i)) for i in range(1, 11)])
    return patients, env_ids


def get_patient_index(patient_type=None):
    low_index, high_index = -1, -1
    if patient_type == 'adult':
        low_index, high_index = 20, 29
    elif patient_type == 'child':
        low_index, high_index = 10, 19
    elif patient_type == 'adolescent':
        low_index, high_index = 0, 9
    else:
        print('Error in assigning the patient!')
    return low_index, high_index


def risk_index(BG, horizon):
    # BG is in mg/dL, horizon in samples
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        BG_to_compute = np.array(BG[-horizon:])
        BG_to_compute[BG_to_compute < 1] = 1
        fBG = 1.509 * (np.log(BG_to_compute)**1.084 - 5.381)
        rl = 10 * fBG[fBG < 0]**2
        rh = 10 * fBG[fBG > 0]**2
        LBGI = np.nan_to_num(np.mean(rl))
        HBGI = np.nan_to_num(np.mean(rh))
        RI = LBGI + HBGI
    return LBGI, HBGI, RI


def custom_reward(bg_hist, **kwargs):
    return -risk_index([bg_hist[-1]], 1)[-1]


def get_basal(patient_name='none'):
    if patient_name == 'none':
        print('Patient name not provided')
    quest = pd.read_csv(CONTROL_QUEST)
    patient_params = pd.read_csv(PATIENT_PARA_FILE)
    q = quest[quest.Name.str.match(patient_name)]
    params = patient_params[patient_params.Name.str.match(patient_name)]
    u2ss = params.u2ss.values.item()
    BW = params.BW.values.item()
    basal = u2ss * BW / 6000
    return basal
