import csv
import random
import numpy as np
import pandas as pd
import sys
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from environment.utils import get_env, get_patient_env
from metrics.metrics import time_in_range
from utils.core import combined_shape
from clinical.carb_counting import carb_estimate
from clinical.basal_bolus_treatment import BasalBolusController
from utils import core

import warnings
warnings.simplefilter('ignore', Warning)


def run_simulation(args, id=0, rollout_steps=288*30, n_trials=10, seed=0):

    patients, env_ids = get_patient_env()
    env = get_env(args.env, worker_id=seed, env_type='testing')

    results = np.zeros(combined_shape(n_trials, 11), dtype=np.float32)
    complete_results = np.zeros(combined_shape(n_trials, 11), dtype=np.float32)
    failures = 0

    for trial in range(0, n_trials):
        trial_history = np.zeros(combined_shape(rollout_steps, 4), dtype=np.float32)

        counter = 0
        state = env.reset()  # fresh env and ensures all the trials start within normoglycemia.
        glucose = core.inverse_linear_scaling(y=state[-1][0], x_min=args.env.glucose_min, x_max=args.env.glucose_max)
        controller = BasalBolusController(args.agent, patient_name=patients[id], sampling_rate=env.sampling_time)
        action = controller.get_action(meal=0, glucose=glucose)

        for n_steps in range(0, rollout_steps):

            next_state, reward, is_done, info = env.step(action[0])

            carbs = info['meal'] * info['sample_time']  # current meal which is occurring
            # prepare for future meals
            if args.env.t_meal == 0:  # no meal announcement
                bolus_carbs = carbs
            elif args.env.t_meal == info['remaining_time_to_meal']:  # meal announcement: remain time to meal = meal announcement time
                bolus_carbs = info['future_carb']
            else:
                bolus_carbs = 0

            # carbohydrate estimation
            if bolus_carbs != 0:
                bolus_carbs = carb_estimate(bolus_carbs, info['day_hour'], patients[id], type=args.agent.carb_estimation_method)

            trial_history[counter] = [info['cgm'].CGM, carbs, action[0], counter]
            action = controller.get_action(meal=bolus_carbs, glucose=info['cgm'].CGM)  # BB Controller: next action.
            counter += 1  # increment counter

            if is_done or counter > (rollout_steps - 1):  # trial termination & logging trajectory info + metrics.
                df = pd.DataFrame(trial_history[0:counter], columns=['cgm', 'meal', 'ins', 't'])
                normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper = time_in_range(df['cgm'])
                df.to_csv(args.experiment.experiment_dir + '/testing/logs_worker_' + str(id)+'_'+str(trial) + '.csv', mode='a', header=False, index=False)
                results[trial] = [id, trial, counter, normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper]
                if counter < rollout_steps:
                    failures += 1
                else:
                    complete_results[trial] = [id, trial, counter, normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper]
                break

    print('Subject: {}, Mean Normo: {}, RI {}, failures {}%'.format(id, np.mean(complete_results[:, 3]), np.mean(complete_results[:, 9]),
                                                                    (failures/n_trials)*100))
    return results
