import csv
import random
import numpy as np
import pandas as pd
import sys
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)
from utils.options import Options
from utils.core import time_in_range, get_patient_env, set_logger, custom_reward, get_env, combined_shape
from agents.std_bb import carb_estimate
from agents.std_bb.BBController import BasalBolusController

import warnings
warnings.simplefilter('ignore', Warning)


def run_simulation(args, id=0, rollout_steps=288*30, n_trials=10, seed=0):

    patients, env_ids = get_patient_env()
    env = get_env(args, worker_id=seed, env_type='testing')
    args.sampling_rate = env.sampling_time
    results = np.zeros(combined_shape(n_trials, 11), dtype=np.float32)
    complete_results = np.zeros(combined_shape(n_trials, 11), dtype=np.float32)
    failures = 0

    for trial in range(0, n_trials):
        trial_history = np.zeros(combined_shape(rollout_steps, 4), dtype=np.float32)
        state = env.reset()  # fresh env and ensures all the trials start within normoglycemia.

        counter = 0
        controller = BasalBolusController(args, patient_name=patients[id], use_bolus=True, use_cf=False)
        action = controller.get_action(meal=0, glucose=state.CGM)

        for n_steps in range(0, rollout_steps):
            next_state, reward, is_done, info = env.step(action)

            carbs = info['meal'] * info['sample_time']
            if args.t_meal == 0:  # no meal announcement
                bolus_carbs = carbs
            elif args.t_meal == info['remaining_time']:  # meal announcement
                bolus_carbs = info['future_carb']
            else:
                bolus_carbs = 0
            real_bolus = bolus_carbs

            # carbohydrate estimation
            if bolus_carbs != 0:
                bolus_carbs = carb_estimate(bolus_carbs, info['day_hour'], patients[id], type=args.carb_estimation_method)
                #print('The real carbs : {}, estimated carbs: {}'.format(real_bolus, bolus_carbs))

            trial_history[counter] = [next_state.CGM, carbs, action[0], counter]

            counter += 1
            action = controller.get_action(meal=bolus_carbs, glucose=next_state.CGM)  # BB Controller.

            if counter > (rollout_steps - 1) or (next_state.CGM <= 40) or (next_state.CGM >= 600):
                # episode termination criteria. next_state.CGM <= 40 or next_state.CGM >= 600 or
                df = pd.DataFrame(trial_history[0:counter], columns=['cgm', 'meal', 'ins', 't'])
                normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper = time_in_range(df['cgm'], df['meal'], df['ins'],
                                                                             episode=trial, counter=counter, display=False)

                df.to_csv(args.experiment_dir + '/logs_worker_' + str(id)+'_'+str(trial) + '.csv', mode='a', header=False, index=False)

                results[trial] = [id, trial, counter, normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper]
                if counter < rollout_steps:
                    failures += 1
                else:
                    complete_results[trial] = [id, trial, counter, normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper]
                #print('Subject: {}, Trial: {}, Completed!'.format(id, trial))
                #print(trial_history)
                break
    print('Subject: {}, Mean Normo: {}, RI {}, failures {}%'.format(id, np.mean(complete_results[:, 3]), np.mean(complete_results[:, 9]),
                                                                    (failures/n_trials)*100))
    return results


def save_log(dir, worker_id, log_name, file_name):
        with open(dir + file_name + str(worker_id) + '.csv', 'a+') as f:
            csvWriter = csv.writer(f, delimiter=',')
            csvWriter.writerows(log_name)
            f.close()


def main():
    random.seed(0)
    np.random.seed(0)
    args = Options().parse_clinical()

    temp_var = 'OFF' if args.glucose_cf_target == 1800 else 'ON'
    print('Carb est method: {}, trial: {}, correction: {}'.format(args.carb_estimation_method, args.n_trials, temp_var ))

    columns = [['PatientID', 'trial', 'survival', 'normo', 'hypo', 'sev_hypo', 'hyper', 'lgbi', 'hgbi', 'ri', 'sev_hyper']]
    combined_df = pd.DataFrame(columns=columns)
    for id in range(0, 10):
        res = run_simulation(args, id=id, rollout_steps=args.duration, n_trials=args.n_trials)
        df = pd.DataFrame(res, columns=columns)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    #print(combined_df)
    combined_df.to_csv(args.experiment_dir + '/'+args.carb_estimation_method+'.csv')
    print('Experiment ran successfully')


if __name__ == '__main__':
    main()

# python run_manualControl.py --folder_id temp carb_estimation_method real
