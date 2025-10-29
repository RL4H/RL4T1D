""" 
This code is written as an example usage of the data importing code attached to the D4RL paper.

This script generates a csv file giving meal and time length data.
"""

from decouple import config
import sys
import numpy as np
import gc

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from utils.sim_data import DataImporter, MODEL_TYPES, INDIVIDUALS



MEAL_COLUMN = 1

USE_ALGORITHMS = MODEL_TYPES[:]
USE_ALGORITHMS.remove("DPG") 
USE_ALGORITHMS.remove("DDPG")

USE_INDIVIDUALS = INDIVIDUALS

DATA_DEST = "../SimulatedData" #FIXME change data destination for your script

SAVE_PATH = DATA_DEST + "/summaries/"
SAVE_FILE_NAME = "sample_table"


all_data = DataImporter(verbose=True, models=USE_ALGORITHMS, data_folder=DATA_DEST)


individual_minutes_row_tst = []
individual_minutes_row_trn = []

individual_meals_row_tst = []
individual_meals_row_trn = []

individual_trials_row_tst = []
individual_trials_row_trn = []

TRAINING_EXPERTS = ["training"]
TESTING_EXPERTS = ["testing","evaluation","clinical"]

for individual_data in all_data:
    individual_name = all_data.current_individual
    #initialize row data
    individual_minutes_tst = 0
    individual_minutes_trn = 0

    individual_meals_tst = 0
    individual_meals_trn = 0

    individual_trials_tst = 0
    individual_trials_trn = 0

    for model in USE_ALGORITHMS:
        for expert in individual_data.experts: #take all data belonging to this model
            if model in individual_data.data_dict[individual_name][expert]:
                trials = individual_data.data_dict[individual_name][expert][model]
                for trial in trials:
                    rows, _ = trial.shape
                    if rows > 0: minutes = 5 * (rows - 1)
                    meal_data = trial[:, MEAL_COLUMN]
                    meal_num = np.count_nonzero(meal_data)
                    
                    if expert in TESTING_EXPERTS:
                        individual_minutes_tst += minutes
                        individual_meals_tst += meal_num
                        individual_trials_tst += 1
                    elif expert in TRAINING_EXPERTS:
                        individual_minutes_trn += minutes
                        individual_meals_trn += meal_num
                        individual_trials_trn += 1
    
    #calculate total stats for individual
    individual_minutes_row_tst.append(individual_minutes_tst)
    individual_minutes_row_trn.append(individual_minutes_trn)

    individual_meals_row_tst.append(individual_meals_tst)
    individual_meals_row_trn.append(individual_meals_trn)

    individual_trials_row_tst.append(individual_trials_tst)
    individual_trials_row_trn.append(individual_trials_trn)

    #garbage collection
    individual_data = None
    gc.collect()



#generate csv file for testing

lines = [
    ','.join(["Individual"]     + USE_INDIVIDUALS + ["total"]),
    ','.join(["Minutes (m)"]    + [str(i) for i in individual_minutes_row_tst + [sum(individual_minutes_row_tst)]]),
    ','.join(["Meals (count)"]  + [str(i) for i in individual_meals_row_tst + [sum(individual_meals_row_tst)]]),
    ','.join(["Trials (count)"] + [str(i) for i in individual_trials_row_tst + [sum(individual_trials_row_tst)]])

]
with open(SAVE_PATH + SAVE_FILE_NAME + "_stats_testing.csv",'w') as f:
    f.write('\n'.join(lines))

#generate csv file for training
lines = [
    ','.join(["Individual"]     + USE_INDIVIDUALS + ["total"]),
    ','.join(["Minutes (m)"]    + [str(i) for i in individual_minutes_row_trn + [sum(individual_minutes_row_trn)]]),
    ','.join(["Meals (count)"]  + [str(i) for i in individual_meals_row_trn + [sum(individual_meals_row_trn)]]),
    ','.join(["Trials (count)"] + [str(i) for i in individual_trials_row_trn + [sum(individual_trials_row_trn)]])
]


with open(SAVE_PATH + SAVE_FILE_NAME + "_stats_training.csv",'w') as f:
    f.write('\n'.join(lines))





