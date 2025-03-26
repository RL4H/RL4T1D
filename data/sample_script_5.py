""" 
This code is written as an example usage of the data importing code attached to the D4RL paper.

This script generates a csv file giving meal and time length data.
"""

from import_data import DataImporter, MODEL_TYPES, INDIVIDUALS
import numpy as np
import gc



MEAL_COLUMN = 1

USE_ALGORITHMS = MODEL_TYPES[:]
USE_ALGORITHMS.remove("DPG") 
USE_ALGORITHMS.remove("DDPG")

USE_INDIVIDUALS = INDIVIDUALS

DATA_DEST = "../SimulatedData" #FIXME change data destination for your script

SAVE_PATH = DATA_DEST + "/summaries/"
SAVE_FILE_NAME = "sample_table"


all_data = DataImporter(verbose=True, models=USE_ALGORITHMS, data_folder=DATA_DEST)


individual_minutes_row = []
individual_meals_row = []
for individual_data in all_data:
    individual_name = all_data.current_individual
    #initialize row data
    individual_minutes = 0
    individual_meals = 0

    for model in USE_ALGORITHMS:
        for expert in individual_data.experts: #take all data belonging to this model
            if model in individual_data.data_dict[individual_name][expert]:
                trials = individual_data.data_dict[individual_name][expert][model]
                for trial in trials:
                    rows, _ = trial.shape
                    if rows > 0: minutes += 5 * (rows - 1)
                    meal_data = trial[:, MEAL_COLUMN]
                    meal_num = len(meal_data) - meal_data.count(0)

                    individual_minutes += minutes
                    individual_meals += meal_num
    
    #calculate total stats for individual
    individual_minutes_row.append(individual_minutes)
    individual_meals_row.append(individual_meals)

    #garbage collection
    individual_data = None
    gc.collect()



#generate year csv file
lines = [
    ','.join(USE_INDIVIDUALS + ["total"]),
    ','.join([str(i) for i in individual_minutes_row])
]


with open(SAVE_PATH + SAVE_FILE_NAME + "_years",'w') as f:
    f.write('\n'.join(lines))

#generate meals csv file
lines = [
    ','.join(USE_INDIVIDUALS + ["total"]),
    ','.join([str(i) for i in individual_meals_row])
]


with open(SAVE_PATH + SAVE_FILE_NAME + "_meals",'w') as f:
    f.write('\n'.join(lines))



