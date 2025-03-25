""" 
This code is written as an example usage of the data importing code attached to the D4RL paper.

This script goes through the glucose and insulin data, taking mean and standard deviation, and propogating it through the data to improve performance.
"""

from import_data import DataImporter, MODEL_TYPES
import numpy as np
import gc
from functools import reduce



CGM_COLUMN = 0
INS_COLUMN = 2

USE_ALGORITHMS = MODEL_TYPES
COLUMN_HEADINGS = USE_ALGORITHMS + ["OVERALL"]

DATA_DEST = "../SimulatedData" #FIXME change data destination for your script

SAVE_PATH = DATA_DEST + "/summaries/"
SAVE_FILE_NAME = "summary"


all_data = DataImporter(verbose=True, models=USE_ALGORITHMS, data_folder=DATA_DEST)

def combine_mn(avg_1, n_1, avg_2, n_2):
    return ((avg_1 * n_1) + (avg_2 * n_2)) / (n_1 + n_2) 
def combine_mn_n(avg_1_n_1, avg_2_n_2):
    avg_1,n_1 = avg_1_n_1
    avg_2,n_2 = avg_2_n_2
    return combine_mn(avg_1, n_1, avg_2, n_2), (n_1+n_2)

def combine_sd(sd_1, n_1, sd_2, n_2):
    return ((sd_1**2 * n_1 + sd_2**2 * n_2) / (n_1 + n_2)) ** 0.5
def combine_sd_n(sd_1_n_1, sd_2_n_2):
    sd_1,n_1 = sd_1_n_1
    sd_2,n_2 = sd_2_n_2
    return combine_sd(sd_1, n_1, sd_2, n_2), (n_1+n_2)



glucose_rows = []
insulin_rows = []
for individual_data in all_data:
    individual_name = all_data.current_individual
    #initialize row data
    model_entry_ns = []
    glucose_row = []
    insulin_row = []

    for model in USE_ALGORITHMS:
        #initialize model info
        model_total_n = 0
        glucose_model_current_mn = 0
        glucose_model_current_sd = 0
        insulin_model_current_mn = 0
        insulin_model_current_sd = 0
        for expert in individual_data.experts: #take all data belonging to this model
            if model in individual_data.data_dict[individual_name][expert]:
                trials = individual_data.data_dict[individual_name][expert][model]
                for trial in trials:
                    trial_n, _ = trial.shape

                    #take mean and standard deviation of data
                    glucose_trial_mn = np.mean(trial[:, CGM_COLUMN])
                    glucose_trial_sd  = np.std( trial[:, CGM_COLUMN])
                    insulin_trial_mn = np.mean(trial[:, INS_COLUMN])
                    insulin_trial_sd  = np.std( trial[:, INS_COLUMN])

                    #combine mean and standard deviation with previous data
                    glucose_model_current_mn = combine_mn(glucose_model_current_mn, model_total_n, glucose_trial_mn, trial_n)
                    glucose_model_current_sd  = combine_sd( glucose_model_current_sd,  model_total_n, glucose_trial_sd,  trial_n)
                    insulin_model_current_mn = combine_mn(insulin_model_current_mn, model_total_n, insulin_trial_mn, trial_n)
                    insulin_model_current_sd  = combine_sd( insulin_model_current_sd,  model_total_n, insulin_trial_sd,  trial_n)

                    #increment total size of model data
                    model_total_n += trial_n
        
        #add info for model to row of table
        model_entry_ns.append(model_total_n)
        glucose_row.append((glucose_model_current_mn, glucose_model_current_sd))
        insulin_row.append((insulin_model_current_mn, insulin_model_current_sd))
    
    #calculate total stats for individual
    glucose_total_mn = reduce(combine_mn_n, zip(map(lambda mn_sd : mn_sd[0],glucose_row), model_entry_ns))[0]
    glucose_total_sd = reduce(combine_sd_n, zip(map(lambda mn_sd : mn_sd[1],glucose_row), model_entry_ns))[0]
    insulin_total_mn = reduce(combine_mn_n, zip(map(lambda mn_sd : mn_sd[0],insulin_row), model_entry_ns))[0]
    insulin_total_sd = reduce(combine_sd_n, zip(map(lambda mn_sd : mn_sd[1],insulin_row), model_entry_ns))[0]

    glucose_row.append((glucose_total_mn, glucose_total_sd))
    insulin_row.append((insulin_total_mn, insulin_total_sd))

    #add
    glucose_rows.append(glucose_row)
    insulin_rows.append(insulin_row)
    
    individual_data = None
    gc.collect()

print("Done!")

print(COLUMN_HEADINGS, USE_ALGORITHMS)
#save md file
column_n = len(COLUMN_HEADINGS)
print("column lengths:",column_n, len(glucose_rows[0]))

md_lines = ["Readings are given as mean±sd"]

#Make glucose table
md_lines.append("## Glucose Table Readings")
md_lines.append(' | '.join(["Individual"] + COLUMN_HEADINGS)) 
md_lines.append(' | '.join([":---"]*(column_n + 1)))
for c,row in enumerate(glucose_rows):
    md_lines.append(' | '.join([all_data.individuals[c]] + [str(round(mn,2)) + "±" + str(round(sd,2)) for mn,sd in row]))

md_lines.append('')
#Make insulin table
md_lines.append("## Insulin Table Readings")
md_lines.append(' | '.join(["Individual"] + COLUMN_HEADINGS)) 
md_lines.append(' | '.join([":---"]*(column_n + 1)))
for c,row in enumerate(insulin_rows):
    md_lines.append(' | '.join([all_data.individuals[c]] + [str(round(mn,2)) + "±" + str(round(sd,2)) for mn,sd in row]))

with open(SAVE_PATH+SAVE_FILE_NAME+".md",'w',encoding="utf8") as f:
    f.write('\n'.join(md_lines))


#make csv files
csv_lines = [','.join(["Individual"] + COLUMN_HEADINGS)]
for c,row in enumerate(glucose_rows):
    csv_lines.append(','.join([all_data.individuals[c]] + [str(mn)+'_'+str(sd) for mn,sd in row]))

with open(SAVE_PATH+SAVE_FILE_NAME+"_glucose.csv",'w',encoding="utf8") as f:
    f.write('\n'.join(csv_lines))


csv_lines = [','.join(["Individual"] + COLUMN_HEADINGS)]
for c,row in enumerate(insulin_rows):
    csv_lines.append(','.join([all_data.individuals[c]] + [str(mn)+'_'+str(sd) for mn,sd in row]))

with open(SAVE_PATH+SAVE_FILE_NAME+"_insulin.csv",'w',encoding="utf8") as f:
    f.write('\n'.join(csv_lines))

