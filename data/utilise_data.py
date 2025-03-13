# from import_data import DataImporter

# all_data = DataImporter(verbose=True)

# for individual_data in all_data:
#     print("Imported data for", all_data.current_individual)
#     del individual_data


# EXAMPLE 1 - Calculates full time of simulations.
from import_data import DataImporter

all_data = DataImporter(verbose=False)

total_simulated_minutes = 0

for individual_data in all_data:
    print("Imported data for", all_data.current_individual)
    individual_data.flatten()
    raw_data = individual_data.get_raw()

    individual_minutes = 0
    for trial in raw_data:
        rows, _ = trial.shape
        if rows > 0:
            individual_minutes += 5 * (rows - 1) #each row is 5 minutes of data, except first row
    
    print("Counted", individual_minutes, "minutes of data for", all_data.current_individual)
    total_simulated_minutes += individual_minutes
    del individual_data


print(f"Total minutes: {total_simulated_minutes}m")
years = total_simulated_minutes // (60 * 24 * 365)
weeks = (total_simulated_minutes - (years * (60 * 24 * 365))) // (60 * 24 * 7)
days = (total_simulated_minutes - (years * (60 * 24 * 365)) - (weeks * (60 * 24 * 7))) // (60 * 24)
hours = (total_simulated_minutes % (60 * 24)) // 60
minutes = (total_simulated_minutes % (60 * 24)) % 60
print(f"Total time: {years}y {weeks} weeks, {days}d {hours}h {minutes}m")

# EXAMPLE 2 - Retrieve raw data
from import_data import DataImporter

all_data = DataImporter(verbose=True)

SAVE_PATH = "../data/raw_csv/"
SAVE_FLAT = True

for individual_data in all_data:
    if SAVE_FLAT:
        individual_data.flatten()
    individual_name = all_data.current_individual
    individual_data.save_as_csv(individual_name,SAVE_PATH)
    del individual_data

# EXAMPLE 3 - Generate table of glucose and insulin data by person and algorithm
from import_data import DataImporter, MODEL_TYPES
import numpy as np

SAVE_PATH = "../data/summaries/"
FILE_NAME = "summary"

CGM_COLUMN = 0
INS_COLUMN = 2

USE_ALGORITHMS = MODEL_TYPES
COLUMN_HEADINGS = USE_ALGORITHMS + ["OVERALL"]

def combine_avg(avg_1, n_1, avg_2, n_2):
    return ((avg_1 * n_1) + (avg_2 * n_2)) / (n_1 + n_2) 

def combine_sd(sd_1, n_1, sd_2, n_2):
    return ((sd_1**2 * n_1 + sd_2**2 * n_2) / (n_1 + n_2)) ** 0.5

all_data = DataImporter(verbose=True, models=USE_ALGORITHMS)

glucose_rows = []
insulin_rows = []


for individual_data in all_data:
    model_entry_ns = []
    raw_dict = individual_data.get_raw()
    individual_name = all_data.current_individual
    glucose_row = []
    insulin_row = []
    for model in all_data.models:
        model_total_n = 0
        glucose_model_current_avg = 0
        glucose_model_current_sd = 0
        insulin_model_current_avg = 0
        insulin_model_current_sd = 0
        for expert in all_data.experts:
            trials = raw_dict[individual_name][expert][model]
            for trial in trials:
                trial_n, _ = trial.shape
                glucose_trial_avg = np.mean(trial[:, CGM_COLUMN])
                glucose_trial_sd  = np.std( trial[:, CGM_COLUMN])
                insulin_trial_avg = np.mean(trial[:, INS_COLUMN])
                insulin_trial_sd  = np.std( trial[:, INS_COLUMN])

                glucose_model_current_avg = combine_avg(glucose_model_current_avg, model_total_n, glucose_trial_avg, trial_n)
                glucose_model_current_sd  = combine_sd( glucose_model_current_sd,  model_total_n, glucose_trial_sd,  trial_n)
                insulin_model_current_avg = combine_avg(insulin_model_current_avg, model_total_n, insulin_trial_avg, trial_n)
                insulin_model_current_sd  = combine_sd( insulin_model_current_sd,  model_total_n, insulin_trial_sd,  trial_n)

                model_total_n += trial_n
        model_entry_ns.append(model_total_n)
        glucose_row.append((glucose_model_current_avg, glucose_model_current_sd))
        insulin_row.append((insulin_model_current_avg, insulin_model_current_sd))
    
    running_n = 0
    insulin_running_avg = 0
    insulin_running_sd = 0
    glucose_running_avg = 0
    glucose_running_sd = 0
    for n in range(len(model_entry_ns)):
        pass

        



            

            


    del individual_data
