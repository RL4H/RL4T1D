""" 
This code is written as an example usage of the data importing code attached to the D4RL paper.

This script counts the total simulated time for the recorded trials.
"""

from import_data import DataImporter

DATA_DEST = "../SimulatedData" #FIXME change data destination for your script
all_data = DataImporter(verbose=True, data_folder=DATA_DEST)

total_simulated_minutes = 0

for individual_data in all_data:
    print("Imported data for", all_data.current_individual)
    individual_data.flatten()

    individual_minutes = 0
    for trial in individual_data.flat_trials:
        rows, _ = trial.shape
        if rows > 0:
            individual_minutes += 5 * (rows - 1) #each row is 5 minutes of data, except first row
    
    print("Counted", individual_minutes, "minutes of data for", all_data.current_individual)
    total_simulated_minutes += individual_minutes
    del individual_data


print(f"Total minutes: {total_simulated_minutes}m")
print(f"Total hours: {total_simulated_minutes//60}h")
years = total_simulated_minutes // (60 * 24 * 365)
weeks = (total_simulated_minutes - (years * (60 * 24 * 365))) // (60 * 24 * 7)
days = (total_simulated_minutes - (years * (60 * 24 * 365)) - (weeks * (60 * 24 * 7))) // (60 * 24)
hours = (total_simulated_minutes % (60 * 24)) // 60
minutes = (total_simulated_minutes % (60 * 24)) % 60
print(f"Total time: {years}y {weeks} weeks, {days}d {hours}h {minutes}m")