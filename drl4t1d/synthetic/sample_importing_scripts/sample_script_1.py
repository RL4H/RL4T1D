""" 
This code is written as an example usage of the data importing code attached to the D4RL paper.

This script goes through each individual's trials and display the first results.
"""

from decouple import config
import sys

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from utils.sim_data import DataImporter

DATA_DEST = "../SimulatedData" #FIXME change data destination for your script
all_data = DataImporter(verbose=True, data_folder=DATA_DEST)

for individual_data in all_data:
    print("===Imported data for", all_data.current_individual)
    individual_data.flatten()
    print(individual_data.flat_trials[0])

    del individual_data
