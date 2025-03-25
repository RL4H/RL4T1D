""" 
This code is written as an example usage of the data importing code attached to the D4RL paper.

This script converts the data for each individual into csv format and saves it.
"""
from import_data import DataImporter

DATA_DEST = "../SimulatedData" #FIXME change data destination for your script
all_data = DataImporter(verbose=True, data_folder=DATA_DEST)

SAVE_PATH = "../data/raw_csv/"
SAVE_FLAT = True

for individual_data in all_data:
    if SAVE_FLAT:
        individual_data.flatten()
    individual_name = all_data.current_individual
    individual_data.save_as_csv(individual_name,SAVE_PATH)
    del individual_data