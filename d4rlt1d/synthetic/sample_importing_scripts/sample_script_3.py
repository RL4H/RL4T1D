""" 
This code is written as an example usage of the data importing code attached to the D4RLT1D paper.

This script converts the data for each individual into csv format and saves it.
"""
from decouple import config
import sys

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from d4rlt1d.synthetic.sim_data import DataImporter

DATA_DEST = "../SimulatedData" #FIXME change data destination for your script
all_data = DataImporter(verbose=True, data_folder=DATA_DEST)

SAVE_PATH = "../data/raw_csv/" # Ensure path already exists before running this script
SAVE_FLAT = True

for individual_data in all_data:
    if SAVE_FLAT:
        individual_data.flatten()
    individual_name = all_data.current_subject
    individual_data.save_as_csv(individual_name,SAVE_PATH)
    del individual_data