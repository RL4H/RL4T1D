from datetime import datetime
from import_data import import_from_obj, INDIVIDUALS
import os
import numpy as np

def convert_to_frames(data_obj, window_size=16, default_starting_window=True, default_starting_value=0):
    #data_obj is a 2D numpy array , rows x columns. Columns are :  cgm, meal, ins, t, meta_data
    rows, columns = data_obj.shape
    ins_column = None
    cgm_column = None

    for row in range(rows):
        if row < window_size and default_starting_window:
            ins_window = np.array([default_starting_value]*(16-row)).concat(ins_column[0: row])
        else:
            ins_window = ins_column[row-16: row]



    
    raise NotImplementedError


if __name__ == "__main__":

    SAVE_TO_PICKLE = False #decides if converted data using pickle or not at all

    ### Data Importing
    individual = "adult0"

    start_time = datetime.now() #start the read timer

    overall_data_dict = dict()
    file_dest="../data/object_save/data_dictionary_" + individual + "_data.pkl"
    print("Starting import for",individual,"from",file_dest)
    
    data = import_from_obj(file_dest) #import data from pickle object
    file_size = os.path.getsize(file_dest) #obtain file size of read file
    print(f"\t{file_dest} has size {file_size / (1024 * 1024):.2f}MB")
    overall_data_dict[individual] = data[individual]


    end_time = datetime.now() #end the read timer
    duration = end_time - start_time
    print("Executed in",duration.total_seconds(), "seconds")

    ### Data Conversion



    ### Data Saving




