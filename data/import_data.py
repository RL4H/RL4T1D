import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import pickle
import json
from omegaconf import OmegaConf

AGE_VALUES = ["adolescent", "adult"] 




MODEL_TYPES = ["A2C", "AUXML", "BBHE", "BBI", "G2P2C", "PPO", "SAC", "TD3", "DDPG", "DPG"]

# stores which encoding version is used for each model's data
FOLDER_TYPE_MODELS = ["A2C", "AUXML", "G2P2C", "PPO", "SAC", "DDPG", "TD3", "DPG"]
FOLDER_TYPE_EXPERTS = ["training","testing","evaluation"]

ONLY_ADOLESCENT_MODELS = ["DDPG", "DPG"]
ONLY_ADULT_MODELS = []

CSV_TYPE_MODELS = ["BBHE", "BBI"]
CSV_TYPE_EXPERTS = ["clinical"]

EXPERTS = FOLDER_TYPE_EXPERTS + CSV_TYPE_EXPERTS


ADOLESCENT_INDIVIDUAL_NUMBER = 10
ADULT_INDIVIDUAL_NUMBER = 10

ADULT_INDIVIDUALS = ["adult" + str(num) for num in range(ADULT_INDIVIDUAL_NUMBER)]
ADOLESCENT_INDIVIDUALS = ["adolescent" + str(num) for num in range(ADOLESCENT_INDIVIDUAL_NUMBER)]

INDIVIDUALS = ADULT_INDIVIDUALS + ADOLESCENT_INDIVIDUALS



# stores where object is saved to when run as main
OBJECT_SAVE_FILE = "../data" + "/object_save/data_dictionary.pkl"

# stores which file names are excluded for csv type model data folders
EXCLUDE_FILES = ["quadratic.csv", "real.csv"]

# stores which words are exclusionary to be contained in file names for the folder type data, within the testing/data/ and training/data/ subfolders.
EXCLUDE_IN_FILES = "summary"

def import_all_data(
        dest="../data", 
        age_range = AGE_VALUES,
        model_range = MODEL_TYPES,
        individual_range = INDIVIDUALS,
        csv_type_list = CSV_TYPE_MODELS,
        folder_type_list = FOLDER_TYPE_MODELS,
        show_progress = False
        ):
    """ Imports simulation data from a given folder.

    Args:
        dest (str, optional): The folder to search through. Defaults to "../data".
        age_range (list[str], optional): The list of ages to retrieve data from. Defaults to AGE_VALUES.
        model_range (list[str], optional): The models to retrieve data from. Defaults to MODEL_TYPES.
        individual_range (list[str], optional): The individuals to retrieve data from for folder type data. Defaults to INDIVIDUALS.
        csv_type_list (list[str], optional): The list of model names encoded as csv type data. Defaults to CSV_TYPE_MODELS.
        folder_type_list (list[str], optional): The list of model names encoded as folder type data. Defaults to FOLDER_TYPE_MODELS.
        show_progress (bool, optional): Decides if progress is printed to console. Defaults to False.

    Returns:
        _type_: A layered dictionary contains numpy arrays of simulation data as columns in each trial. Follows individual > expert > model > [trial data].
    """
    
    data_dict = dict() #consider file type for this! will be very slow
    
    #set up dictionary structure
    # individual > expert > model > [trial data]
    for individual in individual_range:
        data_dict[individual] = dict()
        for expert in CSV_TYPE_EXPERTS:
            data_dict[individual][expert] = dict()
            for model in CSV_TYPE_MODELS:
                data_dict[individual][expert][model] = []
        for expert in FOLDER_TYPE_EXPERTS:
            data_dict[individual][expert] = dict()
            for model in FOLDER_TYPE_MODELS:
                data_dict[individual][expert][model] = []

    for age in age_range:
        if show_progress: print("=Age:",age)
        
        #removed excluded models from range given age group
        use_model_range = model_range[:] #copies list so it doesn't mutate the original object
        for model in (ONLY_ADOLESCENT_MODELS if age == "adult" else ONLY_ADULT_MODELS):
            use_model_range.remove(model)


        for model in use_model_range:
            model_folder = dest + '/' + age + '/' + model + '/'

            if show_progress: print(model_folder)
            if model in csv_type_list:

                available_files = os.listdir(model_folder)

                for excl_file in EXCLUDE_FILES:
                    if excl_file in available_files: 
                        available_files.remove(excl_file)
                        if show_progress: print("\t>>Excluded",excl_file,"from",model_folder)

                for file in available_files:
                    individual_number = int(file.split('_')[-2]) - (20 if age == "adult" else 0)
                    run_individual = age + str(individual_number)
                    trial_number = file.split('_')[-1].split('.')[0]
                    expert_type = "clinical"
                    file_dest = model_folder + file
                    if run_individual in individual_range:
                        if show_progress: print('\t' + file_dest, run_individual)
                        meta_col = '_'.join([
                            model, 
                            expert_type, 
                            "0", #seed number
                            trial_number,
                            run_individual
                        ])
                        data_dict[run_individual][expert_type][model].append(import_from_csv_as_rows(file_dest, meta_col=meta_col))

                # data_dict[age][model] = np.array(data_dict[age])


            elif model in folder_type_list:
                available_folders = os.listdir(model_folder)
                for folder in available_folders:
                    folder_dest = model_folder + folder
                    run_seed = folder[-1]
                    run_individual = age + folder[-3]

                    if run_individual in individual_range:
                        if show_progress: print('\t' + folder_dest)
                        args = open_arg_file(folder_dest + '/args.json')
                        #taking validation and testing together in one list
                        for trial_folder in ['/testing/data/', '/training/data/']:
                            trial_folder_dest = folder_dest + trial_folder
                            available_files = os.listdir(trial_folder_dest)
                            for file in available_files:
                                if not EXCLUDE_IN_FILES in file:
                                    running_trial_number = 0

                                    #decide expert type
                                    worker_number = int(file.split('_')[2][:-4])
                                    if trial_folder == "/training/data/": expert_type = "training"
                                    elif 6000 > worker_number >= 5000: expert_type = "testing"
                                    elif worker_number >= 6000: expert_type = "evaluation"

                                    file_dest = trial_folder_dest + file
                                    meta_col_func = (lambda epi : '_'.join([
                                        model, 
                                        expert_type,
                                        run_seed,
                                        str(running_trial_number + epi), 
                                        run_individual
                                    ]))
                                    new_data_arrays = import_from_big_csv_as_rows(file_dest, meta_col_func=meta_col_func)
                                    for new_data_array in new_data_arrays:
                                        data_dict[run_individual][expert_type][model].append(new_data_array)

                                    running_trial_number += len(new_data_arrays) #update running trial number based on number of trials retrieved from file



    return data_dict

CSV_HEADERS = ["cgm", "carbs", "ins", "t"]
def import_from_csv_as_rows(file_dest, headers=CSV_HEADERS, meta_col=""):
    df = pd.read_csv(file_dest, header=None, names=headers)
    df["meta"] = meta_col
    df = df[headers + ["meta"]] #makes order consistent to other imports
    data_array = df.to_numpy()
    return data_array

def import_from_big_csv_as_rows(file_dest, columns=["cgm","meal","rl_ins","t"], meta_col_func= lambda x : str(x)):
    use_columns = ["epi"] + columns
    df = pd.read_csv(file_dest, usecols=use_columns)
    df["meta"] = df["epi"].apply(lambda epi : meta_col_func(int(float(epi)))) #adds meta column, including episode
    df = df[use_columns + ["meta"]] #reorders dataframe to same as other setup

    end_episodes = max([int(float(i)) for i in df["epi"]])
    start_episode = min([int(float(i)) for i in df["epi"]])
    n_episodes = end_episodes - start_episode + 1
    episode_list = []

    #obtain index boundaries for each episode
    episode_indices = [0]
    current_episode = start_episode
    for c,row_episode in enumerate(df["epi"]):
        if int(float(row_episode)) != current_episode:
            current_episode += 1
            episode_indices.append(c)
    episode_indices.append(None)

    del df["epi"] #remove episode from final object
    data_array = df.to_numpy()

    #assign rows for each episode
    for n in range(n_episodes):
        current_slice = slice(episode_indices[n], episode_indices[n+1])
        episode_list.append( data_array[current_slice])

    return episode_list


def import_from_obj(file_dest=OBJECT_SAVE_FILE):
    with open(file_dest, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

def save_to_obj_file(data_dict, file_dest=OBJECT_SAVE_FILE):
    with open(file_dest, 'wb') as f:
        data_dict = pickle.dump(data_dict, f)
    return data_dict

def open_arg_file(file_dest):
    with open(file_dest,'r') as fp:
        args_dict = json.load(fp)
        fp.close()
    return OmegaConf.create(args_dict)
    


def import_pickle_files(file_dest_folder="../data/object_save/", file_name_start="data_dictionary_", file_name_end="_data"):
    start_time = datetime.now() #start the read timer
    overall_file_size = 0

    overall_data_dict = dict()
    for individual in INDIVIDUALS:
        file_dest = file_dest_folder + file_name_start + individual + file_name_end + ".pkl"
        print("Starting import for",individual,"from",file_dest)
        
        data = import_from_obj(file_dest) #import data from pickle object
        file_size = os.path.getsize(file_dest) #obtain file size of read file
        print(f"\t{file_dest} has size {file_size / (1024 * 1024):.2f}MB")
        overall_file_size += file_size

        overall_data_dict[individual] = data[individual]

    end_time = datetime.now() #end the read timer
    duration = end_time - start_time
    print("Executed in",duration.total_seconds(), "seconds")
    print(f"Overall files have size {overall_file_size / (1024 * 1024):.2f}MB")

def import_raw_files(file_dest_folder="../data/object_save/", file_name_start="data_dictionary_", file_name_end="_data"):
    overall_start_time = datetime.now()

    total_importing_time = 0
    total_saving_time = 0

    for individual in INDIVIDUALS:
        print("Starting import for",individual)
        start_time = datetime.now() #start the write timer
        file_dest = file_dest_folder + file_name_start + individual + file_name_end + ".pkl"
        
        data = import_all_data("../data", show_progress=False, individual_range=[individual]) #import data from files
        print("\nSuccesfully imported.")

        end_time = datetime.now() #end the read timer
        duration = end_time - start_time
        total_importing_time += duration.total_seconds()
        print(f"Data for {individual} model imported in {int(duration.total_seconds() // 60)}m {duration.total_seconds() % 60 :.1f}s")

        obj_size = sys.getsizeof(data) #get size of object #FIXME doesn't seem to work correctly
        obj_size_mb = obj_size / (1024 ** 2)
        print("Returned object is",round(obj_size_mb,10),"MB")

        print("===Data Breakdown:===")
        for individual in data:
            print("Individual:",individual)
            for expert in data[individual]:
                print("\t",expert,':',sep='')
                for model in data[individual][expert]:
                    specific_length = len(data[individual][expert][model])
                    print("\t\tModel", model,"with length",specific_length)
                    if specific_length > 0:
                        print("\t\t\tItem 0 has shape",data[individual][expert][model][0].shape)
        
        #save the data to pickle
        start_time = datetime.now() #start the write timer
        print("Writing file object, this may take some time.")
        save_to_obj_file(data, file_dest)
        file_size = os.path.getsize(file_dest) #calculate size of saved file
        print(f"Object saved to {file_dest} with size {file_size / (1024 * 1024):.2f}MB") \
        
        end_time = datetime.now() #end the read timer
        duration = end_time - start_time
        total_saving_time += duration.total_seconds()
        print(f"Saved object in {int(duration.total_seconds() // 60)}m {duration.total_seconds() % 60 :.1f}s\n")
        
        del data #wipe data from memory to clear space

    overall_end_time = datetime.now()
    duration = overall_end_time - overall_start_time
    print(f"Whole procedure executed in {int(duration.total_seconds() // 60)}m {duration.total_seconds() % 60 :.1f}s")
    print(f"Importing took {int(total_importing_time // 60)}m {total_importing_time % 60 :.1f}s")
    print(f"Saving took {int(total_saving_time // 60)}m {total_saving_time % 60 :.1f}s")


PICKLE_FILE_NAME_END = "_data"
PICKLE_FILE_NAME_START = "data_dictionary_"
class DataImporter:
    def __init__(self, 
            source = "pickle", 
            data_folder="../data/",
            verbose = True,
            individuals = INDIVIDUALS, ages=AGE_VALUES, models = MODEL_TYPES, experts=EXPERTS,
        ):
        self.individuals, self.ages, self.models, self.experts = individuals, ages, models, experts
        self.source = source
        self.source_folder = (data_folder) if source == "files" else (data_folder + "object_save/")
        self.current_data = None
        self.current_index = 0
        self.current_individual = self.individuals[0]
        
    def start(self):
        self.current_index = -1

    def __iter__(self):
        self.start()
        print("Starting!")
        return self

    def __next__(self):
        print("next!")
        self.current_index += 1
        if self.current_index < len(self.individuals):
            self.current_individual = self.individuals[self.current_index]
            self.import_current()
            return self.current_data
        else:
            self.current_individual = None
            self.current_data = None
            raise StopIteration

    def import_current(self):
        if self.source == "pickle":
            #import file
            file_dest = self.source_folder + PICKLE_FILE_NAME_START + self.current_individual + PICKLE_FILE_NAME_END + ".pkl"
            if self.verbose: print("Starting import for",self.current_individual,"from",file_dest)
        
            self.current_data = import_from_obj(file_dest) #import data from pickle object
            if self.verbose:
                file_size = os.path.getsize(file_dest) #obtain file size of read file
                print(f"\t{file_dest} has size {file_size / (1024 * 1024):.2f}MB")

        elif self.source == "files":
            if self.verbose: print("Starting import for",self.current_individual)
            self.current_data = import_all_data(self.source_folder, individual_range=[self.current_individual],model_range=self.models, show_progress=False)
            if self.verbose: print("Import Completed")

        #strip irrelevant parts of imported object based on defined parameter ranges 
        for expert in self.current_data[self.current_individual]:
            if not (expert in self.experts):
                del self.current_data[self.current_individual][expert]
            else:
                for model in self.current_data[self.current_individual][expert]:
                    if not (model in self.models):
                        del self.current_data[self.current_individual][expert][model]

    def check_finished(self):
        return self.current_individual == None


if __name__ == "__main__":

    SAVE_TO_PICKLE = True #decides if data imported from files is saved using pickle or not at all
    IMPORT_MODE = "files" #can be `files` or `pickle`
    SINGLE_INDIVIDUAL_FILES = True #decides if files are read per individual or all at once

    if IMPORT_MODE == "pickle": #reading data from pickle
        if SINGLE_INDIVIDUAL_FILES: #read from the files for each individual
            import_pickle_files()
        elif not SINGLE_INDIVIDUAL_FILES: #read from the single overall file
            start_time = datetime.now() #start the read timer

            file_dest="../data/object_save/data_dictionary.pkl"
            print("Starting read from file",file_dest)
            data = import_from_obj(file_dest) #import data from pickle object

            end_time = datetime.now() #end the read timer
            duration = end_time - start_time
            print("Executed in",duration.total_seconds(), "seconds")

            file_size = os.path.getsize(file_dest) #obtain file size of read file
            print(f"Read file has size {file_size / (1024 * 1024):.2f}MB")
    elif IMPORT_MODE == "files" and SINGLE_INDIVIDUAL_FILES: #import data of each individuals seperately (useful if you have limited RAM on your computer)
        import_raw_files()
    elif IMPORT_MODE == "files" and not SINGLE_INDIVIDUAL_FILES: #read all data as single object and saves it as such

        start_time = datetime.now() #start the write timer
        file_dest="../data/object_save/data_dictionary_whole.pkl"
        
        data = import_all_data("../data", show_progress=True) #import data from files
        print("\nSuccesfully imported.")

        end_time = datetime.now() #end the read timer
        duration = end_time - start_time
        print(f"Executed in {int(duration.total_seconds() // 60)}m {duration.total_seconds() % 60 :.1f}s")

        obj_size = sys.getsizeof(data) #get size of object #FIXME doesn't seem to work correctly
        obj_size_mb = obj_size / (1024 ** 2)
        print("Returned object is",round(obj_size_mb,10),"MB")

        for age_k in data: #display length of data by age group and model name
            print("Age:",age_k)
            for model_k in data[age_k]:
                print("\tModel:", model_k, "with length", len(data[age_k][model_k]))
        
        if SAVE_TO_PICKLE:
            start_time = datetime.now() #start the write timer
            print("Writing file object, this may take some time.")
            save_to_obj_file(data, file_dest)
            file_size = os.path.getsize(file_dest) #calculate size of saved file
            print(f"Object saved to {file_dest} with size {file_size / (1024 * 1024):.2f}MB") \
            
            end_time = datetime.now() #end the read timer
            duration = end_time - start_time
            print(f"Saved object in {int(duration.total_seconds() // 60)}m {duration.total_seconds() % 60 :.1f}s")
        

