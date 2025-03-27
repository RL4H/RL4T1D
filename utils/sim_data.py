import os
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import pickle
import json
from omegaconf import OmegaConf
import gc
from random import randrange
from decouple import config

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

SIM_DATA_PATH = config("SIM_DATA_PATH")
# CLN_DATA_PATH = config("CLN_DATA_PATH")

DATA_DEST = SIM_DATA_PATH

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



# default file dest for saving pkl objects
OBJECT_SAVE_FILE = DATA_DEST + "/object_save/data_dictionary.pkl"

# stores which file names are excluded for csv type model data folders
EXCLUDE_FILES = ["quadratic.csv", "real.csv"]

# stores which words are exclusionary to be contained in file names for the folder type data, within the testing/data/ and training/data/ subfolders.
EXCLUDE_IN_FILES = "summary"

CSV_HEADERS = ["cgm", "carbs", "ins", "t"]
CSV_COLUMN_TYPES = {
    'cgm'       : "float64",
    "carbs"     : "float32",
    "ins"       : "float64",
    "t"         : "int32"
}
BIG_CSV_COLUMN_TYPES ={
    "cgm" : CSV_COLUMN_TYPES["cgm"],
    "meal" : CSV_COLUMN_TYPES["carbs"],
    "rl_ins" : CSV_COLUMN_TYPES["ins"],
    "t" : CSV_COLUMN_TYPES["t"],
}

#designate file naming conventions to read from
PICKLE_FILE_NAME_END = "_data"
PICKLE_FILE_NAME_START = "data_dictionary_"


### Helpers

def convert_mins_to_string(raw_mins):
    days = raw_mins//(60*24)
    hours = (raw_mins % (60*24)) // 60
    mins = raw_mins % (60)
    seconds = 0
    return f"{days}:{hours:02}:{mins:02}"

def convert_string_to_mins(time_string):
    days,hours,mins,seconds = tuple([int(i) for i in time_string.split(':')])
    return days*60*24 + hours*60 + mins

def import_from_csv_as_rows(file_dest, headers=CSV_HEADERS, meta_col=""):
    df = pd.read_csv(file_dest, header=None, names=headers, dtype=CSV_COLUMN_TYPES)
    df["meta"] = meta_col
    df["t"] = [convert_mins_to_string(t*5) for t in df["t"]] #convert time intervals to times. Each interval is 5 minutes.
    df = df[headers + ["meta"]] #makes order consistent to other imports
    data_array = df.to_numpy()
    return data_array

def import_from_big_csv_as_rows(file_dest, columns=["cgm","meal","rl_ins","t"], meta_col_func= lambda x : str(x)):
    use_columns = ["epi"] + columns
    df = pd.read_csv(file_dest, usecols=use_columns,dtype=BIG_CSV_COLUMN_TYPES)
    df["meta"] = df["epi"].apply(lambda epi : meta_col_func(int(float(epi)))) #adds meta column, including episode
    df["t"] = [convert_mins_to_string(t*5) for t in df["t"]] #convert time intervals to times. Each interval is 5 minutes.
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

def import_raw_files(file_dest_folder=DATA_DEST+"/object_save/", file_name_start="data_dictionary_", file_name_end="_data"):
    overall_start_time = datetime.now()

    total_importing_time = 0
    total_saving_time = 0

    for individual in INDIVIDUALS:
        print("Starting import for",individual)
        start_time = datetime.now() #start the write timer
        file_dest = file_dest_folder + file_name_start + individual + file_name_end + ".pkl"
        
        data = import_all_data(DATA_DEST, show_progress=True, individual_range=[individual]) #import data from files
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

def import_all_data(
        dest=DATA_DEST, 
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

def convert_to_frames(data_obj, window_size=16, default_starting_window=True, default_starting_value=0):
    #data_obj is a 2D numpy array , rows x columns. Columns are :  cgm, meal, ins, t, meta_data
    rows, _ = data_obj.shape
    ins_column = data_obj[:, 2]
    cgm_column = data_obj[:, 0]

    assert rows > window_size
    
    data_frames = np.zeros((rows, 2, window_size)) if default_starting_window else np.zeros((rows-window_size, 2, window_size))

    for row in range(rows):
        if row < window_size and default_starting_window:
            ins_window = np.append(np.array([default_starting_value]*(window_size-row)), ins_column[0: row])
            cgm_window = np.append(np.array([default_starting_value]*(window_size-row)), cgm_column[0: row])
        else:
            ins_window = ins_column[row-window_size: row]
            cgm_window = cgm_column[row-window_size: row]

        data_frames[row] = np.array([ins_window, cgm_window])
    
    return data_frames

### Classes

class DataImporter:
    """
    Manages the parameterised importing of data.
    """
    def __init__(self, 
            data_folder=DATA_DEST,
            verbose = True,
            individuals = INDIVIDUALS, ages=AGE_VALUES, models = MODEL_TYPES, experts=EXPERTS,
        ):
        self.individuals, self.ages, self.models, self.experts = individuals, ages, models, experts
        self.verbose = verbose
        self.source_folder = data_folder + "/object_save/"
        self.current_data = None
        self.current_index = 0
        self.current_individual = self.individuals[0]
        
    def start(self):
        self.current_index = -1

    def __iter__(self):
        self.start()
        return self

    def __next__(self):
        self.current_index += 1
        if self.current_index < len(self.individuals):
            self.current_individual = self.individuals[self.current_index]
            self.import_current()
            return self.current_data
        else:
            self.current_individual = None
            self.delete_current()
            raise StopIteration
    
    def delete_current(self):
        if self.current_data != None:
            self.current_data.delete()
            self.current_data = None
            gc.collect()

    def import_current(self):
        if self.verbose: print("\tDeleting previous data.")
        self.delete_current()
        if self.verbose: print("\tFinished deleting previous data.")
        #import file
        file_dest = self.source_folder + PICKLE_FILE_NAME_START + self.current_individual + PICKLE_FILE_NAME_END + ".pkl"
        if self.verbose: 
            print("\tStarting import for",self.current_individual,"from",file_dest)
            start_time = datetime.now() #start the write timer
    
        raw_data = import_from_obj(file_dest) #import data from pickle object
        if self.verbose:
            end_time = datetime.now() #end the read timer
            duration = end_time - start_time
            print(f"Read object in {int(duration.total_seconds() // 60)}m {duration.total_seconds() % 60 :.1f}s\n")
            file_size = os.path.getsize(file_dest) #obtain file size of read file
            print(f"\t{file_dest} has size {file_size / (1024 * 1024):.2f}MB")

        #strip irrelevant parts of imported object based on defined parameter ranges 
        if self.verbose: print("\tStripping Irrelevant Sections")
        for expert in list(raw_data[self.current_individual].keys()):
            if not (expert in self.experts):
                print("\t\tDeleted expert",expert)
                del raw_data[self.current_individual][expert]
            else:
                for model in list(raw_data[self.current_individual][expert].keys()):
                    if not (model in self.models):
                        print("\t\tDeleted model",model)
                        del raw_data[self.current_individual][expert][model]
        if self.verbose: print("\tFinished stripping sections.")

        self.current_data = DataHandler(raw_data)
        raw_data = None

    def check_finished(self):
        return self.current_individual == None

    def get_trials(self, individuals_override = None, ages_override=None, models_override= None, experts_override=None):
        individuals = self.individuals if individuals_override == None else individuals_override
        ages = self.ages if ages_override == None else ages_override
        models = self.models if models_override == None else models_override
        experts = self.experts if experts_override == None else experts_override

        #TODO: add a layer to filter out individuals if ages, models, or experts limit them, since reading the files is the main bottleneck.

        trial_data = dict()
        for individual in individuals:
            file_dest = self.source_folder + PICKLE_FILE_NAME_START + self.individual + PICKLE_FILE_NAME_END + ".pkl"
            
            current_data = import_from_obj(file_dest) #import data from pickle object

            if self.verbose: print("\tStripping Irrelevant Sections")

            for expert in current_data[individual]:
                if not (expert in experts):
                    del current_data[individual][expert]
                else:
                    for model in current_data[individual][expert]:
                        if not (model in models):
                            del current_data[individual][expert][model]
            if self.verbose: print("\tFinished stripping sections.")

            trial_data[individual] = current_data[individual]
            del current_data

        return DataHandler(trial_data)

class DataHandler:
    """
    Manages the handling of data once imported.
    """
    def __init__(self, data_dict, verbose=False):
        self.data_dict = data_dict
        self.trial_labels = CSV_HEADERS
        self.flat = False
        self.verbose = verbose

        #detect range of individuals, experts, models, and ages
        self.individuals, self.experts, self.models, self.ages = [], [] , [], []
        for individual in list(data_dict.keys()):
            for expert in list(data_dict[individual].keys()):
                for model in list(data_dict[individual][expert].keys()):
                    if data_dict[individual][expert][model] != []: #only add if empty
                        if not individual in self.individuals: self.individuals.append(individual)
                        age = individual[:-1]
                        if not age in self.ages: self.ages.append(age)
                        if not expert in self.experts: self.experts.append(expert)
                        if not model in self.models: self.models.append(model)
                    else:
                        del data_dict[individual][expert][model]
                        if self.verbose: print("Pruned empty dictionary entry",individual,expert,model)
        
        self.gen_summaries()

    def get_raw(self):
        """
        Returns the raw data object held by the handler.

        Returns:
            a list of 2D numpy arrays representing trials if flat, a tiered dictionary with keys of invividual name -> expert type -> model type -> trial (2D numpy array) if not flat.
        """
        if self.flat:
            return self.flat_trials
        else:
            return self.data_dict
        
    def gen_summaries(self):
        if self.flat: raise NotImplementedError("gen_summaries() only implemented for unflattened data.")
        self.trials_count = 0
        self.minutes_count = 0

        for individual in self.data_dict:
            for expert in self.data_dict[individual]:
                for model in self.data_dict[individual][expert]:
                    trial_data = self.data_dict[individual][expert][model]

                    self.trials_count += len(trial_data)
                    for trial in trial_data:
                        rows, _ = trial.shape
                        self.minutes_count += (rows - 1)*5 #5 minute time interval

    def print_summary(self):
        print(f"{self.trials_count} trials stored.")
        print(f"{self.minutes_count // 60}h {self.minutes_count % 60}m worth of data stored.")

    def flatten(self):
        """
        Converts data from a dictionary tiered format to a flattened list of trials. Destructive to list of trials.
        """
        if self.verbose: print("Starting to Flatten")
        output_list = []
        for individual in self.individuals:
            for expert in list(self.data_dict[individual].keys()):
                for model in list(self.data_dict[individual][expert].keys()):
                    output_list += self.data_dict[individual][expert][model]
                    del self.data_dict[individual][expert][model]
        del self.data_dict
        self.flat_trials = output_list
        print("made flat", len(self.flat_trials))
        self.flat = True
        if self.verbose: print("Flattening Complete")

    def save_as_csv(self, name, dest_folder=DATA_DEST + "/csv_saves/",seperate_flat_files = False):
        if self.verbose: print("Starting CSV Writing")
        use_columns = self.trial_labels + ["meta"]
        if self.flat and seperate_flat_files: #saves each trial in a seperate folder
            num = 0
            for trial in self.flat_trials:
                df = pd.DataFrame(trial, columns=use_columns)
                df.to_csv(dest_folder + name + str(num) + ".csv", sep=',', index=False, header=True)
                num += 1
        elif self.flat: #saves all trials in a single, large, .csv file
            print(self.flat_trials[0])
            df = pd.DataFrame(np.vstack(self.flat_trials), columns=use_columns)
            df.to_csv(dest_folder + name + ".csv", sep=',', index=False, header=True)
        else: #saves trials in seperate files, organised by folders.
            for individual in self.data_dict:
                os.makedirs(dest_folder + individual)
                for expert in self.data_dict[individual]:
                    os.makedirs(dest_folder + name + '/' + individual + '/' + expert)
                    for model in self.data_dict[individual][expert]:
                        os.makedirs(dest_folder + name + '/' + individual + '/' + expert + '/' + model)
                        data = self.data_dict[individual][expert][model]
                        df = pd.DataFrame(np.vstack(data), columns=use_columns)
                        df.to_csv(dest_folder + name + '/' + individual + '/' + expert + '/' + model + "/trials.csv", sep=',', index=False, header=True)
        if self.verbose: print("Finished CSV Writing to",dest_folder + name)

    def delete(self):
        print("deleting time!")
        self.flat_trials = None
        self.data_dict = None
    
    def print_structure(self):
        print("Structure")
        if not self.flat:
            for individual in self.data_dict:
                print(individual)
                for expert in self.data_dict[individual]:
                    print("\t",expert)
                    for model in self.data_dict[individual][expert]:
                        print("\t\t",model,"with",len(self.data_dict[individual][expert][model]),"trials")
        print()



if __name__ == "__main__":
    main_function = input("| pickle | convert | import |\nChoose: \n").lower()

    if main_function == "convert":
        individual = "adult0"
        overall_data_dict = dict()
        file_dest=DATA_DEST + "/object_save/data_dictionary_" + individual + "_data.pkl"
        data = import_from_obj(file_dest) #import data from pickle object
        overall_data_dict[individual] = data[individual]

        #Follows individual > expert > model > [trial data].
        expert = "clinical"
        model = "BBHE"
        example_trials = data[individual][expert][model]
        trial_len = len(example_trials)
        print("Trials have length",trial_len)

        trial_ind = randrange(0,trial_len)
        chosen_trial = example_trials[trial_ind]
        print("Chose trial",trial_ind,"with shape", chosen_trial.shape)

        print(chosen_trial)

        conv = convert_to_frames(chosen_trial)

        print("Converted data")
        print(conv)
    
    elif main_function == "pickle":
        import_raw_files()

    elif main_function == "import":
        all_data = DataImporter(verbose=True, data_folder=DATA_DEST)
        for individual_data in all_data:
            print("===Imported data for", all_data.current_individual)
            individual_data.flatten()
            print(individual_data.flat_trials[0])

            del individual_data