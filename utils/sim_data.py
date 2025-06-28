import os
import numpy as np
import pandas as pd
import random
from datetime import datetime
import sys
import pickle
import json
from omegaconf import OmegaConf
import gc
from random import randrange
from decouple import config
from collections import namedtuple, deque
from environment.reward_func import composite_reward
from utils.core import linear_scaling, calculate_features, pump_to_rl_action
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


SIM_DATA_PATH = config("SIM_DATA_PATH")
# CLN_DATA_PATH = config("CLN_DATA_PATH")

DATA_DEST = SIM_DATA_PATH

COHORT_VALUES = ["adolescent", "adult"] 


AGENT_TYPES = ["A2C", "AUXML", "BBHE", "BBI", "G2P2C", "PPO", "SAC", "TD3", "DDPG", "DPG"]

# stores which encoding version is used for each agent's data
FOLDER_TYPE_AGENTS = ["A2C", "AUXML", "G2P2C", "PPO", "SAC", "DDPG", "TD3", "DPG"]
FOLDER_TYPE_PROTOCOLS = ["training","testing","evaluation"]

ONLY_ADOLESCENT_AGENTS = ["DDPG", "DPG"]
ONLY_ADULT_AGENTS = []

CSV_TYPE_AGENTS = ["BBHE", "BBI"]
CSV_TYPE_PROTOCOLS = ["clinical"]

PROTOCOLS = FOLDER_TYPE_PROTOCOLS + CSV_TYPE_PROTOCOLS


ADOLESCENT_SUBJECT_NUMBER = 10
ADULT_SUBJECT_NUMBER = 10

ADULT_SUBJECTS = ["adult" + str(num) for num in range(ADULT_SUBJECT_NUMBER)]
ADOLESCENT_SUBJECTS = ["adolescent" + str(num) for num in range(ADOLESCENT_SUBJECT_NUMBER)]

SUBJECTS = ADULT_SUBJECTS + ADOLESCENT_SUBJECTS



# default file dest for saving pkl objects
OBJECT_SAVE_FILE = DATA_DEST + "/object_save/data_dictionary.pkl"

# stores which file names are excluded for csv type agent data folders
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

SHUFFLE_QUEUE_IMPORTS = False
IMPORT_SEED = 0

#designate file naming conventions to read from
PICKLE_FILE_NAME_END = "_data"
PICKLE_FILE_NAME_START = "data_dictionary_"

#import patient attributes
with open(MAIN_PATH + "/utils/patient_attrs.csv",'r') as f:
    lines = [line.split(',') for line in f.read().splitlines()]

patient_attr_names = lines[0]
patient_attr_dict = dict()
for line in lines[1:]:
    patient_attr_dict[line[0].lower()] = dict()
    for c,var in enumerate(patient_attr_names):
        patient_attr_dict[line[0].lower()][var] = line[c]

def get_patient_attrs(subject): return patient_attr_dict[subject.lower()]
    

### Helpers

def convert_mins_to_string(raw_mins):
    days = raw_mins//(60*24)
    hours = (raw_mins % (60*24)) // 60
    mins = raw_mins % (60)
    seconds = 0
    return f"{days}:{hours:02}:{mins:02}"

def convert_string_to_mins(time_string):
    days,hours,mins = tuple([int(i) for i in time_string.split(':')])
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

    for subject in SUBJECTS:
        print("Starting import for",subject)
        start_time = datetime.now() #start the write timer
        file_dest = file_dest_folder + file_name_start + subject + file_name_end + ".pkl"
        
        data = import_all_data(DATA_DEST, show_progress=True, subject_range=[subject]) #import data from files
        print("\nSuccesfully imported.")

        end_time = datetime.now() #end the read timer
        duration = end_time - start_time
        total_importing_time += duration.total_seconds()
        print(f"Data for {subject} agent imported in {int(duration.total_seconds() // 60)}m {duration.total_seconds() % 60 :.1f}s")

        obj_size = sys.getsizeof(data) #get size of object #FIXME doesn't seem to work correctly
        obj_size_mb = obj_size / (1024 ** 2)
        print("Returned object is",round(obj_size_mb,10),"MB")

        print("===Data Breakdown:===")
        for subject in data:
            print("Subject:",subject)
            for protocol in data[subject]:
                print("\t",protocol,':',sep='')
                for agent in data[subject][protocol]:
                    specific_length = len(data[subject][protocol][agent])
                    print("\t\tAgent", agent,"with length",specific_length)
                    if specific_length > 0:
                        print("\t\t\tItem 0 has shape",data[subject][protocol][agent][0].shape)
        
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
        cohort_range = COHORT_VALUES,
        agent_range = AGENT_TYPES,
        subject_range = SUBJECTS,
        csv_type_list = CSV_TYPE_AGENTS,
        folder_type_list = FOLDER_TYPE_AGENTS,
        show_progress = False
        ):
    """ Imports simulation data from a given folder.

    Args:
        dest (str, optional): The folder to search through. Defaults to "../data".
        cohort_range (list[str], optional): The list of cohorts to retrieve data from. Defaults to cohort_VALUES.
        agent_range (list[str], optional): The agents to retrieve data from. Defaults to AGENT_TYPES.
        subject_range (list[str], optional): The subjects to retrieve data from for folder type data. Defaults to SUBJECTS.
        csv_type_list (list[str], optional): The list of agent names encoded as csv type data. Defaults to CSV_TYPE_AGENTS.
        folder_type_list (list[str], optional): The list of agent names encoded as folder type data. Defaults to FOLDER_TYPE_AGENTS.
        show_progress (bool, optional): Decides if progress is printed to console. Defaults to False.

    Returns:
        _type_: A layered dictionary contains numpy arrays of simulation data as columns in each trial. Follows subject > protocol > agent > [trial data].
    """
    
    data_dict = dict() #consider file type for this! will be very slow
    
    #set up dictionary structure
    # subject > protocol > agent > [trial data]
    for subject in subject_range:
        data_dict[subject] = dict()
        for protocol in CSV_TYPE_PROTOCOLS:
            data_dict[subject][protocol] = dict()
            for agent in CSV_TYPE_AGENTS:
                data_dict[subject][protocol][agent] = []
        for protocol in FOLDER_TYPE_PROTOCOLS:
            data_dict[subject][protocol] = dict()
            for agent in FOLDER_TYPE_AGENTS:
                data_dict[subject][protocol][agent] = []

    for cohort in cohort_range:
        if show_progress: print("=Cohort:",cohort)
        
        #removed excluded agents from range given cohort group
        use_agent_range = agent_range[:] #copies list so it doesn't mutate the original object
        for agent in (ONLY_ADOLESCENT_AGENTS if cohort == "adult" else ONLY_ADULT_AGENTS):
            use_agent_range.remove(agent)


        for agent in use_agent_range:
            agent_folder = dest + '/' + cohort + '/' + agent + '/'

            if show_progress: print(agent_folder)
            if agent in csv_type_list:

                available_files = os.listdir(agent_folder)

                for excl_file in EXCLUDE_FILES:
                    if excl_file in available_files: 
                        available_files.remove(excl_file)
                        if show_progress: print("\t>>Excluded",excl_file,"from",agent_folder)

                for file in available_files:
                    subject_number = int(file.split('_')[-2]) - (20 if cohort == "adult" else 0)
                    run_subject = cohort + str(subject_number)
                    trial_number = file.split('_')[-1].split('.')[0]
                    protocol_type = "clinical"
                    file_dest = agent_folder + file
                    if run_subject in subject_range:
                        if show_progress: print('\t' + file_dest, run_subject)
                        meta_col = '_'.join([
                            agent, 
                            protocol_type, 
                            "0", #seed number
                            trial_number,
                            run_subject
                        ])
                        data_dict[run_subject][protocol_type][agent].append(import_from_csv_as_rows(file_dest, meta_col=meta_col))

                # data_dict[cohort][agent] = np.array(data_dict[cohort])


            elif agent in folder_type_list:
                available_folders = os.listdir(agent_folder)
                for folder in available_folders:
                    folder_dest = agent_folder + folder
                    run_seed = folder[-1]
                    run_subject = cohort + folder[-3]

                    if run_subject in subject_range:
                        if show_progress: print('\t' + folder_dest)
                        args = open_arg_file(folder_dest + '/args.json')
                        #taking validation and testing together in one list
                        for trial_folder in ['/testing/data/', '/training/data/']:
                            trial_folder_dest = folder_dest + trial_folder
                            available_files = os.listdir(trial_folder_dest)
                            for file in available_files:
                                if not EXCLUDE_IN_FILES in file:
                                    running_trial_number = 0

                                    #decide protocol type
                                    worker_number = int(file.split('_')[2][:-4])
                                    if trial_folder == "/training/data/": protocol_type = "training"
                                    elif 6000 > worker_number >= 5000: protocol_type = "testing"
                                    elif worker_number >= 6000: protocol_type = "evaluation"

                                    file_dest = trial_folder_dest + file
                                    meta_col_func = (lambda epi : '_'.join([
                                        agent, 
                                        protocol_type,
                                        run_seed,
                                        str(running_trial_number + epi), 
                                        run_subject
                                    ]))
                                    new_data_arrays = import_from_big_csv_as_rows(file_dest, meta_col_func=meta_col_func)
                                    for new_data_array in new_data_arrays:
                                        data_dict[run_subject][protocol_type][agent].append(new_data_array)

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

def patient_id_to_label(patient_id):
    if patient_id < 0 or patient_id >= 30: raise ValueError("Invalid patient id")
    return ["adolescent","adult","child"][patient_id//10] + str(patient_id % 10)



def convert_trial_into_transitions(data_obj, args, env_args, reward_func=(lambda cgm : composite_reward(None, cgm))):
    #data_obj is a 2D numpy array , rows x columns. Columns are :  cgm, meal, ins, t, meta_data
    window_size = args.obs_window

    rows, _ = data_obj.shape

    states = np.array([calculate_features(data_row, args, env_args) for data_row in data_obj])

    actions = [pump_to_rl_action(ins, args, env_args) for ins in data_obj[:, 2]]

    rewards = [linear_scaling(cgm, args.glucose_min, args.glucose_max) for cgm in data_obj[:, 0]]

    transitions = []
    for row_n in range(window_size, rows-1):
        state = np.array(states[row_n-window_size: row_n])
        action = np.array([actions[row_n]])
        reward = np.array([rewards[row_n+1]]) #sample from next states reward
        next_state = np.array(states[row_n-window_size+1: row_n+1])
        done = np.array([int(row_n == rows - 2)]) #FIXME change condition
        transitions.append(Transition(state, action, reward, next_state, done))

    return transitions

### Classes

class DataImporter:
    """
    Manages the parameterised importing of data.
    """
    def __init__(self, 
            data_folder=DATA_DEST,
            verbose = True,
            subjects = SUBJECTS, 
            agents = AGENT_TYPES, 
            protocols=PROTOCOLS,
            args=None,
            env_args=None
        ):
        self.args = args

        #ARGS overrides passed subjects and cohorts
        if args != None:
            print("Overwriting DataImporter Values from args.")
            self.subjects = [patient_id_to_label(self.args.patient_id)]
            self.agents = (AGENT_TYPES if args.data_algorithms == [] else args.data_algorithms)
            print(self.agents)
            self.protocols = (PROTOCOLS if args.data_protocols == [] else args.data_protocols)
        else:
            self.subjects, self.agents, self.protocols = subjects, agents, protocols

        self.cohorts = list(set([subj[:-1] for subj in self.subjects]))




        self.env_args = env_args
        self.verbose = verbose
        self.source_folder = data_folder + "/object_save/"
        self.current_data = None
        self.current_index = 0
        self.current_subject = self.subjects[0]
        
    def start(self):
        self.current_index = -1

    def __iter__(self):
        self.start()
        return self

    def __next__(self):
        self.current_index += 1
        if self.current_index < len(self.subjects):
            self.current_subject = self.subjects[self.current_index]
            self.import_current()
            return self.current_data
        else:
            self.current_subject = None
            self.delete_current()
            raise StopIteration
    
    def delete_current(self):
        if self.current_data != None:
            self.current_data.delete()
            self.current_data = None
            gc.collect()

    def import_subject(self, subject):
        #import file
        file_dest = self.source_folder + PICKLE_FILE_NAME_START + subject + PICKLE_FILE_NAME_END + ".pkl"
        if self.verbose: 
            print("\tStarting import for",self.current_subject,"from",file_dest)
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
        for protocol in list(raw_data[self.current_subject].keys()):
            if not (protocol in self.protocols):
                print("\t\tDeleted protocol",protocol)
                del raw_data[self.current_subject][protocol]
            else:
                for agent in list(raw_data[self.current_subject][protocol].keys()):
                    if not (agent in self.agents):
                        print("\t\tDeleted agent",agent)
                        del raw_data[self.current_subject][protocol][agent]
        if self.verbose: print("\tFinished stripping sections.")

        handled_data = DataHandler(raw_data)
        raw_data = None
        return handled_data
    
    def import_current(self):
        if self.verbose: print("\tDeleting previous data.")
        self.delete_current()
        if self.verbose: print("\tFinished deleting previous data.")

        handled_data = self.import_current(self.current_subject)
        self.current_data = handled_data

    def check_finished(self):
        return self.current_subject == None

    def get_trials(self, subjects_override = None, cohorts_override=None, agents_override= None, protocols_override=None):
        subjects = self.subjects if subjects_override == None else subjects_override
        cohorts = self.cohorts if cohorts_override == None else cohorts_override
        agents = self.agents if agents_override == None else agents_override
        protocols = self.protocols if protocols_override == None else protocols_override

        #TODO: add a layer to filter out subjects if cohorts, agents, or protocols limit them, since reading the files is the main bottleneck.

        trial_data = dict()
        for subject in subjects:
            file_dest = self.source_folder + PICKLE_FILE_NAME_START + self.subject + PICKLE_FILE_NAME_END + ".pkl"
            
            current_data = import_from_obj(file_dest) #import data from pickle object

            if self.verbose: print("\tStripping Irrelevant Sections")

            for protocol in current_data[subject]:
                if not (protocol in protocols):
                    del current_data[subject][protocol]
                else:
                    for agent in current_data[subject][protocol]:
                        if not (agent in agents):
                            del current_data[subject][protocol][agent]
            if self.verbose: print("\tFinished stripping sections.")

            trial_data[subject] = current_data[subject]
            del current_data

        return DataHandler(trial_data)

    def get_current_subject_attrs(self):
        return get_patient_attrs(self.current_subject)
    
    def create_queue(self, minimum_length=100, maximum_length=2000, mapping=convert_trial_into_transitions):
        self.queue = DataQueue(self, minimum_length, maximum_length, mapping)
        return self.queue

class DataHandler:
    """
    Manages the handling of data once imported.
    """
    def __init__(self, data_dict, verbose=False):
        self.data_dict = data_dict
        self.trial_labels = CSV_HEADERS
        self.flat = False
        self.verbose = verbose

        #detect range of subjects, protocols, agents, and cohorts
        self.subjects, self.protocols, self.agents, self.cohorts = [], [] , [], []
        for subject in list(data_dict.keys()):
            for protocol in list(data_dict[subject].keys()):
                for agent in list(data_dict[subject][protocol].keys()):
                    if data_dict[subject][protocol][agent] != []: #only add if empty
                        if not subject in self.subjects: self.subjects.append(subject)
                        cohort = subject[:-1]
                        if not cohort in self.cohorts: self.cohorts.append(cohort)
                        if not protocol in self.protocols: self.protocols.append(protocol)
                        if not agent in self.agents: self.agents.append(agent)
                    else:
                        del data_dict[subject][protocol][agent]
                        if self.verbose: print("Pruned empty dictionary entry",subject,protocol,agent)
        
        self.gen_summaries()
    def get_raw(self):
        """
        Returns the raw data object held by the handler.

        Returns:
            a list of 2D numpy arrays representing trials if flat, a tiered dictionary with keys of invividual name -> protocol type -> agent type -> trial (2D numpy array) if not flat.
        """
        if self.flat:
            return self.flat_trials
        else:
            return self.data_dict      
    def gen_summaries(self):
        if self.flat: raise NotImplementedError("gen_summaries() only implemented for unflattened data.")
        self.trials_count = 0
        self.minutes_count = 0

        for subject in self.data_dict:
            for protocol in self.data_dict[subject]:
                for agent in self.data_dict[subject][protocol]:
                    trial_data = self.data_dict[subject][protocol][agent]

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
        for subject in self.subjects:
            for protocol in list(self.data_dict[subject].keys()):
                for agent in list(self.data_dict[subject][protocol].keys()):
                    output_list += self.data_dict[subject][protocol][agent]
                    del self.data_dict[subject][protocol][agent]
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
            for subject in self.data_dict:
                os.makedirs(dest_folder + subject)
                for protocol in self.data_dict[subject]:
                    os.makedirs(dest_folder + name + '/' + subject + '/' + protocol)
                    for agent in self.data_dict[subject][protocol]:
                        os.makedirs(dest_folder + name + '/' + subject + '/' + protocol + '/' + agent)
                        data = self.data_dict[subject][protocol][agent]
                        df = pd.DataFrame(np.vstack(data), columns=use_columns)
                        df.to_csv(dest_folder + name + '/' + subject + '/' + protocol + '/' + agent + "/trials.csv", sep=',', index=False, header=True)
        if self.verbose: print("Finished CSV Writing to",dest_folder + name)
    def delete(self):
        print("deleting time!")
        self.flat_trials = None
        self.data_dict = None  
    def print_structure(self):
        print("Structure")
        if not self.flat:
            for subject in self.data_dict:
                print(subject)
                for protocol in self.data_dict[subject]:
                    print("\t",protocol)
                    for agent in self.data_dict[subject][protocol]:
                        print("\t\t",agent,"with",len(self.data_dict[subject][protocol][agent]),"trials")
        print()
    def get_subject_attrs(self):
        """Returns list of dictionary objects giving attributes of current subjects.
        """
        return [get_patient_attrs(subject) for subject in self.subjects]



class DataQueue: 
    def __init__(self, importer, minimum_length=1024, maximum_length = 8192, mapping = convert_trial_into_transitions):
        self.importer, self.minimum_length, self.maximum_length, self.mapping = importer, minimum_length, maximum_length, mapping
        self.queue = []
        self.queue_revolutions = 0
        self.subjects_n = len(self.importer.subjects)
        assert maximum_length >= minimum_length
    def start(self,count_transitions=True):
        self.current_subject_ind = 0
        self.current_subject = self.importer.subjects[self.current_subject_ind]
        self.current_subject_trial_ind = 0

        if count_transitions and len(self.importer.subjects) == 1: #run an altered first sync to minimise needed imports and count maximum transitions
            
            # import data
            remaining_length = self.maximum_length - len(self.queue)
            handled_data = self.importer.import_subject(self.current_subject)
            handled_data.flatten()

            #count transitions
            window_size = self.importer.env_args.obs_window
            transitions = 0
            n = 0
            for trial in handled_data.flat_trials:
                transitions += max(0, len(trial) - window_size - 1)
                # actual_transitions = len(self.mapping(trial, self.importer.args, self.importer.env_args))
                # assert transitions == actual_transitions
                n += 1
            
            print(transitions, "transitions counted.")
            self.total_transitions = transitions

            #add to queue

            handled_len = len(handled_data.flat_trials)
            gc.collect()

            while remaining_length > 0 and self.current_subject_trial_ind < handled_len:
                trial_mapping = self.mapping(handled_data.flat_trials[self.current_subject_trial_ind], self.importer.args, self.importer.env_args)
                mapping_len = len(trial_mapping)

                self.queue += trial_mapping
                remaining_length -= mapping_len
                self.current_subject_trial_ind += 1
            
            if self.current_subject_trial_ind >= handled_len:
                self.current_subject_trial_ind = 0

            del handled_data.flat_trials
            del handled_data
            gc.collect()

        elif count_transitions:
            raise NotImplementedError
        else:
            self.sync_queue()
    def next_subject(self):
        self.current_subject_ind += 1
        if self.current_subject_ind >= self.subjects_n:
            self.current_subject_ind = 0
            self.queue_revolutions += 1
        self.current_subject_trial_ind = 0
        self.current_subject = self.importer.subjects[self.current_subject_ind]
    def sync_queue(self):
        if len(self.queue) < self.minimum_length:
            remaining_length = self.maximum_length - len(self.queue)
            while remaining_length > 0:
                # print("Importing data for",self.current_subject,"at index",self.current_subject_ind)
                handled_data = self.importer.import_subject(self.current_subject)
                handled_data.flatten()
                
                if SHUFFLE_QUEUE_IMPORTS:
                    random.seed(IMPORT_SEED)
                    random.shuffle(handled_data.flat_trials)


                handled_len = len(handled_data.flat_trials)
                # print("Data Imported and flattened with", handled_len,"trials.")
                gc.collect()

                while remaining_length > 0 and self.current_subject_trial_ind < handled_len:
                    # print("\tMapping step",self.current_subject_trial_ind, handled_len)
                    trial_mapping = self.mapping(handled_data.flat_trials[self.current_subject_trial_ind], self.importer.args, self.importer.env_args)
                    if SHUFFLE_QUEUE_IMPORTS and not self.importer.args.preserve_trajectories:
                        random.shuffle(trial_mapping)
                    mapping_len = len(trial_mapping)

                    self.queue += trial_mapping
                    remaining_length -= mapping_len
                    self.current_subject_trial_ind += 1
                
                if self.current_subject_trial_ind >= handled_len:
                    self.current_subject_trial_ind = 0


                del handled_data.flat_trials
                del handled_data
            gc.collect()
            # print("Sync completed")
    def pop(self):
        out = self.queue.pop(0)
        self.sync_queue()
        return out
    def pop_batch(self,n):
        return [self.pop() for _ in range(n)]






if __name__ == "__main__":
    main_function = input("| pickle | convert | import |\nChoose: \n").lower()

    if main_function == "convert":
        subject = "adult0"
        overall_data_dict = dict()
        file_dest=DATA_DEST + "/object_save/data_dictionary_" + subject + "_data.pkl"
        data = import_from_obj(file_dest) #import data from pickle object
        overall_data_dict[subject] = data[subject]

        #Follows subject > protocol > agent > [trial data].
        protocol = "clinical"
        agent = "BBHE"
        example_trials = data[subject][protocol][agent]
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
        for subject_data in all_data:
            print("===Imported data for", all_data.current_subject)
            print("\tSubject attrs:",all_data.get_current_subject_attrs())
            subject_data.flatten()
            print(subject_data.flat_trials[0])

            del subject_data
    
    else:
        raise ValueError("Invalid choice.")