import os
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import gc

AGE_VALUES = ["adolescent", "adult"] 


DATA_DEST = "../SimulatedData" #FIXME change data destination for your script


MODEL_TYPES = ["A2C", "AUXML", "BBHE", "BBI", "G2P2C", "PPO", "SAC", "TD3", "DDPG", "DPG"] #stores the types of models accessed by the script

# stores which encoding version is used for each model's data
FOLDER_TYPE_MODELS = ["A2C", "AUXML", "G2P2C", "PPO", "SAC", "DDPG", "TD3", "DPG"] 
FOLDER_TYPE_EXPERTS = ["training","testing","evaluation"]

CSV_TYPE_MODELS = ["BBHE", "BBI"]
CSV_TYPE_EXPERTS = ["clinical"]

EXPERTS = FOLDER_TYPE_EXPERTS + CSV_TYPE_EXPERTS


ONLY_ADOLESCENT_MODELS = ["DDPG", "DPG"] #stores which models only have data for adolescents
ONLY_ADULT_MODELS = [] #stores which models only have data for adults

#number of individuals for each cohort
ADOLESCENT_INDIVIDUAL_NUMBER = 10
ADULT_INDIVIDUAL_NUMBER = 10

#generate names for individuals, used in file references
ADULT_INDIVIDUALS = ["adult" + str(num) for num in range(ADULT_INDIVIDUAL_NUMBER)] 
ADOLESCENT_INDIVIDUALS = ["adolescent" + str(num) for num in range(ADOLESCENT_INDIVIDUAL_NUMBER)]

INDIVIDUALS = ADULT_INDIVIDUALS + ADOLESCENT_INDIVIDUALS

# stores where object is saved to when run as main
OBJECT_SAVE_FILE = DATA_DEST + "/object_save/data_dictionary.pkl"

#designate file naming conventions to read from
PICKLE_FILE_NAME_END = "_data"
PICKLE_FILE_NAME_START = "data_dictionary_"

#record column names
CSV_HEADERS = ["cgm", "carbs", "ins", "t"]


def import_from_obj(file_dest=OBJECT_SAVE_FILE):
    with open(file_dest, 'rb') as f:
        data_dict = pickle.load(f)
        f.close()
    return data_dict

def save_to_obj_file(data_dict, file_dest=OBJECT_SAVE_FILE):
    with open(file_dest, 'wb') as f:
        data_dict = pickle.dump(data_dict, f)
        f.close()
    return data_dict




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

    SINGLE_INDIVIDUAL_FILES = True #decides if files are read per individual or all at once

    if SINGLE_INDIVIDUAL_FILES: #read from the files for each individual
        all_data = DataImporter(verbose=True, data_folder=DATA_DEST)
        for individual_data in all_data:
            print("===Imported data for", all_data.current_individual)
            individual_data.flatten()
            print(individual_data.flat_trials[0])

            del individual_data
    elif not SINGLE_INDIVIDUAL_FILES: #read from the single overall file
        start_time = datetime.now() #start the read timer

        file_dest= DATA_DEST + "/object_savedata_dictionary.pkl"
        print("Starting read from file",file_dest)
        data = import_from_obj(file_dest) #import data from pickle object

        end_time = datetime.now() #end the read timer
        duration = end_time - start_time
        print("Executed in",duration.total_seconds(), "seconds")

        file_size = os.path.getsize(file_dest) #obtain file size of read file
        print(f"Read file has size {file_size / (1024 * 1024):.2f}MB")
    

