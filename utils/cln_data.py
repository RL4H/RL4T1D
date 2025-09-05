import os
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import xport
import xport.v56
from datetime import datetime, timezone, timedelta
import math
from decouple import config
import random
from omegaconf import DictConfig, OmegaConf

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from visualiser.core import plot_episode
from utils.sim_data import convert_trial_into_transitions, convert_trial_into_windows

SIM_DATA_PATH = config('SIM_DATA_PATH')
if SIM_DATA_PATH == '':
    raise ImportError("Environment variable 'SIM_DATA_PATH' not defined.")

CLN_DATA_DEST = config('CLN_DATA_PATH')

CLN_DATA_SAVE_DEST = "/home/users/u7482502/data/cln_pickled_data" #FIXME make into env variable 

SHUFFLE_QUEUE_IMPORTS = False
IMPORT_SEED = 0

AGE_VALUES = ["adolescent", "adult"] 
CGM_TOLERANCE = 140
MLS_TOLERANCE = 140
INS_TOLERANCE = 140
MINIMUM_EPI_LEN = 120
BLANK_RESULT = {
    "data" : np.array([]),
    "meta" : {
        "blank" : True
    }
}
COLUMN_NAMES = ["epi", "cgm", "carbs", "ins", 't', "meta"]
CSV_COLUMN_TYPES = {
    "epi"       : "int32",
    "cgm"       : "float32",
    "carbs"     : "float32",
    "ins"       : "float64",
    "t"         : "string",
    "meta"      : "string"
}
SUBJECTS_N = 502
SUMMARY_HEADERS = ["name", "age", "bw", "tdi", "icr", "isf", "height", "sex", "system_name", "injections_type", "epi_n", "total_time", "ind", "subj_id"]
SUMMARY_NAME = "patient_attrs.csv"
SUMMARY_FILE_DEST = CLN_DATA_SAVE_DEST + '/' + SUMMARY_NAME

COL_ORDERING = ["cgm", "meal", "ins", "full_time", "meta"]

SUBJS_PUMP_NONEMPTY = [ "clinical" + ind for ind in ["3","4","5","10","11","12","13","16","17","18","19","20","21","22","23","24","25","27","28","31","34","35","36","37","38","39","40","41","42","43","44","48","49","50","51","54","55","57","62","63","66","70","73","74","75","78","79","81","82","86","87","88","89","90","91","94","95","96","97","98","99","100","101","102","103","104","105","106","107","108","111","112","113","114","115","116","119","120","121","122","123","124","125","126","133","134","135","136","139","140","141","142","145","146","147","148","149","150","151","152","155","156","158","161","162","163","164","168","169","172","173","174","175","176","177","178","182","183","184","187","188","189","190","193","194","195","196","197","198","199","200","201","202","203","204","205","206","207","208","209","210","211","212","214","215","216","217","220","221","226","227","228","229","230","231","232","236","239","240","241","242","247","250","256","257","258","259","260","261","262","263","264","265","266","267","268","269","275","276","277","278","279","280","283","284","285","288","289","291","292","293","294","295","296","297","301","302","303","308","309","310","311","312","313","314","317","320","321","322","323","324","325","328","329","330","331","332","333","334","335","339","340","341","342","343","347","348","349","350","353","354","355","358","359","360","363","364","365","366","367","368","371","372","373","374","379","380","381","382","385","386","387","388","389","390","391","392","395","402","403","404","407","410","411","412","413","414","415","418","419","420","423","424","425","426","427","428","431","434","437","438","439","440","443","444","445","446","447","448","449","450","451","452","457","459","461","462","465","466","469","472","473","477","478","483","484","485","486","490","491","492","499","500"] ]
SUBJS_INJ_NONEMPTY = [ "clinical" + ind for ind in ["2","7","9","15","30","33","47","53","59","61","65","68","72","77","110","118","130","132","138","144","154","160","167","171","181","186","192","219","223","225","244","246","249","253","255","272","274","282","287","316","319","327","338","346","352","357","362","370","376","378","384","394","397","399","401","406","409","417","422","430","433","442","456","468","471","480","482","489","498"] ]
SUBJS_EMPTY = [ "clinical" + ind for ind in ["0","1","6","8","14","26","29","32","45","46","52","56","58","60","64","67","69","71","76","80","83","84","92","109","117","127","128","129","131","137","143","153","157","159","165","166","170","179","180","185","191","213","218","222","224","233","234","235","237","243","245","248","251","252","254","270","271","273","281","286","290","298","299","304","305","306","315","318","326","336","337","344","345","351","356","361","369","375","377","383","393","396","398","400","405","408","416","421","429","432","435","436","441","453","455","458","460","463","467","470","474","475","476","479","481","487","488","493","494","495","497","501"] ]
SUBJS_OTHER = [ "clinical" + ind for ind in ["85","93","238","300","307","454","464","496"] ]

# General Helpers

def force_st_length(st, le):  return st + ' '*(max(0,le - len(st)))

def read_time(epoch_num):
    return datetime.fromtimestamp(epoch_num, tz=timezone.utc)

def convert_mins_to_string(raw_mins):
    days = raw_mins//(60*24)
    hours = (raw_mins % (60*24)) // 60
    mins = raw_mins % (60)
    seconds = 0
    return f"{days}:{hours:02}:{mins:02}"

def convert_string_to_mins(time_string):
    days,hours,mins = tuple([int(i) for i in time_string.split(':')])
    return days*60*24 + hours*60 + mins

def pretty_print_table(tab): #assumes rectangular
    if len(tab) == 0: return
    col_lengths = [max([len(str(tab[y][x])) for y in range(len(tab))]) for x in range(len(tab[0]))]
    for tab_row in tab: print(''.join([force_st_length(str(i), col_lengths[x] + 3) for x,i in enumerate(tab_row)]))

# XPT Importers
def import_xpt_file(file_dest,file_name,show=False):
    with open(file_dest + '/' + file_name + '.xpt', 'rb') as f:
        library = xport.v56.load(f)
    df = library[file_name]
    if show: print("Dataset parsed"); print(df)

    return df

def import_large_xpt_file(folder_dest, file_name, filter_func, chunk_size=10000, debug_show=False, declare=True):
    if declare: print("Beginning import for",file_name)
    file_dest = folder_dest + '/' + file_name + '.xpt'
    xpt_chunks = pd.read_sas(file_dest, format='xport', chunksize=chunk_size)

    total_df = pd.DataFrame()
    
    # Process each chunk
    for i, chunk_df in enumerate(xpt_chunks):
        filtered_chunk = chunk_df[chunk_df.apply(filter_func,axis=1)]
        if debug_show and i%100 == 0: print(f"Processing chunk {i+1} with {len(chunk_df)} rows, retained {len(filtered_chunk)} rows after filter.")

        total_df = pd.concat( (total_df, filtered_chunk) )
        

    if declare: print("Completed import for",file_name)
    return total_df.reset_index(drop=True)

def import_for_subject(file_dest,file_name, USUBJID):
    with open(file_dest + '/' + file_name + '.xpt', 'rb') as f:
        df = xport.v56.load(f)[file_name]
        df = df[df['USUBJID'] == str(USUBJID)]
    return df

# Subject Data Handling
def filter_for_subject(df, USUBJID):
    new_df = df[df['USUBJID'] == str(USUBJID)]
    return new_df.reset_index(drop=True)

def seperate_df_epis(p_df):
    return [ df[df["epi"] == epi].reset_index(drop=True) for epi in range(max(df["epi"])+1)]
 
def find_windows(df_epi,window_size= ((24*60)//5) ):
    le = len(df_epi)
    current_index = 0 #FIXME decide if buffer from start is appropiate
    days = []
    while current_index < le-window_size:
        next_index = current_index + window_size
        days.append((current_index, next_index))
        current_index = next_index
    return days
    
def calculate_average_tdi(p_df):
    #find 24 hour daily windows
    day_len = (24 * 60) // 5 #5 minutes per index
    
    epi_dfs = seperate_df_epis(p_df)
    tdi_estimates = []
    for epi_df in epi_dfs:
        windows = find_windows(epi_df, day_len)
        for start_ind,end_ind in windows:
            ins_values = list(epi_df["ins"].loc[start_ind:end_ind])
            tdi_estimates.append(sum(ins_values))

    if len(tdi_estimates) == 0:
        return float('nan')
    else:
        mean, std = np.mean(tdi_estimates), np.std(tdi_estimates)
        # print(f"TDI: {mean}±{std}")
        return mean

# Subject Data Imports and Exports   

with open(SUMMARY_FILE_DEST,'r') as f:
    lines = [line.split(',') for line in f.read().splitlines()]

#name, age, bw, tdi, icr, isf, height, sex, system_name, injections_type, epi_n, total_time, ind, subj_id
patient_attr_names = lines[0]
patient_attr_dict = dict()
for line in lines[1:]:
    subj_name = line[0]
    patient_attr_dict[subj_name] = dict()
    for c,var in enumerate(patient_attr_names):
        patient_attr_dict[subj_name][var] = line[c]

def get_patient_attrs(subject): return patient_attr_dict[subject.lower()]

def read_individual(subj_id,debug_show=True):
    # read relevant data
    subj_LB = filter_for_subject(LB, subj_id) #cgm data
    subj_LB_len = len(subj_LB)
    if subj_LB_len == 0: 
        print("No subject glucose data found, returning blank data")
        return BLANK_RESULT
    subj_LB = subj_LB.drop(subj_LB_len-1)
    
    subj_FACM = filter_for_subject(FACM,subj_id) #insulin data
    #subj_FACM = subj_FACM[subj_FACM['FASTRESN'].notna()].reset_index(drop=True)
    subj_FACM = subj_FACM.sort_values(by="FADTC").reset_index()
    
    subj_ML = filter_for_subject(ML,subj_id) #meals data
    subj_ML = subj_ML.sort_values(by="MLDTC").reset_index()
    subj_ML = subj_ML[subj_ML['MLCAT'] != "USUAL DAILY CONSUMPTION"].reset_index(drop=True) #remove usual daily consumption lines; not relevant here
    
    
    # check type of treatment
    treatment_types = list(set(subj_FACM["INSDVSRC"]))
    if '' in treatment_types: treatment_types.remove('')
    if ["Pump"] == treatment_types: treatment_type = "Pump"
    elif ["Injections"] == treatment_types: treatment_type = "Injections"
    else: 
        print("\tNo treatment type found, returning blank result")
        return BLANK_RESULT #treatment_type = "Blank"

    if debug_show: print("\tTreatment Type:",treatment_type)


    # read time information
    cgm_time = subj_LB["LBDTC"]
    ins_time = subj_FACM["FADTC"]
    mls_time = subj_ML["MLDTC"]

    #check meal data is present
    if len(mls_time) < 5:
        print("\tInsufficient meal data Found.")
        return BLANK_RESULT
    
    subj_start_time = cgm_time.loc[0]
    subj_end_time = cgm_time.loc[ len(cgm_time) - 1]
    subj_duration = subj_end_time - subj_start_time

    # sample readings in 5 minute intervals, sepearting by episodes when longer gaps appear
    cur_episode = 0
    cur_episode_len = 0
    cur_time = subj_start_time

    cgm_ind = 0
    ins_ind = 0
    mls_ind = 0
    while mls_time[mls_ind] < subj_start_time and mls_ind < len(mls_time): mls_ind += 1 #skip meals set before cgm data is recorded.
    mls_skip = mls_ind

    if treatment_type == "Pump":
        current_insulin_rate = subj_FACM["FASTRESN"].loc[ins_ind]
        while ins_time[ins_ind] < subj_start_time: 
            if subj_FACM["FATEST"].loc[ins_ind] == "BASAL FLOW RATE":
                next_insulin_datapoint = subj_FACM["FASTRESN"].loc[ins_ind] 
                current_insulin_rate = (next_insulin_datapoint if not (math.isnan(next_insulin_datapoint)) else 0.0)
            ins_ind += 1 #skip insulin readings set before cgm data is recorded.
    else: #Injections
        while ins_time[ins_ind] < subj_start_time: ins_ind += 1 #skip insulin readings set before cgm data is recorded.
        
    episodes = [] # [ [ (epi, cgm, meal, ins, t, meta), ... ] ]
    cur_episode_start_time = subj_start_time
    cur_episode_time_obj = datetime.fromtimestamp(cur_episode_start_time, tz=timezone.utc)
    if debug_show: print("\tFirst episode starting at",cur_episode_time_obj)
    cur_episode_time_offset = cur_episode_time_obj.hour * (60*60) + cur_episode_time_obj.minute * 60 + cur_episode_time_obj.second
    rows = [] 

    while cur_time < subj_end_time:
        cur_cgm_time = cgm_time[cgm_ind]
        
        if cur_cgm_time - cur_time > CGM_TOLERANCE: #check if timeskip in cgm monitor
            if debug_show: print("\tAfter episode of",cur_episode_len*5,"m, skip of", (cur_cgm_time - cur_time)/60,"m detected, new episode starting at",datetime.fromtimestamp(cur_cgm_time, tz=timezone.utc))
            episodes.append(rows)
            rows = []
            cur_episode_len = 0
            cur_episode += 1

            cur_time = cur_cgm_time
            cur_episode_start_time = cur_time
            cur_episode_time_obj = datetime.fromtimestamp(cur_time, tz=timezone.utc)
            cur_episode_time_offset = cur_episode_time_obj.hour * (60*60) + cur_episode_time_obj.minute * 60 + cur_episode_time_obj.second
            
            while mls_ind < len(mls_time) and mls_time[mls_ind] < cur_time - MLS_TOLERANCE: mls_ind += 1 #skip meals between episodes
            while ins_ind < len(ins_time) and ins_time[ins_ind] < cur_time - INS_TOLERANCE: ins_ind += 1 #skip insulin between episodes

        if treatment_type == "Pump":
            ins_reading = 0
            while ins_ind < len(ins_time) and ins_time[ins_ind] < cur_time + INS_TOLERANCE: #find latest pump rate reading
                next_insulin_datapoint = subj_FACM["FASTRESN"].loc[ins_ind]
                next_insulin_datapoint = (next_insulin_datapoint if not (math.isnan(next_insulin_datapoint)) else 0.0) 
                if subj_FACM["FATEST"].loc[ins_ind] == "BASAL FLOW RATE":
                    current_insulin_rate = next_insulin_datapoint#FIXME account for proportions of active time instead
                elif subj_FACM["FATEST"].loc[ins_ind] in ["BASAL INSULIN", "BOLUS INSULIN"]:
                    ins_reading += next_insulin_datapoint
                ins_ind += 1
            ins_reading += current_insulin_rate
        else:
            ins_reading = 0
            while ins_ind < len(ins_time) and ins_time[ins_ind] < cur_time + INS_TOLERANCE:
                ins_reading += subj_FACM["FASTRESN"].loc[ins_ind]
                ins_ind += 1

        mls_total = 0
        while mls_ind < len(mls_time) and mls_time[mls_ind] < cur_time + MLS_TOLERANCE:
            mls_total += subj_ML["MLDOSE"].loc[mls_ind]
            mls_ind += 1
         
        rows.append([
            cur_episode,
            subj_LB["LBORRES"][cgm_ind],
            mls_total,
            ins_reading,
            convert_mins_to_string(int(cur_time - cur_episode_start_time + cur_episode_time_offset)// 60),
            "meta_TODO"
        ])

        cur_time += 300 # increment by 5 minutes
        cgm_ind += 1 #index cgm index

        cur_episode_len += 1

    if debug_show: print("\tFinal episode of",cur_episode_len*5,"m\n")
    episodes.append(rows)

    # filter out short episodes
    total_rows = []
    current_epi = 0
    for n_epi, epi_rows in enumerate(episodes):
        epi_len = len(epi_rows)*5
        if epi_len < MINIMUM_EPI_LEN:
            if debug_show: print("\tEpisode",n_epi,"removed for being too short (",epi_len,"m)")
        else:
            for row in epi_rows: row[0] = current_epi #change episode to reconsider removed episodes
            total_rows += epi_rows
            current_epi += 1
    
    # convert trial data to numpy
    trial_data = np.array(total_rows)

    
    if debug_show: 
        print("\tTotal rows after filtration are",len(total_rows)*5,"m, removed",(sum([len(epi_rows) for epi_rows in episodes]) - len(total_rows))*5, 'm' )
        print(f"Meal index {mls_ind}/{len(mls_time)} reached. Started at index {mls_skip}.")
        ml_col = trial_data[:,2]
        ml_count = len(list(filter(lambda x : float(x) != 0.0, ml_col)))
        print(f"{ml_count} meals found.")
        
        

    # generate meta data

        
    meta_data = {
        "treatment type": treatment_type,
        "subject id" : subj_id,
        "blank" : False
    }

    return {
        "data": trial_data,
        "meta": meta_data
    }

def save_subj_file(subj_info, filepath, filename):
    meta_content = "subject_id_" + str(subj_info["meta"]["subject id"]) + "_" + str(subj_info["meta"]["treatment type"])
    for line in subj_info["data"]: 
        line[-1] = meta_content
    txt = ','.join(COLUMN_NAMES) + '\n' + '\n'.join([','.join(line) for line in (subj_info['data'])])
    with open(filepath + '/' + filename, 'w') as f:
        f.write(txt)
    return 1

def read_subj_file(file_num,generate_as_epi_list=True,show_warnings=True):
    file_name = "subj_ind_" + str(file_num) + ".csv"
    file_dest = CLN_DATA_SAVE_DEST + '/' +  file_name

    df = pd.read_csv(file_dest, header="infer", dtype=CSV_COLUMN_TYPES)
    df = df.rename(columns={"t" : "full_time", "carbs" : "meal"})
    
    if len(list(df["epi"])) == 0: 
        if show_warnings: print(f"Importing empty data (subject ind {file_num})")
        return pd.DataFrame()

    t_col = []
    current_t = 0
    current_episode = df["epi"].loc[0]
    for epi in list(df["epi"]):
        if epi != current_episode:
            current_t = 0
            current_episode = epi
            
        t_col.append(current_t)
        current_t += 1
        
    df["t"] = t_col
    
    df["day_hour"] = df["full_time"].apply(lambda t : int(t.split(':')[1]))
    df["day_min"] = df["full_time"].apply(lambda t : int(t.split(':')[2]))

    if generate_as_epi_list:
        grouped = df.groupby('epi')
        return [group.reset_index(drop=True) for _, group in grouped]
    else:
        return df

def convert_df_to_arr(df):
    # rows = len(df)
    # cols = len(COL_ORDERING)

    # arr = np.zeros( (rows, cols) )
    # for col, header in enumerate(COL_ORDERING):
    #     arr[:, col] = list(df[header])
    
    # return arr

    return df[COL_ORDERING].values.tolist()





# Graphing
def display_subj_epi_graph(file_num,epi=0):
    file_name = "subj_ind_" + str(file_num) + ".csv"
    file_dest = CLN_DATA_SAVE_DEST + '/' +  file_name

    df = pd.read_csv(file_dest, header="infer", dtype=CSV_COLUMN_TYPES)

    df = df.rename(columns={"epi" : "episode", "carbs" : "meal"})
    if not epi in df["episode"]: raise IndexError(f"Episode {epi} does not exist.")
    # df = df[df['episode'] == epi].reset_index()

    base_t = datetime(2020, 1, 1, 0, 0)

    df["time"] = df["t"].apply(lambda t : base_t + timedelta(minutes=convert_string_to_mins(t)))
    df["day_hour"] = df["t"].apply(lambda t : int(t.split(':')[1]))
    df["day_min"] = df["t"].apply(lambda t : int(t.split(':')[2]))
    
    # df["t"] = df["t"].apply(lambda t : convert_string_to_mins(t) // 5)
    
    new_t_col = []
    current_t = 0
    current_episode = df["episode"].loc[0]
    for epi in list(df["episode"]):
        if epi != current_episode:
            current_t = 0
            current_episode = epi
            
        new_t_col.append(current_t)
        current_t += 1
        
    df["t"] = new_t_col

    for col_name in ["rew","rl_ins","mu","sigma","prob","state_val"]: df[col_name] = 0.0

    tester = 0
    dummy = DummyClass(df)
    plot_episode(dummy, tester, episode=epi)

def display_all_subj_epi_graph(file_num,cap=3):
    file_name = "subj_ind_" + str(file_num) + ".csv"
    file_dest = CLN_DATA_SAVE_DEST + '/' +  file_name

    df = pd.read_csv(file_dest, header="infer", dtype=CSV_COLUMN_TYPES)

    max_epi = max(df["epi"])
    del df
    for t_epi in range(min(max_epi,cap-1)+1):
        display_subj_epi_graph(file_num, t_epi)

# Classes
class DummyClass:
    def __init__(self, df):
        self.df = df
        self.plot_version = 1
    def get_test_episode(self,tester,epi):
        return self.df[self.df['episode'] == epi]

class ClnDataImporter:
    def __init__(self, args, env_args):
        self.args, self.env_args = args, env_args
        self.subj_ind = args.patient_id
        self.subj_name = "clinical" + str(self.subj_ind)
        self.attrs = get_patient_attrs(self.subj_name)
    def calculate_vld_split(self, mapping):
        vld_split = None

        self.load()
        for c,df in enumerate(self.df_list):
            trial_transitions = max(0, len(df) - self.env_args.window_size - 1)
            if trial_transitions < self.args.vld_interactions:
                vld_split = (c, self.args.vld_interactions)
                break
        self.clear()

        if vld_split == None:
            raise ValueError("Not enough data to split validation trial.")
        return vld_split
    def load(self, vld_split_indicies=None, is_vld=False):
        self.df_list = read_subj_file(self.subj_ind)
        if SHUFFLE_QUEUE_IMPORTS:
            random.seed(IMPORT_SEED)
            random.shuffle(self.df_list)
            self.df_seeds = [randrange(0,10000) for _ in range(len(self.df_list))]
        
        if vld_split_indicies != None:
            target_epi = self.df_list.pop(vld_split_indicies[0])
            split_epi_vld = target_epi.loc[:vld_split_indicies[1]]
            split_epi_vld['epi'] = '0' #FIXME check if this should be a string

            split_epi_trn = target_epi.loc[vld_split_indicies[1]:].reset_index(drop=True)
            self.df_list.insert(vld_split_indicies[0], split_epi_trn)

            if is_vld: 
                self.df_list = [split_epi_vld]
                return split_epi_vld
            else: 
                return self.df_list
        else:
            return self.df_list
    def clear(self):
        del self.df_list
    def create_queue(self, minimum_length=1024, maximum_length=8192, mapping=convert_trial_into_transitions, reserve_validation=0):
        self.queue = ClnDataQueue(self, minimum_length, maximum_length, mapping, reserve_validation)
        return self.queue 
        
class ClnDataQueue: 
    def __init__(self, importer, minimum_length=1024, maximum_length = 8192, mapping = convert_trial_into_transitions, reserve_validation=0):
        self.importer, self.minimum_length, self.maximum_length, self.mapping = importer, minimum_length, maximum_length, mapping
        self.queue = []
        self.queue_revolutions = 0
        self.subjects_n = 1 #len(self.importer.subjects)
        assert maximum_length >= minimum_length

        self.reserve_validation = reserve_validation
        self.reserve_validation_trials = 0 #gets reassigned in start()
        self.reset_validation()
        self.vld_split = self.importer.calculate_vld_split(mapping)
    def start(self,count_transitions=True):
        self.trial_ind = self.reserve_validation_trials #start index after validation trials


        if count_transitions: #run an altered first sync to minimise needed imports and count maximum transitions
            
            # import data
            remaining_length = self.maximum_length - len(self.queue)
            df_list = self.importer.load(self.vld_split, False)

            #count transitions
            window_size = self.importer.env_args.obs_window
            transitions = 0
            n = 0
            self.reserve_validation_trials = 0
            reserve_validation_trial_count = 0
            for trial in df_list:
                trial_transitions = max(0, len(trial) - window_size - 1)
                transitions += trial_transitions
                if reserve_validation_trial_count < self.reserve_validation:
                    reserve_validation_trial_count += trial_transitions
                    self.reserve_validation_trials += 1
                n += 1
            
            print(transitions, "transitions counted,", self.reserve_validation_trials, "trials reserved for validation.")
            self.total_transitions = transitions
            self.trial_ind = self.reserve_validation_trials #start index after validation trials

            #add to queue

            handled_len = len(df_list)

            while remaining_length > 0 and self.trial_ind < handled_len:
                trial_mapping = self.mapping(df_list.flat_trials[self.trial_ind], self.importer.args, self.importer.env_args)
                mapping_len = len(trial_mapping)

                self.queue += trial_mapping
                remaining_length -= mapping_len
                self.trial_ind += 1
            
            if self.trial_ind >= handled_len:
                self.trial_ind = self.reserve_validation_trials

            self.importer.clear()

        elif count_transitions:
            raise NotImplementedError
        else:
            self.sync_queue()
    def sync_queue(self):
        if len(self.queue) < self.minimum_length:
            df_list = self.importer.load(self.vld_split, False)
            remaining_length = self.maximum_length - len(self.queue)
            while remaining_length > 0:

                while remaining_length > 0 and self.trial_ind < df_list:
                    # print("\tMapping step",self.trial_ind, handled_len)
                    trial_mapping = self.mapping(df_list[self.trial_ind], self.importer.args, self.importer.env_args)

                    if SHUFFLE_QUEUE_IMPORTS:
                        random.seed(self.importer.df_seeds[self.trial_ind])
                        random.shuffle(trial_mapping)
                    mapping_len = len(trial_mapping)

                    self.queue += trial_mapping
                    remaining_length -= mapping_len
                    self.trial_ind += 1
                
                if self.trial_ind >= len(df_list):
                    self.trial_ind = self.reserve_validation_trials

            self.importer.clear()
            # print("Sync completed")
    def pop(self):
        out = self.queue.pop(0)
        self.sync_queue()
        return out
    def pop_batch(self,n):
        return [self.pop() for _ in range(n)]
    
    def reset_validation(self):
        self.validation_trial_ind = 0
        self.validation_in_trial_ind = 0
        self.vld_queue = []
    def start_validation(self):
        self.reset_validation()
        self.sync_validation()
    def sync_validation(self):
        if len(self.vld_queue) <= 0:
            # assumes only one individual being used
            df_list = self.importer.load(self.vld_split, True)
            
            if SHUFFLE_QUEUE_IMPORTS:
                random.seed(IMPORT_SEED)
                random.shuffle(df_list)
            
            trial_mapping = self.mapping(df_list[self.validation_trial_ind], self.importer.args, self.importer.env_args)

            while len(self.vld_queue) < self.reserve_validation + 20: #and len(self.vld_queue) < self.maximum_length
                if self.validation_in_trial_ind > len(trial_mapping):
                    self.validation_trial_ind = (self.validation_trial_ind + 1) % self.reserve_validation_trials
                    trial_mapping = self.mapping(df_list[self.validation_trial_ind], self.importer.args, self.importer.env_args)
                    self.validation_in_trial_ind = 0

                    if SHUFFLE_QUEUE_IMPORTS:
                        random.seed(self.importer.df_seeds[self.trial_ind])
                        random.shuffle(trial_mapping)
                
                self.vld_queue.append(trial_mapping[self.validation_in_trial_ind])     

            self.importer.clear()
    def pop_validation(self):
        out = self.vld_queue.pop(0)
        self.sync_validation()
        return out
    def pop_validation_queue(self, n):
        return [self.pop_validation() for _ in range(n)]

# Main 
if __name__ == "__main__":
    option = input("Select:\n| convert | summarise | display | convert 2 | generate types |\n: ").lower().strip()
    if option == "convert":

        print("Converting Subject Data to .csv format.")
        
        LB = import_xpt_file(CLN_DATA_DEST,'LB') # glucose levels
        ML = import_xpt_file(CLN_DATA_DEST,'ML') # meal data
        FACM = import_xpt_file(CLN_DATA_DEST,'FACM') # insulin data

        subject_ids = list(set(LB['USUBJID']))
        subject_ids = sorted([int(i) for i in subject_ids])
        subject_len = len(subject_ids)
        print(subject_len," subjects found.")

        print("Reading from",CLN_DATA_DEST)
        print("Writing to",CLN_DATA_SAVE_DEST)
        
        for n in range(subject_len):
            #break
            subject_id = subject_ids[n]
            
            subject_data = read_individual(subject_id, False)
            file_name = "subj_ind_" + str(n) + ".csv"

            if not subject_data["meta"]["blank"]:

                save_subj_file(subject_data, CLN_DATA_SAVE_DEST, file_name)

                print(f"Subject {subject_id} ({n}/{subject_len-1}) info saved to {file_name}")
            else:
                print(f"Subject {subject_id} ({n}/{subject_len-1}) info skipped, saving blank file")

                with open(CLN_DATA_SAVE_DEST + '/' + file_name, 'w') as f:
                    f.write(','.join(COLUMN_NAMES))
    
    elif option == "display":
        from random import randrange

        print("Generating graphic for random patient.")

        ind_n = randrange(0, SUBJECTS_N)
        display_all_subj_epi_graph(ind_n,cap=3)

    elif option == "summarise":

        print("Creating data summary table.")
        DM = import_xpt_file(CLN_DATA_DEST,'DM') #demographics
        FACM = import_xpt_file(CLN_DATA_DEST,'FACM') #insulin
        CM = import_xpt_file(CLN_DATA_DEST,'CM') #insulin device type
        DI = import_xpt_file(CLN_DATA_DEST,'DI') #device type mapping
        DX = import_xpt_file(CLN_DATA_DEST,'DX') #non standard device names

        VS_filt = import_large_xpt_file(CLN_DATA_DEST, 'VS', lambda row : row["VSTESTCD"] in [b"HEIGHT",b"WEIGHT"], debug_show=False, declare=True) #vital signs. takes ~400 seconds to run, or ~7 minutes. 11336 iterations. ~ 113356252 total rows
        VS_filt["USUBJID"] = VS_filt["USUBJID"].astype(str)

        rows = [ SUMMARY_HEADERS ]

        subj_ids = sorted(list(set(CM["USUBJID"].astype(int))))

        # print('\t'.join([force_st_length(str(i), 20) for i in rows[-1]]))
        # for ind_n in range(0,12):
        for ind_n in range(SUBJECTS_N):
            df = read_subj_file(ind_n)

            subj_id = subj_ids[ind_n]
            if len(df) != 0:
                subj_meta = df["meta"].loc[0]
                subj_id_alt = int(subj_meta.split('_')[2])
                assert subj_id_alt == int(subj_id) #FIXME remove later

                
                tdi = calculate_average_tdi(ind_n)
                isf = round(100 / tdi,2) #FIXME dont use 100 rule
                """
                www.diabetesqualified.com.au/insulin-sensitivity-factor-explained
                maybe use a combination of IOB, and meal gaps to measure the isf.
                """
                icr = round(500 / tdi,2) #FIXME don't use 500 rule
                """
                Consider taking a suitable time in simulation where a single large meal is taken, and observe rise in insulin. Have to account for isf first to use it as a correction factor.
                """
                tdi = round(tdi,2)
                
                injections_type_list = list(set(subj_FACM["INSDVSRC"]))
                if '' in injections_type_list: injections_type_list.remove('')
                injections_type = ';'.join(injections_type_list)  if len(injections_type_list) > 0 else "NA"

                df_epis = seperate_df_epis(df)
                epi_n = len(df_epis)
            
                total_time = len(df)*5 #5 minutes per index
            else:
                tdi = isf = icr = float('nan')
                injections_type = 'NA'
                epi_n = total_time = 0
                
                
            
            subj_DM = filter_for_subject(DM,subj_id).loc[0]
            subj_VS = filter_for_subject(VS_filt,subj_id)
            subj_FACM = filter_for_subject(FACM,subj_id)
            subj_CM = filter_for_subject(CM,subj_id)
            subj_DX = filter_for_subject(DX,subj_id)
            
            name = f"CLINICAL_SUBJ_IND_{ind_n}_{subj_id}"
            age = float(subj_DM['AGE'])

            if any(subj_VS["VSTESTCD"] == b"WEIGHT"): bw = round(np.mean(list(subj_VS[subj_VS["VSTESTCD"] == b"WEIGHT"]["VSSTRESN"])) * 0.45359237,2) # converting bw from LB to kg
            else: weight = float('nan')
            
            if any(subj_VS["VSTESTCD"] == b"HEIGHT"): height = round(np.mean(list(subj_VS[subj_VS["VSTESTCD"] == b"HEIGHT"]["VSSTRESN"])) * 2.54,2) # converting height from in to cm
            else: height = float('nan')
                                                                                            
            sex = subj_DM["SEX"]

            system_names_list = list(subj_DX["DXTRT"])
            for excl_name in ["INSULIN PUMP", "MULTIPLE DAILY INJECTIONS", "CLOSED LOOP INSULIN PUMP"]:
                if excl_name in system_names_list: system_names_list.remove(excl_name)
            system_name = ';'.join(system_names_list) if len(system_names_list) > 0 else "NA"

            rows.append( (name, age, bw, tdi, icr, isf, height, sex, system_name, injections_type, epi_n, total_time, ind_n, subj_id) )
            # print('\t'.join([force_st_length(str(i), 20) for i in rows[-1]]))

        pretty_print_table(rows)

        txt = '\n'.join([','.join([str(i) for i in row]) for row in rows])
        with open(SUMMARY_FILE_DEST, 'w') as f:
            f.write(txt)
        print(f"Summary file saved to {SUMMARY_FILE_DEST}.")
    
    elif option == "convert 2":
        import gc
        from utils.cln_data import ClnDataImporter
        from experiments.glucose_prediction.portable_loader import CompactLoader
        from utils.sim_data import calculate_augmented_features

        SEEDS = [0,1,2]
        # CLN_DATA_PATH = config('CLN_DATA_PATH')
    

        for patient_id in range(SUBJECTS_N):
            gc.collect()
            print("Importing for patient id",patient_id,"index",get_patient_attrs("clinical" + str(patient_id))["subj_id"])
            args = OmegaConf.create({
                "patient_ind" : patient_id,
                "patient_id" : patient_id,
                "batch_size" : 8192,
                "data_type" : "simulated", #simulated | clinical,
                "data_protocols" : ["evaluation","training"], #None defaults to all,
                "data_algorithms" : ["G2P2C","AUXML", "PPO","TD3"], #None defaults to all,
                "obs_window" : 12,
                "control_space_type" : 'exponential_alt',
                "insulin_min" : 0,
                "insulin_max" : 350,
                "glucose_min" : 39,
                "glucose_max" : 600,
                "obs_features" : ['cgm','insulin','day_hour']
            })

            importer = ClnDataImporter(args=args,env_args=args)
            flat_trials = importer.load()
            del importer

            data = CompactLoader(
                args, args.batch_size*1, args.batch_size*4, 
                flat_trials,
                lambda trial : calculate_augmented_features(convert_df_to_arr(trial), args, args),
                1,
                lambda trial : max(0, len(trial) - args.obs_window - 1) if trial['meta'].loc[0].split('_')[-1] == 'Pump' else 0, #exclude non pump data
                0,
                1024,
                folder= CLN_DATA_SAVE_DEST + '/'
            )

            data.save_compact_loader_object()
            print("\tData saved for seed 0.")
            for seed in range(1,3):
                data.reset_shuffle(seed, data.n_list)

                data.save_compact_loader_object()
                print(f"\tData saved for seed {seed}.")
    
    elif option == "generate types":
        import gc
        from utils.cln_data import ClnDataImporter
        from experiments.glucose_prediction.portable_loader import CompactLoader, load_compact_loader_object
        from utils.sim_data import calculate_augmented_features

        pump_nonempty = []
        inj_nonempty = []
        empty = []
        other = []

        for patient_id in range(SUBJECTS_N):
            attrs = get_patient_attrs("clinical" + str(patient_id))
            print("Importing for patient id",patient_id,"index",attrs["subj_id"])
            injections_type = attrs["injections_type"]
            
            data_save_path_args = CLN_DATA_SAVE_DEST + '/' + f"temp_args_{patient_id}_{0}.pkl"

            data = load_compact_loader_object(data_save_path_args)

            data_le = len(data)

            if data_le == 0:
                empty.append(patient_id)
            elif injections_type == "Pump":
                pump_nonempty.append(patient_id)
            elif injections_type == "Injections":
                inj_nonempty.append(patient_id)
            else:
                other.append(patient_id)

            print("pump_nonempty = [" + ','.join(['"' + str(i) + '"' for i in pump_nonempty]) + "]")
            print("inj_nonempty = [" + ','.join(['"' + str(i) + '"' for i in inj_nonempty]) + "]")
            print("empty = [" + ','.join(['"' + str(i) + '"' for i in empty]) + "]")
            print("other = [" + ','.join(['"' + str(i) + '"' for i in other]) + "]")



    else: raise ValueError("Invalid input.")