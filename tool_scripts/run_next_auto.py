import sys
import torch
import random
import warnings
import json
import numpy as np
import os
from decouple import config

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf
from utils.logger import setup_folders, Logger
from metrics.statistics_alt import get_summary_stats, read_file

warnings.simplefilter('ignore', Warning)


TRIAL_DIGIT_NUM = 3
CSV_HEADERS = ["Step","Glucose", "CGM", "t", "CHO", "Insulin", "MA"]
DEFAULT_FILE_NAME = "TRIAL_"
LOGS_DEFAULT = True

def int_show(num, force_digits=3):
    str_num = str(num)
    return '0'*max(0, force_digits - len(str_num)) + str_num

def get_next_file_num(folder_dest, starts_with=""): #assumes all csv files in folder end with 
    files = os.listdir(folder_dest)
    start_len = len(starts_with)
    numbers = [0] + [int(file.split('.')[0][-3:]) for file in files if (file[:start_len] == starts_with)]
    print("Files:",files)
    print("Numbers:", numbers)
    return int_show(max(numbers) + 1, TRIAL_DIGIT_NUM)


def write_summary_md(mem_obj, agent_name, file_dest):
    pass

def write_csv(mem_obj, folder_dest, logs=LOGS_DEFAULT):
    lines = [','.join(CSV_HEADERS)]
    data = mem_obj.get_simu_data()

    sim_len = len(data[0])
    for n in range(sim_len):
        lines.append(','.join([str(n)] + [str(float(i[n][0])) for i in data]))

    file_name = DEFAULT_FILE_NAME + get_next_file_num(folder_dest) + ".csv"

    with open(folder_dest+file_name,'w') as f:
        f.write('\n'.join(lines))
    
    if logs:
        print("CSV File", folder_dest + file_name,"written.")



def set_agent_parameters(cfg):
    agent = None

    setup_folders(cfg)
    logger = Logger(cfg)

    if cfg.agent.agent == 'ppo':
        from agents.algorithm.ppo import PPO
        agent = PPO(args=cfg.agent, env_args=cfg.env, logger=logger, load_model=False, actor_path='', critic_path='')

    elif cfg.agent.agent == 'a2c':
        from agents.algorithm.a2c import A2C
        agent = A2C(args=cfg.agent, env_args=cfg.env, logger=logger, load_model=False, actor_path='', critic_path='')

    elif cfg.agent.agent == 'sac':
        from agents.algorithm.sac import SAC
        agent = SAC(args=cfg.agent, env_args=cfg.env, logger=logger, load_model=False, actor_path='', critic_path='')

    elif cfg.agent.agent == 'g2p2c':
        from agents.algorithm.g2p2c import G2P2C
        agent = G2P2C(args=cfg.agent, env_args=cfg.env, logger=logger, load_model=False, actor_path='', critic_path='')

    elif cfg.agent.agent == 'custom':
        from agents.algorithm.custom import Custom
        agent = Custom(args=cfg.agent, env_args=cfg.env, logger=logger, load_model=False, actor_path='', critic_path='')

    else:
        print('Please select an agent for the experiment. Hint: a2c, sac, ppo, g2p2c, custom')
    return agent

with open("../offline_testing/default_args.json",'r') as fp:
    args_dict = json.load(fp)
    fp.close()

args = OmegaConf.create(args_dict)

RESULT_TITLE_BASE = "CustomTesting"
EXPERIMENTS_DIR = MAIN_PATH + '/results/'

new_exp_name = RESULT_TITLE_BASE + '_' + get_next_file_num(EXPERIMENTS_DIR, RESULT_TITLE_BASE)
print("New: ",new_exp_name)

args.env.experiment_folder = EXPERIMENTS_DIR + new_exp_name
args.agent.experiment_folder = EXPERIMENTS_DIR + new_exp_name
args.agent.experiment_dir = EXPERIMENTS_DIR + new_exp_name
args.experiment.name = new_exp_name
args.experiment.folder = new_exp_name
args.experiment.experiment_dir = EXPERIMENTS_DIR + new_exp_name

SHOW_DICT_ATTRS = True

@hydra.main()
def main(cfg : DictConfig) -> None:
    if cfg.agent.debug:  # if debug override params to run experiment in a smaller scale.
        for key in cfg.agent.debug_params:
            if key in cfg.agent:
                cfg.agent[key] = cfg.agent.debug_params[key]

    if SHOW_DICT_ATTRS:
        print("===========================\nDICT ATTRIBUTES")
        for k in cfg:
            print('>',k,':')
            for sub_k in cfg[k]:
                print('\t',k,'.',sub_k,': ', cfg[k][sub_k],sep='')
        print("===========================")

    if cfg.experiment.verbose:
        print('\nExperiment Starting...')
        print("\nOptions =================>")
        print(vars(cfg))
        print('\nDevice which the program run on:', cfg.experiment.device)

    agent = set_agent_parameters(cfg)  # load agent

    torch.manual_seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    if cfg.mlflow.track:
        print("Running with mlflow")
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.experiment.name)
        experiment = mlflow.get_experiment_by_name(cfg.experiment.name)
        run_name = cfg.experiment.run_name if cfg.experiment.run_name is not None else cfg.agent.agent

        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name):
            mlflow.log_params(cfg)
            agent.run()
    else:
        print("Running without mlflow")
        agent.run()


def save_to_csv(cfg : DictConfig):

    text_lines = []

    chosen_exp = cfg.experiment.name
    exp_directory = MAIN_PATH + "/results/" + chosen_exp + '/'
    text_lines.append("```")
    text_lines.append("======================================== ARGS for '" + chosen_exp + "'")
    for k in args:
        text_lines.append('>' + str(k) +':')
        for sub_k in args[k]:
            text_lines.append('\t' + str(k) + '.' + str(sub_k) + ': ' + str(args[k][sub_k]))
    text_lines.append("========================================")
    text_lines.append("```")

    exp_args_filepath = exp_directory + "args.md"
    with open(exp_args_filepath, 'w') as fp:
        fp.write('\n'.join(text_lines))

    text_lines = []
    ## Display Test Data
    text_lines.append("### Test Data Trials ###")
    text_lines.append("```")
    exp_test_data = read_file(experiment_name=chosen_exp, algorithm=args.agent.agent, n_trials=args.agent.n_testing_workers, base_num=args.agent.testing_agent_id_offset)
    for nums in exp_test_data:
        text_lines.append("worker_episode_" + str(nums))
        for c,episode_data in enumerate(exp_test_data[nums]):
            text_lines.append("\tEpisode " + str(c+1))
            for k in episode_data:
                text_lines.append("\t\t" + str(k) + ':' + ','.join([str(round(float(i),2)) for i in episode_data[k]]))
            text_lines.append("")
    text_lines.append("========================================")
    text_lines.append("```")

    ## Display Validation Data
    text_lines.append("### Validation Data Trials ###")
    text_lines.append("```")
    exp_vald_data = read_file(experiment_name=chosen_exp, algorithm=args.agent.agent, n_trials=args.agent.debug_params.n_val_trials, base_num=args.agent.validation_agent_id_offset)
    for nums in exp_vald_data:
        text_lines.append("worker_episode_" + str(nums))
        for c,episode_data in enumerate(exp_vald_data[nums]):
            text_lines.append("\tEpisode " + str(c+1))
            for k in episode_data:
                text_lines.append("\t\t" + str(k) + ':' + ','.join([str(round(float(i),2)) for i in episode_data[k]]))
            text_lines.append("")
    text_lines.append("========================================")
    text_lines.append("```")

    exp_preview_filepath = exp_directory + "results.md"
    with open(exp_preview_filepath, 'w') as fp:
        fp.write('\n'.join(text_lines))

    text_lines = []
    text_lines.append("## Summary of Experiment " + args.experiment.name)
    if args.debug: text_lines.append("Experiment ran on debug mode.")
    text_lines.append("Experiment ran on device " + args.experiment.device + " and agent ran on device " + args.agent.device + '.')

    # exp_summary_filepath = exp_directory + "summary.md"
    # with open(exp_summary_filepath, 'w') as fp:
    #     fp.write('\n\n'.join(text_lines))

if __name__ == '__main__':
    main(args)

    save_to_csv(args)

