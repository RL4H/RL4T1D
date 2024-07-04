from utils.logger import setup_folders, copy_folder
import yaml
import json
import argparse
from decouple import config
MAIN_PATH = config('MAIN_PATH')


def load_yaml(file_name, args):
    stream = open(file_name, 'r')
    config_dict = yaml.safe_load(stream)
    for key, value in config_dict.items():
        setattr(args, key, value)
    return args  # return updated args


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):

        # experiment info directories, device, seed, ...
        self.parser.add_argument('--experiment_folder', type=str, default='testing', help='folder path for results and log')
        self.parser.add_argument('--device', type=str, default='cpu', help='give device name')
        self.parser.add_argument('--verbose', type=bool, default=True, help='')
        self.parser.add_argument('--seed', type=int, default=0, help='')
        self.parser.add_argument('--debug', type=bool, default=False, help='if debug ON => 1')

        # environment
        self.parser.add_argument('--patient_id', type=int, default=0, help='id: adolescent(0-9), child(10-19), adult(20-29)')
        self.parser.add_argument('--env_config', type=str, default=MAIN_PATH+'/environment/env_config.yaml', help='')

        self.parser.add_argument('--clinical_config', type=str, default=MAIN_PATH+'/clinical/bb_config.yaml', help='')

        # RL agent
        self.parser.add_argument('--agent', type=str, default='ppo', help='agent used for the experiment.')
        agent_name = self.parser.parse_args().agent
        self.parser.add_argument('--rl_config', type=str, default=MAIN_PATH+'/agents/configs/'+agent_name+'_config.yaml', help='')
        self.parser.add_argument('--debug_config', type=str, default=MAIN_PATH+'/experiments/debug_config.yaml', help='')

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()  # load command line args
        self.opt = load_yaml(self.opt.env_config, self.opt)  # load env configs
        self.opt = load_yaml(self.opt.rl_config, self.opt)  # load RL configs
        if self.opt.debug:
            self.opt = load_yaml(self.opt.debug_config, self.opt)  # load RL configs
        Options.validate_args(self.opt)  # validate the arguments parsed
        self.opt = setup_folders(self.opt)  # set up the folders for exp results
        copy_folder(src=MAIN_PATH + '/agents/'+ self.opt.agent, dst=MAIN_PATH + '/results/' + self.opt.experiment_folder + '/code')  # copy running agent code to outputs

        with open(self.opt.experiment_dir + '/args.json', 'w') as fp:  # save the experiments args.
            json.dump(vars(self.opt), fp, indent=4)
            fp.close()

        return self.opt

    def parse_clinical(self):
        self._initial()
        self.opt = self.parser.parse_args()  # load command line args
        self.opt = load_yaml(self.opt.env_config, self.opt)  # load env configs
        self.opt = load_yaml(self.opt.clinical_config, self.opt)  # load bb/clinical configs
        Options.validate_args(self.opt)  # validate the arguments parsed
        self.opt = setup_folders(self.opt)  # set up the folders for exp results
        copy_folder(src=MAIN_PATH + '/agents/std_bb', dst=MAIN_PATH + '/results/' + self.opt.experiment_folder + '/code')  # copy running agent code to outputs
        return self.opt

    @staticmethod
    def validate_args(args):  # todo: implement validation criteria.
        valid = True
        if args.feature_history != args.calibration:
            valid = False
        if not valid:
            print("Check the input arguments!")
            exit()
