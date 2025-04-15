import torch
import torch.nn as nn

from agents.algorithm.agent import Agent
from agents.models.actor_critic import ActorCritic

from decouple import config
import sys

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from utils.sim_data import DataImporter

# CURRENTLY JUST A COPY OF PPO FILE (+ a few things); just here as a placeholder to set up custom agent registration
class Offline(Agent):
    def __init__(self, args, env_args, logger, load_model, actor_path, critic_path):
        super(Offline, self).__init__(args, env_args=env_args, logger=logger, type="Offline")
        self.device = args.device
        self.completed_interactions = 0

        # training params
        self.train_v_iters = args.n_vf_epochs
        self.train_pi_iters = args.n_pi_epochs
        self.batch_size = args.batch_size
        self.pi_lr = args.pi_lr
        self.vf_lr = args.vf_lr

        # load models and setup optimiser.
        ## initialise phi_0
        ## initialise pi_0
        ## initialise s ~ d_0(s)

        # readout
        print("Setting up offline Agent")
        print(f"Using {args.data_type} data.")

        # import custom data and setup buffer
        try:
            self.importer = DataImporter()
            self.importer.create_queue() #fixme determine minimum and maximum buffer size
            print("Succesfully instantiated data importer object.")
        except:
            print("No object save found at data/object_save/data_dictionary.pkl")

    def update_buffer(self):
        pass

    def update_gradient(self):
        pass

    def update(self):
        self.update_buffer()

        self.update_gradient()

        # data = dict(policy_grad=pi_grad, policy_loss=pi_loss, value_grad=vf_grad, value_loss=vf_loss, explained_var=explained_var, true_var=true_var)
        return {k: v.detach().cpu().flatten().numpy()[0] for k, v in data.items()}




