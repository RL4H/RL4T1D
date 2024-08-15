import sys
import torch
import random
import pandas as pd
import warnings
import numpy as np
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf
from utils.logger import setup_folders, Logger
from clinical.clinical_worker import run_simulation

warnings.simplefilter('ignore', Warning)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:

    if cfg.agent.debug:  # if debug override params to run experiment in a smaller scale.
        for key in cfg.agent.debug_params:
            if key in cfg.agent:
                cfg.agent[key] = cfg.agent.debug_params[key]

    if cfg.experiment.verbose:
        print('\nExperiment Starting...')
        print("\nOptions =================>")
        print(vars(cfg))
        print('\nDevice which the program run on:', cfg.experiment.device)

    setup_folders(cfg)

    random.seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    columns = [['PatientID', 'trial', 'survival', 'normo', 'hypo', 'sev_hypo', 'hyper', 'lgbi', 'hgbi', 'ri', 'sev_hyper']]
    res = run_simulation(cfg, id=cfg.env.patient_id, rollout_steps=cfg.agent.max_test_epi_len, n_trials=cfg.agent.n_trials)
    df = pd.DataFrame(res, columns=columns)
    df.to_csv(cfg.experiment.experiment_dir + '/testing/'+'subject'+str(cfg.env.patient_id)+'_method_'+cfg.agent.carb_estimation_method+'_summary'+'.csv')
    print(df)


if __name__ == '__main__':
    main()

# e.g., command python run_clinical_treatment.py experiment.name=test33 agent=clinical_treatment hydra/job_logging=disabled
