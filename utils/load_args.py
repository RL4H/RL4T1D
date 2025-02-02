import warnings
from hydra import compose, initialize
from omegaconf import OmegaConf
from utils.logger import setup_folders

warnings.simplefilter('ignore', Warning)


def load_arguments(overrides=None):
    with initialize(version_base=None, config_path="../configs", job_name="test_app"):
        cfg = compose(config_name="config", overrides=overrides)
    # print(OmegaConf.to_yaml(cfg))

    if cfg.agent.debug:  # if debug override params to run experiment in a smaller scale.
        for key in cfg.agent.debug_params:
            if key in cfg.agent:
                cfg.agent[key] = cfg.agent.debug_params[key]

    setup_folders(cfg)

    return cfg
