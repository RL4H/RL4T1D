import numpy as np
import pandas as pd
from datetime import datetime
import sys
from decouple import config
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig, OmegaConf

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

from experiments.glucose_prediction.transformer_decoder import MultiBranchAutoregressiveDecoder


"""
Transformer setup:

Given an array of the last 24 hours (24 * (60/5) = 288 data points) of glucose, meal, and insulin values. Predicts the next glucose value, and provided the next meal and insulin values.
Mapping: 3 x 288 -> 1x1

evaluated by RMSE to target values, cross-validated with dataset


"""

@hydra.main(version_base=None, config_path=MAIN_PATH + "/experiments/glucose_prediction", config_name="prediction_config.yaml")
def main(cfg: DictConfig) -> None:

    print(cfg.text)


if __name__ == '__main__':
    main()
