import os
from os.path import join, basename

import yaml
import torch
import random
import numpy as np

import shutil
from dotmap import DotMap


def process_config(config_path):
    r"""Loads a config file and setups the experiment
    Arguments:
    --
    config_path: string
        Path to the config file. Must be a YAML file
    """
    config_dict = from_yaml(config_path)
    config = DotMap(config_dict)

    # Create a checkpoint directory inside the experiment directory
    config.checkpoint_dir = join(config.experiment.exp_dir, "checkpoints/")
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Create a directory for saving predictions
    config.out_dir = join(config.experiment.exp_dir, "out/")
    os.makedirs(config.out_dir, exist_ok=True)

    # Copy the config file to the experiment directory for safe keeping
    shutil.copyfile(config_path, join(config.experiment.exp_dir, basename(config_path)))

    return config


def seed_everything(seed, use_cuda=True):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda: torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    return np.random.RandomState(seed)


def from_yaml(path):
    r"""Load content from YAML file"""

    with open(path) as fp:
        config = yaml.load(fp, Loader=yaml.Loader)

    return config


def collect_metrics(outputs, split):
    r"""At the end of an epoch, use this function to aggregate metrics.
    Arguments:
    --
    outputs: list[dict[string, any]]
    split: string e.g. train
        train | dev | test
    """
    size = len(outputs)
    keys = outputs[0].keys()

    metrics = {}
    for key in keys:
        metrics[f'{split}/{key}'] = 0

    for output in outputs:
        for key in keys:
            metrics[f'{split}/{key}'] += output[key] / size

    return metrics
