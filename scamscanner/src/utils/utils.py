import os
from os.path import join, basename

import yaml
import torch
import random
import numpy as np

import shutil
from dotmap import DotMap

from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)


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
    config.experiment.checkpoint_dir = join(config.experiment.exp_dir, "checkpoints/")
    os.makedirs(config.experiment.checkpoint_dir, exist_ok=True)

    # Create a directory for saving predictions
    config.experiment.out_dir = join(config.experiment.exp_dir, "out/")
    os.makedirs(config.experiment.out_dir, exist_ok=True)

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


def collect_binary_classification_metrics(outputs, split):
    r"""At the end of an epoch, use this function to aggregate metrics
    and compute binary classification metrics.
    Arguments:
    --
    outputs: list[dict[string, any]]
    split: string e.g. train
        train | dev | test
    """
    labels, probs = [], []

    for output in outputs:
        labels.append(output['labels'])
        probs.append(output['probs'])

    labels = np.concatenate(labels)
    probs = np.concatenate(probs)
    preds = np.round(probs)

    metrics = {}
    metrics[f'{split}/acc'] = accuracy_score(labels, preds)
    metrics[f'{split}/precision'] = precision_score(labels, preds)
    metrics[f'{split}/recall'] = recall_score(labels, preds)
    metrics[f'{split}/f1'] = f1_score(labels, preds)

    # Handle any remaining keys
    keys = outputs[0].keys()
    size = len(outputs)

    for key in keys:
        if key not in ['labels', 'probs']:
            metrics[f'{split}/{key}'] = 0

    for output in outputs:
        for key in keys:
            if key not in ['labels', 'probs']:
                metrics[f'{split}/{key}'] += output[key] / size

    return metrics
