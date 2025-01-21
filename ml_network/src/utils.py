from __future__ import annotations

from typing import Type
import sys
import os
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ml_network.src.dataset import DataContainer

__all__ = [
    "load_setup", "load_config", "get_loader", "add_metrics_to_log", "log_to_message", "ProgressBar", "get_device"
]


# Device configuration
def get_device() -> str:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


#############################
# Reader utils for Yaml files
#############################


def path_constructor(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode) -> str:
    node_dict = loader.construct_sequence(node)
    return "".join(node_dict)


def dataset_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> DataContainer:
    return DataContainer(**loader.construct_mapping(node))  # type: ignore


def combine_constructor(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode) -> list:
    var_lists = loader.construct_sequence(node)
    features = []
    field = var_lists[0]
    for var in var_lists[1:]:
        features.append(f"{field}.{var}")
    return features


def ml_obj_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> dict:
    node_dict = loader.construct_mapping(node, deep=True)
    obj_type = node_dict.pop('id')
    return {"class_name": obj_type, "config": node_dict}


def get_setup_loader():
    """Add constructors to PyYAML loader."""
    loader = yaml.SafeLoader
    loader.add_constructor("!join", path_constructor)
    loader.add_constructor("!dataset", dataset_constructor)
    return loader


def get_config_loader():
    loader = yaml.SafeLoader
    loader.add_constructor("!combine", combine_constructor)
    loader.add_constructor("!ml_obj", ml_obj_constructor)
    return loader


#############################
# Config and Setup Loaders
#############################


def load_setup():
    """Load setup.yaml file."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    with open(f'{current_dir}/../setup.yaml') as f:
        config = yaml.load(f, Loader=get_setup_loader())
    return config


def load_config(config_file: str):
    """Load config.yaml file."""
    with open(config_file) as f:
        config = yaml.load(f, Loader=get_config_loader())
    return config


#############################
# Utils for fitting
#############################

# Data utils

def get_loader(
    inp_embed: list,
    inp_num: np.ndarray,
    target: np.ndarray = None,
    weight: np.ndarray = None,
    batch_size: int = 32,
    shuffle: bool = True,
    device: str = get_device(),
) -> DataLoader:
    """ Create a Tensor with predefined settings, e.g. shuffle, batch, prefetch. """
    class InputData(Dataset):
        def __init__(self, device, inp_embed, inp_num, target=None, weight=None):
            to_tensor_f32 = lambda x: (
                torch.from_numpy(np.astype(x, np.float32)) if isinstance(x, np.ndarray) else x
            )
            to_tensor_i32 = lambda x: (
                torch.from_numpy(np.astype(x, np.int32)) if isinstance(x, np.ndarray) else x
            )

            self.num_data = to_tensor_f32(inp_num).to(device)
            self.embed_data = (
                [to_tensor_i32(d).to(device) for d in inp_embed] if isinstance(inp_embed, list)
                else to_tensor_i32(inp_embed).to(device)
            )
            size = self.num_data.size(0)
            self.target = torch.Tensor(size) if target is None else to_tensor_f32(target).to(device)
            self.weight = torch.ones(size) if weight is None else to_tensor_f32(weight).to(device)

        def get_data(self):
            return (self.embed_data, self.num_data), self.target, self.weight

        def get_target(self):
            return self.target

        def __len__(self):
            return self.num_data.size(0)

        def __getitem__(self, idx):
            return (([d[idx] for d in self.embed_data], self.num_data[idx]), self.target[idx], self.weight[idx])

    tensor = InputData(device=device, inp_embed=inp_embed, inp_num=inp_num, target=target, weight=weight)
    dataloader = DataLoader(tensor, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# Logging

def add_metrics_to_log(log, metrics, y_pred, y_true, prefix=''):
    for metric in metrics:
        q = metric(y_pred, y_true)
        if q is None:
            continue
        metric_name = metric.name if hasattr(metric, 'name') else metric.__name__
        log[prefix + metric_name] = q
    return log


def log_to_message(log, precision=4):
    fmt = "{0}: {1:." + str(precision) + "f}"
    return "    ".join(fmt.format(k, v) for k, v in log.items() if isinstance(v, (int, float)))


class ProgressBar(object):

    def __init__(self, n, length=40):
        # Protect against division by zero
        self.n = max(1, n)
        self.nf = float(n)
        self.length = length
        # Precalculate the i values that should trigger a write operation
        self.ticks = set([round(i / 100.0 * n) for i in range(101)])
        self.ticks.add(n - 1)
        self.bar(0)

    def bar(self, i, message=""):
        """Assumes i ranges through [0, n-1]"""
        if i in self.ticks:
            b = int(np.ceil(((i + 1) / self.nf) * self.length))
            sys.stdout.write("\r[{0}{1}] {2}%\t{3}".format(
                "=" * b, " " * (self.length - b), int(100 * ((i + 1) / self.nf)), message
            ))
            sys.stdout.flush()

    def close(self, message=""):
        # Move the bar to 100% before closing
        self.bar(self.n - 1)
        sys.stdout.write("{0}\n\n".format(message))
        sys.stdout.flush()


def build_model_name(setup: dict, model: Type) -> str:
    """
    Build the model name based on the setup and the model
    """
    model_name = f"{model.__module__.split('.')[-1]}__"
    model_name += f"{setup['used_selector']}__datasets_{len(setup['datasets'])}__{setup['model_suffix']}"
    if "limit_dataset" in setup:
        model_name += f"__limit_{setup['limit_dataset']}"
    return model_name


def make_dir(path: str) -> str:
    """
    Create the directory if it does not exist
    """
    import time
    current_time = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(path, current_time)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
