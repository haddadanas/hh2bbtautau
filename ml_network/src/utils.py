from __future__ import annotations

import yaml

from dataset import Dataset

__all__ = ["load_setup", "load_config"]


#############################
# Reader utils for Yaml files
#############################


def path_constructor(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode) -> str:
    node_dict = loader.construct_sequence(node)
    return "".join(node_dict)


def dataset_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Dataset:
    return Dataset(**loader.construct_mapping(node))


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
    with open('../setup.yaml') as f:
        config = yaml.load(f, Loader=get_setup_loader())
    return config


def load_config(config_file: str):
    """Load config.yaml file."""
    with open(config_file) as f:
        config = yaml.load(f, Loader=get_config_loader())
    return config
