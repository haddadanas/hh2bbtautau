from __future__ import annotations

__all__ = [
    "load_setup", "add_metrics_to_log", "ProgressBar", "get_device", "get_logger",
    "timeit", "make_dir", "build_model_name", "setup_parser", "log_parser"
]

import sys
import os
import shutil
import yaml
import argparse
import logging
from collections import defaultdict
from time import perf_counter_ns, strftime
from typing import Any, Callable, Iterable, Tuple, Type

import numpy as np

from ml_network.src.cf_utils import DotDict


#############################
# General Utils
#############################

def get_logger(name: str) -> logging.Logger:
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='####\t %(asctime)s - %(name)s - %(levelname)s - %(funcName)s :\n\t > %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ],
    )
    return logging.getLogger(name)


def timeit(logger=get_logger(__name__)) -> Callable:
    def decorator(method: Callable) -> Callable:
        def timed(*args, **kw) -> Any:
            ts = perf_counter_ns()
            result = method(*args, **kw)
            te = perf_counter_ns()
            logger.info(f"{method.__name__} took {(te - ts) / 1e9:.6f} seconds")
            return result
        return timed
    return decorator


def cleanup_on_exception(func: Callable) -> Callable:
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            cls_inst = args[0]
            path = cls_inst.path
            logger = cls_inst.trace_func
            logger(
                f"Exception in {func.__name__} with path: {path}\n"
                f"Exception: {e}"
            )
            # remove the directory
            remove: bool = cls_inst._cleanup or (input("Do you want to remove the directory? [y/n] ").lower() == "y")
            if remove and os.path.exists(path):
                logger(f"Removing directory: {path}")
                shutil.rmtree(path)
            raise e
    return wrapper


# Device configuration
def get_device() -> str:
    import torch
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-pt", "-pt", type=int, default=None)
    parser.add_argument("--imgcat", "-i", action="store_true", default=False)
    parser.add_argument("--epochs", "-e", type=int, default=20)
    parser.add_argument("--batch-size", "-b", type=int, default=256)
    parser.add_argument("--patience", "-p", type=int, default=7)
    return parser


def log_parser(logger: logging.Logger, args: argparse.Namespace) -> argparse.Namespace:
    for key, value in vars(args).items():
        if isinstance(value, bool):
            logger.info(f"{key} Enabled" if value else f"{key} Disabled")
            continue
        logger.info(f"Using {key}={value}")
    return args


#############################
# Reader utils for Yaml files
#############################


def _path_constructor(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode) -> str:
    node_dict = loader.construct_sequence(node)
    return "".join(node_dict)


def _get_setup_loader() -> type[yaml.SafeLoader]:
    """Add constructors to PyYAML loader."""
    loader = yaml.SafeLoader
    loader.add_constructor("!join", _path_constructor)
    return loader


def load_setup() -> DotDict:
    """Load setup.yaml file."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    with open(f'{current_dir}/../setup.yaml') as f:
        setup = yaml.load(f, Loader=_get_setup_loader())
    return DotDict.wrap(setup)


#############################
# Logging
#############################


def add_metrics_to_log(
        log: dict,
        metrics: list[Callable],
        y_pred: Iterable,
        y_true: Iterable,
        prefix: str = ''
) -> dict:
    for metric in metrics:
        q = metric(y_pred, y_true)
        if q is None:
            continue
        metric_name = metric.name if hasattr(metric, 'name') else metric.__name__  # type: ignore
        log[prefix + metric_name] = q
    return log


class ProgressBar(object):

    def __init__(self, n: int, length: int = 40, verbose: bool = True):
        # Protect against division by zero
        self.n = max(1, n)
        self.nf = float(n)
        self.length = length
        # Precalculate the i values that should trigger a write operation
        self.ticks = set([round(i / 100.0 * n) for i in range(101)])
        self.ticks.add(n - 1)
        if verbose:
            self.bar(0)

    def _log_to_message(self, log: dict, precision: int = 4) -> str:
        # get log message if it exists
        log_msg = log.get("log_msg", "")
        log_msg = f"\n***\t{log_msg}\t***" if log_msg else ""

        # deal with numerical values
        fmt = "{0}: {1:." + str(precision) + "f}"
        val_logs = {
            k: v for k, v in log.items()
            if isinstance(v, (int, float)) and k.startswith("val")
        }
        val_msg = "\n" if val_logs else ""
        val_msg += "\t".join(fmt.format(k, v) for k, v in val_logs.items() if isinstance(v, (int, float)))
        log_out = (
            "\t".join(fmt.format(k, v) for k, v in log.items() if k not in val_logs and isinstance(v, (int, float))) +
            val_msg
        )
        return f"{log_out}{log_msg}"

    def bar(self, i, log: dict = {}):
        message = "\t".join("{0}: {1:.4f}".format(k, v) for k, v in log.items() if isinstance(v, (int, float)))
        # Assumes i ranges through [0, n-1]
        if i in self.ticks:
            b = int(np.ceil(((i + 1) / self.nf) * self.length))
            sys.stdout.write("\r[{0}{1}] {2}%\t{3}".format(
                "=" * b, " " * (self.length - b), int(100 * ((i + 1) / self.nf)), message
            ))
            sys.stdout.flush()

    def close(self, message: str = "", log: dict = {}):
        message = self._log_to_message(log, 4) + "\n" + message
        # Move the bar to 100% before closing
        self.bar(self.n - 1)
        sys.stdout.write("{0}\n\n".format(message))
        sys.stdout.flush()


#############################
# Model Utils
#############################


def build_model_name(setup: dict, model: Type, **kwargs) -> str:
    """
    Build the model name based on the setup and the model
    """
    post_fix = "__".join([f"{key}_{val}" for key, val in kwargs.items()])
    model_name = f"{model.__module__.split('.')[-1]}__"
    model_name += f"{setup['used_selector']}__datasets_{len(setup['datasets'])}__{post_fix}"
    return model_name


def make_dir(path: str, add_time: bool = False) -> str:
    """
    Create the directory if it does not exist
    """
    if add_time:
        current_time = strftime("%Y%m%d_%H%M%S")
        path = os.path.join(path, current_time)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


#############################
# Legacy Utils
#############################

def get_loader(
    inp_embed: list,
    inp_num: np.ndarray,
    target: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    device: str = get_device(),
):
    import torch
    from torch.utils.data import Dataset, DataLoader
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
            self.target = torch.Tensor(size) if target is None else to_tensor_i32(target).to(device)

            self.weight = None if weight is None else to_tensor_f32(weight).to(device)
            # Dynamically set the __getitem__ method
            if weight is None:
                self._getitem = self._getitem_no_weight
            else:
                self._getitem = self._getitem_with_weight

        def get_data(self):
            return (self.embed_data, self.num_data), self.target, self.weight

        def get_target(self):
            return self.target

        def _getitem_no_weight(self, idx) -> tuple:
            return (([d[idx] for d in self.embed_data], self.num_data[idx]), self.target[idx])

        def _getitem_with_weight(self, idx):
            return (
                ([d[idx] for d in self.embed_data],
                self.num_data[idx]),
                self.target[idx],
                self.weight[idx],  # type: ignore
            )

        def __len__(self):
            return self.num_data.size(0)

        def __getitem__(self, idx) -> tuple:
            return self._getitem(idx)

    tensor = InputData(device=device, inp_embed=inp_embed, inp_num=inp_num, target=target, weight=weight)
    dataloader = DataLoader(tensor, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def merge_means_and_stds(
    lengths: dict[str, int],
    means: dict[str, dict[str, float]],
    stds: dict[str, dict[str, float]]
) -> Tuple[dict[str, float], dict[str, float]]:
    _sum = defaultdict(float)
    _sum2 = defaultdict(float)
    for key, d in means.items():
        for k, v in d.items():
            _sum[k] += v * lengths[key]
        for k, v in stds[key].items():
            _sum2[k] += lengths[key] * (v ** 2) + lengths[key] * (d[k] ** 2)

    # Normalize the means and stds by the total number of events
    total_events = sum(lengths.values())
    merged_means = {}
    merged_stds = {}
    for k, v in _sum.items():
        merged_means[k] = v / total_events
    for k, v in _sum2.items():
        merged_stds[k] = np.sqrt(v / total_events - merged_means[k] ** 2)
    return merged_means, merged_stds


def merge_event_stats(
        dataset_dict: dict[str, Any],
) -> Tuple[dict[str, float], dict[str, float]]:
    """
    Merge the event statistics from the dataset dictionary.
    """
    dataset_length: dict[str, int] = {key: sum(d.meta_data.col_counts) for key, d in dataset_dict.items()}
    means: dict[str, dict[str, float]] = {key: d.column_mean for key, d in dataset_dict.items()}
    stds: dict[str, dict[str, float]] = {key: d.column_std for key, d in dataset_dict.items()}

    # check all dicts have the same keys
    means_iterator = iter(means.values())
    if not all([means[key].keys() == next(means_iterator).keys() for key in means]):
        raise ValueError("All means dictionaries must have the same keys")

    return merge_means_and_stds(
        lengths=dataset_length,
        means=means,
        stds=stds
    )


def get_padding_values(
        means: dict[str, float],
        stds: dict[str, float],
        n_sigma: int = 5,
) -> dict[str, float]:
    """
    Get the padding values for the input data.
    The padding values are calculated as the mean - n_sigma * std.
    """
    padding_values = {}
    for key in means.keys():
        padding_values[str(key)] = means[key] - n_sigma * stds[key]
    return padding_values
