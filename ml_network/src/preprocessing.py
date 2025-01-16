from __future__ import annotations
from typing import Callable
from collections import defaultdict

import awkward as ak
import numpy as np
import numpy.lib.recfunctions as nprec

from ml_network.src.dataset import DataContainer

EMPTY_FLOAT = -99999.0


def build_field_names(dtype) -> list[str]:
    fields = []
    for (field, typ) in dtype.descr:
        if isinstance(typ, list):
            fields.extend([f"{field}_{subfield[0]}" for subfield in typ])
        else:
            fields.append(field.replace("_", ".", 1))
    return fields


def merge_weight_and_class_weights(data: dict, class_weight: dict[DataContainer, float]) -> np.ndarray:
    """ Merge the class weights with the event weights. """
    indecies = data["dataset_id"]
    event_weights = data["weight"]
    weight = np.ones_like(event_weights)
    for d, w in class_weight.items():
        weight[indecies == d.id] = w * event_weights[indecies == d.id]

    return weight


def prepare_input(
    config: dict,
    inputs: dict[str, DataContainer],
    target_mapping: Callable = lambda target, inp: int(inp.is_signal),
    validation_split: float | None = 0.2,
) -> (dict[str, tuple], list[str]):
    """ Prepare the input for the ML model. """
    weight_sum: dict[str, float] = {}
    training: defaultdict[str, list] = defaultdict(list)
    valid: defaultdict[str, list] = defaultdict(list)
    fields: list | None = None

    # get needed config inforamtion
    target_nodes = config["target"]

    for name, inp in inputs.items():
        # calculate the sum of weights for each dataset
        weight_sum[inp] = ak.sum(inp.weights)

        weights = ak.to_numpy(inp.weights)
        channel_id = ak.to_numpy(inp.channel_id)
        dataset_id = np.full(len(inp), inp.id, dtype=np.int32)
        inp_features = []
        val_features = []
        arr_fields = []

        # split into training and validation set
        if validation_split:
            split = int(len(inp) * validation_split)
            choice = np.random.choice(range(len(inp)), size=(split,), replace=False)
            ind = np.zeros(len(inp), dtype=bool)
            ind[choice] = True

        for field in inp.array.fields:
            column = inp.get_column(field)
            arr_fields.append([f"{field}_{subfield}" for subfield in column.fields])
            arr = ak.to_numpy(column, allow_missing=False)
            arr = nprec.structured_to_unstructured(arr)

            # set EMPTY_FLOAT to -10
            if np.any(arr == EMPTY_FLOAT):
                arr[arr == EMPTY_FLOAT] = -10

            if validation_split:
                val_features.append(arr[ind])
                arr = arr[~ind]
            inp_features.append(arr)

        # check for infinite values in weights
        if np.any(~np.isfinite(weights)):
            raise Exception(f"Infinite values found in weights from {inp}")

        # create target array
        target = np.zeros((len(inp), len(target_nodes)), dtype=np.int32)
        target[:, :] = target_mapping(target, inp)

        if validation_split:
            for i, val in enumerate(val_features):
                valid[f"events_{i}"].append(val)
            valid["target"].append(target[ind])
            target = target[~ind]
            valid["weight"].append(weights[ind])
            weights = weights[~ind]
            valid["channel_id"].append(channel_id[ind])
            channel_id = channel_id[~ind]
            valid["dataset_id"].append(dataset_id[ind])
            dataset_id = dataset_id[~ind]

            print(f"*{inp}* is split into {len(target)} training and {split}"
                " validation events")
        for i, inp_feature in enumerate(inp_features):
            training[f"events_{i}"].append(inp_feature)
        training["target"].append(target)
        training["weight"].append(weights)
        training["channel_id"].append(channel_id)
        training["dataset_id"].append(dataset_id)

        if not fields:
            fields = arr_fields

    # Merge over datasets
    mean_weight: np.ndarray = np.mean(list(weight_sum.values()))
    class_weight = {d: mean_weight / w for d, w in weight_sum.items()}

    # concatenate all events and targets
    for i in range(len(fields)):
        training[f"events_{i}"] = np.concatenate(training[f"events_{i}"])
    training["embed_1"] = np.concatenate(training["channel_id"])
    training["target"] = np.concatenate(training["target"])
    training["weight"] = np.concatenate(training["weight"])
    training["dataset_id"] = np.concatenate(training["dataset_id"])
    training["weight"] = merge_weight_and_class_weights(training, class_weight)

    # Shuffle the training data
    shuffle = np.random.permutation(len(training["target"]))
    # create ML tensors
    train_tensor = {
        "inp_embed": [val[shuffle] for key, val in training.items() if key.startswith("embed")],
        "inp_num": np.concatenate([val for key, val in training.items() if key.startswith("events_")], axis=1)[shuffle],
        "target": training["target"][shuffle],
        "weight": training["weight"][shuffle],
    }

    # create an output for the fit function of ML
    result = {"x": train_tensor}  # weights are included in the training tensor

    if validation_split:
        for i in range(len(fields)):
            valid[f"events_{i}"] = np.concatenate(valid[f"events_{i}"])
        valid["embed_1"] = np.concatenate(valid["channel_id"])
        valid["target"] = np.concatenate(valid["target"])
        valid["weight"] = np.concatenate(valid["weight"])
        valid["dataset_id"] = np.concatenate(valid["dataset_id"])
        valid["weight"] = merge_weight_and_class_weights(valid, class_weight)

        valid_tensor = {
            "inp_embed": [val for key, val in valid.items() if key.startswith("embed")],
            "inp_num": np.concatenate([val for key, val in valid.items() if key.startswith("events_")], axis=1),
            "target": valid["target"],
            "weight": valid["weight"],
        }
        result["validation_data"] = valid_tensor

    return result, fields
