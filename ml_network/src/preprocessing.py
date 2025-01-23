from __future__ import annotations
from typing import Callable
from collections import defaultdict

import awkward as ak
import numpy as np

from ml_network.src.dataset import DataContainer

EMPTY_FLOAT = -99999.0


def merge_weight_and_class_weights(data: dict, class_weight: dict[DataContainer, float]) -> np.ndarray:
    """ Merge the class weights with the event weights. """
    indecies = data["dataset_id"]
    event_weights = data["weight"]
    weight = np.ones_like(event_weights)
    for d, w in class_weight.items():
        weight[indecies == d.id] = w * event_weights[indecies == d.id]

    return weight


def prepare_input(
    target_nodes: list[str],
    inputs: dict[str, DataContainer],
    target_mapping: Callable = lambda target, inp: int(inp.is_signal),
    validation_split: float | None = 0.2,
    fields: dict[str, list] | None = None,
    embedding_fields: list[str] = [],
    *args,
    **kwargs,
) -> (dict[str, tuple], list[str]):
    """ Prepare the input for the ML model. """
    weight_sum: dict[str, float] = {}
    training: defaultdict[str, list] = defaultdict(list)
    valid: defaultdict[str, list] = defaultdict(list)

    for name, inp in inputs.items():
        # calculate the sum of weights for each dataset
        use_weights = False
        if inp.weights is not None:
            weight_sum[inp] = ak.sum(inp.weights)
            weights = ak.to_numpy(inp.weights)
            use_weights = True

        channel_id = ak.to_numpy(inp.channel_id)
        dataset_id = np.full(len(inp), inp.id, dtype=np.int32)
        inp_features = []
        embed_features = []
        val_features = []
        val_embed_features = []
        num_fields = []
        embed_fields = []

        # split into training and validation set
        if validation_split:
            split = int(len(inp) * validation_split)
            choice = np.random.choice(range(len(inp)), size=(split,), replace=False)
            ind = np.zeros(len(inp), dtype=bool)
            ind[choice] = True

        for field_name, column in inp.get_features_dict().items():
            arr = ak.to_numpy(column, allow_missing=False)
            # arr = nprec.structured_to_unstructured(arr)

            # set EMPTY_FLOAT to -10
            if np.any(arr == EMPTY_FLOAT):
                arr[arr == EMPTY_FLOAT] = -10

            if validation_split:
                if field_name in embedding_fields:
                    val_embed_features.append(arr[ind])
                else:
                    val_features.append(arr[ind])
                arr = arr[np.logical_not(ind)]
            if field_name in embedding_fields:
                embed_fields.append(field_name)
                embed_features.append(arr)
            else:
                num_fields.append(field_name)
                inp_features.append(arr)

        # check for infinite values in weights
        if use_weights and np.any(~np.isfinite(weights)):
            raise Exception(f"Infinite values found in weights from {inp}")

        # create target array
        target = np.zeros((len(inp), len(target_nodes)), dtype=np.int32)
        target[:, :] = target_mapping(target, inp)

        # treat the channel_id in a special way as it can be a categorical variable
        if validation_split:
            valid["channel_id"].append(channel_id[ind])
            channel_id = channel_id[np.logical_not(ind)]

        if "channel_id" in embedding_fields:
            embed_fields.append("channel_id")
            embed_features.append(channel_id)
            if validation_split:
                val_embed_features.append(valid["channel_id"][-1])

        if validation_split:
            for i, val in enumerate(val_features):
                valid[f"events_{i}"].append(val)
            for i, val in enumerate(val_embed_features):
                valid[f"embed_{i}"].append(val)
            valid["target"].append(target[ind])
            target = target[np.logical_not(ind)]
            valid["dataset_id"].append(dataset_id[ind])
            dataset_id = dataset_id[np.logical_not(ind)]
            if use_weights:
                valid["weight"].append(weights[ind])
                weights = weights[np.logical_not(ind)]

            print(f"*{inp}* is split into {len(target)} training and {split}"
                " validation events")

        for i, inp_feature in enumerate(inp_features):
            training[f"events_{i}"].append(inp_feature)
        for i, embed_feature in enumerate(embed_features):
            training[f"embed_{i}"].append(embed_feature)
        training["target"].append(target)
        training["channel_id"].append(channel_id)
        training["dataset_id"].append(dataset_id)
        if use_weights:
            training["weight"].append(weights)

        if not fields:
            fields = {"num_fields": num_fields, "embed_fields": embed_fields}
        else:
            assert fields["num_fields"] == num_fields, "Numerical Fields are not the same"
            assert fields["embed_fields"] == embed_fields, "Embedding Fields are not the same"

    # Merge over datasets
    if use_weights:
        mean_weight: np.ndarray = np.mean(list(weight_sum.values()))  # type: ignore
        class_weight = {d: mean_weight / w for d, w in weight_sum.items()}

    # concatenate all events and targets
    for i in range(len(num_fields)):  # type: ignore
        training[f"events_{i}"] = np.concatenate(training[f"events_{i}"])
    for i in range(len(embed_fields)):  # type: ignore
        training[f"embed_{i}"] = np.concatenate(training[f"embed_{i}"])
    training["target"] = np.concatenate(training["target"])
    training["dataset_id"] = np.concatenate(training["dataset_id"])

    # Shuffle the training data
    shuffle = np.random.permutation(len(training["target"]))
    # create ML tensors
    train_tensor = {
        "inp_embed": [val[shuffle] for key, val in training.items() if key.startswith("embed")],
        "inp_num": np.stack([val for key, val in training.items() if key.startswith("events_")], axis=1)[shuffle],
        "target": training["target"][shuffle],
    }
    if use_weights:
        training["weight"] = np.concatenate(training["weight"])
        training["weight"] = merge_weight_and_class_weights(training, class_weight)
        train_tensor["weight"] = training["weight"][shuffle]

    # create an output for the fit function of ML
    result = {"x": train_tensor}  # weights are included in the training tensor

    if validation_split:
        for i in range(len(num_fields)):  # type: ignore
            valid[f"events_{i}"] = np.concatenate(valid[f"events_{i}"])
        for i in range(len(embed_fields)):  # type: ignore
            valid[f"embed_{i}"] = np.concatenate(valid[f"embed_{i}"])
        valid["target"] = np.concatenate(valid["target"])
        valid["dataset_id"] = np.concatenate(valid["dataset_id"])

        valid_tensor = {
            "inp_embed": [val for key, val in valid.items() if key.startswith("embed")],
            "inp_num": np.stack([val for key, val in valid.items() if key.startswith("events_")], axis=1),
            "target": valid["target"],
        }
        if use_weights:
            valid["weight"] = np.concatenate(valid["weight"])
            valid["weight"] = merge_weight_and_class_weights(valid, class_weight)
            valid_tensor["weight"] = valid["weight"]

        result["validation_data"] = valid_tensor

    return result, fields
