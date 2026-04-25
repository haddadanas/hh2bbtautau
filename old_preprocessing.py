from __future__ import annotations
from typing import Any
from collections import defaultdict

import awkward as ak
import numpy as np
import numpy.lib.recfunctions as nprec

from dataset import DataContainer

EMPTY_FLOAT = -9999.0

# get the used DeepTau wp
tau_iso_wp = {
    "e": {"vvvloose": 1, "vvloose": 2, "vloose": 3, "loose": 4, "medium": 5, "tight": 6, "vtight": 7, "vvtight": 8},  # noqa
    "jet": {"vvvloose": 1, "vvloose": 2, "vloose": 3, "loose": 4, "medium": 5, "tight": 6, "vtight": 7, "vvtight": 8},  # noqa
    "mu": {"vloose": 1, "loose": 2, "medium": 3, "tight": 4},
}

channel_id_map = {
    "mutau": 1,
    "etau": 2,
    "tautau": 3,
}


def build_field_names(dtype) -> list[str]:
    fields = []
    for (field, typ) in dtype.descr:
        if isinstance(typ, list):
            fields.extend([f"{field}_{subfield[0]}" for subfield in typ])
        else:
            fields.append(field.replace("_", ".", 1))
    return fields


def merge_weight_and_class_weights(data: dict, class_weight: dict[int, float]) -> np.ndarray:
    """ Merge the class weights with the event weights. """
    indecies = data["dataset_id"]
    event_weights = data["weight"]
    weight = np.ones_like(event_weights)
    for d, w in class_weight.items():
        weight[indecies == d.id] = w * event_weights[indecies == d.id]

    return weight


def restructure_channel_id(data: DataContainer, columns: list[str], *args) -> np.ndarray:
    """ Map the channel_id to the corresponding integer. """
    result = np.zeros(len(data), dtype=[("channel_id", np.int32)])
    result["channel_id"] = data.channel_id - 1

    assert columns == ["channel_id"], f"Fields do not match: {columns} != {['channel_id']}"
    return result


def restructure_lepton_array(data: DataContainer, columns: list[str], lepton: str = "1", *args) -> np.recarray:
    """
        Restructure the array to remove the jagged structure and to create a first and second lepton column
        from the available Muons, ELectrons and Taus.
    """
    # get Array and channel_id
    array = data.array
    channel_id = data.channel_id
    lepton_number = 0 if lepton == "1" else 1

    # Create the result array
    fields = {
        "pt": np.float32,
        "eta": np.float32,
        "dz": np.float32,
        "dxy": np.float32,
        "tauVSjet": np.float32,
        "tauVSe": np.float32,
        "tauVSmu": np.float32,
        "is_iso": np.int32,
    }
    if lepton == "1":
        fields["iso_score"] = np.float32

    result = np.zeros(len(array), dtype=[
        (f"l{lepton}_{var}", f_type)
        for var, f_type in fields.items()
    ])
    result[:] = -10

    # helper functions
    tau_matcher = lambda tag, wp: array.Tau[f"idDeepTau2018v2p5VS{tag}"] >= tau_iso_wp[tag][wp]
    channel_matcher = lambda ch: channel_id == channel_id_map[ch]

    # Create the Tau is Iso column
    tau_is_iso = 1 * ((tau_matcher("jet", "loose")) & (
        (
            (channel_id == 3) & (tau_matcher("e", "vvloose") | tau_matcher("mu", "vloose"))
        ) | (
            (channel_id != 3) & (tau_matcher("e", "vloose") | tau_matcher("mu", "tight"))
        )
    ))

    # Take care of the different channels seperately
    # etau and mutau channel
    for ch in ["etau", "mutau"]:
        ch_mask = channel_matcher(ch)
        iso_tag = "mvaIso" if ch == "etau" else "pfRelIso04_all"
        is_iso_tag = "mvaIso_WP80" if ch == "etau" else "tightId"
        if lepton_number == 0:
            lep = array.Electron if ch == "etau" else array.Muon
            lep = lep[ch_mask][:, lepton_number]
            iso_array = lep[iso_tag] / ak.max(lep[iso_tag])
            result[f"l{lepton}_iso_score"][ch_mask] = iso_array
            result[f"l{lepton}_is_iso"][ch_mask] = 1 * lep[is_iso_tag]
        else:
            lep = array.Tau[ch_mask][:, 0]
            result[f"l{lepton}_is_iso"][ch_mask] = tau_is_iso[ch_mask][:, 0]
            result[f"l{lepton}_tauVSjet"][ch_mask] = lep["idDeepTau2018v2p5VSjet"]
            result[f"l{lepton}_tauVSe"][ch_mask] = lep["idDeepTau2018v2p5VSe"]
            result[f"l{lepton}_tauVSmu"][ch_mask] = lep["idDeepTau2018v2p5VSmu"]

        for f in ["pt", "eta", "dz", "dxy"]:
            result[f"l{lepton}_{f}"][ch_mask] = lep[f]

    # tautau channel
    is_tautau = channel_matcher("tautau")
    lep = array.Tau[is_tautau][:, lepton_number]
    for f in ["pt", "eta", "dz", "dxy"]:
        result[f"l{lepton}_{f}"][is_tautau] = lep[f]
    for f in ["jet", "e", "mu"]:
        result[f"l{lepton}_tauVS{f}"][is_tautau] = lep[f"idDeepTau2018v2p5VS{f}"]
    result[f"l{lepton}_is_iso"][is_tautau] = tau_is_iso[is_tautau][:, lepton_number]

    fields = build_field_names(result.dtype)
    assert fields == columns, f"Fields do not match: {fields} != {columns}"

    return result


def restructure_jet_array(data: DataContainer, columns: list[str], jet: str = "1", *args) -> np.recarray:
    """
        Restructure the array to remove the jagged structure and to create a first and second jet column
        from the available Jets.
    """
    # get Array and channel_id
    array = data.array
    jet_number = 0 if jet == "1" else 1

    # Create the result array
    fields = {
        "pt": np.float32,
        "eta": np.float32,
        "mass": np.float32,
        "hhbtag": np.float32,
        "btagPNetB": np.float32,
    }

    result = np.zeros(len(array), dtype=[
        (f"j{jet}_{var}", f_type)
        for var, f_type in fields.items()
    ])
    result[:] = -10

    # Jet information
    jet_array = array.Jet[ak.argsort(array.Jet.hhbtag, ascending=False)]
    for f in ['btagPNetB', 'eta', 'hhbtag', 'mass', 'pt']:
        result[f"j{jet}_{f}"] = jet_array[f][:, jet_number]

    fields = build_field_names(result.dtype)
    assert fields == columns, f"Fields do not match: {fields} != {columns}"

    return result


def prepare_input(
    config: dict,
    inputs: dict[str, DataContainer],
    target_mapping: callable = lambda target, inp: int(inp.is_signal),
    validation_split: float | None = 0.2,
) -> (dict[str, Any], list[str]):
    """ Prepare the input for the ML model. """

    weight_sum: dict[str, float] = {}
    training: defaultdict[str, list] = defaultdict(list)
    valid: defaultdict[str, list] = defaultdict(list)
    fields: list = None

    # get needed config inforamtion
    input_features = config["features"]
    target_features = config["inputs"]
    target_nodes = config["target"]

    for name, inp in inputs.items():
        # calculate the sum of weights for each dataset
        weight_sum[inp] = ak.sum(inp.weights)

        events = inp.get_sub_dataset(input_features)
        weights = ak.to_numpy(inp.weights)
        channel_id = ak.to_numpy(inp.channel_id)
        dataset_id = np.full(len(events), inp.id, dtype=np.int32)
        inp_features = []
        val_features = []
        arr_fields = []

        # split into training and validation set
        if validation_split:
            split = int(len(events) * validation_split)
            choice = np.random.choice(range(len(events)), size=(split,), replace=False)
            ind = np.zeros(len(events), dtype=bool)
            ind[choice] = True

        for columns in input_features:
            from IPython import embed; embed(header="preprocessing.py	l:209")
            arr_fields.append(columns)
            arr = events.get_features([columns])
            if not isinstance(arr, np.ndarray):
                arr = arr.to_numpy()
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
        target = np.zeros((len(events), len(target_nodes)), dtype=np.int32)
        target[:, :] = target_mapping(target, inp)
        if len(target_nodes) == 1:
            target = target[:, 0]

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

        for i, inp in enumerate(inp_features):
            training[f"events_{i}"].append(inp)
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
    training["target"] = np.concatenate(training["target"])
    training["weight"] = np.concatenate(training["weight"])
    training["channel_id"] = np.concatenate(training["channel_id"])
    training["dataset_id"] = np.concatenate(training["dataset_id"])
    training["weight"] = merge_weight_and_class_weights(training, class_weight)

    # create ML tensors
    train_tensor = training

    # create an output for the fit function of ML
    result = {"x": train_tensor}  # weights are included in the training tensor

    if validation_split:
        for i in range(len(fields)):
            valid[f"events_{i}"] = np.concatenate(valid[f"events_{i}"])
        valid["target"] = np.concatenate(valid["target"])
        valid["weight"] = np.concatenate(valid["weight"])
        valid["channel_id"] = np.concatenate(valid["channel_id"])
        valid["dataset_id"] = np.concatenate(valid["dataset_id"])
        valid["weight"] = merge_weight_and_class_weights(valid, class_weight)

        valid_tensor = valid
        result["validation_data"] = valid_tensor

    return result, fields
