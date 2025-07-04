# coding: utf-8

"""
Wrappers for some default sets of producers.
"""
from functools import partial

import gc
from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.util import dev_sandbox, maybe_import
from columnflow.columnar_util import set_ak_column, remove_ak_column
from law.util import InsertableDict

from hbt.production.preprocessing.preprocess_funcs import preprocess
from hbt.production.default import default

ak = maybe_import("awkward")
np = maybe_import("numpy")
torch = maybe_import("torch")
pickle = maybe_import("pickle")


@producer(
    uses={preprocess},
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_torch_dev.sh"),
    model_path=(
        "/afs/desy.de/user/h/haddadan/hh2bbtautau/ml_network/models/nn_models//token_model__loose__datasets_3__seed_1234_logs/20250623_162530/"
        # "/afs/desy.de/user/h/haddadan/hh2bbtautau/ml_network/models/saved_models/ml_model_batch_norm__loose__datasets_3__tau_pt_cut_None_logs/20250321_011854/"
        # "/afs/desy.de/user/h/haddadan/hh2bbtautau/ml_network/models/saved_models/"
        # "ml_model_batch_norm__default__datasets_3__20_epochs_change_feature_set_v1__limit_100000_logs/20250204_164743"
    ),
    node_name="bin_dnn",
)
def ml_classify(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # get the data ready for the evaluation
    ml_events = self[preprocess](events, **kwargs)
    feature_bases = set(r.fields[0] for r in self.needed_features)
    for f in ml_events.fields:
        if f in feature_bases:
            continue
        ml_events = remove_ak_column(ml_events, f)
    dataset = self.data_cls(ml_events)

    # predict the data
    y_pred = self.model(*dataset.input_data)
    y_pred_lora = self.lora_model(*dataset.input_data)

    # add the prediction to the events
    for i in range(y_pred.size(1)):
        events = set_ak_column(events, f"{self.node_name}_{i}", y_pred[:, i].contiguous().cpu().numpy())
        events = set_ak_column(events, f"lora_{self.node_name}_{i}", y_pred_lora[:, i].contiguous().cpu().numpy())

    return events


@ml_classify.setup
def ml_classify_setup(self: Producer, task, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    from cfg.token_model import ANet
    from torch_fit.torch_src.transforms import RemoveEmptyValues
    from torch_fit.torch_src.datasets import TensorParquetDataset

    # set the model device
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set the number of threads to 1
    torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)
    # turn off gradient computation
    torch.autograd.set_grad_enabled(False)

    # load the model configuration
    with open(f"{self.model_path}/setup_config.pkl", "rb") as f:
        ml_config = pickle.load(f)
    model_cfg = ml_config["model_cfg_run"]
    model_cfg["means"] = torch.tensor(model_cfg["means"], device=self.device)
    model_cfg["stds"] = torch.tensor(model_cfg["stds"], device=self.device)
    # load the model to the predict function
    print(f"using model: {self.model_path}")
    model = ANet("model", **model_cfg).to(self.device).eval()
    model.load_state_dict(torch.load(f"{self.model_path}/best_model.pt", map_location=self.device))
    model.eval()
    lora_model = ANet("model", **model_cfg).to(self.device)
    lora_model.init_lora(**ml_config["lora_cfg"]["lora_config"])
    lora_model.load_state_dict(torch.load(f"{self.model_path}/best_lora_model.pt", map_location=self.device))
    lora_model.enable_disable_lora(True)
    lora_model.eval()
    self.model = model
    self.lora_model = lora_model
    # f_names.remove("channel_id")

    # create the dataset
    self.needed_features = model.categorical_features + model.continuous_features
    dataset_dict = {
        "target": int(self.dataset_inst.has_tag("signal")),
        "device": self.device,
        "pad_flags": True,
        "continuous_features": ml_config["handler_cfg"]["continuous_features"],
        "categorical_features": ml_config["handler_cfg"]["categorical_features"],
        "input_data_transform": RemoveEmptyValues(padding_values=ml_config["padding_values"]),
    }
    self.data_cls = partial(TensorParquetDataset, **dataset_dict)

    # cross-check the features
    static_input = (
        torch.ones((1, len(model.categorical_features)), device=self.device).long(),
        torch.zeros((1, len(model.continuous_features)), device=self.device).float()
    )
    assert torch.allclose(
        model(*static_input), torch.tensor(ml_config["best_static_input"]).to(self.device), atol=1e-5
    ), "Model static input does not match the expected values. Model loading might have failed."
    assert torch.allclose(
        lora_model(*static_input), torch.tensor(ml_config["best_static_input_lora"]).to(self.device), atol=1e-5
    ), "LoRA model static input does not match the expected values. Model loading might have failed."


@ml_classify.teardown
def ml_classify_teardown(self: Producer, **kwargs) -> None:
    # remove the model from the device
    torch.cuda.empty_cache()
    if hasattr(self, "model"):
        del self.model
    if hasattr(self, "lora_model"):
        del self.lora_model
    # remove the model from the config
    if hasattr(self, "ml_config"):
        del self.ml_config
    gc.collect()


@ml_classify.init
def ml_classify_init(self: Producer) -> None:
    self.produces.add(f"{self.node_name}_*")
    self.produces.add(f"lora_{self.node_name}_*")


@producer(
    uses={
        default, normalization_weights, category_ids,
    },
    produces={
        default, normalization_weights, category_ids,
    },
    model_name=None,
)
def ml_producer(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # preprocess the data
    events = self[normalization_weights](events, **kwargs)

    if "normalization_weight_inclusive" in events.fields:
        events["normalization_weight"] = events["normalization_weight_inclusive"]

    # preprocess the data
    events = self[self.classify](events, **kwargs)

    events = self[default](events, **kwargs)

    events = self[category_ids](events, **kwargs)

    return events


@ml_producer.pre_init
def ml_producer_pre_init(self: Producer) -> None:
    threshold_ints = [int(x * 100) for x in self.config_inst.x.dnn_thresholds_wps]
    if not self.config_inst.has_category(f"ml_selected_{threshold_ints[0]}"):
        for threshold_int in threshold_ints:
            self.config_inst.add_category(
                name=f"ml_selected_{threshold_int}",
                id=600 + threshold_int,
                selection=f"cat_ml_selected_{threshold_int}", label=f"ML selected (DNN > {threshold_int / 100:.2f})",
            )
            self.config_inst.add_category(
                name=f"ml_rejected_{threshold_int}",
                id=700 + threshold_int,
                selection=f"cat_ml_rejected_{threshold_int}", label=f"ML rejected (DNN < {threshold_int / 100:.2f})",
            )


@ml_producer.init
def ml_producer_init(self: Producer) -> None:
    if self.model_name is None:
        self.classify = ml_classify
        return
    model_path = self.config_inst.x.ml_dnn.get(self.model_name, None)
    if model_path is None:
        raise ValueError(f"Model path for '{self.model_name}' not found in config.")
    self.classify = ml_classify.derive(
        f"ml_classify_{self.model_name}",
        cls_dict={
            "model_path": model_path,
        },
    )
    self.uses.add(self.classify)
    self.produces.add(self.classify)


ml_producers = []
for name in (
    ["FocalLoss_1", "FocalLoss_g3", "BCE_1", "BCE_2", "BCE_3"] +
    [f"rank_{i}" for i in [1, 3, 5, 10, 20, 50, 100]] +
    [f"rank_{i}_2" for i in [1, 3, 5, 10, 20, 50, 100]] +
    [f"th_0p{i}" for i in [0, 1, 2, 25, 4, 5, 6, 7]] +
    [f"r10_s{i}" for i in [42, 1337, 8675309, 9001, 123456789]]
):
    ml_producer_instance = ml_producer.derive(
        name.lower(),
        cls_dict={
            "model_name": name,
        },
    )
    ml_producers.append(ml_producer_instance)
