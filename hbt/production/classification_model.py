# coding: utf-8

"""
Wrappers for some default sets of producers.
"""
from columnflow.config_util import create_category_combinations
from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights, stitched_normalization_weights  # noqa: F401
from columnflow.util import dev_sandbox, maybe_import
from columnflow.columnar_util import set_ak_column
from law.util import InsertableDict

from hbt.production.preprocessing.preprocess_funcs import preprocess, channel_id_mask
from hbt.production.default import default
from hbt.production.hh_mass import hh_mass

ak = maybe_import("awkward")
np = maybe_import("numpy")
torch = maybe_import("torch")


@producer(
    uses={preprocess},
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_torch_dev.sh"),
    model_path=(
        "/afs/desy.de/user/h/haddadan/hh2bbtautau/ml_network/models/saved_models/ml_model_batch_norm__default__datasets_3__tau_pt_cut_None_logs/20250320_220413/"
        # "/afs/desy.de/user/h/haddadan/hh2bbtautau/ml_network/models/saved_models/ml_model_batch_norm__loose__datasets_3__tau_pt_cut_None_logs/20250321_011854/"
        # "/afs/desy.de/user/h/haddadan/hh2bbtautau/ml_network/models/saved_models/"
        # "ml_model_batch_norm__default__datasets_3__20_epochs_change_feature_set_v1__limit_100000_logs/20250204_164743"
    ),
    node_name="bin_dnn",
)
def ml_classify(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    from ml_network.src.torch_util import FlatTorchDataset, EvaluationDataLoader
    from ml_network.src.ml_utils import TorchFitting
    from ml_network.src.torch_transforms import RemoveEmptyValues

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # preprocess the data
    drop_fields = events.fields + ["delta_r_bjets"] + [f"tau_pt{pt}" for pt in self.config_inst.x.selection_cuts]
    if "normalization_weight" not in events.fields:
        drop_fields.append("normalization_weight")
    ml_events = self[preprocess](events, **kwargs)
    embed_f = self.ml_config["feature_names"]["embed_fields"]
    f_names = sum(self.ml_config["feature_names"].values(), [])
    # f_names.remove("channel_id")
    dataset_dict = {
        "target": int(self.dataset_inst.has_tag("signal")),
        "columns": f_names,
        "embedding_fields": embed_f,
        "device": device,
        "num_transform": RemoveEmptyValues(),
        "embed_transform": RemoveEmptyValues(embed_input=True),
    }
    dataset = FlatTorchDataset(ml_events, **dataset_dict)
    dataloader = EvaluationDataLoader(
        data_map={self.dataset_inst.name: dataset},
        batch_size=512,
        device=device,
    )
    fitting = TorchFitting(self.model, device=device, save_logs=False, tensorboard=False, training_mode=False)
    y_pred = fitting.predict(dataloader.data_loader)
    y_pred = y_pred[self.dataset_inst.name]

    ch_mask = events.channel_id <= 3

    # add the prediction to the events
    for i in range(y_pred.size(1)):
        col = np.full(len(events), -10, dtype=np.float32)
        col[ch_mask] = y_pred[:, i].detach().contiguous().cpu().numpy()
        events = set_ak_column(events, f"{self.node_name}_{i}", col)

    return events


@ml_classify.setup
def ml_classify_setup(self: Producer, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    if getattr(self, "dataset_inst", None) is None:
        return
    json = maybe_import("json")
    from ml_network.models.ml_model_batch_norm import CustomModel

    # set the number of threads to 1
    torch.set_num_threads(1)

    with open(f"{self.model_path}/config.json") as f:
        config = json.load(f)
    # load the model
    self.ml_config = config
    print(f"using model: {self.model_path}")
    model = CustomModel("model", **self.ml_config["feature_names"])
    model.load_state_dict(torch.load(f"{self.model_path}/model.pt", weights_only=True, map_location=device))
    self.model = model


@ml_classify.init
def ml_classify_init(self: Producer) -> None:
    self.produces.add(f"{self.node_name}_*")


@producer(
    uses={
        category_ids, hh_mass, default, channel_id_mask, normalization_weights,
    },
    produces={
        category_ids, hh_mass, default, normalization_weights,
    },
    tau_pt=20,
    dnn_threshold=0.5,
)
def ml_selection(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # preprocess the data
    events = self[normalization_weights](events, **kwargs)

    if "normalization_weight_inclusive" in events.fields:
        events["normalization_weight"] = events["normalization_weight_inclusive"]

    # preprocess the data
    events = self[self.classify](events, **kwargs)

    events = self[default](events, **kwargs)

    events = self[hh_mass](events, **kwargs)

    events = self[category_ids](events, **kwargs)

    return events


@ml_selection.init
def ml_selection_init(self: Producer) -> None:
    # model_path = self.config_inst.x.ml_models[self.tau_pt]
    threshold_int = int(self.dnn_threshold * 100)
    ml_classify_pt = ml_classify
    self.uses.add(ml_classify_pt)
    self.produces.add(ml_classify_pt)
    self.classify = ml_classify_pt

    if not self.config_inst.has_category(f"ml_selected_{threshold_int}"):
        self.config_inst.add_category(name=f"ml_selected_{threshold_int}", id=600 + threshold_int,
                                      selection=f"cat_ml_selected_{threshold_int}", label=f"ML selected (DNN > {self.dnn_threshold})")

        def name_fn(categories):
            return "__".join(cat.name for cat in categories.values() if cat)

        def kwargs_fn(categories):
            # build auxiliary information
            aux = {}
            # return the desired kwargs
            return {
                # just increment the category id
                # NOTE: for this to be deterministic, the order of the categories must no change!
                "id": "+",
                # join all tags
                "tags": set.union(*[cat.tags for cat in categories.values() if cat]),
                # auxiliary information
                "aux": aux,
                # label
                "label": ", ".join([
                    cat.label or cat.name
                    for cat in categories.values()
                ]) or None,
            }
        create_category_combinations(
            config=self.config_inst,
            categories={
                # "channel": [self.config_inst.get_category("tautau")],
                "selection": [self.config_inst.get_category(f"ml_selected_{threshold_int}"), self.config_inst.get_category("signal")],
                # "jets": [self.config_inst.get_category("1bjet")],
                # "tau_pt": [
                #     self.config_inst.get_category("tau_pt_35"),
                #     self.config_inst.get_category("tau_pt_36"),
                #     self.config_inst.get_category("tau_pt_37"),
                #     self.config_inst.get_category("tau_pt_38"),
                #     self.config_inst.get_category("tau_pt_40"),
                #     self.config_inst.get_category("tau_pt_45"),
                #     self.config_inst.get_category("tau_pt_50"),
                #     self.config_inst.get_category("tau_pt_80"),
                # ],
            },
            name_fn=name_fn,
            kwargs_fn=kwargs_fn,
        )


ml_selection_producers = []
for i in [35, 36, 37, 38, 40, 45, 50, 80]:
    ml_selection_producers.append(
        ml_selection.derive(
            f"ml_selection_{i}",
            cls_dict={
                "tau_pt": i,
            },
        )
    )


@producer(
    uses={
        category_ids, hh_mass, default, normalization_weights,
    },
    produces={
        category_ids, hh_mass, default, normalization_weights,
    },
)
def selection(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # preprocess the data
    events = self[normalization_weights](events, **kwargs)

    if "normalization_weight_inclusive" in events.fields:
        events["normalization_weight"] = events["normalization_weight_inclusive"]

    events = self[default](events, **kwargs)

    events = self[hh_mass](events, **kwargs)

    events = self[category_ids](events, **kwargs)

    return events


@selection.init
def selection_init(self: Producer) -> None:
    if not self.config_inst.has_category("signal"):
        def name_fn(categories):
            return "__".join(cat.name for cat in categories.values() if cat)

        def kwargs_fn(categories):
            # build auxiliary information
            aux = {}
            # return the desired kwargs
            return {
                # just increment the category id
                # NOTE: for this to be deterministic, the order of the categories must no change!
                "id": "+",
                # join all tags
                "tags": set.union(*[cat.tags for cat in categories.values() if cat]),
                # auxiliary information
                "aux": aux,
                # label
                "label": ", ".join([
                    cat.label or cat.name
                    for cat in categories.values()
                ]) or None,
            }
        create_category_combinations(
            config=self.config_inst,
            categories={
                "selection": [self.config_inst.get_category("signal")],
                # "jets": [self.config_inst.get_category("1bjet")],
            },
            name_fn=name_fn,
            kwargs_fn=kwargs_fn,
        )
