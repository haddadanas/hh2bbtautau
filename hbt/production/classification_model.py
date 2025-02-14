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
    produces={
        "bin_dnn_*",
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_torch_dev.sh"),
    model_path=(
        "/afs/desy.de/user/h/haddadan/hh2bbtautau/ml_network/models/saved_models/"
        "ml_model_batch_norm__loose__datasets_3__20_epochs_change_feature_set_v1__limit_100000_logs/20250204_135247/"
        # "/afs/desy.de/user/h/haddadan/hh2bbtautau/ml_network/models/saved_models/"
        # "ml_model_batch_norm__default__datasets_3__20_epochs_change_feature_set_v1__limit_100000_logs/20250204_164743"
    ),
)
def ml_classify(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    from ml_network.src.dataset import DataContainer, remove_ak_fields
    from ml_network.src.preprocessing import prepare_input
    from ml_network.src.ml_utils import Fitting
    from ml_network.src.utils import get_loader

    # preprocess the data
    drop_fields = events.fields + ["jet0", "jet1", "jet2", "jet3", "delta_r_bjets"]
    if "normalization_weight" not in events.fields:
        drop_fields.append("normalization_weight")
    ml_events = self[preprocess](events, **kwargs)
    dataset_dict = {
        "name": self.dataset_inst.name,
        "cls_id": self.dataset_inst.id,
        "is_signal": self.dataset_inst.has_tag("signal"),
        "channel_id": ml_events.channel_id,
        "process_id": ml_events.process_id,
        "weights": ml_events.normalization_weight,
        "array": remove_ak_fields(ml_events, drop_fields),
    }
    del ml_events

    dataset_dict = {dataset_dict["name"]: DataContainer(**dataset_dict)}
    inp, fields = prepare_input(**self.ml_config, inputs=dataset_dict, validation_split=None)
    loader = get_loader(**inp["x"], batch_size=128, shuffle=False)
    (x_embed, x_num), y, _ = loader.dataset.get_data()  # type: ignore

    fitting = Fitting(self.model, device="cpu")
    y_pred = fitting.predict(x_embed, x_num, batch_size=512)

    ch_mask = events.channel_id <= 3

    # add the prediction to the events
    for i in range(y_pred.size(1)):
        col = np.full(len(events), -10, dtype=np.float32)
        col[ch_mask] = y_pred[:, i].detach().contiguous().numpy()
        events = set_ak_column(events, f"bin_dnn_{i}", col)

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
    model = CustomModel("model", self.ml_config["feature_names"])
    model.load_state_dict(torch.load(f"{self.model_path}/model.pt", weights_only=True, map_location="cpu"))
    self.model = model


@producer(
    uses={
        category_ids, hh_mass, default, ml_classify, channel_id_mask, stitched_normalization_weights,
    },
    produces={
        category_ids, hh_mass, default, ml_classify, stitched_normalization_weights,
    },
)
def ml_selection(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # preprocess the data
    events = self[stitched_normalization_weights](events, **kwargs)

    if "normalization_weight_inclusive" in events.fields:
        events["normalization_weight"] = events["normalization_weight_inclusive"]

    # preprocess the data
    events = self[ml_classify](events, **kwargs)

    events = self[default](events, **kwargs)

    events = self[hh_mass](events, **kwargs)

    events = self[category_ids](events, **kwargs)

    return events


@ml_selection.init
def ml_selection_init(self: Producer) -> None:
    if not self.config_inst.has_category("ml_selected"):
        self.config_inst.add_category(name="ml_selected", id=42, selection="cat_ml_selected", label="ML selected")
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
                "selection": [self.config_inst.get_category("ml_selected"), self.config_inst.get_category("signal")],
                "jets": [self.config_inst.get_category("1bjet")],
            },
            name_fn=name_fn,
            kwargs_fn=kwargs_fn,
        )


@producer(
    uses={
        category_ids, hh_mass, default, stitched_normalization_weights,
    },
    produces={
        category_ids, hh_mass, default, stitched_normalization_weights,
    },
)
def selection(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # preprocess the data
    events = self[stitched_normalization_weights](events, **kwargs)

    if "normalization_weight_inclusive" in events.fields:
        events["normalization_weight"] = events["normalization_weight_inclusive"]

    events = self[default](events, **kwargs)

    events = self[hh_mass](events, **kwargs)

    events = self[category_ids](events, **kwargs)

    return events


@selection.init
def selection_init(self: Producer) -> None:
    if not self.config_inst.has_category("signal__1bjet"):
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
                "jets": [self.config_inst.get_category("1bjet")],
            },
            name_fn=name_fn,
            kwargs_fn=kwargs_fn,
        )
