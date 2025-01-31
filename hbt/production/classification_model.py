# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.util import dev_sandbox, maybe_import
from columnflow.columnar_util import set_ak_column
from law.util import InsertableDict

from hbt.production.preprocessing.preprocess_funcs import preprocess, channel_id_mask
from hbt.production.default import default
from hbt.production.hh_mass import hh_mass

ak = maybe_import("awkward")
torch = maybe_import("torch")


@producer(
    uses={
        preprocess, channel_id_mask,
    },
    produces={
        "bin_dnn_*", "normalization_weight",
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_torch_dev.sh"),
    model_path=(
        "/afs/desy.de/user/h/haddadan/hh2bbtautau/ml_network/models/saved_models/"
        "ml_model_batch_norm__loose__datasets_3__20_epochs_cf_v2__limit_100000_logs/20250129_165159"
    ),
)
def ml_classify(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    from ml_network.src.dataset import DataContainer, remove_ak_fields
    from ml_network.src.preprocessing import prepare_input
    from ml_network.src.ml_utils import Fitting
    from ml_network.src.utils import get_loader

    # preprocess the data
    drop_fields = ["n_taus", "n_bjets"] + events.fields
    ml_events = self[preprocess](events, **kwargs)
    dataset_dict = {
        "name": self.dataset_inst.name,
        "cls_id": self.dataset_inst.id,
        "is_signal": self.dataset_inst.has_tag("signal"),
        "channel_id": ml_events.channel_id,
        "process_id": ml_events.process_id,
        "weights": ml_events.normalization_weight,
        "array": remove_ak_fields(ml_events, ["normalization_weight"] + drop_fields),
    }

    events = set_ak_column(events, "normalization_weight", ml_events.normalization_weight)
    del ml_events

    dataset_dict = {dataset_dict["name"]: DataContainer(**dataset_dict)}
    inp, fields = prepare_input(**self.ml_config, inputs=dataset_dict, validation_split=None)
    loader = get_loader(**inp["x"], batch_size=128, shuffle=False)
    (x_embed, x_num), y, _ = loader.dataset.get_data()  # type: ignore

    fitting = Fitting(self.model, device="cpu")
    y_pred = fitting.predict(x_embed, x_num, batch_size=512)

    events = self[channel_id_mask](events, **kwargs)

    # add the prediction to the events
    for i in range(y_pred.size(1)):
        col = ak.Array(y_pred[:, i].detach().contiguous().numpy())
        events = set_ak_column(events, f"bin_dnn_{i}", col)

    return events


@ml_classify.setup
def ml_classify_setup(self: Producer, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    if getattr(self, "dataset_inst", None) is None:
        return
    torch = maybe_import("torch")
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
        category_ids, hh_mass, default, ml_classify, channel_id_mask,
    },
    produces={
        category_ids, hh_mass, default, ml_classify,
    },
)
def ml_selection(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # preprocess the data
    events = self[channel_id_mask](events, **kwargs)

    events = self[ml_classify](events, **kwargs)

    events = self[default](events, **kwargs)

    events = self[hh_mass](events, **kwargs)

    events = self[category_ids](events, **kwargs)

    return events


@ml_selection.init
def ml_selection_init(self: Producer) -> None:
    if not self.config_inst.has_category("ml_selected"):
        self.config_inst.add_category(name="ml_selected", id=42, selection="cat_ml_selected", label="ML selected")


@producer(
    uses={
        category_ids, hh_mass, default, channel_id_mask,
    },
    produces={
        category_ids, hh_mass, default,
    },
)
def selection(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # preprocess the data
    events = self[channel_id_mask](events, **kwargs)

    events = self[default](events, **kwargs)

    events = self[hh_mass](events, **kwargs)

    events = self[category_ids](events, **kwargs)

    return events
