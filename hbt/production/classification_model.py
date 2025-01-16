# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.util import dev_sandbox, maybe_import
from columnflow.columnar_util import set_ak_column

from hbt.production.preprocessing.preprocess_funcs import preprocess

ak = maybe_import("awkward")
torch = maybe_import("torch")


@producer(
    uses={
        preprocess,
    },
    produces={
        "bin_dnn",
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_torch_dev.sh"),
    model_path=(
        "/afs/desy.de/user/h/haddadan/hh2bbtautau/ml_network/models/saved_models/"
        "ml_loose__datasets_2__30_epochs_cf__limit_10000.pt"
    ),
)
def Classify(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    from ml_network.src.dataset import DataContainer, remove_ak_fields
    from ml_network.src.preprocessing import prepare_input
    from ml_network.src.utils import get_loader, get_device
    from ml_network.src.ml_utils import Fitting
    from ml_network.models.ml_model import CustomModel

    # preprocess the data
    drop_fields = ["n_taus", "n_bjets"] + events.fields
    events = self[preprocess](events, **kwargs)
    dataset_dict = {
        "name": self.dataset_inst.name,
        "cls_id": self.dataset_inst.id,
        "is_signal": self.dataset_inst.has_tag("signal"),
        "channel_id": events.channel_id,
        "process_id": events.process_id,
        "weights": events.normalization_weight,
        "array": remove_ak_fields(events, ['normalization_weight'] + drop_fields),
    }
    target_config = {
        "target": ["signal"],
    }

    dataset_dict = {dataset_dict["name"]: DataContainer(**dataset_dict)}
    inp, fields = prepare_input(target_config, dataset_dict, validation_split=None)
    loader = get_loader(**inp["x"], batch_size=128, shuffle=False)
    (x_embed, x_num), y, _ = loader.dataset.get_data()

    # load the model
    model = CustomModel("model", fields)
    model.load_state_dict(torch.load(self.model_path, weights_only=True, map_location=get_device()))
    fitting = Fitting(model)
    y_pred = fitting.predict(x_embed, x_num, batch_size=512)

    # add the prediction to the events
    events = set_ak_column(events, "bin_dnn", y_pred.cpu().flatten())

    return events
