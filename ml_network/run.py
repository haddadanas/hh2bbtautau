from __future__ import annotations
import sys

sys.path.append("/afs/desy.de/user/h/haddadan/hh2bbtautau")

from ml_network.src.torch_transforms import RemoveEmptyValues, GetSelectedEvents
from ml_network.src.utils import load_setup, get_device, build_model_name, setup_parser, log_parser, get_logger, timeit
from ml_network.src.ml_utils import TorchFitting
from ml_network.models.ml_model_batch_norm import CONFIG, CustomModel as Model
from ml_network.src.cf_utils import DotDict
from ml_network.src.torch_util import CompositeDataLoader, EvaluationDataLoader, FlatTorchDataset, NestedDictMapAndCollate
from ml_network.src.torch_callbacks import signal_purity, signal_acceptance, selection_efficiency, auc_score, accuracy, get_ROC_Plotter

# Set some constants and global variables
EMPTY_FLOAT = -10.0
EMPTY_INT = -10
LOGGER = get_logger("ML Network")
SETUP = load_setup()
DEVICE = get_device()
EMBED_FIELDS = SETUP["embed_fields"]
NUM_FIELDS = SETUP["num_fields"]
COLUMN_NAMES = {"num_fields": NUM_FIELDS, "embed_fields": EMBED_FIELDS}
CONFIG["feature_names"] = COLUMN_NAMES


def setup_model(setup: dict, device: str = DEVICE, **kwargs):
    # Get the model
    model_name = build_model_name(setup, Model, **kwargs)
    model = Model(model_name, save_path=setup["model_save_path"])
    model.to(device)
    model.compile(backend="aot_eager")
    return model


def get_datasets(setup: dict, selection_field: str | None = None):
    # Needed Columns
    needed_columns = set(EMBED_FIELDS) | set(NUM_FIELDS)
    if selection_field is not None:
        needed_columns.add(selection_field)

    val_map = DotDict()

    # Get the datasets
    datasets = setup["datasets"]
    data_map = DotDict()
    weight_dict = {}
    for name, dataset in datasets.items():
        train_val_split = dataset["train_val_split"]
        splitter = (
            GetSelectedEvents(train_val_split=train_val_split, selection_field=selection_field)
            if selection_field or train_val_split
            else None
        )
        data_map[name] = FlatTorchDataset(
            dataset["path"],
            columns=needed_columns,
            embedding_fields=EMBED_FIELDS,
            padd_value_float=EMPTY_FLOAT,
            padd_value_int=EMPTY_INT,
            device=DEVICE,
            target=int(dataset["is_signal"]),
            global_transform=splitter,
            num_transform=RemoveEmptyValues(),
            embed_transform=RemoveEmptyValues(embed_input=True),
        )
        weight_dict[name] = dataset["weight"]
        LOGGER.info(f"Loaded dataset (Training) {name} with {len(data_map[name])} entries")
        if val_map is not None:
            splitter.set_validation_mode()
            val_map[name] = FlatTorchDataset(
                dataset["path"],
                columns=needed_columns,
                embedding_fields=EMBED_FIELDS,
                padd_value_float=EMPTY_FLOAT,
                padd_value_int=EMPTY_INT,
                device=DEVICE,
                target=int(dataset["is_signal"]),
                global_transform=splitter,
                num_transform=RemoveEmptyValues(),
                embed_transform=RemoveEmptyValues(embed_input=True),
            )
            LOGGER.info(f"Loaded dataset (Validation) {name} with {len(val_map[name])} entries")
    if not val_map:
        val_map = None
    return {"data_map": data_map, "weight_dict": weight_dict}, val_map


@timeit(LOGGER)
def main(fit_inst: TorchFitting, training_loader, validation_loader=None, n_epochs=100):
    logs = fit_inst.fit(
        training_data=training_loader,
        validation_data=validation_loader,
        epochs=n_epochs,
        metrics=[signal_purity, signal_acceptance, selection_efficiency, auc_score, accuracy],
        # plots=[get_ROC_Plotter()],
        **CONFIG,
    )
    # Save the model and logs
    fit_inst.save_best_model()
    return logs


if __name__ == "__main__":
    LOGGER.info(f"Using {DEVICE} device")

    # Set up the parser
    parser = setup_parser()
    argparser = parser.parse_args()
    log_parser(LOGGER, argparser)
    if argparser.imgcat:
        import matplotlib
        matplotlib.use("module://imgcat")
    tau_selection = f"tau_pt{argparser.tau_pt}" if argparser.tau_pt is not None else None

    dataloader_dict, validation_dict = get_datasets(SETUP, selection_field=tau_selection)
    composite_loader = CompositeDataLoader(
        **dataloader_dict,
        map_and_collate_cls=NestedDictMapAndCollate,
        batch_size=argparser.batch_size,
        shuffle=True,
        device=DEVICE,
    )
    validation_loader = EvaluationDataLoader(
        data_map=validation_dict,
        batch_size=argparser.batch_size,
        device=DEVICE,
    ) if validation_dict else None
    # Set up the model
    model = setup_model(SETUP, device=DEVICE, tau_pt_cut=argparser.tau_pt)
    # Fit the model
    fitting = TorchFitting(
        model,
        DEVICE,
        save_logs=False,
        trace_func=LOGGER.info,
        tensorboard=False,
        early_stopping=True,
        cleanup_on_exception=True,
    )

    # Run the main function
    main(fitting, training_loader=composite_loader, validation_loader=validation_loader, n_epochs=argparser.epochs)
    from IPython import embed
    embed(header="run.py	l:173")
