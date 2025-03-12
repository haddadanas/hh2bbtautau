from ml_network.src.utils import load_setup, get_device, build_model_name, setup_parser, log_parser, DotDict, get_logger, timeit
from ml_network.src.preprocessing import prepare_input
from ml_network.src.ml_utils import (
    Fitting, MLPLotting, auc_score, signal_purity, signal_acceptance, selection_efficiency,
)
from ml_network.models.ml_model_batch_norm import CONFIG, NUM_FIELDS, EMBED_FIELDS, CustomModel as Model
from ml_network.src.torch_transforms import AkToProcessedTensor
from ml_network.src.torch_util import CompositeDataLoader, FlatTorchDataset, NestedDictMapAndCollate, NestedMapAndCollate


# Set some constants and global variables
LOGGER = get_logger("ML Network")
EMPTY_FLOAT = -10.0
EMPTY_INT = -10
SETUP = load_setup()
DEVICE = get_device()
COLUMN_NAMES = {"num_fields": NUM_FIELDS, "embed_fields": EMBED_FIELDS}


def setup_model(setup: dict, fields, device: str = DEVICE, **kwargs):
    # Get the model
    model_name = build_model_name(setup, Model, **kwargs)
    model = Model(model_name, fields, save_path=setup["model_save_path"])
    model.to(device)
    model.compile(backend="aot_eager")
    return model


# Define some plotting functions
def get_ROC_Plotter():
    from ml_network.src.ml_utils import roc_curve_auc
    # helper function
    def plot_ROC(y_pred, y_true, ax, epoch=""):
        tpr, tnr, auc = roc_curve_auc(y_pred, y_true)
        ax.plot(tpr, tnr, label=f"AUC: {auc:.3f} @ ep {epoch}")
        return tpr, tnr, auc

    return MLPLotting(
        title="ROC curve",
        xlabel="True positive rate",
        ylabel="True negative rate",
        plot_func=plot_ROC,
        validation=True,
    )


def get_ROC_Plotter_log():
    from ml_network.src.ml_utils import roc_curve_auc_log
    # helper function
    def plt_ROC_log(y_pred, y_true, ax):
        tpr, e_bkg, auc = roc_curve_auc_log(y_pred, y_true)
        ax.plot(tpr, e_bkg, label=f"AUC: {auc:.3f}")
        return tpr, e_bkg, auc

    return MLPLotting(
        title="ROC Log",
        xlabel="True positive rate",
        ylabel=r"Background rejection $\frac{1}{FPR}$",
        plot_func=plt_ROC_log,
        log_axis=True,
        validation=True,
    )


def get_datasets(setup: dict):
    # Needed Columns
    needed_columns = set(EMBED_FIELDS) | set(NUM_FIELDS)
    if setup["selection_field"] is not None:
        needed_columns.add(setup["selection_field"])

    # Get the datasets
    datasets = setup["datasets"]
    data_map = DotDict()
    weight_dict = {}
    for name, dataset in datasets.items():
        data_map[name] = FlatTorchDataset(
            dataset["path"],
            columns=needed_columns,
            embedding_fields=EMBED_FIELDS,
            padd_value_float=EMPTY_FLOAT,
            padd_value_int=EMPTY_INT,
            device=DEVICE,
            target=int(dataset["is_signal"]),
            transform=AkToProcessedTensor(),
        )
        weight_dict[name] = dataset["weight"]
        LOGGER.info(f"Loaded dataset {name} with {len(data_map[name])} entries")

    return {"data_map": data_map, "weight_dict": weight_dict}


@timeit(LOGGER)
def main(logger, setup: dict, model, input_data, fields, device="cpu"):
    model = setup_model(setup, fields, device=device)
    # Fit the model
    fitting = Fitting(model, device, training=True)

    logs = fitting.fit(
        training_data=input_data['x'],
        validation_data=input_data['validation_data'],
        metrics=[signal_purity, signal_acceptance, selection_efficiency, auc_score],
        # plots=[roc_plot],
        use_weights=SETUP["use_weights"],
        **CONFIG,
    )
    # Save the model and logs
    fitting.save_best_model()
    fitting.save_logs(logs)
    fitting.save_config(CONFIG, suffix="config")
    input("Press Enter to end...")




if __name__ == "__main__":
    LOGGER.info(f"Using {DEVICE} device")

    # Set up the parser
    parser = setup_parser()
    argparser = parser.parse_args()
    log_parser(LOGGER, argparser)
    if argparser.imgcat:
        import matplotlib
        matplotlib.use("module://imgcat")

    dataloader_dict = get_datasets(SETUP)
    composite_loader = CompositeDataLoader(
        **dataloader_dict,
        map_and_collate_cls=NestedDictMapAndCollate,
        batch_size=argparser.batch_size,
        shuffle=True,
        device=DEVICE,
    )
    from IPython import embed; embed(header="run.py	l:109")
    pt_cut = argparser.pt_cut

    sel_mask = f"tau_pt{pt_cut}"
    # Prepare the input
    inp, fields = prepare_input(**CONFIG, inputs=SETUP['datasets'],
                                validation_split=0.1, selection_mask=sel_mask, dataset_limits=None)
    CONFIG["feature_names"] = fields

    # Run the main function
    main(setup=SETUP, model=Model, input_data=inp, fields=fields, device=DEVICE)
