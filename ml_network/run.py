from time import perf_counter_ns

from ml_network.src.utils import load_setup, get_device, build_model_name
from ml_network.src.preprocessing import prepare_input
from ml_network.src.ml_utils import (
    Fitting, MLPLotting, roc_curve_auc, signal_purity, signal_acceptance, selection_efficiency,
)
from ml_network.models.ml_model_batch_norm import CONFIG, CustomModel as Model

# import matplotlib
# matplotlib.use("module://imgcat")

# Load some configuration
SETUP = load_setup()
device = get_device()
print(f"Using {device} device")

# Prepare the input
inp, fields = prepare_input(**CONFIG, inputs=SETUP['datasets'])
CONFIG["feature_names"] = fields

# Get the model
model_name = build_model_name(SETUP, Model)
model = Model(model_name, fields, save_path=SETUP["model_save_path"])
model.to(device)
model.compile(backend="aot_eager")


# Define the plotting
def plot_ROC(y_pred, y_true, ax, epoch=""):
    tpr, tnr, auc = roc_curve_auc(y_pred, y_true)
    ax.plot(tpr, tnr, label=f"AUC: {auc:.3f} @ ep {epoch}")
    return tpr, tnr, auc


# def plt_ROC_log(y_pred, y_true, ax):
#     tpr, e_bkg, auc = roc_curve_auc_log(y_pred, y_true)
#     ax.plot(tpr, e_bkg, label=f"AUC: {auc:.3f}")
#     return tpr, e_bkg, auc


roc_plot = MLPLotting(
    title="ROC curve",
    xlabel="True positive rate",
    ylabel="True negative rate",
    plot_func=plot_ROC,
    validation=True,
)

# roc_log_plot = MLPLotting(
#     title="ROC Log",
#     xlabel="True positive rate",
#     ylabel=r"Background rejection $\frac{1}{FPR}$",
#     plot_func=plt_ROC_log,
#     log_axis=True,
#     validation=True,
# )

# Fit the model
fitting = Fitting(model, device, training=True)

start = perf_counter_ns()
logs = fitting.fit(
    training_data=inp['x'],
    validation_data=inp['validation_data'],
    metrics=[signal_purity, signal_acceptance, selection_efficiency],
    plots=[roc_plot],
    use_weights=SETUP["use_weights"],
    **CONFIG,
)
print(f"Training took {(perf_counter_ns() - start) / 1e9:.2f} seconds")

# Save the model and logs
fitting.save_model()
fitting.save_logs(logs)
fitting.save_config(CONFIG, suffix="config")

input("Press Enter to end...")
