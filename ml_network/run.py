from ml_network.src.utils import load_setup, get_device
from ml_network.src.preprocessing import prepare_input
from ml_network.src.ml_utils import (
    Fitting, MLPLotting, roc_curve_auc, signal_purity, signal_acceptance, selection_efficiency,
)
from ml_network.models.ml_model import CustomModel, CONFIG

# import matplotlib
# matplotlib.use("module://imgcat")

# Load some configuration
SETUP = load_setup()
device = get_device()
print(f"Using {device} device")
base_name = f"ml_{SETUP['used_selector']}__datasets_{len(SETUP['datasets'])}__{SETUP['model_suffix']}"
if "limit_dataset" in SETUP:
    base_name += f"__limit_{SETUP['limit_dataset']}"

# Prepare the input
inp, fields = prepare_input(CONFIG, SETUP['datasets'])
# Get the model
model = CustomModel(base_name, fields, save_path=SETUP["model_save_path"])
model.to(device)
model.compile(backend="aot_eager")


# Define the plotting
def plot_ROC(y_true, y_pred, ax):
    fpr, tpr, auc = roc_curve_auc(y_true, y_pred)
    ax.plot(fpr, tpr, label=f"AUC: {auc:.3f}")
    return fpr, tpr, auc


roc_plot = MLPLotting(
    title="ROC curve",
    xlabel="False positive rate",
    ylabel="True positive rate",
    plot_func=plot_ROC,
    validation=True,
)

# Fit the model
fitting = Fitting(model, device)
logs = fitting.fit(
    training_data=inp['x'],
    validation_data=inp['validation_data'],
    metrics=[signal_purity, signal_acceptance, selection_efficiency],
    plots=[roc_plot],
    **CONFIG,
)


# Save the model and logs
fitting.save_model()
fitting.save_logs(logs)


input("Press Enter to end...")
