from src.utils import load_setup, get_device
from src.preprocessing import prepare_input
from src.ml_utils import Fitting, roc_curve_auc, signal_purity, signal_acceptance, selection_efficiency
from models.ml_model import CustomModel, CONFIG

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("module://imgcat")

# Load some configuration
SETUP = load_setup()
device = get_device()
print(f"Using {device} device")


# Prepare the input
inp, fields = prepare_input(CONFIG, SETUP['datasets'])
# Get the model
model = CustomModel(fields)
model.to(device)


# define easy metrices
def AUC(y_true, y_pred):
    return roc_curve_auc(y_true, y_pred)[2].item()


def plot_ROC(y_true, y_pred):
    fpr, tpr, auc = roc_curve_auc(y_true, y_pred)
    plt.plot(fpr, tpr, label=f"AUC: {auc}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.show()


# Fit the model
fitting = Fitting(model, device)
logs = fitting.fit(
    training_data=inp['x'],
    validation_data=inp['validation_data'],
    metrics=[signal_purity, signal_acceptance, selection_efficiency, plot_ROC],
    **CONFIG,
)
