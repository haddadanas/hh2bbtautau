__all__ = [
    "accuracy", "selection_efficiency", "signal_acceptance", "signal_purity", "background_rejection",
    "roc_curve", "roc_curve_auc", "roc_curve_auc_wiki", "roc_curve_auc_log", "auc_score",
    "get_ROC_Plotter", "get_ROC_Plotter_log",
]

import torch

from ml_network.src.ml_utils import MLPLotting

# useful type definitions
Tensor = torch.Tensor


# torch scripts
@torch.jit.script
def accuracy(y_pred: Tensor, y_true: Tensor) -> float:
    threshold = 0.5  # torch.linspace(0, 1, 101)
    # signal_acceptances = torch.zeros(101)

    if y_pred.ndim == 2 and y_pred.size(1) == 2:
        y_pred = y_pred[:, 1]
    elif y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_true.size() != y_pred.size():
        raise ValueError("y_true and y_pred must have the same shape")

    signal_mask = y_true.to(torch.bool)
    # for i, threshold in enumerate(thresholds):
    mask = y_pred > threshold
    accuracy = torch.sum(mask == signal_mask).float()
    accuracy /= y_true.size(0)

    return accuracy.item()


@torch.jit.script
def selection_efficiency(y_pred: Tensor, y_true: Tensor) -> float:
    threshold = 0.5  # torch.linspace(0, 1, 101)
    # selection_efficiencies = torch.zeros(101)

    if y_pred.ndim == 2 and y_pred.size(1) == 2:
        y_pred = y_pred[:, 1]
    elif y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_true.size() != y_pred.size():
        raise ValueError("y_true and y_pred must have the same shape")

    # for i, threshold in enumerate(thresholds):
    mask = y_pred > threshold
    selection_efficiency = torch.mean(mask.float(), dim=0)

    return selection_efficiency.item()


@torch.jit.script
def signal_acceptance(y_pred: Tensor, y_true: Tensor) -> float:
    threshold = 0.5  # torch.linspace(0, 1, 101)
    # signal_acceptances = torch.zeros(101)

    if y_pred.ndim == 2 and y_pred.size(1) == 2:
        y_pred = y_pred[:, 1]
    elif y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_true.size() != y_pred.size():
        raise ValueError("y_true and y_pred must have the same shape")

    signal_mask = y_true.to(torch.bool)
    # for i, threshold in enumerate(thresholds):
    mask = y_pred > threshold
    signal_selection_mask = mask[signal_mask]
    signal_acc = torch.mean(signal_selection_mask.float(), dim=0)

    return signal_acc.item()


@torch.jit.script
def signal_purity(y_pred: Tensor, y_true: Tensor) -> float:
    threshold = 0.5  # torch.linspace(0, 1, 101)
    # signal_purities = torch.zeros(101)

    if y_pred.ndim == 2 and y_pred.size(1) == 2:
        y_pred = y_pred[:, 1]
    elif y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_true.size() != y_pred.size():
        raise ValueError("y_true and y_pred must have the same shape")

    signal_mask = y_true.to(torch.bool)
    # for i, threshold in enumerate(thresholds):
    mask = y_pred > threshold
    signal_purity_array = signal_mask[mask].float()
    signal_pur = torch.mean(signal_purity_array, dim=0)

    return signal_pur.item()


@torch.jit.script
def background_rejection(y_pred: Tensor, y_true: Tensor) -> Tensor:
    thresholds = torch.linspace(0, 1, 101)
    background_rejections = torch.zeros(101)

    if y_pred.ndim == 2 and y_pred.size(1) == 2:
        y_pred = y_pred[:, 1]
    elif y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_true.size() != y_pred.size():
        raise ValueError("y_true and y_pred must have the same shape")

    background_mask = ~y_true.to(torch.bool)
    for i, threshold in enumerate(thresholds):
        mask = y_pred > threshold
        background_rejection_mask = ~mask[background_mask]
        background_rejections[i] = torch.mean(background_rejection_mask.float(), dim=0)

    return background_rejections


@torch.jit.script
def roc_curve(y_pred: Tensor, y_true: Tensor) -> tuple[Tensor, Tensor]:
    thresholds = torch.linspace(0, 1, 101)
    tpr = torch.zeros(101)
    fpr = torch.zeros(101)

    if y_pred.ndim == 2 and y_pred.size(1) == 2:
        y_pred = y_pred[:, 1]
    elif y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_true.size() != y_pred.size():
        raise ValueError("y_true and y_pred must have the same shape")

    positive = y_true.to(torch.bool)
    negative = ~positive
    for i, threshold in enumerate(thresholds):
        predicted_positive = y_pred > threshold
        tp = predicted_positive[positive]
        fp = predicted_positive[negative]
        tpr[i] = torch.mean(tp.float(), dim=0)
        fpr[i] = torch.mean(fp.float(), dim=0)

    return fpr, tpr


@torch.jit.script
def roc_curve_auc_wiki(y_pred: Tensor, y_true: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    thresholds = torch.linspace(0, 1, 101)
    tpr = torch.zeros(101)
    fpr = torch.zeros(101)

    if y_pred.ndim == 2 and y_pred.size(1) == 2:
        y_pred = y_pred[:, 1]
    elif y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_true.size() != y_pred.size():
        raise ValueError("y_true and y_pred must have the same shape")

    positive = y_true.to(torch.bool)
    negative = ~positive
    for i, threshold in enumerate(thresholds):
        predicted_positive = y_pred > threshold
        tp = predicted_positive[positive]
        fp = predicted_positive[negative]
        tpr[i] = torch.mean(tp.float(), dim=0)
        fpr[i] = torch.mean(fp.float(), dim=0)

    # calculate the area under the curve
    fpr_diff = fpr[1:] - fpr[:-1]
    tpr_sum = tpr[1:] + tpr[:-1]
    auc = torch.abs(torch.sum(fpr_diff * tpr_sum) / 2)

    return fpr, tpr, auc


@torch.jit.script
def roc_curve_auc(y_pred: Tensor, y_true: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    thresholds = torch.linspace(0, 1, 101)
    tpr = torch.zeros(101)
    tnr = torch.zeros(101)

    if y_pred.ndim == 2 and y_pred.size(1) == 2:
        y_pred = y_pred[:, 1]
    elif y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_true.size() != y_pred.size():
        raise ValueError("y_true and y_pred must have the same shape")

    positive = y_true.to(torch.bool)
    negative = ~positive
    for i, threshold in enumerate(thresholds):
        predicted_positive = y_pred > threshold
        tp = predicted_positive[positive]
        tn = ~predicted_positive[negative]
        tpr[i] = torch.mean(tp.float(), dim=0)
        tnr[i] = torch.mean(tn.float(), dim=0)

    # calculate the area under the curve
    tpr_diff = tpr[1:] - tpr[:-1]
    tnr_sum = tnr[1:] + tnr[:-1]
    auc = torch.abs(torch.sum(tpr_diff * tnr_sum) / 2)

    return tpr, tnr, auc


@torch.jit.script
def roc_curve_auc_log(y_pred: Tensor, y_true: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    thresholds = torch.linspace(0, 1, 101)
    tpr = torch.zeros(101)
    fpr = torch.zeros(101)

    if y_pred.ndim == 2 and y_pred.size(1) == 2:
        y_pred = y_pred[:, 1]
    elif y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_true.size() != y_pred.size():
        raise ValueError("y_true and y_pred must have the same shape")

    positive = y_true.to(torch.bool)
    negative = ~positive
    for i, threshold in enumerate(thresholds):
        predicted_positive = y_pred > threshold
        tp = predicted_positive[positive]
        fp = predicted_positive[negative]
        tpr[i] = torch.mean(tp.float(), dim=0)
        fpr[i] = torch.mean(fp.float(), dim=0)
    eps_b = 1 / (fpr + 1e-5)

    # calculate the area under the curve
    fpr_diff = fpr[1:] - fpr[:-1]
    tpr_sum = tpr[1:] + tpr[:-1]
    auc = torch.abs(torch.sum(fpr_diff * tpr_sum) / 2)

    return tpr, eps_b, auc


def auc_score(y_pred, y_true):
    tpr, tnr, auc = roc_curve_auc(y_pred, y_true)
    return auc.item()


# Define some plotting functions
def get_ROC_Plotter():
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
        silent=False,
    )


def get_ROC_Plotter_log():
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
