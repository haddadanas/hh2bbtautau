import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import re
import os

# Define a CF custom color map
import matplotlib.colors as colors
cf_colors = {"cf_green_cmap": colors.ListedColormap(["#212121",
                                                     "#242723",
                                                     "#262D25",
                                                     "#283426",
                                                     "#2A3A26",
                                                     "#2C4227",
                                                     "#2E4927",
                                                     "#305126",
                                                     "#325A25",
                                                     "#356224",
                                                     "#386B22",
                                                     "#3B7520",
                                                     "#3F7F1E",
                                                     "#43891B",
                                                     "#479418",
                                                     "#4C9F14",
                                                     "#52AA10",
                                                     "#58B60C",
                                                     "#5FC207",
                                                     "#67cf02"]),  # type: ignore
             "cf_ygb_cmap": colors.ListedColormap(["#003675",
                                                   "#005B83",
                                                   "#008490",
                                                   "#009A83",
                                                   "#00A368",
                                                   "#00AC49",
                                                   "#00B428",
                                                   "#00BC06",
                                                   "#0CC300",
                                                   "#39C900",
                                                   "#67cf02",
                                                   "#72DB02",
                                                   "#7EE605",
                                                   "#8DF207",
                                                   "#9CFD09",
                                                   "#AEFF0B",
                                                   "#C1FF0E",
                                                   "#D5FF10",
                                                   "#EBFF12",
                                                   "#FFFF14"]),  # type: ignore
             "cf_cmap": colors.ListedColormap(["#002C9C",
                                               "#00419F",
                                               "#0056A2",
                                               "#006BA4",
                                               "#0081A7",
                                               "#0098AA",
                                               "#00ADAB",
                                               "#00B099",
                                               "#00B287",
                                               "#00B574",
                                               "#00B860",
                                               "#00BB4C",
                                               "#00BD38",
                                               "#00C023",
                                               "#00C20D",
                                               "#06C500",
                                               "#1EC800",
                                               "#36CA00",
                                               "#4ECD01",
                                               "#67cf02"]),  # type: ignore
             "viridis": colors.ListedColormap(["#263DA8",
                                               "#1652CC",
                                               "#1063DB",
                                               "#1171D8",
                                               "#1380D5",
                                               "#0E8ED0",
                                               "#089DCC",
                                               "#0DA7C2",
                                               "#1DAFB3",
                                               "#2DB7A3",
                                               "#52BA91",
                                               "#73BD80",
                                               "#94BE71",
                                               "#B2BC65",
                                               "#D0BA59",
                                               "#E1BF4A",
                                               "#F4C53A",
                                               "#FCD12B",
                                               "#FAE61C",
                                               "#F9F90E"]),  # type: ignore
             }


def plot_confusion_matrix(cm: np.ndarray,
                          process_labels,
                          class_labels,
                          save_path: str = "./cm_plot.png",
                          normalized=False,
                          title="Confusion matrix",
                          colormap: str = "cf_cmap",
                          cmap_label: str = "Accuracy",
                          digits: int = 3,
                          skip_uncertainties: bool = False):
    """
    Plots a confusion matrix.

    Args:
      cm: Confusion matrix.
      classes: List of class labels.
      normalized: Whether to normalize the confusion matrix.
      title: Title of the confusion matrix.
      cmap: Colormap.
    """
    assert (cm.ndim == 2), (
      f"Input matrix should be of dimension 2. Received dimension {cm.ndim}!")
    n_processes = cm.shape[0]
    n_classes = cm.shape[1]
    assert (n_processes == len(process_labels)), (
      f"Array length of the process labels (={len(process_labels)}) does not match the number of rows givin in the confusion matrix (={n_processes})!")
    assert (n_classes == len(class_labels)), (
      f"Array length of the classes labels (={len(class_labels)}) does not match the number of columns givin in the confusion matrix (={n_processes})!")

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(save_path)

    def scale_font(class_number: int) -> int:
        if class_number > 10:
            return max(8, int(- 8/10 * class_number + 23))
        else:
            return int(class_number/14*(9 * class_number - 177) + 510/7)

    def get_errors(matrix):
        from scinum import UP
        get_errors = np.vectorize(lambda x: x.get(UP, unc=True))
        return get_errors(matrix)

    cmap = cf_colors.get(colormap, cf_colors["cf_cmap"])

    # Get values and their uncertenties
    values = cm.astype(np.float32)
    if cm.dtype.name == "object":
        uncs = get_errors(cm)
    else:
        uncs = np.zeros_like(values)

    def value_text(i, j):
        def fmt(v):
            s = "{{:.{}f}}".format(digits).format(v)
            return s if re.sub(r"(0|\.)", "", s) else ("<" + s[:-1] + "1")
        if skip_uncertainties:
            return fmt(values[i][j])
        else:
            return "{}\n\u00B1{}".format(fmt(values[i][j]), fmt(np.nan_to_num(uncs[i][j])))
            # return "{}".format(fmt(values[i][j])),"\u00B1{}".format(fmt(uncs[i][j]))
    plt.style.use(hep.style.CMS)
    plt.imshow(values, interpolation="nearest", cmap=cmap)

    # Setting some values
    thresh = values.max() / 2.
    font_size = scale_font(n_classes)

    # Remove Major ticks and edit minor ticks
    minor_tick_length = max(int(120/n_classes), 12)
    minor_tick_width = max(6/n_classes, 0.6)
    xtick_marks = np.arange(n_classes)
    ytick_marks = np.arange(n_processes)
    plt.tick_params(axis="both", which="major",
                    bottom=False, top=False, left=False, right=False)
    plt.tick_params(axis="both", which="minor",
                    bottom=True, top=True, left=True, right=True,
                    length=minor_tick_length, width=minor_tick_width)
    plt.xticks(xtick_marks + 0.5, minor=True)
    plt.yticks(ytick_marks + 0.49, minor=True)
    plt.xticks(xtick_marks, class_labels, rotation=0, fontsize=font_size)
    plt.yticks(ytick_marks, process_labels, fontsize=font_size)

    # Justify Color bar
    colorbar = plt.colorbar(fraction=0.0471, pad=0.01)
    colorbar.set_label(label=cmap_label, fontsize=font_size+3)
    colorbar.ax.tick_params(labelsize=font_size)
    plt.clim(0, max(1, values.max()))

    # Add Matrix Elemtns
    # offset = 0.12 if len(class_labels) > 2 and len(class_labels) < 6 else 0.1
    # size_offset = 1 if len(class_labels) > 5 else 3
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            plt.text(j, i, value_text(i, j), fontdict={"size": font_size},
                     horizontalalignment="center", verticalalignment="center",
                     color="white" if values[i, j] < thresh else "black")
            """
                  val, err = value_text(i,j)
            plt.text(j, i-offset, val, fontdict={"size": font_size},
                    horizontalalignment="center", verticalalignment="center",
                    color="white" if values[i, j] < thresh else "black")
            plt.text(j, i+offset, err, fontdict={"size": font_size-size_offset},
                    horizontalalignment="center", verticalalignment="center",
                    color="white" if values[i, j] < thresh else "black")
                    """
    # Add Axes and plot labels
    hep.cms.label(llabel="private work",
                  rlabel=title if title is not None else "")
    plt.xlabel("Predicted process", loc="right", labelpad=10,
               fontsize=font_size+3)
    plt.ylabel("True process", loc="top", labelpad=15, fontsize=font_size+3)

    # Saving
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.clf()


def plot_roc_curve(save_path: str = "./roc_plot.png",
                   fpr=None,
                   tpr=None,
                   *args,
                   label: str = "ROC Curve",
                   grid: tuple = None,  # type: ignore
                   auc_scores: dict or float = None,  # type: ignore
                   shared_y: bool = False,
                   input_dict: dict = None):  # type: ignore
    """Creats the plot for givin ROC curve data

    Args:
        save_path (str, optional): Path to save the plot. Defaults to "./roc_plot.png".
        fpr (_type_, optional): array of list with the FPR values. Defaults to None.
        tpr (_type_, optional): array of list with the TPR values. Defaults to None.
        label (str, optional): plot title of the ROC Curve. Defaults to "ROC Curve"
        grid (tuple, optional): The layout grid for the plots.
                                If not specified, the number of rows and columns will be set to an optimum.
                                Defaults to None.
        auc_scores (dictorfloat, optional): AUC scores for the givin ROC curve.
                                            The parameter should be givin either as a float for a single ROC curve plot or
                                            if `input_dict` is givin as a dictionary with the same keys as `input_dict`.
                                            Defaults to None.
        input_dict (dict, optional): 6Defaults to None.

    Returns:
        _type_: None
    """
    assert (input_dict is not None or (fpr is not None and tpr is not None)), (
      "Either `input_dict` or `fpr` and `tpr` should be givin!")
    assert (input_dict is not None or len(fpr) == len(tpr)), (
      f"The length of `fpr` and `tpr` should be equal! Received arrays with length {len(fpr)} and {len(tpr)}")
    assert isinstance(grid, (tuple, type(None))), (
      f"the `grid` argument must be a tuple. Givin {grid}")
    assert (auc_scores is None or isinstance(auc_scores, float) or auc_scores.keys() == input_dict.keys()), (
      "The parameter `auc_score` should have the same keys as the `input_dict`!")

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(save_path)

    def fmt_auc(num):
        def fmt(v):
            s = "{:.3f}".format(v)
            return s if re.sub(r"(0|\.)", "", s) else ("<" + s[:-1] + "1")
        return "{}\u00B1{}".format(fmt(num.n), fmt(np.nan_to_num(num.u()[0])))

    def get_grid(n):
        nrow = round(np.sqrt(n))
        ncol = int(n/nrow)
        while (nrow*ncol < n):
            ncol += 1
        return nrow, ncol

    def plot_roc(ax: plt.Axes, x, y, axtitle):
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.locator_params(axis="both", nbins=6)
        ax.tick_params(axis="x", pad=10)
        ax.tick_params(axis="y", pad=10)
        ax.set_title(axtitle)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.plot(x, y)

    if input_dict is None:
        input_dict = {label: {"fpr": fpr, "tpr": tpr}}
        if not isinstance(auc_scores, (dict, type(None))):
            auc_scores = {label: auc_scores}

    nrow, ncol = get_grid(len(input_dict.keys())) if grid is None else grid

    plt.style.use(hep.style.CMS)
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(6*ncol, 6*nrow+1),
                            dpi=300, sharey=shared_y)
    axs = np.array(axs).flatten()
    hep.cms.label(llabel="private work", rlabel="", ax=axs[0],
                  pad=0.2 if auc_scores is not None else 0.1)
    for (key, item), ax in zip(input_dict.items(), axs):
        plot_roc(ax, item["fpr"], item["tpr"],
                 f"{key}\n(AUC = {fmt_auc(auc_scores[key])})" if auc_scores else key)

    # Saving
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.clf()


def plot_SIF(thresholds,
             fpr,
             tpr,
             *args,
             save_path: str = "./sfi_plot.png",
             label: str = "SIF Curve"):
    """Creats the plot for givin ROC curve data

    Args:
        save_path (str, optional): Path to save the plot. Defaults to "./roc_plot.png".
        fpr (_type_, optional): background rejection efficiency. Defaults to None.
        tpr (_type_, optional): signal efficiency. Defaults to None.
        label (str, optional): plot title of the ROC Curve. Defaults to "ROC Curve"

    Returns:
        _type_: None
    """
    assert (len(fpr) == len(tpr) and len(fpr) == len(thresholds)), (
      f"The length of `fpr`, `tpr` and `thresholds` should be equal! Received arrays with length {len(fpr)}, {len(tpr)} and {len(thresholds)}")

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(save_path)

    min_value = np.argmin(thresholds)
    norm_factor = tpr[min_value]/np.sqrt(fpr[min_value])
    sif = (tpr/np.sqrt(fpr))/norm_factor
    plt.style.use(hep.style.CMS)
    plt.plot(thresholds, sif)

    hep.cms.label(llabel="private work", rlabel=label)
    plt.xlabel("Thresholds", loc="right")
    plt.ylabel(r"$S.I.F. \sim \frac{\varepsilon_{S}}{\varepsilon_{B}}$",
               loc="top")
    # Saving
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    trues = np.random.random(size=1000) > 0.5
    pred = np.random.random(size=1000)
    from sklearn.metrics import roc_curve
    f, t, t = roc_curve(trues, pred)

    plot_SIF(t, f, t)
