from hbt.ml.plotting.metrices import (get_conf_matrix,
                                      roc_curve_data,
                                      mdim_auc_score,
                                      np)
from hbt.ml.plotting.plotting import (plot_confusion_matrix,
                                      plot_roc_curve,
                                      plot_SIF)


def Confusion_Matrix(true_labels: np.ndarray,
                     model_output: np.ndarray,
                     process_labels: list,
                     class_labels: list,
                     sample_weights: np.ndarray = None,
                     normalization: str = None,
                     skip_uncertenties: bool = False,
                     output_path: str = './cm_plot.png',
                     plot_title='Confusion matrix',
                     colormap='cf_cmap',
                     z_title: str = 'Accuracy',
                     digits: int = 3
                     ) -> np.ndarray:
    """_summary_

    Args:
        true_labels (np.ndarray): _description_
        model_output (np.ndarray): _description_
        process_labels (list): True Processes
        class_labels (list): Predicted Processes
        sample_weights (np.ndarray, optional): _description_. Defaults to None.
        normalization (str, optional): _description_. Defaults to None.
        skip_uncertenties (bool, optional): _description_. Defaults to False.
        output_path (str, optional): _description_. Defaults to './cm_plot.png'.
        plot_title (str, optional): _description_. Defaults to 'Confusion matrix'.
        color_map (_type_, optional): _description_. Defaults to cf_cmap.
        z_title (str, optional): _description_. Defaults to 'Accuracy'.
        digits (int, optional): _description_. Defaults to 3.

    Returns:
        np.ndarray: An array containing the plotted confusion matrix
    """

    cm = get_conf_matrix(true_labels=true_labels,
                         model_output=model_output,
                         sample_weights=sample_weights,
                         normalization=normalization,
                         errors=not (skip_uncertenties))

    if output_path is not None:
        if normalization is not None:
            z_title += f'({normalization}-normalized)'

        plot_confusion_matrix(cm=cm,
                              process_labels=process_labels,
                              class_labels=class_labels,
                              save_path=output_path,
                              normalized=normalization is not None,
                              title=plot_title,
                              colormap=colormap,
                              cmap_label=z_title,
                              digits=digits,
                              skip_uncertainties=skip_uncertenties
                              )

    return cm


def ROC_Curve(evaluation_type: str,
              true_labels: np.ndarray,
              model_output: np.ndarray,
              class_names: list = None,
              thresholds: np.ndarray = None,
              sample_weights: np.ndarray = None,
              skip_uncertenties: bool = False,
              output_length: int = 100 + 1,
              output_path: str = './roc_plot.png',
              plot_title: str = 'ROC Curve',
              figure_grid: tuple = None,  # type: ignore
              ) -> dict:

    roc_dict = roc_curve_data(evaluation_type=evaluation_type,
                              true_labels=true_labels,
                              model_output=model_output,
                              class_names=class_names,
                              thresholds=thresholds,
                              sample_weights=sample_weights,
                              errors=not (skip_uncertenties),
                              output_length=output_length)

    auc_dict = mdim_auc_score(roc_dict)
    if output_path is not None:
        plot_roc_curve(save_path=output_path,
                       input_dict=roc_dict,
                       auc_scores=auc_dict,
                       grid=figure_grid
                       )

    for key, data in roc_dict.items():
        data.update({'auc_score': auc_dict[key]})

    return roc_dict
