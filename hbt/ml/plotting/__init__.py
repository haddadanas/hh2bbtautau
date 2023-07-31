from metrices import *
from plotting import *


def Confusion_Matrix(true_labels: np.ndarray, 
                     model_output: np.ndarray, 
                     process_labels:list,
                     class_labels:list,
                     sample_weights: np.ndarray = None, 
                     *args, 
                     normalization: str = None, 
                     skip_uncertenties: bool = False,
                     output_path:str='./cm_plot.png',
                     plot_title='Confusion matrix',
                     color_map=cf_cmap,
                     z_title:str='Accuracy',
                     digits:int = 3
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
                         errors=not(skip_uncertenties))
    
    if output_path != None:
        if normalization != None:
            z_title += f'({normalization}-normalized)'

        plot_confusion_matrix(cm=cm,
                            process_labels=process_labels,
                            class_labels=class_labels,
                            save_path=output_path,
                            normalize= normalization != None,
                            title=plot_title,
                            cmap=color_map,
                            cmap_label= z_title,
                            digits=digits
                            )
        
    return cm
