import numpy as np
from scinum import Number, UP

def calc_uncert(matrix : list or np.array , func):
    #result = np.zeros_like(matrix.size, dtype = np.float)
    matrix = np.array(matrix)
    errors = func(matrix)
    row_sum = np.sum(matrix, axis = 1)
    result = ((errors*np.sqrt((np.sum(matrix,axis = 1)- matrix.T).T)).T/np.power(row_sum, 3/2)).T
    return result



#TODO add uncertenties to Matrix
def get_conf_matrix(true_labels: np.array, model_output: np.array, weights: np.array = None, normalization: str = None, errors: bool = True) -> np.array:
    """
    Generates the confusion matrix given the output of the nodes and a true labels array.
    The Cronfusion matrix can also be weighted and 

    Args:
        true_labels (np.array): an `Array` with the true labels of the events
        model_output (np.array): output of the model with propability predictions for each event.
        errors (bool): calculate errors of the ROC entries
        weights (np.array): weights of the events

    Returns:
        An `Array` with shape `[m, n]` representing the confusion matrix, where `m` is number of expected classes in the true labels and `n` is the number of possible labels in the classification

    Raises:
        ValueError: If both predictions and labels have mismatched shapes, or if `weights` is not `None` and its shape doesn't match `predictions`.
    """    

    if (true_labels.shape[0] != model_output.shape[0]):
        raise ValueError(f'Mismatched shapes of the inputs! Inputs can not be compared with shapes {true_labels.shape[0]} and {model_output.shape[0]}')
    
    is_weighted = type(weights) != type(None)
    if (is_weighted and weights.shape[0] != model_output.shape[0]):
        raise ValueError(f'Mismatched shapes of the weights! Weights can not be broadcast to the inputs with shapes {true_labels.shape[0]} and {model_output.shape[0]}')
    
    return_type = np.float32 if is_weighted else np.int32

    # define shape of the output
    mat_shape = (true_labels.max() + 1, model_output.shape[1])
    # create predictions of the model output
    predictions = np.argmax(model_output, axis = 1)
    
    result = np.zeros(shape = mat_shape, dtype = return_type)
    counts = np.zeros(shape = mat_shape, dtype = return_type)

    #indecies = np.stack((true_labels, predictions), axis = 1)
    for ind, (true, pred) in enumerate(zip(true_labels, predictions)):
        result[true][pred] += weights[ind] if is_weighted else 1
        counts[true][pred] += 1

    #Normalize Matrix if needed
    if normalization is not None:
        valid = {'row': 1, 'column': 0}
        if normalization not in valid.keys():
            raise ValueError(f'\'{normalization}\' is no valid argument for normalization. If givin, normalization should only take \'row\' or \'column\'')
        
        row_sums = result.sum(axis=valid.get(normalization))
        result = result / row_sums[:, np.newaxis]

    if errors:
        vNumber = np.vectorize(lambda num, count: Number(num, float(np.divide(num, np.sqrt(count)))))
        return vNumber(result, counts)
        

    return result
    
#TODO add function for multi-dim ROC curve
def get_roc_data(true_labels: np.array, model_output_positive: np.array, model_output_negative: np.array = None, thresholds: np.array = None, weights: np.array = None, errors: bool = True, *args: list, output_length: int = 10 + 1) -> tuple:
    """
    Compute Receiver operating characteristic (ROC) values givin the nodes outputs and the true labels for a binary classification

    Args:
        true_labels (np.array): an `Array` with the true labels of the events. Entries should be of type true/flase or 1/0
        model_output_positive (np.array): output of the model with propability predictions for positive events.
        model_output_negative (np.array): output of the model with propability predictions for negative events. If not specified complementary propabilitues are chosen.
        thresholds (np.array): array with custom thresholds, at which the ROC curve points shall be calculated. If specified, this will overwrite the parameter `output_points`, else a linear spacewill be created with `output_points` entries.
        weights (np.array): weights of the events
        errors (bool): calculate errors of the ROC entries
        args (y): list of additional parameters.
        output_length (int): number of points generated for the ROC curve

    Returns:
        An tuple of 1D `np.array` each with length `output_length` representing the FPR and TPR.

    Raises:
        ValueError: If both predictions and labels have mismatched shapes.
    """   
    #Helper functions
    def cast_to_Number(array, get_errors) -> Number:
        if get_errors:
            return Number(np.sum(array), float(np.divide(array.sum(), array.size)))
        return Number(np.sum(array))
    
    #Define Thresholds if None
    if thresholds is None:
        thresholds = np.linspace(0, 1, output_length)

    #Define model_output_negatives if None
    if model_output_negative is None:
        model_output_negative = 1- model_output_positive

    #Define weights if None
    if weights is None:
        weights = np.ones_like(model_output_positive)
    
    #Cast trues labels
    trues = true_labels.astype(dtype=np.bool)

    #Check the input on correctness
    if (true_labels.shape[0] != model_output_positive.shape[0]):
        raise ValueError(f'Mismatched shapes of the positive inputs! Inputs can not be compared with shapes {true_labels.shape[0]} and {model_output_positive.shape[0]}')
    
    if (true_labels.shape[0] != model_output_negative.shape[0]):
        raise ValueError(f'Mismatched shapes of the negative inputs! Inputs can not be compared with shapes {true_labels.shape[0]} and {model_output_negative.shape[0]}')
    
    if (weights.shape[0] != model_output_positive.shape[0]):
        raise ValueError(f'Mismatched shapes of the weights! Weights can not be broadcast to the inputs with shapes {true_labels.shape[0]} and {model_output_positive.shape[0]}')    
    
    
    tpr = []
    fpr = []

    for t in thresholds:
        positives = model_output_positive >= t
        tp = cast_to_Number(weights[np.logical_and(positives, trues)], errors)
        tn = cast_to_Number(weights[np.logical_and(positives == False, trues == False)], errors)
        fp = cast_to_Number(weights[np.logical_and(positives, trues == False)], errors)
        fn = cast_to_Number(weights[np.logical_and(positives == False, trues)], errors)

        tpr.append(tp/(tp + fn))
        fpr.append(fp/(fp + tn))

    return fpr, tpr, thresholds








if __name__ == '__main__':

    test = np.random.randint(0,100, size=(100,5))
    trues = np.random.randint(1,5,size= 100)
    x = np.divide(test.T, np.sum(test, axis=1)).T

    from sklearn.metrics import roc_curve

    true = np.random.randint(0,2,size =1000)
    pred = np.random.random(size = 1000)
    weights = np.random.random(size = 1000)
    x = get_roc_data(true, pred,weights=weights,output_length=13)
    print(x)

