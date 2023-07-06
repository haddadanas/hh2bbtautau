import numpy as np
from scinum import Number, UP

def calc_uncert(matrix : list or np.array , func):
    #result = np.zeros_like(matrix.size, dtype = np.float)
    matrix = np.array(matrix)
    errors = func(matrix)
    row_sum = np.sum(matrix, axis = 1)
    result = ((errors*np.sqrt((np.sum(matrix,axis = 1)- matrix.T).T)).T/np.power(row_sum, 3/2)).T
    return result

test = np.random.randint(0,100, size=(100,5))
trues = np.random.randint(1,5,size= 100)
x = np.divide(test.T, np.sum(test, axis=1)).T


def get_conf_matrix(true_labels: np.array, model_output: np.array, weights: np.array = None) -> np.array:
    """
    Generates the confusion matrix given the output of the nodes and a true labels array.
    The Cronfusion matrix can also be weighted and 

    Args:
        model_output (np.array): output of the model with propability predictions for each event.
        true_labels (np.array): an `Array` with the true labels of the events
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

    #indecies = np.stack((true_labels, predictions), axis = 1)
    for ind, (true, pred) in enumerate(zip(true_labels, predictions)):
        result[true][pred] += weights[ind] if is_weighted else 1

    return result
    




if __name__ == '__main__':
    print(calc_uncert([[1, 2], [3, 4]], lambda x: 1))

