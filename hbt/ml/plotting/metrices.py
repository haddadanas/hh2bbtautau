import numpy as np
from scinum import Number


# Helper functions
def calc_uncert(matrix, func):
    # result = np.zeros_like(matrix.size, dtype = np.float)
    matrix = np.array(matrix)
    errors = func(matrix)
    row_sum = np.sum(matrix, axis=1)
    result = ((errors * np.sqrt((np.sum(matrix, axis=1) - matrix.T).T)).T / np.power(row_sum, 3 / 2)).T
    return result


def is_input_valid(true_labels, input):
    assert (true_labels.shape[0] == input.shape[0]), (
        f"Mismatched shapes of the inputs! Inputs can not be compared with shapes \
            {true_labels.shape[0]} and {input.shape[0]}")


def is_weights_valid(input, weights) -> bool:
    # Check if weights are givin
    is_weighted = not isinstance(weights, type(None))

    # Check if shapes are valid
    assert ((not is_weighted) or weights.shape[0] == input.shape[0]), (
        f"Mismatched shapes of the weights! Weights can not be broadcast to the \
            inputs with shapes {weights.shape[0]} and {input.shape[0]}")
    return is_weighted


def get_conf_matrix(true_labels: np.ndarray,
                    model_output: np.ndarray,
                    sample_weights: np.ndarray = None,
                    *args,
                    normalization: str = None,
                    errors: bool = True) -> np.ndarray:
    """
    Generates the confusion matrix given the output of the nodes and a true labels array.
    The Cronfusion matrix can also be weighted and

    Args:
        true_labels (np.array): an `Array` with the true labels of the events
        model_output (np.array): output of the model with propability predictions for each event.
        errors (bool): calculate errors of the ROC entries
        weights (np.array): weights of the events

    Returns:
        An `Array` with shape `[m, n]` representing the confusion matrix, where `m` is number of expected
        classes in the true labels and `n` is the number of possible labels in the classification

    Raises:
        AssertionError: If both predictions and labels have mismatched shapes, or if `weights` is not `None`
        and its shape doesn't match `predictions`.
    """

    is_input_valid(true_labels, model_output)

    is_weighted = is_weights_valid(model_output, sample_weights)

    return_type = np.float32 if is_weighted else np.int32

    # define shape of the output
    mat_shape = (true_labels.max() + 1, model_output.shape[1])
    # create predictions of the model output
    predictions = np.argmax(model_output, axis=1)

    result = np.zeros(shape=mat_shape, dtype=return_type)
    counts = np.zeros(shape=mat_shape, dtype=return_type)

    # indecies = np.stack((true_labels, predictions), axis = 1)
    for ind, (true, pred) in enumerate(zip(true_labels, predictions)):
        result[true][pred] += sample_weights[ind] if is_weighted else 1
        counts[true][pred] += 1

    # Normalize Matrix if needed
    if normalization is not None:
        valid = {"row": 1, "column": 0}
        assert (normalization in valid.keys()), (
            f"\"{normalization}\" is no valid argument for normalization. If givin, normalization \
            should only take \"row\" or \"column\"")

        row_sums = result.sum(axis=valid.get(normalization))
        result = result / row_sums[:, np.newaxis]

    if errors:
        vNumber = np.vectorize(lambda num,
                               count: Number(num, float(np.divide(num,
                                                                  np.sqrt(count)))))
        return vNumber(result, counts)

    return result


def binary_roc_data(true_labels: np.ndarray,
                    model_output_positive: np.ndarray,
                    sample_weights: np.ndarray = None,
                    *args,
                    thresholds: np.ndarray = None,
                    errors: bool = True,
                    output_length: int = 10 + 1,
                    ) -> tuple:
    """
    Compute Receiver operating characteristic (ROC) values givin the nodes outputs and the
    true labels for a binary classification

    Args:
        true_labels (np.array): an `Array` with the true labels of the events. Entries should be of type true/flase
        model_output_positive (np.array): output of the model with propability predictions for positive events.
        thresholds (np.array, optional): array with custom thresholds, at which the ROC curve points
        shall be calculated. If specified, this will overwrite the parameter `output_points`,
        else a linear spacewill be created with `output_points` entries. Defaults to None.
        weights (np.array): weights of the events
        errors (bool): calculate errors of the ROC entries
        args (y): list of additional parameters.
        output_length (int): number of points generated for the ROC curve

    Returns:
        An tuple of 1D `np.array` each with length `output_length` representing the FPR and TPR.

    Raises:
        AssertionError: If both predictions and labels have mismatched shapes.
    """

    # Helper functions
    def cast_to_Number(array, get_errors) -> Number:
        if get_errors:
            if array.size != 0:
                return Number(np.sum(array),
                              float(np.divide(array.sum(), np.sqrt(array.size)),
                                    ))
            return Number(np.sum(array), 0.0)
        return Number(np.sum(array))

    # Define Thresholds if None
    if thresholds is None:
        assert isinstance(output_length, int), (
            "If `thresholds`  is not givin, `output_length` must be givin an int, from which the thresholds \
                are determined! Both arguments cannot be `None`!")
        thresholds = np.linspace(0, 1, output_length)

    # Define weights if None
    if sample_weights is None:
        sample_weights = np.ones_like(model_output_positive)

    # Cast trues labels
    trues = true_labels.astype(dtype=bool)

    # Check the input on correctness
    model_output_positive = model_output_positive.flatten()
    is_input_valid(true_labels, model_output_positive)
    is_weights_valid(model_output_positive, sample_weights)

    tpr = []
    fpr = []

    for t in thresholds:
        positives = model_output_positive >= t
        tp = cast_to_Number(sample_weights[np.logical_and(positives, trues)],
                            errors)
        tn = cast_to_Number(sample_weights[np.logical_and(~positives, ~trues)],
                            errors)
        fp = cast_to_Number(sample_weights[np.logical_and(positives, ~trues)],
                            errors)
        fn = cast_to_Number(sample_weights[np.logical_and(~positives, trues)],
                            errors)

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    return fpr, tpr, thresholds


def roc_curve_data(evaluation_type: str,
                   true_labels: np.ndarray,
                   model_output: np.ndarray,
                   class_names: list = None,
                   thresholds: np.ndarray = None,
                   sample_weights: np.ndarray = None,
                   errors: bool = True,
                   *args,
                   output_length: int = 10 + 1,
                   ) -> dict:
    """
    Compute Receiver operating characteristic (ROC) values givin the nodes outputs and the
    true labels for a multi-class classification

    Args:
        evaluation_type (str): type of evaluation. Valid keys are \"OvO\" (One vs One) or \"OvR\" (One vs Rest).
        For a binary classification choose \"OvR\" or the dedicated function `binary_roc_data`
        true_labels (np.array): an `Array` with the true labels of the events. If a boolean array is givin
        model_output (np.array): output of the model with propability predictions for each event.
        class_names (np.array): name for the givin classes. Should be givin in the same order as the column.
        If not specified the index of the column will be used instead
        thresholds (np.array, optional): array with custom thresholds, at which the ROC curve points shall
        be calculated. If specified, this will overwrite the parameter `output_points`,
        else a linear spacewill be created with `output_points` entries. Defaults to None.
        sample_weights (np.array): weights of the events
        errors (bool): calculate errors of the ROC entries. Defaults to True.
        output_length (int): number of points generated for the ROC curve
    """

    def one_vs_rest(names):
        result = {}
        for ind, cls_name in enumerate(names):
            positiv_inputs = model_output[:, ind]
            fpr, tpr, th = binary_roc_data(true_labels=(true_labels == ind),
                                           model_output_positive=positiv_inputs,
                                           sample_weights=sample_weights,
                                           *args,
                                           thresholds=thresholds,
                                           errors=errors,
                                           output_length=output_length)
            result[f"{cls_name}_vs_rest"] = {"fpr": fpr,
                                             "tpr": tpr,
                                             "thresholds": th}
        return result

    def one_vs_one(names):
        result = {}
        for pos_ind, cls_name in enumerate(names):
            for neg_ind, cls_name2 in enumerate(names):
                if (pos_ind == neg_ind):
                    continue

                # Event selection masks only for the 2 classes analysed
                inputs_mask = np.logical_or(true_labels == pos_ind,
                                            true_labels == neg_ind)
                select_input = model_output[inputs_mask]
                select_labels = true_labels[inputs_mask]
                select_weights = None if sample_weights is None else sample_weights[inputs_mask]

                positiv_inputs = select_input[:, pos_ind]
                fpr, tpr, th = binary_roc_data(true_labels=(select_labels == pos_ind),
                                               model_output_positive=positiv_inputs,
                                               sample_weights=select_weights,
                                               *args,
                                               thresholds=thresholds,
                                               errors=errors,
                                               output_length=output_length)
                result[f"{cls_name}_vs_{cls_name2}"] = {"fpr": fpr,
                                                        "tpr": tpr,
                                                        "thresholds": th}
        return result

    is_input_valid(true_labels, model_output)

    # reshape in case only predictions for the positive class are givin
    if model_output.ndim != 2:
        model_output = model_output.reshape((model_output.size, 1))

    # Generate class names if not givin
    if class_names is None:
        class_names = list(range(model_output.shape[1]))

    assert (len(class_names) == model_output.shape[1]), (
        "Number of givin class names does not match the number of output nodes in the `model_output` argument!")

    # Cast trues labels to class numers
    if true_labels.dtype.name == "bool":
        true_labels = np.logical_not(true_labels).astype(dtype=np.int32)

    # Map true labels to integers if needed
    if "int" not in true_labels.dtype.name:
        for ind, name in enumerate(class_names):
            true_labels = np.where(true_labels == name, ind, true_labels)

    # Choose the evaluation type
    if (evaluation_type == "OvO"):
        return one_vs_one(class_names)
    elif (evaluation_type == "OvR"):
        return one_vs_rest(class_names)
    else:
        raise ValueError("Illeagal Argument! Evaluation Type can only be choosen as \"OvO\" (One vs One) \
                         or \"OvR\" (One vs Rest)")


def binary_auc_score(fpr: list, tpr: list, *args) -> Number or np.float64:
    """Calculates the Area Under the Curve (AUC) for givin False Positive Rate (fpr) and True Positive Rate (tpr)

    Args:
        fpr (list): False Positive Rate
        tpr (list): True Positive Rate

    Raises:
        AssertionError: If `tpr` or `fpr` is empty
        AssertionError: If the length of `tpr` does not match the length of `fpr`
        ValueError: If `fpr` is not monoton

    Returns:
        Number or np.float64: If the givin list contain scinum.Number a Number instance is returend
        with the calculated error on the auc score. Else a numpy.float64 is returend.
    """

    assert (fpr and tpr), (
        "Neither `tpr` nor `fpr` can be an empty list!")
    assert (len(fpr) == len(tpr)), (
        "Mismatch in the list length of tpr and fpr!")

    sign = 1
    if np.any(np.diff(fpr) < 0):
        if np.all(np.diff(fpr) <= 0):
            sign = -1
        else:
            raise ValueError("x is neither increasing nor decreasing : {}.".format(fpr))

    return sign * np.trapz(tpr, fpr)


def mdim_auc_score(inputs: dict, *args) -> dict:
    """Calculates the Area Under the Curve (AUC) for givin dictionary containing the False Positive Rate (fpr)
    and True Positive Rate (tpr) from a multi-dimentional ROC curve

    Args:
        inputs (dict): _description_

    Raises:
        AssertionError: If `inputs` is an empty dictionary

    Returns:
        dict: dictionary containing the calculated AUC scores.
    """

    assert inputs, "`inputs` is empty!"
    result = {}
    for cls_name, values in inputs.items():
        result[cls_name] = binary_auc_score(fpr=values["fpr"],
                                            tpr=values["tpr"])

    return result
