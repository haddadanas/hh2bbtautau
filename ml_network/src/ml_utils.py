from __future__ import annotations

from collections import OrderedDict
from functools import partial

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD

from src.utils import add_metrics_to_log, get_loader, log_to_message, ProgressBar, get_device

device = get_device()

DEFAULT_LOSS = CrossEntropyLoss(reduction="none")
DEFAULT_OPTIMIZER = partial(SGD, lr=0.001, momentum=0.9)


class Fitting:

    def __init__(self, model: Module, device: str = get_device()) -> None:
        model.to(device)
        self.model = model
        self.device = device

    def fit(
        self,
        training_data: dict,
        batch_size=32,
        epochs=1,
        verbose=1,
        validation_data=None,
        shuffle=True,
        initial_epoch=0,
        seed=None,
        loss=DEFAULT_LOSS,
        optimizer=DEFAULT_OPTIMIZER,
        metrics=None,
        **kwargs,
    ):
        """Trains the model similar to Keras' .fit(...) method

        # Arguments
            training_data: dictionary containing training data.
            batch_size: integer. Number of samples per gradient update.
            epochs: integer, the number of times to iterate
            over the training data arrays.
            verbose: 0, 1. Verbosity mode.
            0 = silent, 1 = verbose.
            validation_data: tuple (X_val, y_val, weight_val) on which to evaluate
            the loss and any model metrics
            at the end of each epoch. The model will not
            be trained on this data.
            shuffle: boolean, whether to shuffle the training data
            before each epoch.
            initial_epoch: epoch at which to start training
            (useful for resuming a previous training run)
            seed: random seed.
            loss: training loss
            optimizer: training optimizer
            metrics: list of functions with signatures `metric(y_true, y_pred)`
            where y_true and y_pred are both Tensors

        # Returns
            list of OrderedDicts with training metrics
        """
        if seed and seed >= 0:
            torch.manual_seed(seed)

        # Prepare validation data
        if validation_data:
            valid_data = get_loader(**validation_data, batch_size=batch_size, shuffle=False, device=self.device)
        else:
            valid_data = None
        # Build DataLoaders
        train_data = get_loader(**training_data, batch_size=batch_size, shuffle=shuffle, device=self.device)
        # Compile optimizer
        opt = optimizer(self.model.parameters())
        # Run training loop
        logs = []
        self.model.train()
        for t in range(initial_epoch, epochs):
            if verbose:
                # Setup logger
                pb = ProgressBar(len(train_data))

                print("Epoch {0} / {1}".format(t + 1, epochs))

            log = OrderedDict()
            epoch_loss = 0.0
            # Run batches
            for batch_i, batch_data in enumerate(train_data):
                # Get batch data
                X_embed_batch = batch_data[0][0]
                X_batch = batch_data[0][1].requires_grad_(True)
                y_batch = batch_data[1].requires_grad_(True)
                weight_batch = batch_data[2].requires_grad_(True)
                # Backprop
                opt.zero_grad()
                y_batch_pred = self.model(X_embed_batch, X_batch)
                batch_loss_array = loss(y_batch_pred, y_batch) * weight_batch
                batch_loss = batch_loss_array.mean()
                batch_loss.backward()
                opt.step()
                # Update status
                epoch_loss += batch_loss.item()
                log['loss'] = float(epoch_loss) / (batch_i + 1)
                if verbose:
                    pb.bar(batch_i, log_to_message(log))
            # Run metrics
            if metrics:
                _, Y_train, _ = train_data.dataset.get_data()  # type: ignore
                y_train_pred = self._predict(train_data, batch_size)
                add_metrics_to_log(log, metrics, Y_train, y_train_pred)
            if valid_data:
                _, Y_val, weight_val = valid_data.dataset.get_data()  # type: ignore
                y_val_pred = self._predict(valid_data, batch_size)
                val_loss_arr = loss(Variable(y_val_pred), Variable(Y_val)) * Variable(weight_val)
                val_loss = val_loss_arr.mean()
                log['val_loss'] = val_loss.item()
                if metrics:
                    add_metrics_to_log(log, metrics, Y_val, y_val_pred, 'val_')
            logs.append(log)
            if verbose:
                pb.close(log_to_message(log))
        return logs

    def _predict(self, dataloader, batch_size=32):
        self.model.eval()
        r, n = 0, len(dataloader.dataset)
        for batch_data in dataloader:
            # Predict on batch
            embed_batch, num_batch = batch_data[0]
            X_embed_batch = [Variable(embed) for embed in embed_batch]
            X_num_batch = Variable(num_batch)
            y_batch_pred = self.model(X_embed_batch, X_num_batch).data
            # Infer prediction shape
            if r == 0:
                y_pred = torch.zeros((n,) + y_batch_pred.size()[1:], device=self.device)
            # Add to prediction tensor
            y_pred[r: min(n, r + batch_size)] = y_batch_pred
            r += batch_size
        return y_pred

    def predict(self, X_embed, X_num, batch_size=32):
        """Generates output predictions for the input samples.

        Computation is done in batches.

        # Arguments
            X: input data Tensor.
            batch_size: integer.

        # Returns
            prediction Tensor.
        """
        # Build DataLoader
        data = get_loader(inp_embed=X_embed, inp_num=X_num, batch_size=batch_size, shuffle=False, device=self.device)
        # Batch prediction
        self.model.eval()
        r, n = 0, X_num.size()[0]
        for batch_data in data:
            # Predict on batch
            embed_batch, num_batch = batch_data[0]
            X_embed_batch = [Variable(embed) for embed in embed_batch]
            X_num_batch = Variable(num_batch)
            y_batch_pred = self.model(X_embed_batch, X_num_batch).data
            # Infer prediction shape
            if r == 0:
                y_pred = torch.zeros((n,) + y_batch_pred.size()[1:], device=self.device)
            # Add to prediction tensor
            y_pred[r: min(n, r + batch_size)] = y_batch_pred
            r += batch_size
        return y_pred


# torch scripts
@torch.jit.script
def selection_efficiency(y_true: Tensor, y_pred: Tensor) -> float:
    threshold = 0.5  # torch.linspace(0, 1, 101)
    # selection_efficiencies = torch.zeros(101)

    if y_pred.ndim > 1:
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
def signal_acceptance(y_true: Tensor, y_pred: Tensor) -> float:
    threshold = 0.5  # torch.linspace(0, 1, 101)
    # signal_acceptances = torch.zeros(101)

    if y_pred.ndim > 1:
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
def signal_purity(y_true: Tensor, y_pred: Tensor) -> float:
    threshold = 0.5  # torch.linspace(0, 1, 101)
    # signal_purities = torch.zeros(101)

    if y_pred.ndim > 1:
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
def background_rejection(y_true: Tensor, y_pred: Tensor) -> Tensor:
    thresholds = torch.linspace(0, 1, 101)
    background_rejections = torch.zeros(101)

    if y_pred.ndim > 1:
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
def roc_curve(y_true: Tensor, y_pred: Tensor) -> tuple[Tensor, Tensor]:
    thresholds = torch.linspace(0, 1, 101)
    tpr = torch.zeros(101)
    fpr = torch.zeros(101)

    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_true.size() != y_pred.size():
        raise ValueError("y_true and y_pred must have the same shape")

    signal_mask = y_true.to(torch.bool)
    background_mask = ~y_true.to(torch.bool)
    for i, threshold in enumerate(thresholds):
        mask = y_pred > threshold
        signal_selection_mask = mask[signal_mask]
        background_rejection_mask = ~mask[background_mask]
        tpr[i] = torch.mean(signal_selection_mask.float(), dim=0)
        fpr[i] = torch.mean(background_rejection_mask.float(), dim=0)

    return tpr, fpr


@torch.jit.script
def roc_curve_auc(y_true: Tensor, y_pred: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    thresholds = torch.linspace(0, 1, 101)
    tpr = torch.zeros(101)
    fpr = torch.zeros(101)

    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_true.size() != y_pred.size():
        raise ValueError("y_true and y_pred must have the same shape")

    signal_mask = y_true.to(torch.bool)
    background_mask = ~y_true.to(torch.bool)
    for i, threshold in enumerate(thresholds):
        mask = y_pred > threshold
        signal_selection_mask = mask[signal_mask]
        background_rejection_mask = ~mask[background_mask]
        tpr[i] = torch.mean(signal_selection_mask.float(), dim=0)
        fpr[i] = torch.mean(background_rejection_mask.float(), dim=0)

    # calculate the area under the curve
    tpr_diff = tpr[1:] - tpr[:-1]
    fpr_sum = fpr[1:] + fpr[:-1]
    auc = torch.abs(torch.sum(tpr_diff * fpr_sum) / 2)

    return tpr, fpr, auc
