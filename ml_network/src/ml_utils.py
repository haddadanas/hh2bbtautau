from __future__ import annotations

from collections import OrderedDict
import copy
from functools import partial
from typing import Callable

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore

import matplotlib.pyplot as plt

from ml_network.src.utils import add_metrics_to_log, get_loader, log_to_message, ProgressBar, get_device, make_dir

device = get_device()

DEFAULT_LOSS = CrossEntropyLoss(reduction="none")
DEFAULT_OPTIMIZER = partial(SGD, lr=0.001, momentum=0.9)


class Fitting:

    def __init__(self, model: Module, device: str = get_device()) -> None:
        model.to(device)
        self.model = model
        self.device = device
        self.path = make_dir(f"{self.model.save_path}/{self.model.model_name}_logs")

    def fit(
        self,
        training_data: dict,
        batch_size=32,
        epochs=1,
        verbose=1,
        validation_data=None,
        use_weights=False,
        shuffle=True,
        initial_epoch=0,
        seed=None,
        loss=DEFAULT_LOSS,
        optimizer=DEFAULT_OPTIMIZER,
        metrics=None,
        plots: None | list[MLPLotting] = None,
        tensorboard: bool = True,
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
            validation_data: dictionary containing validation data.
            use_weights: boolean, whether to use weights in the loss calculation.
            shuffle: boolean, whether to shuffle the training data
            before each epoch.
            initial_epoch: epoch at which to start training
            (useful for resuming a previous training run)
            seed: random seed.
            loss: training loss function.
            optimizer: training optimizer.
            metrics: list of functions with signatures `metric(y_pred, y_true)`
            where y_true and y_pred are both Tensors.
            plots: list of MLPLotting objects for visualizing metrics.
            tensorboard: boolean, whether to use TensorBoard for logging.
            **kwargs: additional arguments.

        # Returns
            list of OrderedDicts with training metrics.
        """
        if seed and seed >= 0:
            torch.manual_seed(seed)

        if tensorboard:
            self._tensorboard_setup()
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
        # torch.autograd.detect_anomaly(True)

        # define usefull functions
        loss_func = self._build_loss_function(loss, use_weights)
        split_data = self._split_data_tuple(use_weights)

        for t in range(initial_epoch, epochs):
            if verbose:
                print("Epoch {0} / {1}".format(t + 1, epochs))
                # Setup logger
                pb = ProgressBar(len(train_data))

            log = OrderedDict()
            epoch_loss = 0.0
            self.model.train()
            # Run batches
            for batch_i, batch_data in enumerate(train_data):
                # Get batch data
                X_embed_batch, X_batch, Y_batch, W_batch = split_data(batch_data)
                # Backprop
                opt.zero_grad()
                Y_batch_pred = self.model(X_embed_batch, X_batch)
                batch_loss = loss_func(Y_batch_pred, Y_batch, W_batch)
                batch_loss.backward()
                opt.step()
                # Update status
                epoch_loss += batch_loss.item()
                log['loss'] = float(epoch_loss) / (batch_i + 1)
                if verbose:
                    pb.bar(batch_i, log_to_message(log))

            # Run metrics and plots
            if metrics or plots:
                Y_train = train_data.dataset.get_target()  # type: ignore
                y_train_pred, _ = self._predict(train_data, batch_size * 2)
            if metrics:
                add_metrics_to_log(log, metrics, y_train_pred, Y_train)
            if plots:
                for plot in plots:
                    plot.update(y_train_pred, Y_train, mode="train")

            # Run validation
            if valid_data:
                Y_val = valid_data.dataset.get_target()  # type: ignore
                Y_val_pred, val_loss = self._predict(
                    valid_data, batch_size * 2, loss_func=loss_func, use_weights=use_weights,
                )
                log['val_loss'] = val_loss
                if metrics:
                    add_metrics_to_log(log, metrics, Y_val_pred, Y_val, 'val_')
                if plots:
                    for plot in plots:
                        plot.update(Y_val_pred, Y_val, mode="valid")

            if tensorboard:
                for key, value in log.items():
                    self.writer.add_scalar(key, value, t)
                if plots:
                    for plot in plots:
                        self.writer.add_figure(plot.title, copy.deepcopy(plot.fig), t)

            logs.append(log)
            if verbose:
                pb.close(log_to_message(log))
        if tensorboard:
            self.writer.flush()
        return logs

    def _predict(self, dataloader, batch_size=32, loss_func=None, use_weights: bool = False) -> tuple[Tensor, float]:
        self.model.eval()
        r, n = 0, len(dataloader.dataset)
        loss = 0.0
        # Rebuild DataLoader with turned off shuffling to keep the order
        _dataloader = DataLoader(dataloader.dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch_data in _dataloader:
                # Predict on batch
                embed_batch, num_batch = batch_data[0]
                X_embed_batch = [Variable(embed) for embed in embed_batch]
                X_num_batch = Variable(num_batch)
                y_batch_pred = self.model(X_embed_batch, X_num_batch).data

                # Calculate loss
                if loss_func:
                    y_batch = Variable(batch_data[1])
                    w_batch = None if not use_weights else Variable(batch_data[2])
                    batch_loss = loss_func(y_batch_pred, y_batch, w_batch)
                    loss += batch_loss.item()

                # Infer prediction shape
                if r == 0:
                    y_pred = torch.zeros((n,) + y_batch_pred.size()[1:], device=self.device)
                # Add to prediction tensor
                y_pred[r: min(n, r + batch_size)] = y_batch_pred
                r += batch_size

        return y_pred, loss / len(_dataloader)

    def _build_loss_function(self, loss: Callable, use_weights: bool = False) -> Callable:
        if use_weights:
            def weighted_loss(y_pred, y_true, weight):
                return torch.mean(loss(y_pred, y_true) * weight)
            return weighted_loss
        return lambda y_pred, y_true, weight: loss(y_pred, y_true)

    def _split_data_tuple(self, use_weights: bool) -> Callable:
        if use_weights:
            return lambda data: (
                data[0][0], data[0][1].requires_grad_(True), data[1].requires_grad_(True), data[2].requires_grad_(True)
            )
        return lambda data: (data[0][0], data[0][1].requires_grad_(True), data[1].requires_grad_(True), None)

    def _tensorboard_setup(self) -> None:
        self.writer = SummaryWriter(self.path)
        print(f"Tensorboard logs are saved to {self.path}")

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
        with torch.no_grad():
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

    def save_model(self) -> None:
        path = f"{self.path}/model.pt"
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def save_logs(self, logs: list[dict], suffix: str = "logs") -> None:
        import pickle
        path = f"{self.path}/{suffix}.pickle"
        with open(path, "wb") as f:
            pickle.dump(logs, f)
        print(f"Logs saved to {path}")

    def save_config(self, logs: dict, suffix: str = "config") -> None:
        import json
        path = f"{self.path}/{suffix}.json"
        str_dict = {k: (str(v) if isinstance(v, Callable) else v) for k, v in logs.items()}
        with open(path, "w") as f:
            json.dump(str_dict, f)
        print(f"JSON saved to {path}")


class MLPLotting:
    """Plotting class for Training and Validation metrics"""
    def __init__(
            self,
            title: str,
            xlabel: str,
            ylabel: str,
            plot_func: Callable,
            log_axis: bool = False,
            validation: bool = False,
            data_metric: str = "max",
    ) -> None:
        """
        Initializes the plotting utility with the given parameters.
        Args:
            title (str): The title of the plot.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            plot_func (Callable): The function used to generate the plot.
                The function should have the signature
                `plot_func(y_true: Tensor, y_pred: Tensor, ax: Axes) -> result array, metric score`.
            validation (bool, optional): If True, creates an additional subplot for validation. Defaults to False.
            data_metric (str, optional): The metric to use for saving the best data. Either "max" or "min".
                Defaults to "max".
        """
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.log_axis = log_axis
        self.fig, axs = plt.subplots(
            1,
            2 if validation else 1,
            figsize=(12 if validation else 6, 6),
            dpi=150,
            sharey=True,
        )
        if validation:
            self.ax, self.ax_val = axs
            self._set_up_axes(self.ax)
            self._set_up_axes(self.ax_val, validation=True)
        else:
            self.ax = axs
            self._set_up_axes(self.ax)
        self.plot_func = plot_func
        self.data_metric = data_metric
        self.prev_train = None
        self.prev_valid = None
        self.best_train = None
        self.best_valid = None

        self.fig.tight_layout()
        self.fig.show()

    def _set_up_axes(self, ax, validation=False):
        ax.set_title(f"{self.title} {'(Validation)' if validation else ''}")
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_yscale("log" if self.log_axis else "linear")
        ax.grid()

    def _get_best_data(self, mode, x, y, metric_score: dict) -> dict:
        best = getattr(self, f"best_{mode}")
        if self.data_metric == "max":
            if best is None or metric_score > best["metric"]:
                return {"x": x, "y": y, "metric": metric_score}
        elif self.data_metric == "min":
            if best is None or metric_score < best["metric"]:
                return {"x": x, "y": y, "metric": metric_score}
        return best  # type: ignore

    def _update_state(self, mode, ax, x, y, metric_score):
        prev = getattr(self, f"prev_{mode}")
        if prev is not None:
            ax.plot(
                prev["x"],
                prev["y"], "k--",
                alpha=0.5,
                label=f"last epoch AUC {prev['metric']:.3f}",
            )
        setattr(self, f"prev_{mode}", {"x": x, "y": y, "metric": metric_score})
        setattr(self, f"best_{mode}", self._get_best_data(mode, x, y, metric_score))
        best = getattr(self, f"best_{mode}")
        if best is not None:
            ax.plot(
                best["x"],
                best["y"],
                "g--",
                alpha=0.7,
                label=f"Best: {best['metric']:.3f}",
            )

    def update(self, y_pred: Tensor, y_true: Tensor, mode: str = "train") -> None:
        """
        Updates the plot with the given data.
        Args:
            data (dict): The data to plot.
            epoch (int): The epoch number.
            best (bool, optional): If True, saves the data as the best data. Defaults to False.
        """
        if mode not in ["train", "valid"]:
            raise ValueError(f"mode must be either 'train' or 'valid', got {mode}")

        is_validation = mode == "valid"
        ax = self.ax_val if is_validation else self.ax
        ax.cla()
        self._set_up_axes(ax, validation=is_validation)
        x, y, metric_score = self.plot_func(y_pred, y_true, ax)
        self._update_state(mode, ax, x, y, metric_score)

        ax.legend(loc="lower left")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# torch scripts
@torch.jit.script
def selection_efficiency(y_pred: Tensor, y_true: Tensor,) -> float:
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
