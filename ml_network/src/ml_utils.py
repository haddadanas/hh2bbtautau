from __future__ import annotations

__all__ = ["TorchFitting", "MLPLotting"]

from collections import OrderedDict
import copy
from functools import partial
from typing import Callable

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD

import matplotlib.pyplot as plt

from ml_network.src.utils import add_metrics_to_log, log_to_message, log_batch_loss, ProgressBar, get_device, make_dir
from ml_network.src.torch_util import CompositeDataLoader, EvaluationDataLoader

DEVICE = get_device()

DEFAULT_LOSS = CrossEntropyLoss(reduction="mean")
DEFAULT_OPTIMIZER = partial(SGD, lr=0.001, momentum=0.9)


class TorchFitting:
    """Fitting class for PyTorch models"""

    def __init__(
        self,
        model: Module,
        device: str = DEVICE,
        trace_func=None,
        verbose: bool = True,
        *args,
        save_logs: bool = True,
        tensorboard: bool = True,
        early_stopping: bool = False,
        patience: int = 7,
        delta: float = 0,
        cleanup_on_exception: bool = False,
    ) -> None:
        """
        Args:
            model (Module): PyTorch model to train.
            device (str): Device to run the model on.
            trace_func (function): Function for tracing logs. Default is print.
            verbose (bool): Whether to print training logs.
                            Default: True
            *args: Additional arguments passed.
            save_logs (bool): Whether to save training logs.
                            Default: True
            early_stopping (bool): Whether to use early stopping.
                            Default: False
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            **kwargs: Additional arguments.
        """
        # Instance variables
        self._model: Module = model.to(device)
        self.device: str = device
        self._save_logs: bool = save_logs
        self._tensorboard: bool = tensorboard
        self.trace_func: Callable = print if not trace_func else trace_func
        self._verbose: bool = verbose
        self._cleanup: bool = cleanup_on_exception

        # Early stopping setup
        self.is_early_stopping: bool = early_stopping
        if self.is_early_stopping:
            self._set_early_stopping(patience=patience, delta=delta)

        # Training setup
        self.path: str = make_dir(f"{self._model.save_path}/{self._model.name}_logs", add_time=True)
        if self._tensorboard:
            self._set_tensorboard()

        # TODO something to save config / Logs / Setup

    def _set_tensorboard(self) -> None:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
        self.writer = SummaryWriter(self.path)
        self.trace_func(f"Tensorboard logs are saved to {self.path}")

    def _set_early_stopping(self, patience: int = 7, delta: float = 0.0) -> None:
        self.patience: int = patience
        self.delta: float = delta
        self.counter: int = 0
        self.best_val_loss: float | None = None
        self.early_stop: bool = False
        self.val_loss_min: float = float("inf")
        self.best_state_dict: dict | None = None
        self.trace_func(f"Early stopping is set with patience {patience} and delta {delta}")

    def _save_log(self, logs: list | dict, suffix: str = "logs") -> None:
        if not self._save_logs:
            return
        if isinstance(logs, dict):
            import json as save_lib
            ext = "json"
            logs = {k: (str(v) if isinstance(v, Callable) else v) for k, v in logs.items()}
        else:
            import pickle as save_lib
            ext = "pickle"
        path = f"{self.path}/{suffix}.{ext}"
        with open(path, "wb" if ext == "pickle" else "w") as f:
            save_lib.dump(logs, f)
        self.trace_func(f"{suffix}{ext} saved to {path}")

    def _predict(self, dataloader, loss_func=None, class_weight: float = 0) -> tuple[Tensor, float]:
        # define useful variables
        batch_size = dataloader.source.batch_size
        r, n = 0, len(dataloader.map_fn.dataset)
        n_batches = 0
        loss = 0.0
        class_weight = class_weight or 1

        # ensure dataloader is reseted before iterating
        dataloader.reset()

        # ensure model is in evaluation mode
        self._model.eval()

        with torch.no_grad():
            for X_embed_batch, X_num_batch, Y_batch in dataloader:
                # Predict on batch
                y_batch_pred = self._model(X_embed_batch, X_num_batch)

                # Calculate loss
                if loss_func:
                    Y_batch = Y_batch["categorical_target"]
                    batch_loss = loss_func(y_batch_pred, Y_batch)
                    loss += batch_loss.item()

                # Infer prediction shape
                if r == 0:
                    y_pred = torch.zeros((n,) + y_batch_pred.size()[1:], device=self.device)
                # Add to prediction tensor
                y_pred[r: min(n, r + batch_size)] = y_batch_pred
                r += batch_size
                n_batches += 1

        return y_pred, loss * class_weight / n_batches  # TODO devide by or n_batches

    def _get_evaluate_func(
            self,
            metrics: list[Callable] | None,
            plots: list[MLPLotting] | None,
            eval_dataloader: EvaluationDataLoader,
            loss_callable: Callable | None = None,
            validation: bool = False,
            use_weights: bool = False,
    ) -> Callable:
        def _concat_values(d: dict) -> tuple[Tensor, float]:
            tensors = []
            losses = []
            for value in d.values():
                if isinstance(value, tuple):
                    tensors.append(value[0])
                    losses.append(value[1])
                else:
                    tensors.append(value)
            return torch.cat(tensors), sum(losses)
        prefix = "val_" if validation else ""
        true_values, _ = _concat_values(eval_dataloader.get_targets())
        dataloader_dict = eval_dataloader.data_loader
        weight_map = eval_dataloader.weight_map if use_weights else {}

        def _evaluate(epoch, log):
            if not (metrics or plots):
                return
            y_pred = OrderedDict({
                key: self._predict(
                    loader, loss_func=loss_callable, class_weight=weight_map.get(key, 0),
                )
                for key, loader in dataloader_dict.items()
            })
            y_pred, loss = _concat_values(y_pred)
            if loss_callable is not None:
                log[f"{prefix}loss"] = loss
            if metrics:
                add_metrics_to_log(log, metrics, y_pred, true_values, prefix=prefix)
            if plots:
                for plot in plots:
                    plot.update(y_pred, true_values, mode="valid" if validation else "train", epoch=str(epoch))
        return _evaluate

    def _check_early_stopping(self, log: dict):
        log_msg = ""
        val_loss = log.get("val_loss", float("nan"))
        # Check if validation loss is nan
        if val_loss == float("nan"):
            log["log_msg"] = "No Validation loss found / Validation loss is NaN. Ignoring this epoch."
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            log_msg = self._cache_checkpoint(val_loss)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            log_msg = self._cache_checkpoint(val_loss)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            log_msg = f"EarlyStopping counter: {self.counter} out of {self.patience}"
            if self.counter >= self.patience:
                self.early_stop = True

        log["log_msg"] = log_msg

    def _cache_checkpoint(self, val_loss):
        """Saves model to a instance variable when validation loss decreases."""
        log_msg = ""
        if self._verbose:
            log_msg = (
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})."
                "Saving model state to `self.best_state_dict` ..."
            )
        self.best_state_dict = copy.deepcopy(self._model.state_dict())
        self.val_loss_min = val_loss
        return log_msg

    def _log_parameters(self, **kwargs):
        for key, value in kwargs.items():
            self.trace_func(f"{key}: {value}")

    def save_model(self) -> None:
        path = f"{self.path}/model.pt"
        torch.save(self._model.state_dict(), path)
        self.trace_func(f"Model state dictionary saved to {path}")

    def save_best_model(self) -> None:
        path = f"{self.path}/best_model.pt"
        torch.save(self.best_state_dict, path)
        self.trace_func(f"Best model state dictionary saved to {path}")

    @property
    def model(self) -> Module:
        return self._model

    @property
    def best_model(self) -> Module:
        """returns the best model state."""
        if self.is_early_stopping:
            return_model = copy.deepcopy(self._model)
            return return_model.load_state_dict(self.best_state_dict, strict=True)
        return self._model

    def predict(self, dataloader, use_best_model: bool = False) -> Tensor:
        """Generates output predictions for the input samples.

        Computation is done in batches.

        # Arguments
            X: input data Tensor.
            batch_size: integer.

        # Returns
            prediction Tensor.
        """
        r, n = 0, len(dataloader.map_fn.dataset)
        batch_size = dataloader.source.batch_size

        # ensure dataloader is reseted before iterating
        dataloader.reset()

        model = self.best_model if use_best_model else self._model

        # Batch prediction
        model.eval()
        with torch.no_grad():
            for X_embed_batch, X_num_batch, _ in dataloader:
                # Predict on batch
                y_batch_pred = model(X_embed_batch, X_num_batch)

                # Infer prediction shape
                if r == 0:
                    y_pred = torch.zeros((n,) + y_batch_pred.size()[1:], device=self.device)
                # Add to prediction tensor
                y_pred[r: min(n, r + batch_size)] = y_batch_pred
                r += batch_size
        return y_pred

    def predict_legacy(self, X_embed, X_num, batch_size=32):
        """Generates output predictions for the input samples.

        Computation is done in batches.

        # Arguments
            X: input data Tensor.
            batch_size: integer.

        # Returns
            prediction Tensor.
        """
        from ml_network.src.utils import get_loader
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

    # @cleanup_on_exception
    def fit(
        self,
        training_data: CompositeDataLoader,
        epochs=1,
        validation_data: EvaluationDataLoader | None = None,
        use_eval_weights=True,
        loss_func=DEFAULT_LOSS,
        optimizer=DEFAULT_OPTIMIZER,
        metrics=None,
        plots: None | list[MLPLotting] = None,
        initial_epoch: int = 0,
        seed: int | None = None,
        **kwargs,
    ):
        """Trains the model similar to Keras' .fit(...) method

        Args:
            training_data (CompositeDataLoader): DataLoader for training data.
            epochs (int): Number of epochs to train the model.
            validation_data (EvaluationDataLoader, optional): DataLoader for validation data. Defaults to None.
            use_eval_weights (bool, optional): Whether to use weights in the evaluation. Defaults to True.
            loss_func (Callable, optional): Loss function for training. Defaults to DEFAULT_LOSS.
            optimizer (Callable, optional): Optimizer for training. Defaults to DEFAULT_OPTIMIZER.
            metrics (list[Callable], optional): List of metric functions to evaluate. Defaults to None.
            plots (list[MLPLotting], optional): List of MLPLotting objects for visualizing metrics. Defaults to None.
            initial_epoch (int, optional): Epoch at which to start training. Useful for resuming a previous training run. Defaults to 0.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            **kwargs: Additional arguments.

        Returns:
            list[OrderedDict]: List of logs with training metrics for each epoch.
        """
        if seed and seed >= 0:
            torch.manual_seed(seed)
        logs = []
        verbose = self._verbose
        tensorboard = self._tensorboard
        # torch.autograd.detect_anomaly(True)

        if verbose:
            self._log_parameters(loss_func=loss_func, optimizer=optimizer, **kwargs)

        # Get DataLoader
        train_dataloader = training_data.data_loader

        # Define a Dataloader(s) w/o sampling and shuffling for evaluation (if needed)
        # and get the needed evaluation functions for both training and validation
        if metrics or plots:
            train_eval_data = EvaluationDataLoader(
                training_data.data_map,
                batch_size=training_data.batch_size * 2,
                device=self.device
            )
            eval_training = self._get_evaluate_func(metrics, plots, train_eval_data)
        if validation_data:
            eval_validation = self._get_evaluate_func(
                metrics, plots, validation_data, loss_callable=loss_func, validation=True, use_weights=use_eval_weights,
            )

        # Compile optimizer
        opt = optimizer(self._model.parameters())

        for t in range(initial_epoch, epochs):
            log = OrderedDict()
            epoch_loss = 0.0
            if verbose:
                print("Epoch {0} / {1}".format(t + 1, epochs))
                pb = ProgressBar(training_data.num_batches)

            # ensure model is in training mode
            self._model.train(True)

            # reset dataloader before looping over batches
            train_dataloader.reset()
            for batch_i, (X_embed_batch, X_batch, Y_batch) in enumerate(train_dataloader):
                Y_batch = Y_batch["categorical_target"]

                # Backpropagation
                opt.zero_grad()
                Y_batch_pred = self._model(X_embed_batch, X_batch)
                batch_loss = loss_func(Y_batch_pred, Y_batch)
                batch_loss.backward()
                opt.step()

                # Update status
                epoch_loss += batch_loss.item()
                log["loss"] = float(epoch_loss) / (batch_i + 1)
                if verbose:
                    pb.bar(batch_i, log_batch_loss(log))

            # Run metrics and plots
            if metrics or plots:
                eval_training(t, log)
            if validation_data:
                eval_validation(t, log)

            if tensorboard:
                for key, value in log.items():
                    self.writer.add_scalar(key, value, t)
                if plots:
                    for plot in plots:
                        self.writer.add_figure(plot.title, copy.deepcopy(plot.fig), t)
            self._check_early_stopping(log)
            logs.append(log)
            if verbose:
                pb.close(log_to_message(log))
            if self.early_stop:
                self.trace_func(f"Early stopping at epoch {t + 1}")
                break

        # on finish
        if tensorboard:
            self.writer.flush()
        self.trace_func("Training finished.")
        if self._save_logs:
            self._save_log(logs)
            func_config = {
                "device": self.device,
                "early_stopping": self.is_early_stopping,
                "patience": self.patience,
                "delta": self.delta,
                "epochs": epochs,
                "use_eval_weights": use_eval_weights,
                "loss_func": loss_func,
                "optimizer": optimizer,
            } | kwargs
            self._save_log(func_config, suffix="config")

        return logs


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
            silent: bool = False,
    ) -> None:
        """
        Initializes the plotting utility with the given parameters.
        Args:
            title (str): The title of the plot.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            plot_func (Callable): The function used to generate the plot.
                The function should have the signature
                `plot_func(y_true: Tensor, y_pred: Tensor, ax: Axes, epoch: str) -> result array, metric score`.
            validation (bool, optional): If True, creates an additional subplot for validation. Defaults to False.
            data_metric (str, optional): The metric to use for saving the best data. Either "max" or "min".
                Defaults to "max".
        """
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.log_axis = log_axis
        self.silent = silent
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
        if not self.silent:
            self.fig.show()

    def _set_up_axes(self, ax, validation=False):
        ax.set_title(f"{self.title} {'(Validation)' if validation else ''}")
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_yscale("log" if self.log_axis else "linear")
        ax.grid()

    def _get_best_data(self, mode, x, y, metric_score: dict, epoch) -> dict:
        best = getattr(self, f"best_{mode}")
        if self.data_metric == "max":
            if best is None or metric_score > best["metric"]:
                return {"x": x, "y": y, "metric": metric_score, "epoch": epoch}
        elif self.data_metric == "min":
            if best is None or metric_score < best["metric"]:
                return {"x": x, "y": y, "metric": metric_score, "epoch": epoch}
        return best  # type: ignore

    def _update_state(self, mode, ax, x, y, metric_score, epoch):
        prev = getattr(self, f"prev_{mode}")
        if prev is not None:
            ax.plot(
                prev["x"],
                prev["y"], "k--",
                alpha=0.5,
                label=f"last epoch AUC {prev['metric']:.3f}",
            )
        setattr(self, f"prev_{mode}", {"x": x, "y": y, "metric": metric_score})
        setattr(self, f"best_{mode}", self._get_best_data(mode, x, y, metric_score, epoch))
        best = getattr(self, f"best_{mode}")
        if best is not None:
            ax.plot(
                best["x"],
                best["y"],
                "g--",
                alpha=0.7,
                label=f"Best: {best['metric']:.3f} @ ep {best['epoch']}",
            )

    def update(self, y_pred: Tensor, y_true: Tensor, mode: str = "train", epoch: str = "") -> None:
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
        x, y, metric_score = self.plot_func(y_pred, y_true, ax, epoch)
        self._update_state(mode, ax, x, y, metric_score, epoch)

        ax.legend(loc="lower left")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
