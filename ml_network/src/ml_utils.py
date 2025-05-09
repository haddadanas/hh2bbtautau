from __future__ import annotations

__all__ = ["TorchFitting", "MLPLotting"]
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import copy
from functools import partial
from typing import Any, Callable, Iterator

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
import matplotlib.pyplot as plt

from ml_network.src.utils import add_metrics_to_log, ProgressBar, get_device, make_dir
from ml_network.src.torch_src import CompositeDataLoader, EvaluationDataLoader

DEVICE = get_device()

DEFAULT_LOSS = CrossEntropyLoss(reduction="mean")
DEFAULT_OPTIMIZER = partial(torch.optim.Adam, lr=0.001)


class MovingAverage:
    """Computes the moving average of a given value over a specified window size."""

    def __init__(self, window_size: int = 3, weights: list | None = None) -> None:
        self.window_size = window_size
        self.values = []
        self.weights = weights or [1.0] * window_size
        if len(self.weights) != window_size:
            raise ValueError(f"Weights length must be equal to window size ({window_size}).")
        self.average: float | None = None

    def update(self, value: float) -> float:
        """Updates the moving average with a new value."""
        self.values.insert(0, value)
        # Remove the oldest value if the window size is exceeded
        if len(self.values) > self.window_size:
            self.values.pop(-1)
        used_weight = self.weights[: len(self.values)]
        self.average = sum(v * w for v, w in zip(self.values, used_weight)) / sum(used_weight)
        return self.average


class AbstractMeta(ABCMeta):
    """Abstract metaclass that enforces both abstract methods and custom metaclass behavior."""

    def __call__(cls, *args, **kwargs) -> Callable[[TorchFitting], "BaseFitting"]:
        """
        Enforces that the class is instantiated via a factory pattern.
        Returns a callable that takes a TorchFitting object and returns an instance.
        """
        def create_instance(obj: TorchFitting) -> "BaseFitting":
            if obj.__class__.__name__ != "TorchFitting":
                raise TypeError(f"Expected a TorchFitting object, got {type(obj).__name__}")
            return type.__call__(cls, obj=obj, *args, **kwargs)

        return create_instance


class BaseFitting(metaclass=AbstractMeta):
    """Abstract base class for general fitting components."""

    def __init__(self, *args, obj: TorchFitting, **kwargs) -> None:
        self._fitting: TorchFitting = obj
        self._trace_func = obj.trace_func

    def log(self) -> None:
        """Logs a message using the provided trace function."""
        self._trace_func(self._log())

    @abstractmethod
    def _log(self) -> str:
        """Returns a log message."""
        pass

    @abstractmethod
    def __call__(self, log: dict, *args, **kwargs) -> None:
        """Executes the main functionality of the fitting component."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Returns a string representation of the fitting component."""
        pass


class EmptyClass(BaseFitting):
    """Empty class for PyTorch models"""

    def __init__(self, *args, obj) -> None:
        pass

    def __call__(self, log: dict, *args, **kwargs) -> None:
        pass

    def __str__(self) -> str:
        return "No fitting component is set."

    def __bool__(self) -> bool:
        return False

    def _log(self) -> str:
        return ""


class TorchBoard(BaseFitting):
    """Tensorboard class for PyTorch models"""
    def __init__(
        self,
        save_figs: bool = True,
        *args,
        obj: TorchFitting,
    ) -> None:
        """
        Args:
            path (str): Path to save the Tensorboard logs.
            initial_epoch (int): Initial epoch for logging.
            save_figs (bool): Whether to save figures in Tensorboard.
            trace_func (Callable): Function for tracing logs. Default is print.
        """
        from torch.utils.tensorboard import SummaryWriter  # type: ignore

        super(TorchBoard, self).__init__(*args, obj=obj)
        self.writer: SummaryWriter = SummaryWriter(self._fitting.path)
        self._save_figs: bool = save_figs
        self._current_epoch: int = 0
        self._plots: list[MLPLotting] = self._fitting._plots

    def __call__(self, log: dict, *args, **kwargs) -> None:
        for key, value in log.items():
            self.writer.add_scalar(key, value, self._current_epoch)
            if self._save_figs:
                for plot in self._plots:
                    self.writer.add_figure(plot.title, copy.deepcopy(plot.fig), self._current_epoch)
        self._current_epoch += 1

    def __str__(self) -> str:
        return f"Tensorboard logs are saved to {self.writer.log_dir}"

    def set_epoch(self, epoch: int) -> None:
        """Sets the current epoch for logging."""
        self._current_epoch = epoch

    def flush(self) -> None:
        """Flushes the Tensorboard writer."""
        self.writer.flush()
        self.writer.close()

    def _log(self) -> str:
        """Logs the current state of the Tensorboard writer."""
        return f"Tensorboard logs are saved to {self.writer.log_dir}"


class EarlyStopping(BaseFitting):
    """Early stopping class for PyTorch models"""
    def __init__(
        self,
        patience: int = 7,
        delta: float = 0.0,
        mode: str = "min",
        metric: str = "val_loss",
        *args,
        obj: TorchFitting,
    ) -> None:
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): Mode for early stopping, either 'min' for minimizing loss or 'max' for maximizing metrics.
            metric (str): Metric to monitor for early stopping.
        """
        super(EarlyStopping, self).__init__(*args, obj=obj)

        self.patience: int = patience
        self.delta: float = delta
        self.factor: float = 1.0 if mode == "min" else -1.0
        if mode not in ["min", "max"]:
            raise ValueError(f"Mode must be either 'min' or 'max', got {mode}")

        self.counter: int = 0
        self.early_stop: bool = False
        self.score_min: float = self.factor * float("inf")
        self.score_average: MovingAverage = MovingAverage(window_size=3, weights=[0.6, 0.3, 0.1])
        self._best_score: float | None = self.score_average.average
        self.best_state_dict: dict[str, Any] = copy.deepcopy(self._fitting._model.state_dict())
        self.metric: str = metric
        self.get_metric: Callable = lambda log: log.get(metric, float("nan"))

    def __bool__(self) -> bool:
        """Returns True if early stopping is triggered."""
        return self.early_stop

    def __call__(self, log: dict, *args, **kwargs) -> None:
        log_msg = ""
        score = self.factor * self.get_metric(log)
        # Check if validation loss is nan
        if score == float("nan"):
            log["log_msg"] = f"No {self.metric} found / {self.metric} is NaN. Ignoring this epoch."
            return

        if self._best_score is None:
            self._best_score = self.score_average.update(score)
            log_msg = self._cache_checkpoint()
        elif score < self._best_score - self.delta:
            # Significant improvement detected
            self._best_score = score
            log_msg = self._cache_checkpoint()
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            log_msg = f"EarlyStopping counter: {self.counter} out of {self.patience}"
            if self.counter >= self.patience:
                self.early_stop = True

        log["log_msg"] = log_msg

    def __str__(self) -> str:
        return f"EarlyStopping(patience={self.patience}, delta={self.delta}, mode={self.factor}, metric={self.metric})"

    @property
    def best_score(self) -> float | None:
        """Returns the best score."""
        if self._best_score is None:
            return None
        return self.factor * self._best_score

    def _cache_checkpoint(self):
        """Saves model to a instance variable when validation loss decreases."""
        log_msg = ""
        if self._fitting._verbose:
            log_msg = (
                f"{self.metric} (moving average) decreased ({self.score_min:.6f} --> {self._best_score:.6f})."
                "Saving model state to `self.best_state_dict` ..."
            )
        self.best_state_dict = copy.deepcopy(self._fitting._model.state_dict())
        self.score_min = self._best_score if self._best_score is not None else float("inf")
        return log_msg

    def _log(self) -> str:
        """Logs the current state of the early stopping."""
        return f"Early stopping is set with patience {self.patience} and delta {self.delta}"


class TorchFitting:
    """Fitting class for PyTorch models"""

    def __init__(
        self,
        model: Module,
        *args,
        optimizer: Callable = DEFAULT_OPTIMIZER,
        save_logs: bool = True,
        loss_func=DEFAULT_LOSS,
        val_loss_func=DEFAULT_LOSS,
        metrics=None,
        plots: None | list[MLPLotting] = None,
        early_stopping: Callable[[TorchFitting], "BaseFitting"] | None = None,
        tensorboard: Callable[[TorchFitting], "BaseFitting"] | None = None,
        device: str = DEVICE,
        trace_func=None,
        verbose: bool = True,
        cleanup_on_exception: bool = False,
        training_mode: bool = True,
        **kwargs,
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
                            Default: 0.0
            **kwargs: Additional arguments.
        """
        # Instance variables
        self._model: Module = model.to(device)
        self.device: str = device
        self._save_logs: bool = save_logs
        self.trace_func: Callable = print if not trace_func else trace_func
        self._verbose: bool = verbose
        self._cleanup: bool = cleanup_on_exception

        # empty func
        self._empty_func = EmptyClass()(self)

        # fit components
        self._loss_func: Callable = loss_func
        self._val_loss_func: Callable = val_loss_func
        self._optimizer: Callable[[Iterator], torch.optim.Optimizer] = optimizer
        self._metrics: list[Callable] = metrics or []
        self._plots: list[MLPLotting] = plots or []
        self._evaluate_training: bool = bool(metrics or plots)

        # Training setup
        if training_mode:
            self.path: str = make_dir(f"{self._model.save_path}/{self._model.name}_logs", add_time=True)
        else:
            self._save_logs = False
            self.trace_func("Training mode is set to evaluation. No logs will be saved.")

        # Early stopping setup
        self._early_stopping: EarlyStopping | BaseFitting = (
            early_stopping(self) if early_stopping is not None else self._empty_func
        )

        # Tensorboard setup
        self._tensorboard: TorchBoard | BaseFitting = (
            tensorboard(self) if tensorboard is not None else self._empty_func
        )

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

    def _predict(self, dataloader, loss_func=None) -> tuple[Tensor, float]:
        # define useful variables
        batch_size = dataloader.source.batch_size
        r, n = 0, len(dataloader.map_fn.dataset)
        n_batches = 0
        loss = 0.0

        # ensure dataloader is reseted before iterating
        dataloader.reset()

        # infer prediction shape
        output_dim = list(self._model.parameters())[-1].size()
        y_pred = torch.zeros((n,) + output_dim, device=self.device)
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

                # Add to prediction tensor
                y_pred[r: min(n, r + batch_size)] = y_batch_pred
                r += batch_size
                n_batches += 1

        return y_pred, loss  # / n_batches

    def _get_evaluate_func(
            self,
            eval_dataloader: EvaluationDataLoader | None,
            loss_callable: Callable | None = None,
            validation: bool = False,
            use_weights: bool = False,
    ) -> Callable[[dict, int], None]:

        if eval_dataloader is None or (not validation and not self._evaluate_training):
            return self._empty_func

        # helpful function to concatenate the values of a dictionary
        def _concat_values(d: dict, weight_map: dict = {}) -> tuple[Tensor, float]:
            tensors, losses = (
                zip(*((value[0], value[1] * weight_map.get(key, 1)) for key, value in d.items()))
                if d and isinstance(next(iter(d.values())), tuple)
                else (list(d.values()), [])
            )
            return torch.cat(tensors), sum(losses) / len(losses) if losses else 0.0

        # define useful variables
        prefix = "val_" if validation else ""
        true_values, _ = _concat_values(eval_dataloader.get_targets())
        dataloader_dict = eval_dataloader.data_loader
        weight_map = eval_dataloader.weight_map if use_weights else {}
        metrics = self._metrics
        plots = self._plots

        def _evaluate(log: dict, epoch: int) -> None:
            y_pred = OrderedDict({
                key: self._predict(loader, loss_func=loss_callable)
                for key, loader in dataloader_dict.items()
            })
            single_losses_unweighted = {k: v[1] for k, v in y_pred.items()}
            single_losses = {k: v * weight_map.get(k, 1) for k, v in single_losses_unweighted.items()}
            y_pred, loss = _concat_values(y_pred, weight_map=weight_map)
            if loss_callable is not None:
                log[f"{prefix}loss"] = loss
                for k, v in single_losses_unweighted.items():
                    log[f"{prefix}{k}_loss_unweighted"] = v
                    log[f"{prefix}{k}_loss"] = single_losses[k]
            if metrics:
                add_metrics_to_log(log, metrics, y_pred, true_values, prefix=prefix)
            for plot in plots:
                plot.update(y_pred, true_values, mode="valid" if validation else "train", epoch=str(epoch))
        return _evaluate

    def _log_parameters(self, **kwargs):
        for key, value in kwargs.items():
            self.trace_func(f"{key}: {value}")

    @property
    def model(self) -> Module:
        return self._model

    @property
    def best_model(self) -> Module:
        """returns the best model state."""
        if isinstance(self._early_stopping, EarlyStopping):
            return_model = copy.deepcopy(self._model)
            return_model.load_state_dict(self._early_stopping.best_state_dict, strict=True)
            return return_model
        self.trace_func("Early stopping is not set. Returning the current model.")
        return self._model

    def save_model(self) -> None:
        path = f"{self.path}/model.pt"
        torch.save(self._model.state_dict(), path)
        self.trace_func(f"Model state dictionary saved to {path}")

    def save_best_model(self) -> None:
        if not isinstance(self._early_stopping, EarlyStopping):
            self.trace_func("Early stopping is not set. No best model to save.")
            return
        path = f"{self.path}/best_model.pt"
        torch.save(self._early_stopping.best_state_dict, path)
        self.trace_func(f"Best model state dictionary saved to {path}")

    def predict(self, dataloader, use_best_model: bool = False) -> Tensor | dict:
        """Generates output predictions for the input samples.

        Computation is done in batches.

        # Arguments
            X: input data Tensor.
            batch_size: integer.

        # Returns
            prediction Tensor.
        """
        if isinstance(dataloader, dict):
            return {key: self.predict(l, use_best_model) for key, l in dataloader.items()}

        r, n = 0, len(dataloader.map_fn.dataset)
        batch_size = dataloader.source.batch_size

        # ensure dataloader is reseted before iterating
        dataloader.reset()

        # get the right model
        model = self.best_model if use_best_model else self._model

        # Infer prediction shape
        output_dim = list(model.parameters())[-1].size()
        y_pred = torch.zeros((n,) + output_dim, device=self.device)

        # Batch prediction
        model.eval()
        with torch.no_grad():
            for X_embed_batch, X_num_batch, _ in dataloader:
                # Predict on batch
                y_batch_pred = model(X_embed_batch, X_num_batch)

                # Add to prediction tensor
                y_pred[r: min(n, r + batch_size)] = y_batch_pred
                r += batch_size
        return y_pred

    # @cleanup_on_exception
    def fit(
        self,
        training_data: CompositeDataLoader,
        epochs: int = 1,
        initial_epoch: int = 0,
        validation_data: EvaluationDataLoader | None = None,
        use_eval_weights: bool = True,
        seed: int | None = None,
        **kwargs,
    ) -> list[OrderedDict]:
        """
        This method trains the PyTorch model using the provided training data and optional validation data.
        It supports early stopping, logging, and visualization of metrics.

        Args:
            training_data (CompositeDataLoader): DataLoader for training data.
            epochs (int): Number of epochs to train the model.
            initial_epoch (int, optional): Epoch at which to start training. Defaults to 0.
            validation_data (EvaluationDataLoader, optional): DataLoader for validation data. Defaults to None.
            use_eval_weights (bool, optional): Whether to use weights in the evaluation. Defaults to True.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            **kwargs: Additional arguments for logging or configuration.

        Returns:
            list[OrderedDict]: List of logs with training and validation metrics for each epoch.

        Notes:
            - The method supports early stopping if configured during initialization.
            - Metrics and plots are updated after each epoch if provided.
            - Logs and model checkpoints are saved if logging is enabled.
        """
        # shift variables to local scope for better readability
        logs = []
        tensorboard = self._tensorboard
        loss_func = self._loss_func
        val_loss_func = self._val_loss_func
        optimizer = self._optimizer

        if seed and seed >= 0:
            torch.manual_seed(seed)
        verbose = self._verbose
        if verbose:
            self._log_parameters(loss_func=loss_func, optimizer=optimizer, use_eval_weights=use_eval_weights, **kwargs)
        # torch.autograd.detect_anomaly(True)

        # Get DataLoader
        train_dataloader = training_data.data_loader

        # Define a Dataloader(s) w/o sampling and shuffling for evaluation (if needed)
        # and get the needed evaluation functions for both training and validation
        train_eval_data = EvaluationDataLoader(
            training_data.data_map or {},
            batch_size=training_data.batch_size * 2,
            device=self.device
        ) if self._evaluate_training else None
        eval_training = self._get_evaluate_func(train_eval_data)
        eval_validation = self._get_evaluate_func(
            validation_data, loss_callable=val_loss_func, validation=True, use_weights=use_eval_weights,
        )

        # Compile optimizer
        opt = optimizer(self._model.parameters())

        for t in range(initial_epoch, epochs):
            log = OrderedDict()
            epoch_loss = 0.0
            if verbose:
                print("Epoch {0} / {1}".format(t + 1, epochs))
            pb = ProgressBar(training_data.num_batches, verbose=verbose)

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
                    pb.bar(batch_i, log)

            # Run metrics and plots
            if self._evaluate_training:
                eval_training(log, t)
            if validation_data:
                eval_validation(log, t)

            # Log to Tensorboard
            tensorboard(log)
            self._early_stopping(log)
            logs.append(log)
            if verbose:
                pb.close(log=log)
            if self._early_stopping:
                self.trace_func(f"Early stopping at epoch {t + 1}")
                break

        # on finish
        if isinstance(tensorboard, TorchBoard):
            tensorboard.flush()
        self.trace_func("Training finished.")
        if self._save_logs:
            self._save_log(logs)
            func_config = {
                "device": self.device,
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
