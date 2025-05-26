from __future__ import annotations
from functools import partial

from hbt.ml.torch_utils.layers import StandardizeLayer
from hbt.ml.torch_utils.utils import expand_columns, get_standardization_parameter

__all__ = [
]

from columnflow.util import MockModule, maybe_import, DotDict
from columnflow.types import Any
from collections.abc import Container

from columnflow.columnar_util import Route, EMPTY_FLOAT, EMPTY_INT

from hbt.ml.torch_utils.utils import get_standardization_parameter

torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")
ak = maybe_import("awkward")
import law

model_clss: DotDict[str, torch.nn.Module] = DotDict()

if not isinstance(torch, MockModule):
    from torch import nn
    from torch.optim import Adam, AdamW, lr_scheduler
    from torch.utils.tensorboard import SummaryWriter
    from ignite.metrics import Loss, ROC_AUC

    from hbt.ml.torch_utils.transforms import (
        AkToTensor, PreProcessFloatValues, MoveToDevice, TokenizeCategories,
    )
    from hbt.ml.torch_utils.datasets.handlers import (
        FlatListRowgroupParquetFileHandler, FlatArrowParquetFileHandler,
        WeightedFlatListRowgroupParquetFileHandler,
        RgTensorParquetFileHandler, WeightedRgTensorParquetFileHandler,
        WeightedTensorParquetFileHandler,
    )
    from hbt.ml.torch_utils.ignite.metrics import (
        WeightedROC_AUC, WeightedLoss,
    )
    from hbt.ml.torch_utils.utils import (
        embedding_expected_inputs,
    )
    from hbt.ml.torch_utils.functions import generate_weighted_loss
    from hbt.ml.torch_utils.ignite.mixins import IgniteTrainingMixin, IgniteEarlyStoppingMixin
    from hbt.ml.torch_utils.layers import PaddingLayer, InputLayer

    def NLL_Focal_Loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: torch.Tensor | None = None,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
    ) -> torch.Tensor:

        # Calculate the NLL loss
        if targets.dim() == 2:
            targets = targets.squeeze(1)
        nll_loss = nn.functional.nll_loss(inputs.log(), targets.long(), reduction="none")

        # Calculate the Focal Loss
        p_t = inputs[:, 1] * targets + inputs[:, 0] * (1 - targets)
        loss = nll_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if weight is not None:
            loss = loss * weight
        # Check reduction option and return loss accordingly
        if reduction == "none":
            pass
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss

    class NetworkBase(nn.Module):
        def __init__(self, *args, tensorboard_path: str | None = None, logger: Any | None = None, **kwargs):
            super().__init__()
            self.writer = None
            self.logger = logger or law.logger.get_logger(__name__)
            if tensorboard_path:
                self.logger.info(f"Creating tensorboard logger at {tensorboard_path}")
                self.writer = SummaryWriter(log_dir=tensorboard_path)
            self.custom_hooks = list()

    class FeedForwardNet(
        IgniteEarlyStoppingMixin,
        IgniteTrainingMixin,
        NetworkBase,
    ):
        def __init__(
            self,
            *args,
            tensorboard_path: str | None = None, logger: Any | None = None,
            task: law.Task,
            **kwargs,
        ):
            super().__init__(*args, tensorboard_path=tensorboard_path, logger=logger, **task.param_kwargs, **kwargs)

            columns = [
                "lepton1.{px,py,pz,energy,mass}",
                "lepton2.{px,py,pz,energy,mass}",
                "bjet1.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
                "bjet2.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
                "fatjet.{px,py,pz,energy,mass}",
            ]
            self.inputs: set[Route] = set()
            self.inputs.update(*list(map(Route, law.util.brace_expand(obj)) for obj in columns))

            self.init_layers()
            self.logger.info("Constructing loss and optimizer")
            self._loss_fn = nn.BCELoss()
            self.validation_metrics = {
                "loss": Loss(self.loss_fn),
                "roc_auc": ROC_AUC(),
            }
            self.training_epoch_length_cutoff = None
            self.training_weight_cutoff = None
            self.val_epoch_length_cutoff = None
            self.val_weight_cutoff = None
            self.training_logger_interval = 20

        def init_layers(self):
            self.padding_layer = PaddingLayer(padding_value=0, mask_value=EMPTY_FLOAT)
            self.norm_layer = nn.BatchNorm1d(len(self.inputs))
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(len(self.inputs), 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid(),
            )

        def init_optimizer(self, learning_rate=1e-3, weight_decay=1e-5) -> None:
            self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        def _build_categorical_target(self, dataset: str):
            return int(1) if dataset.startswith("hh") else int(0)

        def train_step(self, engine, batch):
            # Set the model to training mode - important for batch normalization and dropout layers
            self.train()
            # Compute prediction and loss
            X, y = batch[0], batch[1]
            self.optimizer.zero_grad()

            pred = self(X)
            target = y["categorical_target"].to(torch.float32)
            if target.dim() == 1:
                target = target.reshape(-1, 1)

            loss = self.loss_fn(pred, target)
            # Backpropagation
            loss.backward()
            self.optimizer.step()

            return loss.item()

        def validation_step(self, engine, batch):
            # Set the model to evaluation mode - important for batch normalization and dropout layers
            self.eval()

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors
            # with requires_grad=True
            with torch.no_grad():
                X, y = batch[0], batch[1]
                pred = self(X)
                target = y["categorical_target"].to(torch.float32)
                if target.dim() == 1:
                    target = target.reshape(-1, 1)
                return pred, target

        def init_dataset_handler(self, task: law.Task, *args, **kwargs):
            group_datasets = {
                "ttbar": [d for d in task.datasets if d.startswith("tt_")],
            }
            device = next(self.parameters()).device
            all_datasets = getattr(task, "resolved_datasets", task.datasets)

            self.dataset_handler = FlatListRowgroupParquetFileHandler(
                task=task,
                columns=self.inputs,
                batch_transformations=AkToTensor(device=device),
                # global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_"])],
            )
            self.training_loader, self.validation_loader = self.dataset_handler.init_datasets()

        def _handle_input(
            self,
            x: dict[str, torch.Tensor],
            feature_list: Container[str] | None = None,
            mask_value: int | float = EMPTY_FLOAT,
            empty_fill_val: float = 0,
            norm_layer: nn.Module | None = None,
            dtype=torch.float32,
        ):
            if not feature_list:
                feature_list = self.inputs
            input_data = x

            if isinstance(x, dict):
                input_data: torch.Tensor = torch.cat([
                    x[str(key)].reshape(-1, 1) for key in sorted(feature_list)],
                    axis=-1,
                )
            # check for dummy values
            input_data = self.padding_layer(input_data)

            if norm_layer:
                input_data = norm_layer(input_data.to(dtype))
                # from IPython import embed
                # embed(header=f"using norm layer to derive default values")

            return input_data.to(dtype)

        def forward(self, x):
            input_data = self._handle_input(x, norm_layer=getattr(self, "norm_layer", None))
            logits = self.linear_relu_stack(input_data)
            return logits

    class TensorFeedForwardNet(FeedForwardNet):
        def init_dataset_handler(self, task: law.Task, *args, **kwargs):
            group_datasets = {
                "ttbar": [d for d in task.datasets if d.startswith("tt_")],
            }
            device = next(self.parameters()).device
            all_datasets = getattr(task, "resolved_datasets", task.datasets)

            self.dataset_handler = RgTensorParquetFileHandler(
                task=task,
                continuous_features=self.inputs,
                batch_transformations=MoveToDevice(device=device),
                # global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_"])],
            )
            self.training_loader, self.validation_loader = self.dataset_handler.init_datasets()

        def forward(self, x):
            x = self.padding_layer(x)
            x = self.norm_layer(x.to(torch.float32))
            logits = self.linear_relu_stack(x)
            return logits

        def train_step(self, engine, batch):
            # Set the model to training mode - important for batch normalization and dropout layers
            self.train()
            # Compute prediction and loss
            X, y = batch[0], batch[1]
            self.optimizer.zero_grad()
            pred = self(X)

            target = y.to(torch.float32)
            if target.dim() == 1:
                target = target.reshape(-1, 1)

            loss = self.loss_fn(pred, target)
            # Backpropagation
            loss.backward()
            self.optimizer.step()

            return loss.item()

        def validation_step(self, engine, batch):
            # Set the model to evaluation mode - important for batch normalization and dropout layers
            self.eval()

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors
            # with requires_grad=True
            with torch.no_grad():
                X, y = batch[0], batch[1]
                pred = self(X)
                target = y.to(torch.float32)
                if target.dim() == 1:
                    target = target.reshape(-1, 1)
                return pred, target

    class WeightedTensorFeedForwardNet(TensorFeedForwardNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._loss_fn = generate_weighted_loss(nn.BCELoss)()

            self.validation_metrics = {
                "loss": WeightedLoss(self.loss_fn),
                "roc_auc": WeightedROC_AUC(),
            }

        def init_dataset_handler(self, task: law.Task):
            all_datasets = getattr(task, "resolved_datasets", task.datasets)
            group_datasets = {
                "hh": [d for d in all_datasets if d.startswith("hh_")],
            }
            device = self.used_device
            self.dataset_handler = WeightedRgTensorParquetFileHandler(
                task=task,
                continuous_features=getattr(self, "continuous_features", self.inputs),
                categorical_features=getattr(self, "categorical_features", None),
                batch_transformations=MoveToDevice(device=device),
                # global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_", "dy_"])],
            )
            self.training_loader, (self.train_validation_loader, self.validation_loader) = self.dataset_handler.init_datasets()  # noqa

            self.dataset_statistics = get_standardization_parameter(self.train_validation_loader.data_map, self.inputs)

        def init_layers(self):
            self.std_layer = StandardizeLayer()
            super().init_layers()

        def setup_preprocessing(self):
            # extract dataset std and mean from dataset
            # extraction happens form no oversampled dataset
            mean, std = [], []
            for _input in sorted(self.inputs, key=str):
                input_statitics = self.dataset_statitics[_input.column]
                mean.append(torch.from_numpy(input_statitics["mean"]))
                std.append(torch.from_numpy(input_statitics["std"]))

            device = next(self.parameters()).device
            mean, std = torch.concat(mean).to(device), torch.concat(std).to(device)

            # set up standardization layer
            self.std_layer.set_mean_std(
                mean.float(),
                std.float(),
            )

        def train_step(self, engine, batch):
            # Set the model to training mode - important for batch normalization and dropout layers
            self.train()
            # Compute prediction and loss
            X, y = batch[0], batch[1]
            self.optimizer.zero_grad()
            pred = self(X)

            target = y.to(torch.float32)
            if target.dim() == 1:
                target = target.reshape(-1, 1)

            loss = self.loss_fn(pred, target)
            # Backpropagation
            loss.backward()
            self.optimizer.step()

            return loss.item()

        def validation_step(self, engine, batch):
            # Set the model to evaluation mode - important for batch normalization and dropout layers
            self.eval()

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors
            # with requires_grad=True
            with torch.no_grad():
                X, y = batch[0], batch[1]
                input_data, weights = X[:-1], X[-1]
                if isinstance(input_data, list) and len(input_data) == 1:
                    input_data = input_data[0]
                pred = self(input_data)
                target = y.to(torch.float32)
                if target.dim() == 1:
                    target = target.reshape(-1, 1)
                return pred, target, {"weight": weights}

    class WeightedTensorFeedForwardNetWithCat(WeightedTensorFeedForwardNet):
        def __init__(self, *args, **kwargs):
            self.categorical_features = {
                "pair_type",
                # "decay_mode1",
                "decay_mode2",
                "lepton1.charge",
                "lepton2.charge",
                "has_fatjet",
                "has_jet_pair",
                "year_flag",
            }
            self.embedding_dims = 50

            super().__init__(*args, **kwargs)

            self.continuous_features = self.inputs

            self.init_layers()

        def init_layers(self):
            self.std_layer = StandardizeLayer()
            self.input_layer = InputLayer(
                categorical_inputs=sorted(self.categorical_features, key=str),
                continuous_inputs=sorted(self.inputs, key=str),
                expected_categorical_inputs=embedding_expected_inputs,
                embedding_dim=self.embedding_dims,
            )
            self.padding_layer_cat = PaddingLayer(padding_value=self.input_layer.empty, mask_value=EMPTY_INT)
            self.padding_layer_cont = PaddingLayer(padding_value=0, mask_value=EMPTY_FLOAT)

            self.linear_relu_stack = nn.Sequential(
                nn.Linear(self.input_layer.ndim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid(),
            )

        def init_optimizer(self, learning_rate=1e-3, weight_decay=1e-5) -> None:
            self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        def forward(self, X):
            # X is a tuple of (input_data, categorical_features)

            cat_features, cont_features = X
            cont_features = self.padding_layer_cont(cont_features)
            cont_features = self.std_layer(cont_features.to(torch.float32))
            cat_features = self.padding_layer_cat(cat_features)

            # pass through the embedding layer
            features = self.input_layer((cat_features.to(torch.int32), cont_features))

            # concatenate the continuous and categorical features

            logits = self.linear_relu_stack(features)
            return logits

        def validation_step(self, engine, batch):
            # Set the model to evaluation mode - important for batch normalization and dropout layers
            self.eval()

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors
            # with requires_grad=True
            with torch.no_grad():
                X, y = batch[0], batch[1]
                input_data, weights = X[:-1], X[-1]
                pred = self(input_data)
                target = y.to(torch.float32)
                if target.dim() == 1:
                    target = target.reshape(-1, 1)

                return pred, target, {"weight": weights}

        def to(self, *args, **kwargs):
            self.std_layer = self.std_layer.to(*args, **kwargs)
            self.input_layer = self.input_layer.to(*args, **kwargs)
            return super().to(*args, **kwargs)

    class WeightedTensorFeedForwardNetWithCatOutsourceTokens(WeightedTensorFeedForwardNetWithCat):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def init_layers(self):
            self.std_layer = StandardizeLayer()

            self.tokenizer = TokenizeCategories(
                categories=sorted(self.categorical_features, key=str),
                expected_categorical_inputs=embedding_expected_inputs,
                cat_feature_idx=0,
            )
            self.input_layer = InputLayer(
                continuous_inputs=sorted(self.inputs, key=str),
                categorical_inputs=sorted(self.categorical_features, key=str),
                embedding_dim=self.embedding_dims,
                category_dims=self.tokenizer.num_dim,
            )
            self.padding_layer_cat = PaddingLayer(padding_value=self.input_layer.empty, mask_value=EMPTY_INT)
            self.padding_layer_cont = PaddingLayer(padding_value=0, mask_value=EMPTY_FLOAT)

            self.linear_relu_stack = nn.Sequential(
                nn.Linear(self.input_layer.ndim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid(),
            )

        def init_dataset_handler(self, task: law.Task, *args, **kwargs):
            group_datasets = {
                "ttbar": [d for d in task.datasets if d.startswith("tt_")],
            }
            device = next(self.parameters()).device
            all_datasets = getattr(task, "resolved_datasets", task.datasets)

            self.dataset_handler = WeightedRgTensorParquetFileHandler(
                task=task,
                continuous_features=getattr(self, "continuous_features", self.inputs),
                categorical_features=getattr(self, "categorical_features", None),
                batch_transformations=MoveToDevice(device=device),
                input_data_transform=self.tokenizer,
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_"])],
            )
            self.training_loader, (self.train_validation_loader, self.validation_loader) = self.dataset_handler.init_datasets() # noqa
            self.dataset_statistics = get_standardization_parameter(self.train_validation_loader.data_map, self.inputs)

    class WeightedTensorFeedForwardNetWithCatReducedEmbedding(WeightedTensorFeedForwardNetWithCat):

        def __init__(self, *args, **kwargs):

            super().__init__(*args, **kwargs)
            self.embedding_dims = 10

            self.init_layers()

    class WeightedTensorFeedForwardNetWithCatReducedEmbedding1F(WeightedTensorFeedForwardNetWithCatOutsourceTokens):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.categorical_features = {
                "pair_type",
                # "decay_mode1",
                # "decay_mode2",
                # "lepton1.charge",
                # "lepton2.charge",
                # "has_fatjet",
                # "has_jet_pair",
                # "channel_id",
                "year_flag",
            }

            self.embedding_dims = 10
            self.init_layers()
            self.custom_hooks.append("perform_scheduler_step")

        def perform_scheduler_step(self):
            if self.scheduler:
                def do_step(engine, logger=self.logger):
                    logger.info(f"Performing scheduler step")
                    self.scheduler.step()

                self.train_evaluator.add_event_handler(
                    event_name="EPOCH_COMPLETED",
                    handler=do_step,
                )

        def init_optimizer(self, learning_rate=1e-3, weight_decay=1e-5) -> None:
            embedding_params = {x for name, x in self.named_parameters() if "embedding" in name and not "bias" in name}
            other_params = {x for x in self.parameters() if not x in embedding_params}
            self.optimizer = AdamW(
                [
                    {
                        "params": list(other_params),
                    },
                    {
                        "params": list(embedding_params),
                        "lr": learning_rate * 10,
                        "weight_decay": weight_decay * 10,
                    },
                ],
                lr=learning_rate, weight_decay=weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=4,
                gamma=0.9,
            )

        def train_step(self, engine, batch):
            # Set the model to training mode - important for batch normalization and dropout layers
            self.train()
            # Compute prediction and loss
            X, y = batch[0], batch[1]
            self.optimizer.zero_grad()
            pred = self(X)

            target = y.to(torch.float32)
            if target.dim() == 1:
                target = target.reshape(-1, 1)

            loss = self.loss_fn(pred, target)
            # from IPython import embed
            # embed(header=f"check back propagation in class {self.__class__.__name__}")
            # Backpropagation
            loss.backward()
            self.optimizer.step()

            return loss.item()

    class TensorFeedForwardNetAdam(TensorFeedForwardNet):
        def init_optimizer(self, learning_rate=0.001, weight_decay=0.00001):
            self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    class DropoutFeedForwardNet(FeedForwardNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.linear_relu_stack = nn.Sequential(

                # nn.Dropout(p=0.2),
                nn.Linear(len(self.inputs), 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                # nn.Dropout(p=0.2),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                # nn.Dropout(p=0.2),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid(),
            )

    class FeedForwardArrow(FeedForwardNet):
        def init_dataset_handler(self, task: law.Task, *args, **kwargs):
            group_datasets = {
                "ttbar": [d for d in task.datasets if d.startswith("tt_")],
            }
            device = next(self.parameters()).device
            self.dataset_handler = FlatArrowParquetFileHandler(
                task=task,
                columns=self.inputs,
                batch_transformations=AkToTensor(device=device),
                global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
            )

    class ANet(WeightedTensorFeedForwardNet, IgniteEarlyStoppingMixin):
        def __init__(
            self,
            *args,
            tensorboard_path: str | None = None, logger: Any | None = None,
            means: torch.Tensor | None = None,
            stds: torch.Tensor | None = None,
            **kwargs,
        ):
            super(FeedForwardNet, self).__init__(*args, tensorboard_path=tensorboard_path, logger=logger, **kwargs)

            self.used_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.categorical_target_map = {
                "hh": 1,
                "tt": 0,
                "dy": 0,
            }

            self.categorical_features = sorted({
                "channel_id",
                # "l{1,2}.tauVS{e,jet,mu}",
            })
            # continuous inputs
            self.continuous_features = set()
            self.continuous_features.update(
                *list(law.util.brace_expand(obj) for obj in [
                    "l1.{pt,eta,dxy,dz,is_iso,iso_score}",
                    "l2.{pt,eta,dxy,dz,is_iso}",
                    "bjet0.{pt,phi,mass,hhbtag,eta,btagPNetB}",
                    "bjet1.{pt,phi,mass,hhbtag,eta,btagPNetB}",
                    "di{BJet,Tau}.{pt,mass,eta}",
                    "hh.{pt,eta,mass}",
                    "n_{bjets,jets,taus}"
                ])
            )
            self.continuous_features = list(map(Route, self.continuous_features))
            self.inputs = set()
            self.inputs.update(self.continuous_features)
            self.inputs.update(*list(map(Route, law.util.brace_expand(obj)) for obj in self.categorical_features))

            self.nodes = kwargs.get("nodes", 256)
            self.activation_functions = kwargs.get("activation_functions", "PReLU")

            embed_map = {
                "channel_id": (3, 2),
                "tauVSjet": (10, 6),
                "tauVSe": (10, 6),
                "tauVSmu": (5, 3),
                "pad_flag": (2, 1),
            }
            # embedding layers
            self.embed: nn.ModuleList = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Embedding(*embed_map[feat.split(".")[-1]]),
                        nn.Flatten(),
                    )
                    for feat in self.categorical_features
                ]
            )
            self.input_shape = len(self.inputs) + \
                sum([embed_map[feat.split(".")[-1]][1] - 1 for feat in self.categorical_features])
            self.standardize = StandardizeLayer(None, None)
            # define the layers with batch normalization
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(self.input_shape, 64),
                nn.BatchNorm1d(64),
                nn.PReLU(),
                nn.Linear(64, 256),
                nn.BatchNorm1d(256),
                nn.PReLU(),
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.PReLU(),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.PReLU(),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.PReLU(),
                nn.Linear(128, 32),
                nn.BatchNorm1d(32),
                nn.PReLU(),
                nn.Linear(32, 2),
                nn.Softmax(1),
            )

            self.logger.info("Constructing loss and optimizer")
            self._loss_fn = partial(NLL_Focal_Loss, alpha=0.25, gamma=2, reduction="mean")
            self.validation_metrics = {
                "loss": WeightedLoss(partial(NLL_Focal_Loss, alpha=0.25, gamma=2, reduction="sum")),
            }
            self.training_epoch_length_cutoff = None
            self.training_weight_cutoff = None
            self.val_epoch_length_cutoff = None
            self.val_weight_cutoff = None
            self.training_logger_interval = 20
            self.placeholder = 15
            self.scheduler_start = 2

        def to(self, *args, **kwargs):
            self.standardize = self.standardize.to(*args, **kwargs)
            self.linear_relu_stack = self.linear_relu_stack.to(*args, **kwargs)
            self.embed = self.embed.to(*args, **kwargs)
            return super().to(*args, **kwargs)

        def init_optimizer(self, learning_rate=5e-3, weight_decay=1e-5) -> None:
            self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=0)
            self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=2, gamma=0.8)

        def train_step(self, engine, batch):
            if engine.state.epoch == self.scheduler_start:
                self.scheduler.step()
                self.scheduler_start += 1
                self.logger.info(f"Learning rate adjusted to {self.scheduler.get_last_lr()}")
            # Set the model to training mode - important for batch normalization and dropout layers
            self.train()
            # Compute prediction and loss
            (X_e, E_n), y = batch[0], batch[1]
            self.optimizer.zero_grad()
            pred = self((X_e, E_n))
            target = y.to(torch.float32)
            if target.dim() == 1:
                target = target.reshape(-1, 1)

            loss = self.loss_fn(pred, target)
            # Backpropagation
            loss.backward()
            self.optimizer.step()

            return loss.item()

        def validation_step(self, engine, batch):
            # Set the model to evaluation mode - important for batch normalization and dropout layers
            self.eval()

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors
            # with requires_grad=True
            with torch.no_grad():
                (X_e, E_n, _), y = batch[0], batch[1]
                pred = self((X_e, E_n))
                target = y.to(torch.float32)
                if target.dim() == 1:
                    target = target.reshape(-1, 1)
                weights = torch.where(y == 1, 2 / self.dataset_length[1], 1 / self.dataset_length[0])
                return pred, target, {"weight": weights}

        def setup_preprocessing(self):
            # extract dataset std and mean from dataset
            # extraction happens form no oversampled dataset
            mean, std = [], []
            for _input in self.continuous_features:
                input_statitics = self.dataset_statitics[_input.column]
                mean.append(torch.from_numpy(input_statitics["mean"]))
                std.append(torch.from_numpy(input_statitics["std"]))

            mean, std = torch.concat(mean), torch.concat(std)
            # set up standardization layer
            self.standardize.set_mean_std(
                mean.float(),
                std.float(),
            )

        def logging(self, *args, **kwargs):
            # output histogram
            for target, index in self.categorical_target_map.items():
                # apply softmax to prediction
                pred_prob = kwargs["prediction"]

                self.writer.add_histogram(
                    f"output_prob_{target}",
                    pred_prob[:, index],
                    self.trainer.state.iteration,
                )
                self.writer.add_histogram(
                    f"output_logit_{target}",
                    logit[:, index],
                    self.trainer.state.iteration,
                )

        def init_dataset_handler(self, task: law.Task, device: str = "cpu") -> None:
            super(ANet, self).init_dataset_handler(task)
            # get statistics for standardization from training dataset without oversampling
            self.dataset_length = {0: 0, 1: 0}
            for d in self.train_validation_loader.data_map:
                self.dataset_length[d.class_target] += len(d)
            self.weight_dict = {
                1: 1 / (2 * self.dataset_length[1]),
                0: 1 / (4 * self.dataset_length[0]),
            }
            self.dataset_statitics = get_standardization_parameter(
                self.train_validation_loader.data_map, self.continuous_features)

        def forward(self, x, *args, **kwargs):
            X_embed, X_num = x
            x_inp = [f(a.int()) for f, a in zip(self.embed, torch.split(X_embed, 1, dim=1))]
            x_inp.append(self.standardize(X_num))
            x_inp = torch.cat(x_inp, dim=1)
            x_inp = x_inp.float()
            logits = self.linear_relu_stack(x_inp)
            return logits
