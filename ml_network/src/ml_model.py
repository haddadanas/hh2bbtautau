from __future__ import annotations

import torch
from torch import nn

from collections import OrderedDict
from functools import partial
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD

from utils import add_metrics_to_log, get_loader, log_to_message, ProgressBar, get_device

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
    ):
        """Trains the model similar to Keras' .fit(...) method

        # Arguments
            X: training data Tensor.
            y: target data Tensor.
            batch_size: integer. Number of samples per gradient update.
            epochs: integer, the number of times to iterate
                over the training data arrays.
            verbose: 0, 1. Verbosity mode.
                0 = silent, 1 = verbose.
            validation_split: float between 0 and 1:
                fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
            validation_data: (x_val, y_val) tuple on which to evaluate
                the loss and any model metrics
                at the end of each epoch. The model will not
                be trained on this data.
            shuffle: boolean, whether to shuffle the training data
                before each epoch.
            initial_epoch: epoch at which to start training
                (useful for resuming a previous training run)
            seed: random seed.
            optimizer: training optimizer
            loss: training loss
            metrics: list of functions with signatures `metric(y_true, y_pred)`
                where y_true and y_pred are both Tensors

        # Returns
            list of OrderedDicts with training metrics
        """
        if seed and seed >= 0:
            torch.manual_seed(seed)

        # Prepare validation data
        if validation_data:
            X_val, y_val, weight_val = validation_data
        else:
            X_val, y_val, weight_val = None, None, None
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
                X_train, Y_train, _ = train_data.dataset.get_data()  # type: ignore
                y_train_pred = self.predict(X_train[0], X_train[1], batch_size)
                add_metrics_to_log(log, metrics, Y_train, y_train_pred)
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val[0], X_val[1], batch_size)
                val_loss_arr = loss(Variable(y_val_pred), Variable(y_val)) * Variable(weight_val)
                val_loss = val_loss_arr.mean()
                log['val_loss'] = val_loss.item()
                if metrics:
                    add_metrics_to_log(log, metrics, y_val, y_val_pred, 'val_')
            logs.append(log)
            if verbose:
                pb.close(log_to_message(log))
        return logs

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
        data = get_loader(inp_embed=X_embed, inp_num=X_num, batch_size=batch_size, device=self.device)
        # Batch prediction
        self.model.eval()
        r, n = 0, X_num.size()[0]
        for batch_data in data:
            # Predict on batch
            X_embed_batch = Variable(batch_data[0][0])
            X_num_batch = Variable(batch_data[0][1])
            y_batch_pred = self.model(X_embed_batch, X_num_batch).data
            # Infer prediction shape
            if r == 0:
                y_pred = torch.zeros((n,) + y_batch_pred.size()[1:])
            # Add to prediction tensor
            y_pred[r: min(n, r + batch_size)] = y_batch_pred
            r += batch_size
        return y_pred


class CustomModel(nn.Module):
    def __init__(self, input_features):
        super(CustomModel, self).__init__()
        self.layers_dict = nn.ModuleDict()
        self.input_length = sum(len(input_spec) for input_spec in input_features)
        self.embedding_out = 2
        self.input_dim = self.input_length + self.embedding_out

        # embedding layers
        self.embed1 = nn.Embedding(3, self.embedding_out)
        self.flatten = nn.Flatten()
        self.concat = torch.cat

        # define the layers
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, ch, x):
        x0 = self.embed1(ch.int())
        x0 = self.flatten(x0)
        x = self.concat([x0] + x, dim=1)
        x = x.float()

        logits = self.linear_relu_stack(x)
        return logits
