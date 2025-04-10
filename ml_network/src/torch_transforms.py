from __future__ import annotations

from collections.abc import Mapping
from typing import NoReturn, Sequence

import numpy as np
import awkward as ak
import torch
from dask_awkward.lib.core import Array
from torch.nested._internal.nested_tensor import NestedTensor

from ml_network.src.cf_utils import flat_np_view


class AkToTorchTensor(torch.nn.Module):

    def __init__(self, requires_grad=False, device=None, *args, preprocess=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_grad = requires_grad
        self.device = device
        self.preprocess = preprocess

        # Dict of NumPy dtype -> torch dtype (when the correspondence exists)
        # from https://github.com/pytorch/pytorch/blob/e180ca652f8a38c479a3eff1080efe69cbc11621/torch/testing/_internal/common_utils.py#L349  # noqa
        self.numpy_to_torch_dtype_dict = {
            np.bool: torch.bool,
            np.uint8: torch.uint8,
            np.int8: torch.int8,
            np.int16: torch.int16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.float16: torch.float16,
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.complex64: torch.complex64,
            np.complex128: torch.complex128
        }

    def _transform_input(self, X):
        return_tensor = None

        if isinstance(X, ak.Array):
            # first, get flat numpy view to avoid copy operations
            view = flat_np_view(X)

            # transform into torch Tensor with same type
            values = torch.tensor(view).to(
                self.numpy_to_torch_dtype_dict.get(view.dtype.type, view.dtype.type)
            )

            # to calculate the offsets, count the elements
            n_elements = ak.num(X, axis=1)

            # the offsets are the cumulated number of elements
            # prepend 0 to also account for first element in array
            try:
                cumsum = np.cumsum(ak.concatenate([0, n_elements], axis=0), axis=0)
            except Exception as e:
                from IPython import embed
                embed(header=f"raised Exception '{e}', debugging")
                raise e

            # alternative way to get underlying offsets and data structure
            # array = ak.to_layout(X)
            # offsets = array.offsets.data
            # view = array.content.content.data
            # this should always yield the correct offsets, regardless of slicing

            # now directly construct NestedTensor
            # DANGERZONE: after comparing hex(id(values)) vs hex(id(return_tensor.values()))
            # realized that there must be a copy going on somewhere...
            return_tensor = torch.nested._internal.nested_tensor.NestedTensor(
                values=values, offsets=torch.tensor(cumsum),
                requires_grad=self.requires_grad, device=self.device,
            )
        elif isinstance(X, torch.Tensor):
            # if the input is already a Tensor, cast it into a nested tensor
            return_tensor = torch.nested.as_nested_tensor(
                X, layout=torch.jagged, device=self.device,
            )
        elif isinstance(X, NestedTensor):
            return_tensor = X

        if return_tensor is None:
            raise ValueError(f"Could not convert input {X=}")

        return return_tensor

    def forward(
        self,
        X: ak.Array | torch.Tensor | NestedTensor | Mapping[str, ak.Array | torch.Tensor | NestedTensor]
    ) -> (NestedTensor | Mapping[str, NestedTensor]):
        return_tensor: NestedTensor | Mapping[str, NestedTensor] | None = None

        if isinstance(X, Mapping):
            return_tensor = {
                key: self._transform_input(data) for key, data in X.items()
            }
        else:
            return_tensor = self._transform_input(X)

        return return_tensor


class AkToTensor(AkToTorchTensor):
    def _transform_input(self, X):
        return_tensor = None
        if isinstance(X, ak.Array):
            # first, get flat numpy view to avoid copy operations
            try:
                return_tensor = ak.to_torch(X)
            except:
                # default back to NestedTensor
                return_tensor = super()._transform_input(X)
        elif isinstance(X, (torch.Tensor, NestedTensor)):
            return_tensor = X
        elif isinstance(X, (list, tuple)):
            return_tensor = [self._transform_input(entry) for entry in X]
        elif isinstance(X, dict):
            return_tensor = {key: self._transform_input(val) for key, val in X.items()}

        if return_tensor is None:
            raise ValueError(f"Could not convert input {X=}")

        return return_tensor


class AkToProcessedTensor(AkToTorchTensor):

    def _transform_input(self, X):
        return_tensor = None
        if isinstance(X, torch.Tensor):
            return X
        elif isinstance(X, ak.Array):
            return_tensor = self._transform_input(ak.to_torch(X))
        elif isinstance(X, (list, tuple)):
            return_tensor = [self._transform_input(entry) for entry in X]
        elif isinstance(X, dict):
            return_tensor = {key: self._transform_input(val) for key, val in X.items()}

        if return_tensor is None:
            raise ValueError(f"Could not convert input {X=}")

        return return_tensor


class RemoveEmptyValues(torch.nn.Module):
    def __init__(self, embed_input: bool = False, physical_padding: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_input = embed_input
        self.pad_dict = {
            "pt": -1,
            "eta": -5,
            "phi": -5,
            "mass": -1,
            "btagPNetB": -1,
            "hhbtag": -1,
        } if physical_padding else dict()

    def _transform_input(
            self, X, pad_key: str | None = None
    ) -> torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | NoReturn:
        return_tensor = None
        if isinstance(X, torch.Tensor):
            if self.embed_input:
                return_type = torch.int
                clamp_min, clamp_max = 0, 9
            else:
                return_type = torch.float
                pad_value = self.pad_dict.get(pad_key.split(".")[-1], -10.0) if pad_key is not None else -10.0
                clamp_min, clamp_max = pad_value, None

            return X.to(return_type).clamp(min=clamp_min, max=clamp_max)

        elif isinstance(X, ak.Array):
            return_tensor = self._transform_input(ak.to_torch(X))
        elif isinstance(X, (list, tuple)):
            return_tensor = [self._transform_input(entry) for entry in X]
        elif isinstance(X, dict):
            return_tensor = {key: self._transform_input(val, pad_key=key) for key, val in X.items()}

        if return_tensor is None:
            raise ValueError(f"Could not convert input {X=}")

        return return_tensor

    def forward(
            self,
            X: ak.Array | torch.Tensor | Mapping[str, ak.Array | torch.Tensor]
    ) -> (torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor]):
        return_tensor: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | None = None
        if isinstance(X, Mapping):
            return_tensor = {
                key: self._transform_input(data, pad_key=key) for key, data in X.items()
            }
        else:
            return_tensor = self._transform_input(X)
        return return_tensor


class GetSelectedEvents:
    def __init__(self, train_val_split: float | None = None, selection_field: str | None = None, random_seed: int = 42):
        self.selection_indecies = None
        self.training_split = train_val_split
        self.selection_field = selection_field
        self.seed = random_seed
        self.training_mode = True

    def _build_split_indecies(self, length: int) -> np.ndarray | None:
        if self.training_split is None:
            return
        if self.selection_indecies is None:
            split = int(length * self.training_split)
            np.random.seed(self.seed)
            choice = np.random.choice(range(length), size=(split,), replace=False)
            ind = np.zeros(length, dtype=bool)
            ind[choice] = True
            self.selection_indecies = ind
        if len(self.selection_indecies) != length:
            raise ValueError(f"Length of selection_indecies {len(self.selection_indecies)} does not match {length=}")
        return self.selection_indecies if self.training_mode else ~self.selection_indecies

    def _get_selection_column(self, array) -> np.ndarray:
        selection_column = None
        if self.selection_field is not None:
            selection_column = array[self.selection_field]
            if isinstance(selection_column, Array):
                selection_column = selection_column.compute()
            selection_column = np.astype(selection_column, bool)
        return selection_column

    def set_training_mode(self):
        self.training_mode = True

    def set_validation_mode(self):
        self.training_mode = False

    def __call__(self, X) -> ak.Array | Mapping[str, ak.Array | Array] | Sequence[ak.Array | Array]:
        return_tensor = None
        if isinstance(X, Array):
            X.eager_compute_divisions()
            X = X.repartition(npartitions=1)
        if isinstance(X, (ak.Array, Array)):
            return_array = X
            length = len(return_array)
            mask = self._get_selection_column(return_array)
            if mask is not None:
                return_array = X[mask]
                length = np.sum(mask)
            if self.training_split is not None:
                mask_indecies = self._build_split_indecies(length)
                return_array = return_array[mask_indecies]
            return_tensor = return_array
        elif isinstance(X, (list, tuple)):
            return_tensor = [self._transform_input(entry) for entry in X]
        elif isinstance(X, dict):
            return_tensor = {key: self._transform_input(val) for key, val in X.items()}

        if return_tensor is None:
            raise ValueError(f"Could not convert input {X=}")

        return return_tensor
