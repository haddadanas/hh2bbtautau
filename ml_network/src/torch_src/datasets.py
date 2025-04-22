from __future__ import annotations

__all__ = [
    "ParquetDataset", "FlatParquetDataset", "FlatTorchDataset", "ListDataset"
]

import re
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Sequence, Literal

import numpy as np
import awkward as ak
import dask_awkward as dak
from dask_awkward.lib.core import Array

import torch
from torch.utils.data import Dataset

from ml_network.src.cf_utils import (
    EMPTY_FLOAT, EMPTY_INT, DotDict, Route, brace_expand, get_ak_routes, remove_ak_column, flat_np_view,
)


class ParquetDataset(Dataset):
    def __init__(
        self,
        input: Sequence[str] | ak.Array,
        columns: Sequence[str] | None = None,
        target: str | int | Iterable[str | int] | None = None,
        open_options: dict[str, Any] | None = None,
        transform: Callable | None = None,
        global_transform: Callable | None = None,
        device: str | None = None,
    ):
        self.open_options = open_options or {}
        self.columns = columns or set()
        # container for target columns
        self.target_columns = set()
        # container for integer targets
        self.int_targets = set()
        self.transform = transform
        self.global_transform = global_transform

        self.input = input
        self._data: ak.Array | Array | None = None
        self._input_data: ak.Array | None = None
        self._target_data: ak.Array | None = None
        self.class_target: int

        # container for meta data of parquet file(s)
        # None if input is an ak.Array
        self.meta_data = None

        if isinstance(input, (str, list)):
            self.path = input
            # idea: write sampler that sub samples each partition individually
            # the __getitem(s)__ method should then check which partition
            # is currently read, open the corresponding partition with
            # line below, and return the requested item(s).
            # If a new partition is requested, close/delete the current array
            # and load the next one.
            # Would require reading the parquet file multiple times after
            # each reset call (= overhead?), but would limit the memory consumption
            self.meta_data = DotDict.wrap(ak.metadata_from_parquet(self.path))
        elif isinstance(input, (ak.Array, Array)):
            self._data = input

        self.all_columns = set()
        self._parse_columns()

        self._parse_target(target=target)

        self._validate()

        if len(self.int_targets) > 0:
            self.class_target = list(self.int_targets)[0]
        self.data_columns = self.all_columns.symmetric_difference(self.target_columns)

        # parse all strings to Route objects
        self.data_columns: set[Route] = set(Route(x) for x in self.data_columns)
        self.target_columns: set[Route] = set(Route(x) for x in self.target_columns)
        self.all_columns: set[Route] = set(Route(x) for x in self.all_columns)

    def _parse_columns(self) -> None:
        if self.columns:
            self.all_columns = set(Route(x) for x in self.columns)
        elif isinstance(self._data, (ak.Array, Array)):
            self.all_columns = set(x for x in get_ak_routes(self._data))

        if self.meta_data:
            self.all_columns = set(x.replace(".list.item", "") for x in self.meta_data.columns)
            # columns are not explicitely considered when loading the meta data
            # so filter the full set of columns accordingly
            tmp_cols = set()
            for x in self.all_columns:
                for col in (self.columns or (".*",)):
                    resolved_route = self._check_against_pattern(x, col)
                    if resolved_route:
                        tmp_cols.add(resolved_route)

            self.all_columns = tmp_cols

            # check if transformations define columns to use and
            # make sure that the needed inputs are loaded, a.k.a. add them to all_columns
            self.all_columns.update(self._extract_transform_columns())

        if len(self.all_columns) == 0:
            raise ValueError("No columns specified and no metadata found")

    def _check_against_pattern(self, target: str, col: Route) -> Route | None:
        slice_dict: dict[int, tuple[slice]] = {}
        str_col: str = str(col)

        if isinstance(col, Route):
            slice_dict = {
                i: field for i, field in enumerate(col.fields) if isinstance(field, tuple)
            }
            str_col = col.string_column

        # make sure there aren't any special characters that aren't caught
        str_col = str_col.replace("{", "(").replace("}", ")").replace(",", "|")
        pattern = re.compile(f"^{str_col}$")

        if not pattern.match(target):
            return
        # if there is a match, insert possible slices
        # from IPython import embed
        # embed(header=f"found match for target '{target}' and pattern '{col}'")
        parts = target.split(".")
        for index in reversed(slice_dict.keys()):
            parts.insert(index, slice_dict[index])

        return Route(Route.join(parts))

    def _extract_transform_columns(self, attr: Literal["uses", "produces"] = "uses") -> set[Route]:
        """
        Small function to extract columns from transformations

        :param attr: attribute to extract from transformations, either "uses" or "produces"
        :returns: Set with resolved Routes to columns in awkward array (braces are expanded)
        """
        transform_inputs: set[Route] = set()
        for t in [self.transform, self.global_transform]:
            transform_inputs.update(
                *list(map(Route, brace_expand(obj)) for obj in getattr(t, attr, []))
            )
        return transform_inputs

    def _parse_target(self, target: str | int | Iterable[str | int] | None) -> None:
        # if the target is not a list, cast it
        def _add_target(target):
            if isinstance(target, str):
                # target might be regex, so resolve it against all columns
                self.target_columns.update(
                    x for x in self.all_columns if self._check_against_pattern(x, target)
                )
            elif isinstance(target, (int, float)):
                self.int_targets.add(int(target))
            else:
                raise ValueError(f"Target must be string or int, received {target=}")

        if target is not None and not isinstance(target, Iterable):
            _add_target(target)
        elif target:
            for t in target:
                _add_target(t)

    def _validate(self) -> None:
        if self.columns and not isinstance(self.columns, Iterable):
            raise ValueError(f"columns must be an iterable of strings, received {self.columns}")
        # sanity checks for targets

        for target in self.target_columns:
            # if target is a string and specific columns are supposed to be
            # loaded, check whether the target is also in the columns

            # targets can also be produced by a transformation, so first collect all
            # columns in one super set
            full_column_set = self.all_columns | self._extract_transform_columns(attr="produces")

            if not any(self._check_against_pattern(target, col) for col in full_column_set):
                raise ValueError(f"target {target} not found in columns {full_column_set=}")

            if not any(self._check_against_pattern(target, col) for col in self.all_columns):
                raise ValueError(f"target {target} not found in columns")

        # if target is an integer, this is a class index
        # this should be >= 0
        if any(target < 0 for target in self.int_targets):
            raise ValueError(f"int targets must be >= 0, received {self.int_targets}")
        if len(self.int_targets) > 1:
            raise ValueError(
                "There cannot be more than one categorical target per dataset"
                f", received {self.int_targets}"
            )

    def _compute(self):
        if isinstance(self._data, ak.Array):
            return
        self._data = self.data.compute()

    @property
    def data(self) -> ak.Array:
        if self._data is None:
            self.open_options["columns"] = [x.string_column for x in self.all_columns]
            self._data = dak.from_parquet(self.path, **self.open_options)
            if self.global_transform:
                self._data = self.global_transform(self._data)
            self._data.eager_compute_divisions()
        return self._data

    @property
    def input_data(self) -> ak.Array:
        if self._input_data is None:
            self._compute()
            self._input_data = self.data
            for col in self.target_columns:
                self._input_data = remove_ak_column(self._input_data, col.string_column)
        return self._input_data

    @property
    def target_data(self) -> ak.Array:
        if self._target_data is None and len(self.target_columns) > 0:
            self._compute()
            self._target_data = self.data
            for col in self.data_columns:
                self._target_data = remove_ak_column(self._target_data, col.string_column)
        return self._target_data

    def __len__(self):
        return len(self.data)

    def _get_data(self, i: int | Sequence[int], input_data: ak.Array | None = None) -> ak.Array:
        data: ak.Array | Array
        if input_data is None:
            data = self.input_data
        else:
            data = input_data
        return data[i]

    def _create_class_target(self, length: int, input_int_targets: int | None = None) -> ak.Array:
        int_target: int = input_int_targets or self.class_target

        return ak.Array([int_target] * int(length))

    def __getitem__(self, i: int | Sequence[int]) -> (
            ak.Array | tuple[ak.Array, ak.Array] | tuple[ak.Array, ak.Array, ak.Array]
    ):
        # from IPython import embed
        # embed(header=f"entering {self.__class__.__name__}.__getitem__ for index {i}")
        return_data = [self._get_data(i)]
        if len(self.target_columns) == 0 and len(self.int_targets) == 0:
            return_data = return_data[0]
        else:
            if self.target_data:
                return_data.append(self._get_data(i, self.target_data))
            if len(self.int_targets) > 0:
                return_data.append(self._create_class_target(ak.num(return_data[0], axis=0)))

        if self.transform:
            return_data = self.transform(return_data)
        return tuple(return_data) if isinstance(return_data, list) else return_data

    def __getitems__(self, idx: Sequence[int]) -> ak.Array:
        return self.__getitem__(idx)

    def to_list(self) -> list[dict[str, Any]]:
        return self.data.to_list()


class FlatParquetDataset(ParquetDataset):
    def __init__(
        self,
        *args,
        padd_value_float: float = EMPTY_FLOAT,
        padd_value_int: int = EMPTY_INT,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.padd_values = {
            t: padd_value_float
            for t in [np.float16, np.float32, np.float64, np.float128]
        }
        self.padd_values.update({
            t: padd_value_int
            for t in [
                np.uint8, np.uint16, np.uint32, np.uint64,
                np.int8, np.int16, np.int32, np.int64,
            ]
        })
        self.padd_values.update({
            np.bool: False,
        })

        self._input_data: Mapping[str, ak.Array] | None = None
        self._target_data: Mapping[str, ak.Array] | None = None

    def _extract_columns(self, array: ak.Array, route: Route):
        # first, get super set of column
        super_route = Route(route.string_column)
        total_array = super_route.apply(array)

        # determine the type of the array values
        view = flat_np_view(total_array)
        val_type = view.dtype.type
        padding = self.padd_values.get(val_type, None)

        if padding is None:
            from IPython import embed
            embed(header=f"Error for route {route}, val_type={val_type}")
            raise ValueError(f"Could not determine padding value for type {val_type}")

        return route.apply(array, padding)

    @property
    def input_data(self) -> Mapping[str, ak.Array]:
        if self._input_data is None:
            self._input_data = super().input_data
            self._input_data = {
                str(r): self._extract_columns(self._input_data, r)
                for r in self.data_columns
            }
        return self._input_data

    @property
    def target_data(self) -> Mapping[str, ak.Array]:
        if self._target_data is None:
            self._target_data = super().target_data
            self._target_data = {
                str(r): self._extract_columns(self._target_data, r)
                for r in self.target_columns
            }
        return self._target_data

    def __getitem__(self, i: int | Sequence[int]) -> Any | tuple | tuple:
        # from IPython import embed
        # embed(header=f"entering {self.__class__.__name__}.__getitem__ for index {i}")
        return_data = [{key: self._get_data(i, data) for key, data in self.input_data.items()}]
        if len(self.target_columns) == 0 and len(self.int_targets) == 0:
            return_data = return_data[0]
        else:
            if self.target_data:
                return_data.append({key: self._get_data(i, data) for key, data in self.target_data.items()})
            if len(self.int_targets) > 0:
                first_key = list(return_data[0].keys())[0]

                return_data.append({
                    "categorical_target": self._create_class_target(
                        ak.num(return_data[0][first_key], axis=0), input_int_targets=self.class_target
                    )
                })

        if self.transform:
            return_data = self.transform(return_data)
        return tuple(return_data) if isinstance(return_data, list) else return_data


class FlatTorchDataset(ParquetDataset):
    def __init__(
        self,
        *args,
        embedding_fields: list[str] = [],
        padd_value_float: float = EMPTY_FLOAT,
        padd_value_int: int = EMPTY_INT,
        num_transform: Callable | None = None,
        embed_transform: Callable | None = None,
        ignored_columns: list[str] | None = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.padd_values = {
            t: padd_value_float
            for t in [np.float16, np.float32, np.float64]
        }
        self.padd_values.update({
            t: padd_value_int
            for t in [
                np.uint8, np.uint16, np.uint32, np.uint64,
                np.int8, np.int16, np.int32, np.int64,
            ]
        })
        self.padd_values.update({
            np.bool: False,
        })

        self.num_transform = num_transform
        self.embed_transform = embed_transform

        self._input_data: tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]] | None = None
        self._numerical_data: Mapping[str, torch.Tensor] | None = None
        self._embed_data: Mapping[str, torch.Tensor] | None = None
        self._target_data: Mapping[str, torch.Tensor] | None = None
        self.device = device
        self.embedding_fields = set(col for col in self.data_columns if col.column in embedding_fields)
        # Remove the fields needed for the transformation but not in the data
        self.data_columns = self.data_columns - {Route(x) for x in ignored_columns or []}

    def _extract_columns(self, array: ak.Array, route: Route) -> torch.Tensor:
        # first, get super set of column
        super_route = Route(route.string_column)
        total_array = super_route.apply(array)

        # determine the type of the array values
        view = flat_np_view(total_array)
        val_type = view.dtype.type
        padding = self.padd_values.get(val_type, None)

        if padding is None:
            raise ValueError(f"Could not determine padding value for type {val_type}")

        return_route = route.apply(array, padding)
        return ak.to_torch(return_route).to(device=self.device)

    def _create_class_target(self, length: int, input_int_targets: int | None = None) -> torch.Tensor:
        int_target: int = input_int_targets or self.class_target

        return torch.tensor([int_target] * int(length), device=self.device)

    @ParquetDataset.input_data.getter
    def input_data(self) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
        if self._input_data is None:
            self._input_data = (self.embed_data, self.numerical_data)
        return self._input_data

    @property
    def numerical_data(self) -> Mapping[str, torch.Tensor]:
        if self._numerical_data is None:
            self._numerical_data = {
                str(r): self._extract_columns(super(FlatTorchDataset, self).input_data, r).unsqueeze(-1)
                for r in sorted(self.data_columns, key=lambda x: x.column)
                if r not in self.embedding_fields
            }
            if self.num_transform:
                self._numerical_data = self.num_transform(self._numerical_data)
            self._numerical_data = {"num": torch.cat(list(self._numerical_data.values()), dim=-1)}
        return self._numerical_data

    @property
    def embed_data(self) -> Mapping[str, torch.Tensor]:
        if self._embed_data is None:
            self._embed_data = {
                str(r).replace(".", "_"): self._extract_columns(super(FlatTorchDataset, self).input_data, r).int()
                for r in sorted(self.embedding_fields, key=lambda x: x.column)
            }
            if self.embed_transform:
                self._embed_data = self.embed_transform(self._embed_data)
        return self._embed_data

    @ParquetDataset.target_data.getter
    def target_data(self) -> Mapping[str, torch.Tensor]:
        if self._target_data is None:
            self._target_data = {
                str(r): self._extract_columns(super(FlatTorchDataset, self).target_data, r)
                for r in self.target_columns
            }
        return self._target_data

    def __getitem__(self, i: int | Sequence[int]) -> Any | tuple | tuple:
        # from IPython import embed
        # embed(header=f"entering {self.__class__.__name__}.__getitem__ for index {i}")
        return_data = [
            {key: self._get_data(i, data) for key, data in inp.items()}
            for inp in self.input_data
        ]
        if len(self.target_columns) == 0 and len(self.int_targets) == 0:
            return_data = return_data[0]
        else:
            if self.target_data:
                return_data.append({key: self._get_data(i, data) for key, data in self.target_data.items()})
            if len(self.int_targets) > 0:
                length = 1 if isinstance(i, int) else len(i)
                return_data.append({
                    "categorical_target": self._create_class_target(
                        length, input_int_targets=self.class_target
                    )
                })

        if self.transform:
            return_data = self.transform(return_data)
        return tuple(return_data) if isinstance(return_data, list) else return_data


class ListDataset(Dataset):

    def __init__(self, len: int, prefix: str = "data"):
        self.len = len
        self.prefix = prefix
        self.data = [f"{self.prefix}_{i}" for i in range(self.len)]
        self.weights = np.linspace(0.1, 1.0, self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, i: int) -> tuple[str, float]:
        return (self.data[i], self.weights[i])

    def to_list(self) -> list[str]:
        return list(zip(self.data, self.weights))
