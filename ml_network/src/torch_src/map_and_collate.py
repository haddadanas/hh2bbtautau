from __future__ import annotations

__all__ = [
    "MapAndCollate", "FlatMapAndCollate", "NestedMapAndCollate", "NestedDictMapAndCollate"
]

from typing import Callable, Collection, Sequence

import torch
import numpy as np
import awkward as ak

from ml_network.src.cf_utils import T, reorganize_idx
from ml_network.src.torch_src.datasets import FlatRowgroupParquetDataset


class MapAndCollate:
    """A simple transform that takes a batch of indices, maps with dataset, and then applies
    collate.
    TODO: make this a standard utility in torchdata.nodes
    """

    def __init__(self, dataset: Collection[T], collate_fn: Callable):
        self.dataset = dataset
        self.collate_fn = collate_fn

    def __call__(self, batch_of_indices: list[int]):
        batch = [self.dataset[i] for i in batch_of_indices]
        return self.collate_fn(batch)


class FlatMapAndCollate(MapAndCollate):
    """A simple transform that takes a batch of indices, maps with dataset, and then applies
    collate.
    TODO: make this a standard utility in torchdata.nodes
    """

    def __call__(self, idx: int):
        batch = self.dataset[idx]
        return self.collate_fn(batch)


class NestedMapAndCollate(MapAndCollate):
    def __init__(
        self,
        dataset: dict[str, Collection[T]],
        collate_fn: Callable | None = None,
    ):
        self.dataset = dataset
        self.collate_fn: Callable = collate_fn or self._default_collate

    def _concat_batches(
        self,
        batch: list[T],
        current_batch: Sequence[T],
        concat_fn: Callable,
        *args,
        **kwargs,
    ) -> Sequence[T]:
        if isinstance(current_batch, (tuple, list)):
            if len(batch) == 0:
                batch = list(current_batch)
            else:
                for idx, item in enumerate(current_batch):
                    batch[idx] = concat_fn((batch[idx], item), *args, **kwargs)
        else:
            batch = concat_fn((batch, current_batch), *args, **kwargs)
        return batch

    def _default_collate(self, idx: dict[str, Sequence[int]]) -> Sequence[object]:
        batch: list[object] = []

        # helper function to concatenate different types of objects

        for key, indices in idx.items():
            current_batch = self.dataset[key][indices]
            concat_fn = ak.concatenate
            if isinstance(current_batch, (list, tuple)):
                if all(isinstance(x, torch.Tensor) for x in current_batch):
                    concat_fn = torch.cat
            elif isinstance(current_batch, torch.Tensor):
                concat_fn = torch.cat

            batch = self._concat_batches(batch=batch, current_batch=current_batch, concat_fn=concat_fn)

        return batch

    def __call__(self, idx: dict[str, Sequence[int]]) -> Sequence[object]:

        return self.collate_fn(idx)


class NestedDictMapAndCollate(NestedMapAndCollate):

    # helper function to concatenate different types of objects
    def _concat_dicts(
        self,
        input_arrays: Sequence[dict[str, T]],
        *args,
        **kwargs,
    ) -> dict[str, T]:
        return_dict = dict()
        first_dict = input_arrays[0]
        for key in first_dict.keys():
            sub_arrays = list(map(lambda x: x.get(key), input_arrays))
            collate_fn = ak.concatenate
            if all(isinstance(x, torch.Tensor) for x in sub_arrays):
                collate_fn = torch.cat
            try:
                return_dict[key] = collate_fn(sub_arrays, *args, **kwargs)
            except Exception as e:
                print(e)
                from IPython import embed
                embed(header=f"Encountered error for key {key} in {self.__class__.__name__}._concat_dict")

        return return_dict

    def _default_collate(self, idx: dict[str, Sequence[int]]) -> Sequence[object]:
        batch: list[object] = []
        # helper function to concatenate different types of objects
        for key, indices in idx.items():
            current_batch = self.dataset[key][indices]
            batch = self._concat_batches(batch=batch, current_batch=current_batch, concat_fn=self._concat_dicts)

        return batch


class FlatListRowgroupMapAndCollate(NestedDictMapAndCollate):
    """A simple transform that takes a batch of indices, maps with dataset, and then applies
    collate.
    TODO: make this a standard utility in torchdata.nodes
    """

    def _default_collate(self, idx: dict[str, dict[tuple[int, int], Sequence[int]]]) -> Sequence[object]:
        batch: list[object] = []

        # the indices are dictionaries with multiple entries, so loop
        idx = reorganize_idx(idx)
        for (dataset_idx, rowgroup), entry_idx in idx.items():
            try:
                dataset = self.dataset[dataset_idx]
                current_batch = dataset[((rowgroup, entry_idx),)]
                batch = self._concat_batches(batch=batch, current_batch=current_batch, concat_fn=self._concat_dicts)
            except Exception:
                from IPython import embed
                embed(header=f"Detected problem in {self.__class__.__name__}")

        return batch


class NestedListRowgroupMapAndCollate(FlatListRowgroupMapAndCollate):
    dataset: dict[str, Sequence[FlatRowgroupParquetDataset]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_datasets = self.dataset

    def _default_collate(self, idx: dict[str, dict[tuple[int, int], Sequence[int]]]) -> Sequence[object]:
        batch: list[object] = []
        keys = np.array(list(idx.keys()))

        worker_info = torch.utils.data.get_worker_info()
        if worker_info and worker_info.num_workers > 1 and worker_info.id is not None:
            key_idx = np.indices(keys.shape)
            mask = ((key_idx + 1) % (worker_info.id + 1)) == 0
            keys = keys[key_idx[mask]]
            print(f"Worker {worker_info.id}: {keys=}")

        for key in keys:
            indices = idx[key]
            self.dataset = self.all_datasets[key]
            current_batch = super()._default_collate(indices)
            batch = self._concat_batches(batch=batch, current_batch=current_batch, concat_fn=self._concat_dicts)

        return batch
