from __future__ import annotations

__all__ = [
    "MapAndCollate", "FlatMapAndCollate", "NestedMapAndCollate", "NestedDictMapAndCollate"
]

from typing import Callable, Sequence, Sized

import awkward as ak

import torch


class MapAndCollate:
    """A simple transform that takes a batch of indices, maps with dataset, and then applies
    collate.
    TODO: make this a standard utility in torchdata.nodes
    """

    def __init__(self, dataset: Sized, collate_fn: Callable):
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
        dataset: dict[str, Sized],
        collate_fn: Callable | None = None,
    ):
        self.dataset = dataset
        self.collate_fn: Callable = collate_fn or self._default_collate

    def _concat_batches(
        self,
        batch: list[object],
        current_batch: Sequence[object],
        concat_fn: Callable,
        *args,
        **kwargs,
    ) -> Sequence[object]:
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
    def _default_collate(self, idx: dict[str, Sequence[int]]) -> Sequence[object]:
        batch: list[object] = []

        # helper function to concatenate different types of objects
        def _concat_dicts(
            input_arrays: Sequence[dict[str, object]],
            *args,
            **kwargs,
        ) -> dict[str, object]:
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

        for key, indices in idx.items():
            current_batch = self.dataset[key][indices]
            batch = self._concat_batches(batch=batch, current_batch=current_batch, concat_fn=_concat_dicts)

        return batch
