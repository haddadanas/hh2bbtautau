from __future__ import annotations
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter


class DynamicTensor():
    """ Tensor with a pytorch tensor backend, which automatically expands, when full.
    This offers a similar functionality to the append method of a list. """
    def __init__(self, size: int, dtype: torch.dtype, device: torch.device):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.tensor = torch.zeros(size, dtype=dtype, device=device)
        self.idx = 0

    def _expand(self):
        self.size = self.size + 50
        new_tensor = torch.zeros(self.size, dtype=self.dtype, device=self.device)
        new_tensor[:self.idx] = self.tensor
        self.tensor = new_tensor

    def append(self, data: float):
        if self.idx == self.size:
            self._expand()
        self.tensor[self.idx] = data
        self.idx += 1

    def __getitem__(self, idx: int):
        return self.tensor[idx]

    def __len__(self):
        return self.idx

    def __iter__(self):
        return iter(self.tensor[:self.idx])

    def __repr__(self):
        return str(self.tensor[:self.idx])


class MetrixTF:

    def __init__(self):
        pass

    def _get_masks_from_scores(self: MetrixTF, scores: Tensor, threshold: float | Tensor) -> Tensor:
        """
        Get selection masks from the network scores.

        Args:
            scores (Iterable): Network scores, typically a list or array of scores for each event.
            threshold (float | Iterable): Threshold or list of thresholds to apply to the scores.
                If a single float is provided, it is used for all scores.

        Returns:
            tensor: Selection masks, where each mask is a boolean array indicating
                        whether each score exceeds the corresponding threshold.
        """

        return torch.gt(scores, threshold)

    def _validate_selection_inputs(
        self: MetrixTF,
        selection_score: Tensor,
        threshold: float | Iterable | None,
    ) -> Tensor:
        """
        Validate the selection mask or score and threshold inputs.

        Args:
            selection_mask (Iterable | None): Selection mask, a boolean array indicating selected events.
            selection_score (Iterable | None): Network score, an array of scores for each event.
            threshold (float | Iterable | None): Threshold or list of thresholds to apply to the scores.

        Returns:
            tuple[tensor, float | Iterable]: Validated selection mask and threshold. If selection_score is provided,
                the selection mask is generated from the scores and thresholds.

        Raises:
            ValueError: If neither selection_mask nor selection_score is provided.
        """

        if threshold is None:
            threshold = 0.5
        return self._get_masks_from_scores(selection_score, threshold)

    def selection_efficiency(
        self: MetrixTF,
        selection_score: Iterable | None = None,
        threshold: float | Iterable | None = None
    ) -> float | Tensor:
        """
        Calculate the selection efficiency of a given selection mask or network score.

        Args:
            selection_mask (Iterable | None): Selection mask, a boolean array indicating selected events.
            selection_score (Iterable | None): Network score, an array of scores for each event.
            threshold (float | Iterable | None): Threshold or list of thresholds to apply to the scores.

        Returns:
            float | tensor: Selection efficiency, the fraction of events that pass the selection criteria.
                If multiple thresholds are provided, returns an array of efficiencies.

        Raises:
            ValueError: If neither selection_mask nor selection_score is provided.
        """
        selection_mask = self._validate_selection_inputs(selection_score, threshold)

        # cast to int
        selection_mask = selection_mask.float()

        # Calculate the selection efficiency
        selection_eff = torch.mean(selection_mask, dim=0)

        return selection_eff

    def signal_acceptance(
        self: MetrixTF,
        y_true: Tensor,
        selection_score: Tensor,
        threshold: float | Iterable | None = None
    ) -> float | Tensor:
        """
        Calculate the signal acceptance of a given selection mask or network score.

        Args:
            process_ids (tensor): 1D Array of process/dataset ids or true labels.
            selection_mask (Iterable | None): Selection mask, a boolean array indicating selected events.
            selection_score (Iterable | None): Network score, an array of scores for each event.
            threshold (float | Iterable | None): Threshold or list of thresholds to apply to the scores.

        Returns:
            float | tensor: Signal acceptance, the fraction of signal events that pass the selection criteria.
                If multiple thresholds are provided, returns an array of acceptances.

        Raises:
            ValueError: If neither selection_mask nor selection_score is provided.
            ValueError: If process_ids is not a 1D array with a length equal to the length of selection masks.
        """
        selection_mask = self._validate_selection_inputs(selection_score, threshold)
        signal_mask = y_true.bool()

        # select only signal events
        signal_selection_mask = selection_mask[signal_mask]
        signal_selection_mask = signal_selection_mask.float()

        # Calculate the signal acceptance
        signal_acceptance = torch.mean(signal_selection_mask, dim=0)

        return signal_acceptance

    def signal_purity(
        self: MetrixTF,
        y_true: Tensor,
        selection_score: Tensor,
        threshold: float | Iterable | None = None
    ) -> float | Tensor:
        """
        Calculate the selection purity of a given selection mask or network score.

        Args:
            process_ids (tensor): 1D Array of process/dataset ids or true labels.
            y_true (Iterable | None): Selection mask, a boolean array indicating selected events.
            selection_score (Iterable | None): Network score, an array of scores for each event.
            threshold (float | Iterable | None): Threshold or list of thresholds to apply to the scores.

        Returns:
            float | tensor: Selection purity, the fraction of selected events that are signal events.
                If multiple thresholds are provided, returns an array of purities.

        Raises:
            ValueError: If neither y_true nor selection_score is provided.
            ValueError: If process_ids is not a 1D array with a length equal to the length of selection masks.
        """
        selection_mask = self._validate_selection_inputs(selection_score, threshold)
        if not torch.any(selection_mask):
            return torch.tensor(0.0, dtype=torch.float32)

        signal_mask = y_true.bool()

        signal_purity_array = signal_mask[selection_mask].float()
        selection_purity = torch.mean(signal_purity_array, dim=0)

        return selection_purity

    def run_metrics(
        self: MetrixTF,
        y_true: Tensor,
        y_pred: Tensor,
        threshold: float = 0.5,
    ) -> dict[str, float | Tensor]:
        """
        Run all selection metrics and return the results as a dictionary.

        Args:
            process_ids (tensor): 1D Array of process/dataset ids or true labels.
            selection_mask (Iterable | None): Selection mask, a boolean array indicating selected events.
            selection_score (Iterable | None): Network score, an array of scores for each event.
            threshold (float | Iterable | None): Threshold or list of thresholds to apply to the scores.

        Returns:
            dict[str, float | tensor]: Dictionary of selection metrics with names as keys and values as values.

        Raises:
            ValueError: If neither selection_mask nor selection_score is provided.
            ValueError: If process_ids is not a 1D array with a length equal to the length of selection masks.
        """
        y_pred = y_pred.flatten()

        selection_eff = self.selection_efficiency(selection_score=y_pred, threshold=threshold)
        signal_acc = self.signal_acceptance(y_true, selection_score=y_pred, threshold=threshold)
        signal_pur = self.signal_purity(y_true, selection_score=y_pred, threshold=threshold)

        return {
            "selection_efficiency": selection_eff,
            "signal_acceptance": signal_acc,
            "signal_purity": signal_pur
        }


class SelectionMetrixMetric():
    def __init__(self, device: torch.device):
        self.metrices = MetrixTF()
        self.selection_efficiency = DynamicTensor(50, dtype=torch.float32, device=device)
        self.signal_acceptance = DynamicTensor(50, dtype=torch.float32, device=device)
        self.signal_purity = DynamicTensor(50, dtype=torch.float32, device=device)

    def append(self, selection_efficiency, signal_acceptance, signal_purity):
        self.selection_efficiency.append(selection_efficiency)
        self.signal_acceptance.append(signal_acceptance)
        self.signal_purity.append(signal_purity)

    def on_epoch_end(self, y_true, y_pred, sample_weight=None):
        results = self.metrices.run_metrics(y_true, y_pred)
        self.append(**results)
        return results

    def result(self):
        return {
            "selection_efficiency": self.selection_efficiency,
            "signal_acceptance": self.signal_acceptance,
            "signal_purity": self.signal_purity,
        }


class TensorBoardCallback():
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def on_epoch_end(self, epoch: int, logs: dict):
        for key, value in logs.items():
            self.writer.add_scalar(f"{key}", value, epoch)

    def on_train_end(self):
        self.writer.flush()
        self.writer.close()
