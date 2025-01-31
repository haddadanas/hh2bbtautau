from __future__ import annotations
from typing import Iterable

import numpy as np
import order as od

from columnflow.util import maybe_import

plt = maybe_import("matplotlib.pyplot")


class SelectionMetrix:

    def __init__(self, config: od.Config):
        self.config = config

    def _check_dataset_type(self: SelectionMetrix, dataset_info: str | int | Iterable[str | int]) -> int | np.ndarray:
        """
        Determine whether the dataset is a signal or background dataset.

        Args:
            dataset_info (str | int | Iterable[str | int]): Dataset identifier(s).
                Can be a single identifier or a list of identifiers.

        Returns:
            int | np.ndarray: 1 if the dataset is a signal dataset, otherwise 0.
                If multiple identifiers are provided, returns an array of 1s and 0s.

        Raises:
            ValueError: If the dataset identifier is not found in the configuration.
        """
        if isinstance(dataset_info, (str, int)):
            dataset_info = [dataset_info]
        dataset_inst = [self.config.get_dataset(dataset_id) for dataset_id in dataset_info]
        signal_tag = [dataset.has_tag("signal") for dataset in dataset_inst]

        if len(signal_tag) == 1:
            return int(signal_tag[0])
        else:
            return np.array(signal_tag).astype(int)

    def _get_masks_from_scores(self: SelectionMetrix, scores: Iterable, threshold: float | Iterable) -> Iterable:
        """
        Get selection masks from the network scores.

        Args:
            scores (Iterable): Network scores, typically a list or array of scores for each event.
            threshold (float | Iterable): Threshold or list of thresholds to apply to the scores.
                If a single float is provided, it is used for all scores.

        Returns:
            Iterable: Selection masks, where each mask is a boolean array indicating
                        whether each score exceeds the corresponding threshold.
        """
        if isinstance(threshold, float):
            threshold = [threshold]

        return np.array([scores > th for th in threshold])

    def _validate_selection_inputs(
        self: SelectionMetrix,
        selection_mask: Iterable | None,
        selection_score: Iterable | None,
        threshold: float | Iterable | None,
    ) -> tuple[Iterable, float | Iterable]:
        """
        Validate the selection mask or score and threshold inputs.

        Args:
            selection_mask (Iterable | None): Selection mask, a boolean array indicating selected events.
            selection_score (Iterable | None): Network score, an array of scores for each event.
            threshold (float | Iterable | None): Threshold or list of thresholds to apply to the scores.

        Returns:
            tuple[Iterable, float | Iterable]: Validated selection mask and threshold. If selection_score is provided,
                the selection mask is generated from the scores and thresholds.

        Raises:
            ValueError: If neither selection_mask nor selection_score is provided.
        """
        if selection_mask is None and selection_score is None:
            raise ValueError("Either selection_mask or selection_score must be given.")

        if selection_score is not None and threshold is None:
            threshold = np.linspace(0, 1, 101)

        if selection_score is not None:
            selection_mask = self._get_masks_from_scores(selection_score, threshold)
        else:
            selection_mask = np.array([selection_mask])

        return selection_mask, threshold

    def _generate_signal_mask(self: SelectionMetrix, process_ids: np.ndarray, selection_mask: np.ndarray) -> np.ndarray:
        """
        Generate a mask for signal events based on process ids.

        Args:
            process_ids (np.ndarray): 1D Array of process/dataset ids or true labels.
            selection_mask (np.ndarray): Selection mask, a boolean array indicating selected events.

        Returns:
            np.ndarray: Signal mask, a boolean array indicating signal events.

        Raises:
            ValueError: If process_ids is not a 1D array with a length equal to the length of selection masks.
        """
        if not len(process_ids) == selection_mask.shape[1] and not process_ids.ndim == 1:
            raise ValueError("The process ids must be a 1D array with a length equal to the length of selection masks.")

        if process_ids.dtype == int:
            # get unique process ids
            signal_process_ids = [pid for pid in set(process_ids) if self._check_dataset_type(pid)]
            signal_mask = np.ones(selection_mask.shape[1], dtype=bool)

            for pid in signal_process_ids:
                signal_mask = np.logical_or(signal_mask, process_ids == pid)
        else:
            signal_mask = process_ids

        return signal_mask

    def selection_efficiency(
        self: SelectionMetrix,
        *args,
        selection_mask: Iterable | None = None,
        selection_score: Iterable | None = None,
        threshold: float | Iterable | None = None,
    ) -> float | Iterable:
        """
        Calculate the selection efficiency of a given selection mask or network score.

        Args:
            selection_mask (Iterable | None): Selection mask, a boolean array indicating selected events.
            selection_score (Iterable | None): Network score, an array of scores for each event.
            threshold (float | Iterable | None): Threshold or list of thresholds to apply to the scores.

        Returns:
            float | Iterable: Selection efficiency, the fraction of events that pass the selection criteria.
                If multiple thresholds are provided, returns an array of efficiencies.

        Raises:
            ValueError: If neither selection_mask nor selection_score is provided.
        """
        selection_mask, threshold = self._validate_selection_inputs(selection_mask, selection_score, threshold)

        # Calculate the selection efficiency
        selection_eff = np.mean(selection_mask, axis=1)

        if selection_eff.shape[0] == 1:
            selection_eff = selection_eff[0]

        return selection_eff

    def signal_acceptance(
        self: SelectionMetrix,
        process_ids: np.ndarray,
        *args,
        selection_mask: Iterable | None = None,
        selection_score: Iterable | None = None,
        threshold: float | Iterable | None = None,
    ) -> float | Iterable:
        """
        Calculate the signal acceptance of a given selection mask or network score.

        Args:
            process_ids (np.ndarray): 1D Array of process/dataset ids or true labels.
            selection_mask (Iterable | None): Selection mask, a boolean array indicating selected events.
            selection_score (Iterable | None): Network score, an array of scores for each event.
            threshold (float | Iterable | None): Threshold or list of thresholds to apply to the scores.

        Returns:
            float | Iterable: Signal acceptance, the fraction of signal events that pass the selection criteria.
                If multiple thresholds are provided, returns an array of acceptances.

        Raises:
            ValueError: If neither selection_mask nor selection_score is provided.
            ValueError: If process_ids is not a 1D array with a length equal to the length of selection masks.
        """
        selection_mask, threshold = self._validate_selection_inputs(selection_mask, selection_score, threshold)
        signal_mask = self._generate_signal_mask(process_ids, selection_mask)

        # select only signal events
        signal_selection_mask = selection_mask[:, signal_mask]

        # Calculate the signal acceptance
        signal_acceptance = np.mean(signal_selection_mask, axis=1)

        if signal_acceptance.shape[0] == 1:
            signal_acceptance = signal_acceptance[0]

        return signal_acceptance

    def signal_purity(
        self: SelectionMetrix,
        process_ids: np.ndarray,
        selection_mask: Iterable | None = None,
        selection_score: Iterable | None = None,
        threshold: float | Iterable | None = None,
    ) -> float | Iterable:
        """
        Calculate the selection purity of a given selection mask or network score.

        Args:
            process_ids (np.ndarray): 1D Array of process/dataset ids or true labels.
            selection_mask (Iterable | None): Selection mask, a boolean array indicating selected events.
            selection_score (Iterable | None): Network score, an array of scores for each event.
            threshold (float | Iterable | None): Threshold or list of thresholds to apply to the scores.

        Returns:
            float | Iterable: Selection purity, the fraction of selected events that are signal events.
                If multiple thresholds are provided, returns an array of purities.

        Raises:
            ValueError: If neither selection_mask nor selection_score is provided.
            ValueError: If process_ids is not a 1D array with a length equal to the length of selection masks.
        """
        selection_mask, threshold = self._validate_selection_inputs(selection_mask, selection_score, threshold)
        signal_mask = self._generate_signal_mask(process_ids, selection_mask)

        selection_purity = []
        for sel in selection_mask:
            selection_purity.append(np.mean(signal_mask[sel], axis=0))

        if len(selection_purity) == 1:
            selection_purity = selection_purity[0]

        return selection_purity

    def plot_curves(self: SelectionMetrix, process_ids: np.ndarray, selection_mask: Iterable | None = None,
                    selection_score: Iterable | None = None, threshold: float | Iterable | None = None,
                    ax: plt.Axes | None = None, **kwargs) -> plt.Axes:
        """
        Plot selection efficiency, signal acceptance, and signal purity curves.

        Args:
            process_ids (np.ndarray): 1D Array of process/dataset ids or true labels.
            selection_mask (Iterable | None): Selection mask, a boolean array indicating selected events.
            selection_score (Iterable | None): Network score, an array of scores for each event.
            threshold (float | Iterable | None): Threshold or list of thresholds to apply to the scores.
            ax (plt.Axes | None): Matplotlib axes to plot the curves.
            **kwargs: Additional keyword arguments to pass to the plot function.

        Returns:
            plt.Axes: Matplotlib axes containing the plotted curves.

        Raises:
            ValueError: If neither selection_mask nor selection_score is provided.
            ValueError: If process_ids is not a 1D array with a length equal to the length of selection masks.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        selection_mask, threshold = self._validate_selection_inputs(selection_mask, selection_score, threshold)

        # Plot selection efficiency
        selection_eff = self.selection_efficiency(selection_mask=selection_mask)
        ax.plot(threshold, selection_eff, label="Selection Efficiency", **kwargs)

        # Plot signal acceptance
        signal_acceptance = self.signal_acceptance(process_ids, selection_mask=selection_mask)
        ax.plot(threshold, signal_acceptance, label="Signal Acceptance", **kwargs)

        # Plot signal purity
        signal_purity = self.signal_purity(process_ids, selection_mask=selection_mask)
        ax.plot(threshold, signal_purity, label="Signal Purity", **kwargs)

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Fraction")
        ax.set_title("Selection Metrics")
        ax.legend()

        return ax
