from __future__ import annotations
from typing import Iterable

import tensorflow as tf


class MetrixTF:

    def __init__(self):
        pass

    def _get_masks_from_scores(self: MetrixTF, scores: Iterable, threshold: float | Iterable) -> tf.Tensor:
        """
        Get selection masks from the network scores.

        Args:
            scores (Iterable): Network scores, typically a list or array of scores for each event.
            threshold (float | Iterable): Threshold or list of thresholds to apply to the scores.
                If a single float is provided, it is used for all scores.

        Returns:
            tf.Tensor: Selection masks, where each mask is a boolean array indicating
                        whether each score exceeds the corresponding threshold.
        """

        return tf.greater(scores, threshold)

    def _validate_selection_inputs(
        self: MetrixTF,
        selection_mask: Iterable | None,
        selection_score: Iterable | None,
        threshold: float | Iterable | None,
    ) -> tuple[tf.Tensor, float | Iterable]:
        """
        Validate the selection mask or score and threshold inputs.

        Args:
            selection_mask (Iterable | None): Selection mask, a boolean array indicating selected events.
            selection_score (Iterable | None): Network score, an array of scores for each event.
            threshold (float | Iterable | None): Threshold or list of thresholds to apply to the scores.

        Returns:
            tuple[tf.Tensor, float | Iterable]: Validated selection mask and threshold. If selection_score is provided,
                the selection mask is generated from the scores and thresholds.

        Raises:
            ValueError: If neither selection_mask nor selection_score is provided.
        """
        if selection_mask is None and selection_score is None:
            raise ValueError("Either selection_mask or selection_score must be given.")

        if selection_score is not None and threshold is None:
            threshold = 0.5

        if selection_score is not None:
            selection_mask = self._get_masks_from_scores(selection_score, threshold)
        else:
            selection_mask = tf.convert_to_tensor(selection_mask)

        return selection_mask, threshold

    def _generate_signal_mask(self: MetrixTF, process_ids: tf.Tensor, selection_mask: tf.Tensor) -> tf.Tensor:
        """
        Generate a mask for signal events based on process ids.

        Args:
            process_ids (tf.Tensor): 1D Array of process/dataset ids or true labels.
            selection_mask (tf.Tensor): Selection mask, a boolean array indicating selected events.

        Returns:
            tf.Tensor: Signal mask, a boolean array indicating signal events.

        Raises:
            ValueError: If process_ids is not a 1D array with a length equal to the length of selection masks.
        """
        if not process_ids.ndim == 1 and not process_ids.shape[0] == selection_mask.shape[0]:
            raise ValueError("The process ids must be a 1D array with a length equal to the length of selection masks.")

        if process_ids.dtype == tf.int32:
            # get unique process ids
            signal_process_ids = [pid for pid in set(process_ids.numpy()) if self._check_dataset_type(pid)]
            signal_mask = tf.ones(selection_mask.shape[0], dtype=tf.bool)

            for pid in signal_process_ids:
                signal_mask = tf.logical_or(signal_mask, tf.equal(process_ids, pid))
        else:
            signal_mask = process_ids

        return signal_mask

    @tf.function
    def selection_efficiency(
        self: MetrixTF,
        *args,
        selection_mask: Iterable | None = None,
        selection_score: Iterable | None = None,
        threshold: float | Iterable | None = None,
    ) -> float | tf.Tensor:
        """
        Calculate the selection efficiency of a given selection mask or network score.

        Args:
            selection_mask (Iterable | None): Selection mask, a boolean array indicating selected events.
            selection_score (Iterable | None): Network score, an array of scores for each event.
            threshold (float | Iterable | None): Threshold or list of thresholds to apply to the scores.

        Returns:
            float | tf.Tensor: Selection efficiency, the fraction of events that pass the selection criteria.
                If multiple thresholds are provided, returns an array of efficiencies.

        Raises:
            ValueError: If neither selection_mask nor selection_score is provided.
        """
        selection_mask, threshold = self._validate_selection_inputs(selection_mask, selection_score, threshold)

        # cast to int
        selection_mask = tf.cast(selection_mask, tf.float32)

        # Calculate the selection efficiency
        selection_eff = tf.reduce_mean(selection_mask, axis=0)

        return selection_eff

    @tf.function
    def signal_acceptance(
        self: MetrixTF,
        process_ids: tf.Tensor,
        *args,
        selection_mask: Iterable | None = None,
        selection_score: Iterable | None = None,
        threshold: float | Iterable | None = None,
    ) -> float | tf.Tensor:
        """
        Calculate the signal acceptance of a given selection mask or network score.

        Args:
            process_ids (tf.Tensor): 1D Array of process/dataset ids or true labels.
            selection_mask (Iterable | None): Selection mask, a boolean array indicating selected events.
            selection_score (Iterable | None): Network score, an array of scores for each event.
            threshold (float | Iterable | None): Threshold or list of thresholds to apply to the scores.

        Returns:
            float | tf.Tensor: Signal acceptance, the fraction of signal events that pass the selection criteria.
                If multiple thresholds are provided, returns an array of acceptances.

        Raises:
            ValueError: If neither selection_mask nor selection_score is provided.
            ValueError: If process_ids is not a 1D array with a length equal to the length of selection masks.
        """
        selection_mask, threshold = self._validate_selection_inputs(selection_mask, selection_score, threshold)
        signal_mask = self._generate_signal_mask(process_ids, selection_mask)

        # select only signal events
        signal_selection_mask = tf.boolean_mask(selection_mask, signal_mask, axis=0)
        signal_selection_mask = tf.cast(signal_selection_mask, tf.float32)

        # Calculate the signal acceptance
        signal_acceptance = tf.reduce_mean(signal_selection_mask, axis=0)

        return signal_acceptance

    @tf.function
    def signal_purity(
        self: MetrixTF,
        process_ids: tf.Tensor,
        selection_mask: Iterable | None = None,
        selection_score: Iterable | None = None,
        threshold: float | Iterable | None = None,
    ) -> float | tf.Tensor:
        """
        Calculate the selection purity of a given selection mask or network score.

        Args:
            process_ids (tf.Tensor): 1D Array of process/dataset ids or true labels.
            selection_mask (Iterable | None): Selection mask, a boolean array indicating selected events.
            selection_score (Iterable | None): Network score, an array of scores for each event.
            threshold (float | Iterable | None): Threshold or list of thresholds to apply to the scores.

        Returns:
            float | tf.Tensor: Selection purity, the fraction of selected events that are signal events.
                If multiple thresholds are provided, returns an array of purities.

        Raises:
            ValueError: If neither selection_mask nor selection_score is provided.
            ValueError: If process_ids is not a 1D array with a length equal to the length of selection masks.
        """
        selection_mask, threshold = self._validate_selection_inputs(selection_mask, selection_score, threshold)
        if not tf.reduce_any(selection_mask):
            return tf.constant(0.0, dtype=tf.float32)

        signal_mask = self._generate_signal_mask(process_ids, selection_mask)

        signal_purity_array = tf.cast(tf.boolean_mask(signal_mask, selection_mask), tf.float32)
        selection_purity = tf.reduce_mean(signal_purity_array, axis=0)

        return selection_purity


class SelectionMetrixMetric(tf.keras.metrics.Metric):
    def __init__(self, name: str = "selection_metrices", **kwargs):
        super(SelectionMetrixMetric, self).__init__(name=name, **kwargs)
        self.metrices = MetrixTF()
        self.selection_efficiency = self.add_variable(
            shape=(),
            name="selection_efficiency",
            initializer="zeros",
            dtype=tf.float32,
            aggregation="mean",
        )
        self.signal_acceptance = self.add_variable(
            shape=(),
            name="signal_acceptance",
            initializer="zeros",
            dtype=tf.float32,
            aggregation="mean",
        )
        self.signal_purity = self.add_variable(
            shape=(),
            name="signal_purity",
            initializer="zeros",
            dtype=tf.float32,
            aggregation="mean",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        if y_true.ndim == 2:
            true = tf.math.argmax(y_true, axis=-1) == 1
            pred = y_pred[:, 1]
        else:
            true = y_true
            pred = y_pred[:, 0]
        sel_eff = self.metrices.selection_efficiency(selection_score=pred, threshold=0.5)
        sig_acc = self.metrices.signal_acceptance(process_ids=true, selection_score=pred, threshold=0.5)
        sig_pur = self.metrices.signal_purity(process_ids=true, selection_score=pred, threshold=0.5)

        self.selection_efficiency.assign(sel_eff)
        self.signal_acceptance.assign(sig_acc)
        self.signal_purity.assign(sig_pur)

    def result(self):
        return {
            "selection_efficiency": self.selection_efficiency,
            "signal_acceptance": self.signal_acceptance,
            "signal_purity": self.signal_purity,
        }
