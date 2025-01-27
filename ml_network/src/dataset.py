from __future__ import annotations

import awkward as ak

# TODO https://pytorch.org/data/0.9/generated/torchdata.datapipes.iter.ParquetDataFrameLoader.html Parquet DataFrame


def remove_ak_fields(array: ak.Array, fields: list[str] | str) -> ak.Array:
    """Remove fields from an awkward array.

    Args:
        array (ak.Array): The array to remove fields from.
        fields (list | str): The fields to remove.

    Returns:
        ak.Array: The array with the fields removed.
    """
    if isinstance(fields, str):
        fields = [fields]
    super_fields: set[str] = set()

    for field in fields:
        if field.count('.') > 0:
            field = field.split('.')
            super_fields.add(field[0])
        array = ak.without_field(array, field)
    for field in super_fields:
        if array[field].fields:  # type: ignore
            continue
        array = ak.without_field(array, field)

    return array


def get_ak_field_names(array: ak.Array):
    """Get the field names of an awkward array.

    Args:
        array (ak.Array): The array to get the field names from.

    Returns:
        list: The field names.
    """
    fields = []
    for field in array.fields:
        if not getattr(array[field], 'fields', None):
            fields.append(field)
            continue
        fields.extend(map(lambda s: f"{field}.{s}", get_ak_field_names(array[field])))
    return fields


class DataContainer:

    # Keep track of assigned ids
    assigned_ids = []

    def __init__(
            self,
            is_signal: bool,
            name: str = "Dataset",
            path: str | None = None,
            max_events: int = None,
            array: ak.Array = None,
            channel_id: ak.Array = None,
            process_id: ak.Array = None,
            weights: ak.Array = None,
            cls_id: int = None,
            drop_fields: list[str] = [],
            use_weights: bool = True,
    ):
        self.name = name
        self.path = path
        self.is_signal = is_signal
        self.channel_id = None
        self.process_id = None
        self.weights = None
        self.feature_names = []

        # Assign an id to the dataset
        if cls_id is not None:
            self.id = cls_id
        else:
            self.id = DataContainer.assigned_ids[-1] + 1 if DataContainer.assigned_ids else 0
        DataContainer.assigned_ids.append(self.id)

        if array is not None:
            self._dataset_from_array(array, channel_id, process_id, weights)
        else:
            # Load the dataset
            self._load_dataset(max_events=max_events, drop_fields=drop_fields, use_weights=use_weights)

    def _load_dataset(self, max_events: int = None, drop_fields: list[str] = [], use_weights: bool = True):
        array = ak.from_parquet(self.path)
        if max_events:
            array = array[:max_events]
        self.channel_id = array.channel_id
        self.process_id = array.process_id
        if use_weights:
            self.weights = array.normalization_weight
        self.features = remove_ak_fields(array, ['channel_id', 'process_id', 'normalization_weight'] + drop_fields)
        self.feature_names = get_ak_field_names(self.features)

    def _dataset_from_array(self, array: ak.Array, channel_id: ak.Array, process_id: ak.Array, weights: ak.Array):
        self.channel_id = channel_id
        self.process_id = process_id
        self.weights = weights
        self.features = array
        self.feature_names = get_ak_field_names(self.features)

    def get_sub_dataset(self, features: list | None = None):
        if not features:
            return self
        return DataContainer(
            is_signal=self.is_signal,
            name=self.name,
            array=self.get_features(features),
            channel_id=self.channel_id,
            process_id=self.process_id,
            weights=self.weights,
            cls_id=self.id,
        )

    def get_features(self, features: list):
        if not features:
            return self.features
        # flatten the features
        keep_features = [_f for f in features for _f in f if isinstance(f, list)]
        remove_fields = [f for f in self.feature_names if f not in keep_features]
        return remove_ak_fields(self.features, remove_fields)

    def get_column(self, column: str) -> ak.Array:
        if column in ["channel_id", "process_id", "weights"]:
            return getattr(self, column)
        if not column.count('.'):
            return getattr(self.features, column)
        array = self.features
        for field in column.split('.'):
            array = getattr(array, field)
        return array

    def get_features_dict(self):
        return {f: self.get_column(f) for f in sorted(self.feature_names)}

    @property
    def n_fields(self):
        return len(self.feature_names)

    @property
    def array(self):
        return self.features

    def __repr__(self):
        return f"{self.name}"

    def __str__(self):
        return f"Dataset {self.name}"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, field):
        return self.features[field]

    def __iter__(self):
        return iter(self.features)
