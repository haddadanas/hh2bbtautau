# coding: utf-8

"""
Test model definition.
"""

from __future__ import annotations

from typing import Any, Sequence
from collections import defaultdict
# from datetime import datetime

import law
import order as od

from columnflow.ml import MLModel
from columnflow.util import DotDict, maybe_import, dev_sandbox
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column, remove_ak_column


ak = maybe_import("awkward")
np = maybe_import("numpy")
tf = maybe_import("tensorflow")
keras = maybe_import("keras")
nprec = maybe_import("numpy.lib.recfunctions")

law.contrib.load("tensorflow")
law.contrib.load("keras")

logger = law.logger.get_logger(__name__)


class SelectionModel(MLModel):

    dataset_names = [
        "hh_ggf_hbb_htt_kl1_kt1_powheg",
        "hh_ggf_hbb_htt_kl0_kt1_powheg",
        "hh_ggf_hbb_htt_kl2p45_kt1_powheg",
        "hh_ggf_hbb_htt_kl5_kt1_powheg",
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        "dy_m50toinf_amcatnlo",
        "tth_hbb_powheg",
    ]

    def __init__(
        self,
        *args,
        folds: int | None = None,
        model_name: str | None = None,
        activation_func: str | None = None,
        batch_norm: bool | None = None,
        input_features: set[str] | None = None,
        nodes: list | None = None,
        L2: bool | None = None,
        validation_split: float | None = None,
        epochs: int = 10,
        version: str = "",
        learningrate: float = 0.001,
        **kwargs,
    ):
        """
        Parameters that need to be set by derived model:
        folds, layers, learningrate, batchsize, epochs, eqweight, dropout,
        processes, ml_process_weights, dataset_names, input_features, store_name,
        """
        single_config = True  # noqa

        super().__init__(*args, **kwargs)

        # class- to instance-level attributes
        # (before being set, self.folds refers to a class-level attribute)
        self.folds = folds or self.folds
        self.model_name = model_name or self.model_name
        self.input_features = input_features or self.input_features
        self.batch_norm = batch_norm or self.batch_norm
        self.activation_func = activation_func or self.activation_func
        self.L2 = L2 or self.L2
        self.nodes = nodes or self.nodes
        self.validation_split = validation_split or self.validation_split
        self.epochs = self.epochs or epochs
        self.version = version
        self.learningrate = learningrate

        # store parameters of interest in the ml_model_inst, e.g. via the parameters attribute
        self.parameters = self.parameters | {
            "batchsize": int(self.parameters.get("batchsize", 512)),
            "layers": tuple(int(layer) for layer in self.parameters.get("layers", self.nodes)),
            "epochs": int(self.parameters.get("epochs", self.epochs)),
            "activation_func": self.parameters.get("activation_func", self.activation_func),
            "L2": self.parameters.get("L2", self.L2),
            "validation_split": self.parameters.get("validation_split", self.validation_split),
            "version": self.parameters.get("version", self.version),
        }

    def setup(self):
        self.dataset_insts = {
            dataset_name: self.config_inst.get_dataset(dataset_name)
            for dataset_name in self.dataset_names
        }
        self.process_insts = {}

        for id, dataset in enumerate(self.dataset_insts.values()):
            proc = dataset.processes.get_first()
            proc.x.ml_id = id
            proc.x.process_weight = 1  # TODO: set process weights
            self.process_insts[dataset.name] = proc

        # dynamically add variables for the quantities produced by this model
        for i in ["background", "signal"]:
            if f"{self.cls_name}.score_{i}" not in self.config_inst.variables:
                self.config_inst.add_variable(
                    name=f"{self.cls_name}.score_{i}",
                    null_value=-1,
                    binning=(10, 0, 1),
                    x_title=f"Selection Model output {i}",
                )

    def sandbox(self, task: law.Task) -> str:
        return dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh")

    def datasets(self, config_inst: od.Config) -> set[od.Dataset]:
        return {config_inst.get_dataset(dataset_name) for dataset_name in self.dataset_names}

    def uses(self, config_inst: od.Config) -> set[Route | str]:
        columns = set(self.input_features) | {"normalization_weight", "process_id", "channel_id"}
        return columns

    def training_producers(self, config_inst: od.Config, requested_producers: Sequence[str]) -> list[str]:
        producers = {"default", "deterministic_event_seeds"} | set(requested_producers)
        # fix MLTraining Phase Space
        return sorted(list(producers))

    def training_selector(self: MLModel, config_inst: od.UniqueObject, requested_selector: str) -> str:
        return "loose"

    def training_calibrators(self: MLModel, config_inst: od.Config, requested_calibrators: Sequence[str]) -> list[str]:
        return sorted(list({"default"} | set(requested_calibrators)))

    def produces(self, config_inst: od.Config) -> set[Route | str]:
        # mark columns that you don't want to be filtered out
        # preserve the networks prediction of a specific feature for each fold
        # cls_name would be the name of your model
        ml_predictions = {
            f"{self.cls_name}.fold{fold}.{feature}"
            for fold in range(self.folds)
            for feature in self.target_columns
        }

        # save indices used to create the folds
        util_columns = {f"{self.cls_name}.fold_indices"}
        additional_columns = {"process_id"}
        # combine all columns to a unique set
        preserved_columns = ml_predictions | util_columns | additional_columns
        return preserved_columns

    def output(self, task: law.Task) -> law.FileSystemDirectoryTarget:
        # needs to be given via config
        max_folds = self.folds
        current_fold = task.fold

        # create directory at task.target, if it does not exist
        target = task.target(f"mlmodel_f{current_fold}of{max_folds}.keras", dir=False)
        return target

    def setup_tensorboard(self, task: law.Task):
        log_dir = f"{task.output().absdirname}/logs/train/model_fold{task.fold}"
        logger.info(f"Tensorboard logs are written to {log_dir}")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
        return tensorboard_callback

    def build_field_names(self, dtype: np.typing.DTypeLike) -> list[str]:
        fields = []
        for (field, typ) in dtype.descr:
            if isinstance(typ, list):
                fields.extend([f"{field}.{subfield[0]}" for subfield in typ])
            else:
                fields.append(field)
        return fields

    def merge_weight_and_class_weights(self, data: dict, class_weight: dict[int, float]) -> np.ndarray:
        """ Merge the class weights with the event weights. """
        indecies = data["dataset_id"]
        event_weights = data["weight"]
        weight = np.ones_like(event_weights)
        for d, w in class_weight.items():
            weight[indecies == d.id] = w * event_weights[indecies == d.id]

        return weight

    def get_tensor(self, arrays: Sequence[np.ndarray]) -> tf.Tensor:
        """ Create a Tensor with predefined settings, e.g. shuffle, batch, prefetch. """
        tensor: tf.Tensor = tf.data.Dataset.from_tensor_slices(arrays)
        tensor = tensor.shuffle(buffer_size=tensor.cardinality())
        tensor = tensor.batch(self.parameters["batchsize"])
        tensor = tensor.prefetch(tf.data.experimental.AUTOTUNE)
        return tensor

    def open_model(self, target: law.FileSystemDirectoryTarget):
        return target.load(formatter="tf_keras_model")

    def get_number_of_features(self) -> int:
        return 17

    # def build_loss_function(self, *args) -> tf.keras.losses.Loss:
    #     from hbt.ml.metrices_callbacks import MetrixTF
    #     losses = [getattr(tf.keras.losses, arg, None) for arg in args]
    #     signal_purity = MetrixTF().signal_purity
    #     if None in losses:
    #         raise ValueError(f"Loss function not found in tf.keras.losses: {args}")
    #     @tf.function
    #     def myloss(y_true, y_pred):
    #         purity_loss = tf.cosh(1 - signal_purity(y_true[:, 0] == 1, selection_score=y_pred[:, 0], threshold=0.5))-1
    #         other_losses = sum([loss(y_true, y_pred) for loss in losses])
    #         return purity_loss + other_losses
    #     return myloss

    def build_model(self) -> tf.Model:
        # helper function to handle model building
        n_features = self.get_number_of_features()
        n_out = len(self.target_features)
        l2_reg = tf.keras.regularizers.l2(0.01) if self.L2 else None
        kernel_init = tf.keras.initializers.Constant(value=0.001)
        last_activation = "softmax" if len(self.target_features) > 1 else "sigmoid"

        # First input
        x_inp = tf.keras.Input(shape=(n_features,))

        # Second input
        ch_id = tf.keras.Input(shape=(1,), dtype=tf.int32)
        embedding = tf.keras.layers.Embedding(input_dim=3, output_dim=2)(ch_id)
        embedding = tf.keras.layers.Flatten()(embedding)

        # Merge inputs
        x = tf.keras.layers.Concatenate()([x_inp, embedding])

        for i, node in enumerate(self.nodes):
            x1 = x
            x = tf.keras.layers.Dense(
                node,
                kernel_regularizer=l2_reg,
                kernel_initializer=kernel_init,
            )(x1)
            if self.batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(self.activation_func)(x)

            if i == 0:
                continue
            x = tf.keras.layers.Add()([x, x1])

        y = tf.keras.layers.Dense(n_out, kernel_initializer=kernel_init, activation=last_activation)(x)
        model = tf.keras.Model(inputs=[x_inp, ch_id], outputs=y)
        return model

    def restructure_array(self, array: ak.Array) -> tuple[np.recarray, ak.Array]:
        """
         Restructure the array to remove the jagged structure and to create a first and second lepton column
         from the available Muons, ELectrons and Taus.
        """
        # Create an empty numpy structured array
        result: np.ndarray = np.zeros(len(array), dtype=[
            (f"l{i}_{var}", type)
            for i in [1, 2]
            for var, type in zip(
                [
                    "pt", "eta", "dz", "dxy",
                    "tauVSjet", "tauVSe", "tauVSmu", "is_iso",
                ],
                [np.float32] * 7 + [np.int32],
            )
        ] + [
            ("l1_iso_score", np.float32),
        ])
        result[:] = -10

        # get the used DeepTau wp
        tau_iso_wp = DotDict.wrap({
            k: self.config_inst.x.tau_id_working_points[f"tau_vs_{k}"] for k in ["e", "mu", "jet"]
        })

        # helper functions
        tau_matcher = lambda tag, wp: array.Tau[f"idDeepTau2018v2p5VS{tag}"] >= tau_iso_wp[tag][wp]
        channel_matcher = lambda ch: array.channel_id == self.config_inst.get_channel(ch).id

        # Create the Tau is Iso column
        tau_is_iso = 1 * ((tau_matcher("jet", "loose")) & (
            (
                (array.channel_id == 3) & (tau_matcher("e", "vvloose") | tau_matcher("mu", "vloose"))
            ) | (
                (array.channel_id != 3) & (tau_matcher("e", "vloose") | tau_matcher("mu", "tight"))
            )
        ))

        # Take care of the different channels seperately
        # etau and mutau channel
        for ch in ["etau", "mutau"]:
            ch_mask = channel_matcher(ch)
            l1 = array.Electron if ch == "etau" else array.Muon
            l1 = l1[ch_mask][:, 0]
            l2 = array.Tau[ch_mask][:, 0]
            iso_tag = "mvaIso" if ch == "etau" else "pfRelIso04_all"
            is_iso_tag = "mvaIso_WP80" if ch == "etau" else "tightId"
            for f in ["pt", "eta", "dz", "dxy"]:
                result[f"l1_{f}"][ch_mask] = l1[f]
                result[f"l2_{f}"][ch_mask] = l2[f]
            iso_array = l1[iso_tag] / ak.max(l1[iso_tag])
            result["l1_iso_score"][ch_mask] = iso_array
            result["l1_is_iso"][ch_mask] = 1 * l1[is_iso_tag]
            result["l2_is_iso"][ch_mask] = tau_is_iso[ch_mask][:, 0]
            result["l2_tauVSjet"][ch_mask] = l2["idDeepTau2018v2p5VSjet"]
            result["l2_tauVSe"][ch_mask] = l2["idDeepTau2018v2p5VSe"]
            result["l2_tauVSmu"][ch_mask] = l2["idDeepTau2018v2p5VSmu"]

        # tautau channel
        is_tautau = channel_matcher("tautau")
        l1 = array.Tau[is_tautau][:, 0]
        l2 = array.Tau[is_tautau][:, 1]
        for f in ["pt", "eta", "dz", "dxy"]:
            result[f"l1_{f}"][is_tautau] = l1[f]
            result[f"l2_{f}"][is_tautau] = l2[f]
        for f in ["jet", "e", "mu"]:
            result[f"l1_tauVS{f}"][is_tautau] = l1[f"idDeepTau2018v2p5VS{f}"]
            result[f"l2_tauVS{f}"][is_tautau] = l2[f"idDeepTau2018v2p5VS{f}"]
        result["l1_is_iso"][is_tautau] = tau_is_iso[is_tautau][:, 0]
        result["l2_is_iso"][is_tautau] = tau_is_iso[is_tautau][:, 1]

        # shift the channel_id by 1 for the embedding layer later
        channel_id = np.asarray(array.channel_id, dtype=np.int8) - 1

        return result, channel_id

    def prepare_input(self, input: ak.Array) -> (dict[str, tf.Tensor | Sequence[tf.Tensor]], list[str]):
        weight_sum: dict[str, float] = {}
        training: defaultdict[str, list] = defaultdict(list)
        valid: defaultdict[str, list] = defaultdict(list)
        fields: list = None

        for dataset, files in input["events"][self.config_inst.name].items():
            dataset_inst = self.dataset_insts[dataset]

            if len(dataset_inst.processes) != 1:
                raise Exception("only 1 process inst is expected for each dataset")

            # calculate the sum of weights for each dataset
            weight_sum[dataset_inst] = sum(ak.sum(file["mlevents"].load().normalization_weight) for file in files)

            for i, inp in enumerate(files):
                events = inp["mlevents"].load()
                weights = ak.to_numpy(events.normalization_weight)
                dataset_id = np.full(len(events), dataset_inst.id, dtype=np.int32)

                events = remove_ak_column(events, "normalization_weight")
                events = remove_ak_column(events, "process_id")
                events, channel_id = self.restructure_array(events)

                # create field names and check if they are matching
                if not fields:
                    fields = self.build_field_names(events.dtype)
                else:
                    if fields != self.build_field_names(events.dtype):
                        raise Exception("fields are not matching")

                events = nprec.structured_to_unstructured(events)

                # set EMPTY_FLOAT to -10
                events[events == EMPTY_FLOAT] = -10

                # check for infinite values in weights
                if np.any(~np.isfinite(weights)):
                    raise Exception(f"Infinite values found in weights from dataset {dataset}")

                # create target array
                target = np.zeros((len(events), 2))
                target[:, int(dataset_inst.has_tag("signal"))] = 1
                if len(self.target_features) == 1:
                    target = np.argmax(target, axis=1)

                # split into training and validation set
                if self.validation_split:
                    split = int(len(events) * self.validation_split)
                    choice = np.random.choice(range(events.shape[0]), size=(split,), replace=False)
                    ind = np.zeros(events.shape[0], dtype=bool)
                    ind[choice] = True
                    valid["events"].append(events[ind])
                    events = events[~ind]
                    valid["target"].append(target[ind])
                    target = target[~ind]
                    valid["weight"].append(weights[ind])
                    weights = weights[~ind]
                    valid["channel_id"].append(channel_id[ind])
                    channel_id = channel_id[~ind]
                    valid["dataset_id"].append(dataset_id[ind])
                    dataset_id = dataset_id[~ind]

                    logger.info(f"file {i} of dataset *{dataset}* split into {len(events)} training and {split}"
                                " validation events")

                training["events"].append(events)
                training["target"].append(target)
                training["weight"].append(weights)
                training["channel_id"].append(channel_id)
                training["dataset_id"].append(dataset_id)

        mean_weight: np.ndarray = np.mean(list(weight_sum.values()))
        class_weight = {d: mean_weight / w for d, w in weight_sum.items()}
        # concatenate all events and targets
        training["events"] = np.concatenate(training["events"])
        training["target"] = np.concatenate(training["target"])
        training["weight"] = np.concatenate(training["weight"])
        training["channel_id"] = np.concatenate(training["channel_id"])
        training["dataset_id"] = np.concatenate(training["dataset_id"])
        training["weight"] = self.merge_weight_and_class_weights(training, class_weight)

        # helper function to get tensor tuple
        get_tensor_tuple = lambda x: tuple(((x["events"], x["channel_id"]), x["target"], x["weight"]))

        # create tf tensors
        train_tensor = self.get_tensor(get_tensor_tuple(training))

        # create an output for the fit function of tf
        result = {"x": train_tensor}  # weights are included in the training tensor

        if self.validation_split:
            valid["events"] = np.concatenate(valid["events"])
            valid["target"] = np.concatenate(valid["target"])
            valid["weight"] = np.concatenate(valid["weight"])
            valid["channel_id"] = np.concatenate(valid["channel_id"])
            valid["dataset_id"] = np.concatenate(valid["dataset_id"])
            valid["weight"] = self.merge_weight_and_class_weights(valid, class_weight)

            valid_tensor = self.get_tensor(get_tensor_tuple(valid))
            result["validation_data"] = valid_tensor

        return result, fields

    def train(
        self,
        task: law.Task,
        input: dict[str, list[law.FileSystemFileTarget]],
        output: law.FileSystemDirectoryTarget,
    ) -> dict[str, Any]:
        from hbt.ml.metrices_callbacks import SelectionMetrixMetric
        # run eagerly
        # tf.config.run_functions_eagerly(True)

        physical_devices = tf.config.list_physical_devices("GPU")
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass
        # use helper functions to define model, open input parquet files and prepare events
        # init a model structure
        model = self.build_model()

        # get data tensors
        data_config, features = self.prepare_input(input)

        model_config = {
            "epochs": self.parameters["epochs"],
            "steps_per_epoch": 100,
            "validation_freq": 5,
            "callbacks": [
                self.setup_tensorboard(task),
            ],
        }

        # setup everything needed for training
        # optimizer = tf.keras.optimizers.SGD()
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learningrate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            amsgrad=False,
        )

        model.compile(
            optimizer,
            # loss=self.build_loss_function("binary_crossentropy"),
            loss="binary_crossentropy",
            steps_per_execution=10,
            metrics=[
                "accuracy",
                SelectionMetrixMetric(),
            ],
        )

        # print model summary
        model.summary()

        # train, throw model_history away
        _ = model.fit(
            **data_config,
            **model_config,
        )

        # save your model and everything you want to keep
        output.dump(model, formatter="tf_keras_model")
        tf.keras.utils.plot_model(model, to_file=f"{output.absdirname}/model.png", show_shapes=True)
        # law.util.interruptable_popen(f"imgcat {output.absdirname}/model.png", shell=True, executable="/bin/bash")

        return {
            "model": model,
            "features": features,
            "data_config": data_config,
            "model_config": model_config,
        }

    def evaluate(
        self,
        task: law.Task,
        events: ak.Array,
        models: list[Any],
        fold_indices: ak.Array,
        events_used_in_training: bool = False,
    ) -> ak.Array:
        # prepare ml_models input features, target is not important
        inputs_tensor, _ = self.prepare_events(events)

        # do evaluation on all data
        # one can get test_set of fold using: events[fold_indices == fold]
        for fold, model in enumerate(models):
            # convert tf.tensor -> np.array -> ak.array
            # to_regular is necessary to make array contigous (otherwise error)
            prediction = ak.to_regular(model(inputs_tensor).numpy())

            # update events with predictions, sliced by feature, and fold_indices for identification purpose
            for index_feature, target_feature in enumerate(self.target_features):
                events = set_ak_column(
                    events,
                    f"{self.cls_name}.fold{fold}.{target_feature}",
                    prediction[:, index_feature],
                )

        events = set_ak_column(
            events,
            f"{self.cls_name}.{fold_indices}",
            fold_indices,
        )

        return events


# usable derivations
selection_model = SelectionModel.derive("selection_model", cls_dict={
    "folds": 1,
    "model_name": "selection_model",
    "version": "zeros_init",
    "activation_func": "relu",
    "batch_norm": True,
    "nodes": [128, 128, 128, 128, 128],
    "validation_split": 0.2,
    "L2": True,
    "epochs": 200,
    "learningrate": 1E-5,
    "input_features": {
        f"Muon.{var}" for var in ("pt", "eta", "dz", "dxy", "tightId", "pfRelIso04_all")
    } | {
        f"Electron.{var}" for var in ("pt", "eta", "dz", "dxy", "mvaIso_WP80", "mvaIso")
    } | {
        f"Tau.{var}" for var in ("pt", "eta", "dz", "dxy")
    } | {
        f"Tau.idDeepTau2018v2p5VS{tag}" for tag in ("e", "mu", "jet")
    },
    "target_features": {
        f"selection_model.score_{i}" for i in ["signal"]
    },
})
