# coding: utf-8

"""
Tasks to plot different types of metrices.
"""

from collections import OrderedDict
from abc import abstractmethod
from enum import Enum

import law
import luigi
from luigi.util import inherits

# from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.framework.base import Requirements, ShiftTask
from columnflow.tasks.framework.plotting import PlotBase
from columnflow.tasks.framework.decorators import view_output_plots
from columnflow.tasks.ml import MLEvaluation, MLEvaluationWrapper
from columnflow.util import DotDict, maybe_import, dev_sandbox

np = maybe_import("numpy")
ak = maybe_import("awkward")


@inherits(MLEvaluationWrapper) #, MLEvaluation)
class PlotMLMetric(PlotBase):

    class MLPLottingChoices(Enum):
        ConfusionMatrix = "pathtoCM"
        ROC = "PathToROC"

    plot_function = luigi.EnumListParameter(
        enum=MLPLottingChoices,
        significant=True,
        description="Metric plotting function to use. Available choices: 'ConfusionMatrix', 'ROC'",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.branch = 0
        # set the sandbox
        self.sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))


    def requires(self):
        # TODO The workflow tree starts with MLEvaluation at branch 0 and PlotMLMetric is not displayed however functions.
        reqs = self.clone_parent()
        return reqs

    def output(self):
        return {"plot": self.target(f"{self.plot_function}_{self.branch}.pdf"),
                "array": self.target(f"array_{self.plot_function}_{self.branch}.parquet")}

    # Should be overwritten when a proper get_target_labels function is defined
    def get_target_labels(self, array, ind) -> np.ndarray:
        target = np.zeros((array.type.length, len(array.fields)))
        target[:, ind] = 1
        # # Define the position of the scores and true labels
        # test_if_int = np.vectorize(float.is_integer)
        # labels_mask = {cl_name: test_if_int(events[self.cls_name][cl_name]).all()
        #             for cl_name in events[self.cls_name].fields}
        # assert (sum(labels_mask.values()) == 1), (
        #     f"Only one `true_labels` column per Network is expected, however {sum(labels_mask.values())} were found!"
        # )
        return target

    def get_plot_data(self, events: dict):
        cls_labels = []
        arrays = ak.Array([])
        true_labels = np.array([])

        for ind, (label, pred) in enumerate(events.items()):
            cls_labels.append(label)
            arrays = ak.concatenate((arrays, pred))
            true_labels = np.concatenate((true_labels, self.get_target_labels(pred, ind))) if ind else self.get_target_labels(pred, ind)

        pred_labels = [l.lstrip('score_') for l in arrays.fields]
        arrays = ak.to_numpy(arrays)
        arrays = arrays.view((arrays.dtype[0], len(arrays.dtype.names)))

        return arrays, true_labels, cls_labels, pred_labels

    @law.decorator.log
    @law.decorator.localize
    @law.decorator.safe_output
    def run(self):
        # prepare inputs and outputs
        inputs = self.input()
        output = self.output()
        events = {}

        for (_,_,label), path in inputs.items():
            _, path = path.popitem()
            array = path[0]['mlcolumns'].load()
            if events.get(label):
                events[label] = ak.concatenate((events[label], array['test']))
            else:
                events[label] = array['test'] #TODO self.ml_model stattdessen verwenden
        array, target, cls_labels, pred_labels = self.get_plot_data(events)

