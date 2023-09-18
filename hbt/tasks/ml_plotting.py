# coding: utf-8

"""
Tasks to plot different types of metrices.
"""

from collections import OrderedDict
from abc import abstractmethod
from enum import Enum

import law
import luigi

from columnflow.tasks.framework.base import Requirements, ShiftTask
from columnflow.tasks.framework.plotting import PlotBase
from columnflow.tasks.framework.decorators import view_output_plots
from columnflow.tasks.ml import MLEvaluation
from columnflow.util import DotDict, maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")


class MLPLottingChoices(Enum):
    ConfusionMatrix = "pathtoCM"
    ROC = "PathToROC"


class PlotMLMetric(PlotBase,
                   MLEvaluation):

    plot_function = luigi.EnumListParameter(
        enum=MLPLottingChoices,
        significant=True,
        description="Metric plotting function to use. Available choices: 'ConfusionMatrix', 'ROC'",
    )

    def requires(self):
        reqs = {
            "events": self.reqs.MLEvaluation.req(
                self,
            ),
        }
        return reqs

    def output(self):
        return {"plot": self.target(f"{self.plot_function}_{self.branch}.pdf"),
                "array": self.target(f"array_{self.plot_function}_{self.branch}.parquet")}

    @law.decorator.log
    @law.decorator.localize
    @law.decorator.safe_output
    def run(self):
        # prepare inputs and outputs
        inputs = self.input()
        output = self.output()
        events = inputs["events"]

        from IPython import embed; embed()
        # Define the position of the scores and true labels
        test_if_int = np.vectorize(float.is_integer)
        labels_mask = {cl_name: test_if_int(events[self.cls_name][cl_name]).all()
                       for cl_name in events[self.cls_name].fields}
        assert (sum(labels_mask.values()) == 1), (
            f"Only one `true_labels` column per Network is expected, however {sum(labels_mask.values())} were found!"
        )
