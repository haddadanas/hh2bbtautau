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
from columnflow.util import DotDict


class MLPLottingChoices(Enum):
    ConfusionMatrix = "pathtoCM"
    ROC = "PathToROC"


class PlotMLMetric(PlotBase):

    plot_function = luigi.EnumListParameter(
        enum=MLPLottingChoices,
        significant=True,
        description="Metric plotting function to use. Available choices: 'ConfusionMatrix', 'ROC'",
    )
