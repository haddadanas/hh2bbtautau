# coding: utf-8

"""
Tasks to plot different types of metrices.
"""

from collections import OrderedDict
from enum import Enum

import law
import luigi
from luigi.util import inherits

# from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.framework.base import Requirements, ShiftTask
from columnflow.tasks.framework.plotting import PlotBase
from columnflow.tasks.framework.decorators import view_output_plots
from columnflow.tasks.ml import MLEvaluation, MLEvaluationWrapper, MergeMLEvaluation
from columnflow.util import maybe_import, dev_sandbox

# from hbt.tasks.ml_merging import MergeMLEvaluation
np = maybe_import("numpy")
ak = maybe_import("awkward")


class MLPLottingChoices(Enum):
    CM = {
        'path': 'hbt.ml.plotting.Confusion_Matrix',
        'params': {
            'normalize': luigi.BoolParameter(
                parsing=luigi.BoolParameter.EXPLICIT_PARSING,
                significant=False
            ),
            'colormap': luigi.ChoiceParameter(default='viridis', significant=False,
                                              choices=['viridis', 'cf_cmap', 'cf_ygb_cmap', 'cf_green_cmap']),
            'z_title': luigi.Parameter(default='Accuracy', significant=False),
            'digits': luigi.IntParameter(default=3, significant=False),
            'title': luigi.Parameter(default='Confusion Matrix', significant=False, description=''),
        }
    }
    ROC = {
        'path': 'hbt.ml.plotting.ROC_Curve',
        'params': {
            'evaluation_type': luigi.ChoiceParameter(default='OvR', choices=['OvR', 'OvO']),
            'array_size': luigi.IntParameter(default=100 + 1, significant=False),
            'title': luigi.Parameter(default='ROC Curve', significant=False, description='')
        }
    }


@inherits(MergeMLEvaluation)
class PlotMLResults(PlotBase):
    dataset = None

    plot_function = luigi.EnumParameter(
        enum=MLPLottingChoices,
        significant=True,
        description="Metric plotting function to use. Available choices: 'CM', 'ROC'",
    )
    datasets = law.CSVParameter(
        default=("*",),
        description="names or name patterns of datasets to use; can also be the key of a "
        "mapping defined in the 'dataset_groups' auxiliary data of the corresponding "
        "config; default: ('*',)",
        brace_expand=True,
    )

    sample_weights = luigi.BoolParameter(
        significant=False,
        description='weights of the events'
    )

    evaluation_type = luigi.ChoiceParameter(default='OvR', choices=['OvR', 'OvO'])  # TODO remove if params worked

    skip_uncertenties = luigi.BoolParameter(significant=False)
    
    import sys
    tmp = 0
    for arg in sys.argv:
        if tmp == '--plot-function':
            plot_param = getattr(MLPLottingChoices, arg).value['params']
            for param_label, param in plot_param.items():
                setattr(cls, param_label, param)
    
    # from IPython import embed; embed()
    # plotting_dict = plot_function[0].value
    # plot_func = get_plot_func(plotting_dict['path'])
    # for param_name, param in plotting_dict['params'].items():
    #     setattr(PlotMLResults, param_name, param)

    # def get_plot_parameters(self) -> DotDict:
    #     # convert parameters to usable values during plotting
    #     params = super().get_plot_parameters()
    #     dict_add_strict(params, "skip_ratio", self.skip_ratio)
    #     dict_add_strict(params, "density", self.density)
    #     dict_add_strict(params, "yscale", None if self.yscale == law.NO_STR else self.yscale)
    #     dict_add_strict(params, "shape_norm", self.shape_norm)
    #     dict_add_strict(params, "hide_errors", self.hide_errors)
    #     return params
    reqs = Requirements(MLEvaluation=MLEvaluation)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.branch = 0

        # set the sandbox
        self.sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

        # Call the choosen plot function and set its Parameters
        plotting_dict = self.plot_function.value
        self.plot_func = self.get_plot_func(plotting_dict['path'])

    def requires(self):
        # TODO The workflow tree starts with MLEvaluation at branch 0 and PlotMLResults is not displayed however functions.
        reqs = dict()

        for dataset in self.datasets:
            self.dataset = dataset
            reqs[self.dataset] = {'events': self.clone_parent(),
                                  'sample_weights': []}
            # reqs[self.dataset]['sample_weights'] = None
        return reqs

    def output(self):
        return {"plot": self.target(f"{self.ml_model}/{self.plot_function.name}_{self.branch}.pdf"),
                "array": self.target(f"{self.ml_model}/array_{self.plot_function.name}_{self.branch}.pickle")}

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

        for ind, (label, data) in enumerate(events.items()):
            cls_labels.append(label)
            arrays = ak.concatenate((arrays, data))
            true_labels = np.concatenate((true_labels, self.get_target_labels(data, ind))) if ind else self.get_target_labels(data, ind)

        pred_labels = [field.lstrip('score_') for field in arrays.fields]
        arrays = ak.to_numpy(arrays)
        arrays = arrays.view((arrays.dtype[0], len(arrays.dtype.names)))

        return arrays, true_labels, pred_labels, cls_labels

    @law.decorator.log
    @law.decorator.localize
    @law.decorator.safe_output
    def run(self):
        # prepare inputs and outputs
        inputs = self.input()
        output = self.output()
        events = OrderedDict()
        from IPython import embed; embed()
        for label, dataset in inputs.items():
            data_inst = dataset['events']['mlcolumns']
            events_array = data_inst.load()
            if events.get(label):
                events[label] = ak.concatenate((events[label], events_array[self.ml_model]))
            else:
                events[label] = events_array[self.ml_model]
        events_array, target, cls_labels, pred_labels = self.get_plot_data(events)
        return 
        # from IPython import embed; embed()
        plot_output = self.plot_func(true_labels=target.argmax(axis=-1),
                     model_output=events_array,
                     process_labels=pred_labels,
                     output_path=output['plot'].abspath,
                     evaluation_type=self.evaluation_type,
                     class_labels=cls_labels,)
        # from IPython import embed; embed()
        output['array'].dump(plot_output, formatter='pickle')
