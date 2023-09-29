# coding: utf-8

"""
Tasks related to reducing events for use on further tasks.
"""

import math
import functools
from collections import OrderedDict, defaultdict

import law
import luigi
from luigi.util import inherits
from luigi.parameter import ParameterVisibility
import order as od

from columnflow.tasks.framework.base import Requirements, AnalysisTask, DatasetTask, wrapper_factory
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, MLModelDataMixin
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.external import GetDatasetLFNs
from columnflow.tasks.ml import MLEvaluation
from columnflow.util import maybe_import, dev_sandbox


ak = maybe_import("awkward")


class MergeMLEvaluation(
    MLModelDataMixin,
    DatasetTask,
    law.tasks.ForestMerge,
    RemoteWorkflow,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    dataset = luigi.Parameter(
        default=law.config.get_expanded("analysis", "default_dataset"),
        visibility=ParameterVisibility.PRIVATE,
        ) # TODO hide from terminal output

    datasets = law.CSVParameter(
        default=("*",),
        description="names or name patterns of datasets to use; can also be the key of a "
        "mapping defined in the 'dataset_groups' auxiliary data of the corresponding "
        "config; default: ('*',)",
        brace_expand=True,
    )

    skip_datasets = law.CSVParameter(
        default=(),
        description="names or name patterns of datasets to skip after evaluating "
        "--datasets; can also be the key of a mapping defined in the 'dataset_groups' "
        "auxiliary data of the corresponding config; empty default",
        brace_expand=True,
    )

    # disable the shift parameter
    shift = None
    effective_shift = None
    allow_empty_shift = True

    # in each step, merge 10 into 1
    merge_factor = 10
    allow_empty_ml_model = False

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MLEvaluation=MLEvaluation,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # find all datasets
        datasets = self.find_config_objects(
            self.datasets,
            self.config_inst,
            od.Dataset,
            self.config_inst.x("dataset_groups", {}),
        )
        if not datasets:
            raise ValueError(
                f"no datasets found in config {self.config_inst} matching "
                f"{self.datasets}",
            )
        if self.skip_datasets:
            skip_datasets = self.find_config_objects(
                self.skip_datasets,
                self.config_inst,
                od.Dataset,
                self.config_inst.x("dataset_groups", {}),
            )
            datasets = [d for d in datasets if d not in skip_datasets]
            if not datasets:
                raise ValueError(
                    f"no datasets found in config {self.config_inst} after skipping "
                    f"{self.skip_datasets}",
                )
        self.selected_datasets = datasets
        # tell ForestMerge to not cache the internal merging structure by default,
        # (this is enabled in merge_workflow_requires)
        self._cache_forest = False

    def local_path(self, *path, **kwargs):
        output_path = super().local_path(*path, **kwargs)
        new_output = '_'.join(self.datasets).replace('*','')
        return output_path.replace(f'{self.dataset}', new_output)
    
    def create_branch_map(self):
        # DatasetTask implements a custom branch map, but we want to use the one in ForestMerge
        return law.tasks.ForestMerge.create_branch_map(self)

    def requires(self):
        if len(self.selected_datasets) == 1:
            return super().requires()
        return [MergeMLEvaluation.req(self, datasets=dataset, dataset=dataset, _exclude={"branches"})
               for dataset in self.selected_datasets]
    
    def merge_workflow_requires(self):
        req = [self.reqs.MLEvaluation.req(self, dataset=dataset, _exclude={"branches"})
               for dataset in self.selected_datasets]
        # if the merging stats exist, allow the forest to be cached
        self._cache_forest = all([task.merging_stats_exist for task in req])
        return req

    def merge_requires(self, start_leaf, end_leaf):
        return [
            self.reqs.MLEvaluation.req(self, dataset=dataset, branch=i)
            for i in range(start_leaf, end_leaf)
            for dataset in self.selected_datasets
        ]
    
    def trace_merge_inputs(self, inputs):
        return super().trace_merge_inputs([inp["mlcolumns"] for inp in inputs])

    def merge_output(self):
        return {"mlcolumns": self.target(f"mlcolumns_{self.branch}.parquet")}

    @law.decorator.log
    def run(self):
        # from IPython import embed; embed()
        return super().run()

    def merge(self, inputs, output):
        if not self.is_leaf():
            inputs = [inp["mlcolumns"] for inp in inputs]

        # from IPython import embed; embed()
        law.pyarrow.merge_parquet_task(self, inputs, output["mlcolumns"])


# @inherits(MLEvaluation)
# class MergeMLEvaluationStats(
#     DatasetTask,
# ):

#     n_inputs = luigi.IntParameter(
#         default=10,
#         significant=True,
#         description="minimal number of input files for sufficient statistics to infer merging "
#         "factors; default: 10",
#     )
#     merged_size = law.BytesParameter(
#         default=law.NO_FLOAT,
#         unit="MB",
#         significant=False,
#         description="the maximum file size of merged files; default unit is MB; default: config "
#         "value 'reduced_file_size' or 512MB'",
#     )

#     # upstream requirements
#     reqs = Requirements(
#         MLEvaluation=MLEvaluation,
#     )

#     @classmethod
#     def resolve_param_values(cls, params):
#         params = super().resolve_param_values(params)

#         # check for the default merged size
#         if "merged_size" in params and params["merged_size"] in (None, law.NO_FLOAT):
#             merged_size = 512.0
#             if "config_inst" in params:
#                 merged_size = params["config_inst"].x("reduced_file_size", merged_size)
#             params["merged_size"] = float(merged_size)

#         return params

#     def requires(self):
#         return self.reqs.MLEvaluation.req(self, branches=((0, self.n_inputs),))

#     def output(self):
#         return {"stats": self.target(f"stats_n{self.n_inputs}.json")}

#     @law.decorator.safe_output
#     def run(self):
#         # get all file sizes in bytes
#         from IPython import embed; embed()
#         coll = self.input()["collection"]
#         n = len(coll)
#         sizes = [
#             inp["mlcolumns"].stat().st_size
#             for inp in self.iter_progress(coll.targets.values(), n, msg=f"loading {n} stats ...")
#         ]

#         # helpers for avg and mean computation
#         def get_avg_std(values):
#             n = len(values)
#             if n < 1:
#                 return 0.0, 0.0
#             avg = sum(values) / n
#             if n < 2:
#                 return avg, 0.0
#             std = (sum((v - avg)**2 for v in values) / (n - 1))**0.5
#             return avg, std

#         # compute some stats
#         tot_size = sum(sizes)
#         avg_size, std_size = get_avg_std(sizes)
#         std_size = (sum((s - avg_size)**2 for s in sizes) / n)**0.5
#         max_size_merged = self.merged_size * 1024**2
#         merge_factor = int(round(max_size_merged / avg_size))
#         merge_factor = min(max(1, merge_factor), self.dataset_info_inst.n_files)
#         n_merged = int(math.ceil(self.dataset_info_inst.n_files / merge_factor))

#         # save them
#         stats = OrderedDict([
#             ("n_test_files", n),
#             ("tot_size", tot_size),
#             ("avg_size", avg_size),
#             ("std_size", std_size),
#             ("max_size_merged", max_size_merged),
#             ("merge_factor", merge_factor),
#         ])
#         self.output()["stats"].dump(stats, indent=4, formatter="json")

#         # print them
#         self.publish_message(f" stats of {n} input files ".center(40, "-"))
#         self.publish_message(f"tot. size: {law.util.human_bytes(tot_size, fmt=True)}")
#         self.publish_message(f"avg. size: {law.util.human_bytes(avg_size, fmt=True)}")
#         self.publish_message(f"std. size: {law.util.human_bytes(std_size, fmt=True)}")
#         self.publish_message(" merging info ".center(40, "-"))
#         self.publish_message(f"target size : {self.merged_size} MB")
#         self.publish_message(f"merging     : {merge_factor} into 1")
#         self.publish_message(f"files before: {self.dataset_info_inst.n_files}")
#         self.publish_message(f"files after : {n_merged}")
#         self.publish_message(40 * "-")


# class MergeMLEvaluationUser(DatasetTask):

#     # recursively merge 20 files into one
#     merge_factor = 20

#     # the initial default value of the cache_branch_map attribute
#     cache_branch_map_default = False

#     # upstream requirements
#     reqs = Requirements(
#         MergeMLEvaluationStats=MergeMLEvaluationStats,
#     )

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # cached value of the file_merging until it's positive
#         self._cached_file_merging = -1

#     @property
#     def file_merging(self):
#         """
#         Needed by DatasetTask to define the default branch map.
#         """
#         if self._cached_file_merging < 0:
#             # check of the merging stats is present and of so, set the cached file merging value
#             output = self.reqs.MergeMLEvaluationStats.req(self).output()
#             if output["stats"].exists():
#                 self._cached_file_merging = output["stats"].load(formatter="json")["merge_factor"]

#                 # as soon as the status file exists, cache the branch map
#                 self.cache_branch_map = True

#         return self._cached_file_merging

#     @property
#     def merging_stats_exist(self):
#         return self.file_merging >= 1

#     def mlevaluation_dummy_output(self):
#         # dummy output to be returned in case the merging stats are not present yet
#         return self.target("DUMMY_UNTIL_MLEVALUATION_MERGING_STATS_EXIST")

#     @classmethod
#     def maybe_dummy(cls, func):
#         # meant to wrap output methods of tasks depending on merging stats
#         # to inject a dummy output in case the stats are not there yet
#         @functools.wraps(func)
#         def wrapper(self):
#             # when the merging stats do not exist yet, return a dummy target
#             if not self.merging_stats_exist:
#                 return self.mlevaluation_dummy_output()

#             # otherwise, bind the wrapped function and call it
#             return func.__get__(self, self.__class__)()

#         return wrapper
