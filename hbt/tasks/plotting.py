# coding: utf-8

"""
Tasks related to plotting selection results.
"""
from __future__ import annotations

import law
import luigi
import order as od

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    # CalibratorsMixin,
    # SelectorStepsMixin,
    # ProducersMixin,
    CategoriesMixin,
    WeightProducerMixin,
)
from columnflow.tasks.framework.plotting import ProcessPlotSettingMixin, PlotBase, VariablePlotSettingMixin
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.framework.decorators import view_output_plots
from columnflow.tasks.production import ProduceColumns
from columnflow.util import dev_sandbox, DotDict, maybe_import
from columnflow.types import Any

from hbt.tasks.base import HBTTask

pd = maybe_import("pandas")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")


class PlotBaseHBT(
    HBTTask,
    VariablePlotSettingMixin,
    ProcessPlotSettingMixin,
    CategoriesMixin,
    WeightProducerMixin,
    # ProducersMixin,
    # SelectorStepsMixin,
    # CalibratorsMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar.sh")
    exclude_index = True

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        ProduceColumns=ProduceColumns,
    )

    def store_parts(self: PlotScatterPlots):
        parts = super().store_parts()
        parts.insert_before("version", "plot", f"datasets_{self.datasets_repr}")
        return parts

    def create_branch_map(self: PlotScatterPlots):
        return [
            DotDict({"category": cat_name})
            for cat_name in sorted(self.categories)
        ]

    def requires(self: PlotScatterPlots):
        return {
            d: self.reqs.ProduceColumns.req(
                self,
                dataset=d,
                branch=-1,
                producer="gen_default",
                selector="gen_default",
                _exclude={"branches"},
            )
            for d in self.datasets
        }

    def workflow_requires(self: PlotScatterPlots, only_super: bool = False):
        reqs = super().workflow_requires()
        if only_super:
            return reqs

        reqs["produce_columns"] = self.requires_from_branch()

        return reqs

    def output(self: PlotScatterPlots) -> dict[str, list]:
        return {
            d: {
                cat: {
                    var: [
                        self.target(name)
                        for name in self.get_plot_names(f"plot__{self.plot_function}__proc_{d}_{cat}_{var}")
                    ]
                    for var in self.variables
                }
                for cat in self.categories
            }
            for d in self.datasets
        }

    def get_input_as_df(self: PlotScatterPlots, inp: DotDict) -> pd.DataFrame:
        # get needed columns
        variables = {val for vals in self.variable_tuples.values() for val in vals}

        # read the data
        selected_columns = ["process_id", "category_ids", "selection_mask", *variables]
        events = pd.read_parquet(inp["collection"][0]["columns"].cache_path, columns=selected_columns)

        # create a column for each category
        category_insts = [self.config_inst.get_category(cat) for cat in self.categories]
        for cat in category_insts:
            events[cat.name] = events["category_ids"].apply(lambda x: cat.id in x)

        # drop the category_ids column as it is not needed anymore
        events = events.drop(columns=["category_ids"])

        return events

    def get_variable_insts(self: PlotScatterPlots, variable_tuple: tuple) -> tuple[od.Variable, od.Variable]:
        return tuple(
            self.config_inst.get_variable(var_name)
            for var_name in variable_tuple
        )

    def get_data_args(self, df: pd.DataFrame, x: str, y: str) -> dict:
        return {"data": df, "x": x, "y": y}

    def update_plot_kwargs(self: PlotScatterPlots, kwargs: dict) -> dict:
        kwargs = super().update_plot_kwargs(kwargs)
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in self.get_plot_func(self.plot_function).__code__.co_varnames
        }
        return kwargs

    @law.decorator.log
    @view_output_plots
    def run(self: PlotScatterPlots):

        for dataset, inp in self.input().items():
            print(f"plotting dataset: {dataset}")
            events = self.get_input_as_df(inp)

            for category in self.categories:
                print(f"├── plotting in {category}")

                for variable in self.variables:
                    print(f"│   ├── Plotting variable: {variable}")
                    variable_tuple = self.variable_tuples[variable]
                    sel_events = events.loc[events[category], ["selection_mask", *variable_tuple]]

                    # call the plot function
                    fig, ax = self.call_plot_func(
                        self.plot_function,
                        **self.get_data_args(sel_events, *variable_tuple),
                        **self.get_plot_parameters(variable_tuple),
                    )
                    # make the plot prettier
                    plt_title = f"{dataset} ({category})"
                    fig, ax = self.make_pretty(fig, ax, variable_tuple, plt_title, sel_events["selection_mask"].mean())

                    # save the outputs
                    for outp in self.output()[dataset][category][variable]:
                        outp.dump(fig, formatter="mpl", dpi=150, bbox_inches="tight")

        print("└── Plotting complete ✅")


class PlotScatterPlots(PlotBaseHBT):
    """
    Task to plot scatter plots of the selection results.
    """

    plot_function = PlotBase.plot_function.copy(
        default="seaborn.scatterplot",
        add_default_to_description=True,
        description="the full path given using the dot notation of the desired plot function.",
    )

    def call_plot_func(self, func_name: str, **kwargs) -> Any:
        plt.style.use(mplhep.style.CMS)
        plt.rcParams.update({"legend.facecolor": "white"})
        fig, ax = plt.subplots(figsize=(15, 15))
        super().call_plot_func(func_name, ax=ax, **kwargs)
        return fig, ax

    def get_plot_parameters(self: PlotScatterPlots, variable_tuple) -> dict:
        x_inst, y_inst = self.get_variable_insts(variable_tuple)
        params = super().get_plot_parameters()
        params["hue"] = "selection_mask"
        params["binrange"] = tuple((var.x_min, var.x_max) for var in (x_inst, y_inst))
        params["binwidth"] = tuple(var.bin_width for var in (x_inst, y_inst))
        params["log_scale"] = tuple(var.log_x for var in (x_inst, y_inst))
        params.update(self.general_settings)
        return params

    def make_pretty(
        self: PlotScatterPlots,
        fig: plt.Figure,
        ax: plt.Axes,
        variable_tuples: tuple,
        plt_title: str = "",
        effeciency: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        x_inst, y_inst = self.get_variable_insts(variable_tuples)
        ax.set_xlabel(x_inst.x_title, fontsize=ax.xaxis.label.get_size() + 4)
        ax.set_ylabel(y_inst.x_title, fontsize=ax.yaxis.label.get_size() + 4)
        ax.set_xlim(x_inst.x_min, x_inst.x_max)
        ax.set_ylim(y_inst.x_min, y_inst.x_max)
        legend = ax.get_legend()
        legend.set_title(plt_title, prop={"size": 35})
        legend.set_loc("lower center")
        legend.set_bbox_to_anchor((0.5, 1.0))
        legend.set_ncols(max(len(legend.get_texts()), 4))
        for text in legend.get_texts():
            text.set_text(f"pass ({effeciency:.2f}%)" if text.get_text() == "True" else "fail")
            text.set_fontsize(text.get_fontsize() + 4)
        # TODO add mean and std of the variables to the axes
        return fig, ax


class PlotFancyPlots(PlotBaseHBT):
    """
    Task to plot scatter plots of the selection results.
    """

    plot_function = PlotBase.plot_function.copy(
        default="seaborn.jointplot",
        add_default_to_description=True,
        description="the full path given using the dot notation of the desired plot function.",
    )

    kind = luigi.ChoiceParameter(
        default="scatter",
        choices=["scatter", "kde", "hist", "hex", "reg", "resid"],
        description="The kind of plot to be created.",
        var_type=str,
    )

    def update_plot_kwargs(self: PlotScatterPlots, kwargs: dict) -> dict:
        kwargs = super().update_plot_kwargs(kwargs)
        subplot_func = f"seaborn.{self.kind}plot" if True
        allowed_kwargs = [
            *self.get_plot_func(self.plot_function).__code__.co_varnames,
            *self.get_plot_func(self.kind + "plot")
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in self.get_plot_func(self.plot_function).__code__.co_varnames
        }
        return kwargs

    def call_plot_func(self, func_name: str, **kwargs) -> Any:
        plt.style.use(mplhep.style.CMS)
        plt.rcParams.update({"legend.facecolor": "white"})
        if "x" not in self.get_plot_func(self.plot_function).__code__.co_varnames:
            kwargs["vars"] = [kwargs.pop("x"), kwargs.pop("y")]
        fig_obj = super().call_plot_func(func_name, **kwargs)
        return fig_obj

    def get_plot_parameters(self: PlotFancyPlots, variable_tuple) -> dict:
        x_inst, y_inst = self.get_variable_insts(variable_tuple)
        params = super().get_plot_parameters()
        params["hue"] = "selection_mask"
        params["height"] = 15
        params["x_lim"] = (x_inst.x_min, x_inst.x_max)
        params["y_lim"] = (y_inst.x_min, y_inst.x_max)
        params["binwidth"] = tuple(var.bin_width for var in (x_inst, y_inst))
        params["log_scale"] = tuple(var.log_x for var in (x_inst, y_inst))
        params.update(self.general_settings)
        return params

    def make_pretty(
        self: PlotFancyPlots,
        fig: plt.Figure,
        ax: plt.Axes,
        variable_tuples: tuple,
        plt_title: str = "",
        effeciency: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        x_inst, y_inst = self.get_variable_insts(variable_tuples)
        ax.set_xlabel(x_inst.x_title, fontsize=ax.xaxis.label.get_size() + 4)
        ax.set_ylabel(y_inst.x_title, fontsize=ax.yaxis.label.get_size() + 4)
        ax.set_xlim(x_inst.x_min, x_inst.x_max)
        ax.set_ylim(y_inst.x_min, y_inst.x_max)
        legend = ax.get_legend()
        legend.set_title(plt_title, prop={"size": 35})
        legend.set_loc("lower center")
        legend.set_bbox_to_anchor((0.5, 1.0))
        legend.set_ncols(max(len(legend.get_texts()), 4))
        for text in legend.get_texts():
            text.set_text(f"pass ({effeciency:.2f}%)" if text.get_text() == "True" else "fail")
            text.set_fontsize(text.get_fontsize() + 4)
        # TODO add mean and std of the variables to the axes
        return fig, ax
