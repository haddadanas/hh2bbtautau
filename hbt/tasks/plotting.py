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
    SelectorStepsMixin,
    ProducersMixin,
    CategoriesMixin,
    WeightProducerMixin,
)
from columnflow.tasks.framework.plotting import ProcessPlotSettingMixin, PlotBase, VariablePlotSettingMixin
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.framework.decorators import view_output_plots
from columnflow.tasks.production import ProduceColumns
from columnflow.util import dev_sandbox, DotDict, maybe_import
from columnflow.columnar_util import Route
from columnflow.types import Any

from hbt.tasks.base import HBTTask

pd = maybe_import("pandas")
ak = maybe_import("awkward")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")


class PlotBaseHBT(
    HBTTask,
    VariablePlotSettingMixin,
    ProcessPlotSettingMixin,
    CategoriesMixin,
    WeightProducerMixin,
    ProducersMixin,
    SelectorStepsMixin,
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

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "plot", f"datasets_{self.datasets_repr}")
        return parts

    def create_branch_map(self):
        return [
            DotDict({"category": cat_name})
            for cat_name in sorted(self.categories)
        ]

    def requires(self):
        return {
            d: self.reqs.ProduceColumns.req(
                self,
                dataset=d,
                branch=-1,
                producer="gen_default",
                _exclude={"branches"},
            )
            for d in self.datasets
        }

    def workflow_requires(self, only_super: bool = False):
        reqs = super().workflow_requires()
        if only_super:
            return reqs

        reqs["produce_columns"] = self.requires_from_branch()

        return reqs

    def output(self) -> dict[str, list]:
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

    def get_input_as_df(self, inp: DotDict) -> pd.DataFrame:
        # get needed columns
        variables = {val for vals in self.variable_tuples.values() for val in vals}

        # read the data
        selected_columns = ["process_id", "category_ids", "selection_mask", *variables]
        events = pd.read_parquet(inp["collection"][0]["columns"].copy_to_local(), columns=selected_columns)

        # create a column for each category
        category_insts = [self.config_inst.get_category(cat) for cat in self.categories]
        for cat in category_insts:
            events[cat.name] = events["category_ids"].apply(lambda x: cat.id in x)

        # drop the category_ids column as it is not needed anymore
        events = events.drop(columns=["category_ids"])

        return events

    def get_variable_insts(self, var_names: tuple | str) -> tuple[od.Variable, od.Variable]:
        if isinstance(var_names, str):
            return self.config_inst.get_variable(var_names)
        return tuple(
            self.config_inst.get_variable(var_name)
            for var_name in var_names
        )

    def get_data_args(self, df: pd.DataFrame, x: str, y: str) -> dict:
        return {"data": df, "x": x, "y": y}

    def update_plot_kwargs(self, kwargs: dict) -> dict:
        kwargs = super().update_plot_kwargs(kwargs)
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in self.get_plot_func(self.plot_function).__code__.co_varnames
        }
        return kwargs

    def make_pretty(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        variable_tuple: tuple,
        plt_title: str = "",
        effeciency: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        x_inst, y_inst = self.get_variable_insts(variable_tuple)
        fig.suptitle(plt_title, size=35, va="top", ha="center")
        ax.set_xlabel(x_inst.x_title, fontsize=ax.xaxis.label.get_size() + 4)
        ax.set_ylabel(y_inst.x_title, fontsize=ax.yaxis.label.get_size() + 4)
        legend = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            shadow=True,
            ncol=5,
        )
        for text in legend.get_texts():
            text.set_text(f"pass ({effeciency:.2f}%)" if text.get_text() == "True" else "fail")
            text.set_fontsize(text.get_fontsize() + 4)
        # TODO add mean and std of the variables to the axes
        return fig, ax

    @law.decorator.log
    @view_output_plots
    def run(self):

        for dataset, inp in self.input().items():
            print(f"plotting dataset: {dataset}")
            events = self.get_input_as_df(inp)

            for category in self.categories:
                print(f"├── plotting in {category}")

                for variable in self.variables:
                    print(f"│   ├── Plotting variable: {variable}", end="  ", flush=True)
                    if all([f.complete() for f in self.output()[dataset][category][variable]]):
                        print("✅")
                        continue
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
                    print("✅")

        print("└── Plotting completed.")


class PlotScatterPlots(PlotBaseHBT):
    """
    Task to plot scatter plots of the selection results.
    """

    plot_function = PlotBase.plot_function.copy(
        default="seaborn.scatterplot",
        add_default_to_description=True,
        description="the full path given using the dot notation of the desired plot function.",
    )

    def call_plot_func(self: PlotScatterPlots, func_name: str, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        plt.style.use(mplhep.style.CMS)
        plt.rcParams.update({"legend.facecolor": "white"})
        fig, ax = plt.subplots(figsize=(15, 15))
        super().call_plot_func(func_name, ax=ax, **kwargs)
        return fig, ax

    def get_plot_parameters(self: PlotScatterPlots, variable_tuple: tuple) -> dict:
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
        variable_tuple: tuple,
        plt_title: str = "",
        effeciency: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        x_inst, y_inst = self.get_variable_insts(variable_tuple)
        fig, ax = super().make_pretty(fig, ax, variable_tuple, plt_title, effeciency)
        ax.set_xlim(x_inst.x_min, x_inst.x_max)
        ax.set_ylim(y_inst.x_min, y_inst.x_max)
        return fig, ax


class PlotKDEPlots(PlotScatterPlots):
    plot_function = PlotBase.plot_function.copy(
        default="seaborn.kdeplot",
        add_default_to_description=True,
        description="the full path given using the dot notation of the desired plot function.",
    )

    def get_input_as_df(self) -> pd.DataFrame:
        # get needed columns
        variables = {val for vals in self.variable_tuples.values() for val in vals}
        route = {Route(self.get_variable_insts(variable).expression) for variable in variables}
        columns = {v.expression for v in self.get_variable_insts(variables)}
        columns |= {"category_ids"}
        category_insts = [self.config_inst.get_category(cat) for cat in self.categories]

        # create a column for each category
        events_dict = {}
        dataset_map = {"tt_sl_powheg": "tt", "tt_dl_powheg": "tt", "tt_fh_powheg": "tt"}  # Merge the tt datasets
        # read the data
        for cat in category_insts:
            events_dict[cat.name] = pd.DataFrame()
            for dataset, inp in self.input().items():
                events = inp["collection"][0]["columns"].load(columns=columns)
                mask = ak.any(events["category_ids"] == cat.id, axis=-1)

                temp = pd.DataFrame()
                for r in route:
                    temp[r.string_column] = r.apply(events[mask]).to_numpy()
                    temp["dataset"] = dataset_map.get(dataset, dataset)
                events_dict[cat.name] = pd.concat([events_dict[cat.name], temp], ignore_index=True)
        return events_dict

    def get_plot_parameters(self: PlotScatterPlots, variable_tuple: tuple) -> dict:
        x_inst, y_inst = self.get_variable_insts(variable_tuple)
        params = super().get_plot_parameters(variable_tuple)
        params["hue"] = "dataset"
        # params["weights"] = "weights"
        params["common_norm"] = False
        params["levels"] = [0.1705, 0.341, 0.682, 0.954, 0.997]
        params["log_scale"] = tuple(var.log_x for var in (x_inst, y_inst))
        params.update(self.general_settings)
        return params

    def make_pretty(
        self: PlotScatterPlots,
        fig: plt.Figure,
        ax: plt.Axes,
        variable_tuple: tuple,
        plt_title: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        x_inst, y_inst = self.get_variable_insts(variable_tuple)
        fig.suptitle(plt_title, size=35, va="top", ha="center")
        ax.set_xlabel(x_inst.x_title, fontsize=ax.xaxis.label.get_size() + 4)
        ax.set_ylabel(y_inst.x_title, fontsize=ax.yaxis.label.get_size() + 4)
        ax.set_xlim(x_inst.x_min, x_inst.x_max)
        ax.set_ylim(y_inst.x_min, y_inst.x_max)
        return fig, ax

    def output(self) -> dict[str, list]:
        return {
            cat: {
                var: [
                    self.target(name)
                    for name in self.get_plot_names(f"plot__{self.plot_function}__proc__{cat}_{var}")
                ]
                for var in self.variables
            }
            for cat in self.categories
        }

    @law.decorator.log
    @view_output_plots
    def run(self):

        events = self.get_input_as_df()
        for category in self.categories:
            print(f"├── plotting in {category}")

            for variable in self.variables:
                print(f"│   ├── Plotting variable: {variable}", end="  ", flush=True)
                if all([f.complete() for f in self.output()[category][variable]]):
                    print("✅")
                    continue
                variable_tuple = self.variable_tuples[variable]
                column_names = tuple(self.get_variable_insts(variable).expression for variable in variable_tuple)
                sel_events = events[category]
                for c in column_names:
                    sel_events = sel_events.loc[sel_events[c] > 0]
                # call the plot function
                fig, ax = self.call_plot_func(
                    self.plot_function,
                    **self.get_data_args(sel_events, *column_names),
                    **self.get_plot_parameters(variable_tuple),
                )
                # make the plot prettier
                plt_title = f"({category})"
                fig, ax = self.make_pretty(fig, ax, variable_tuple, plt_title)

                # save the outputs
                for outp in self.output()[category][variable]:
                    outp.dump(fig, formatter="mpl", dpi=150, bbox_inches="tight")
                print("✅")

        print("└── Plotting completed.")


class PlotHistScatter(PlotKDEPlots):

    plot_function = PlotBase.plot_function.copy(
        default="seaborn.scatterplot",
        add_default_to_description=True,
        description="the full path given using the dot notation of the desired plot function.",
    )

    base_dataset = luigi.Parameter(
        default="hh_ggf_hbb_htt_kl1_kt1_powheg",
        description="The dataset to be used as the base for the KDE plot.",
    )

    base_plot = luigi.Parameter(
        default="seaborn.histplot",
        description="The plot function to be used for the base dataset.",
    )

    def update_plot_kwargs(self: PlotHistScatter, kwargs: dict) -> dict:
        kwargs_new = super().update_plot_kwargs(kwargs)
        kwargs_new["color"] = kwargs["color"]
        if "marker" in kwargs:
            kwargs_new["marker"] = kwargs["marker"]
        if "cbar" in kwargs:
            kwargs_new["cbar"] = kwargs["cbar"]
        kwargs_new.pop("hue")
        return kwargs_new

    def call_plot_func(self: PlotScatterPlots, func_name: str, ax, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        return PlotBaseHBT.call_plot_func(self, func_name, ax=ax, **kwargs)

    def make_pretty(
        self: PlotScatterPlots,
        fig: plt.Figure,
        axs: tuple,
        variable_tuple: tuple,
        plt_title: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        for ax in axs:
            fig, ax = super().make_pretty(fig, ax, variable_tuple, plt_title)
        return fig, axs

    @law.decorator.log
    @view_output_plots
    def run(self):

        events = self.get_input_as_df()
        for category in self.categories:
            print(f"├── plotting in {category}")

            for variable in self.variables:
                print(f"│   ├── Plotting variable: {variable}", end="  ", flush=True)
                if all([f.complete() for f in self.output()[category][variable]]):
                    print("✅")
                    continue
                variable_tuple = self.variable_tuples[variable]
                column_names = tuple(self.get_variable_insts(variable).expression for variable in variable_tuple)
                sel_events = events[category]
                for c in column_names:
                    sel_events = sel_events.loc[sel_events[c] > 0]
                sel_events = self.get_weights(sel_events)
                base_events = sel_events.loc[sel_events["dataset"] == self.base_dataset]
                sel_events = sel_events.loc[sel_events["dataset"] != self.base_dataset]
                # call the plot function
                plt.style.use(mplhep.style.CMS)
                plt.rcParams.update({"legend.facecolor": "white"})
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey="row", figsize=(35, 15))
                for ax, color in zip((ax1, ax2), ("b", "k")):
                    self.call_plot_func(
                        self.base_plot,
                        ax,
                        # cbar=True,
                        # thresh=5,
                        # color="k",
                        **self.get_data_args(base_events, *column_names),
                        **self.get_plot_parameters(variable_tuple),
                    )
                self.call_plot_func(
                    self.plot_function,
                    ax2,
                    # marker="+",
                    # color="r",
                    # legend="full",
                    **self.get_data_args(sel_events, *column_names),
                    **self.get_plot_parameters(variable_tuple),
                )
                # make the plot prettier
                plt_title = f"({category})"
                fig, ax = self.make_pretty(fig, (ax1, ax2), variable_tuple, plt_title)

                # save the outputs
                for outp in self.output()[category][variable]:
                    outp.dump(fig, formatter="mpl", dpi=150, bbox_inches="tight")
                print("✅")

        print("└── Plotting completed.")


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

    def output(self) -> dict[str, list]:
        if self.kind not in self.plot_suffix.rsplit("__", 1):
            self.plot_suffix = self.kind if self.plot_suffix == "NO_STR" else f"{self.plot_suffix}__{self.kind}"
        return super().output()

    def update_plot_kwargs(self: PlotFancyPlots, kwargs: dict) -> dict:
        allowed_args = self.get_plot_func(self.plot_function).__code__.co_varnames
        subplot_func = "matplotlib.pyplot.hexbin" if self.kind == "hex" else f"seaborn.{self.kind}plot"
        kwargs["kind"] = self.kind
        if self.kind in ["hex", "reg", "resid"]:
            kwargs.pop("hue")
        if "x" not in allowed_args:
            kwargs["vars"] = [kwargs.pop("x"), kwargs.pop("y")]
        kwargs_cp = {
            k: v
            for k, v in kwargs.items()
            if k not in allowed_args
        }
        kwargs["joint_kws"] = {
            k: v
            for k, v in kwargs_cp.items()
            if k in self.get_plot_func(subplot_func).__code__.co_varnames
        }
        kwargs["diag_kws"] = {
            k: v
            for k, v in kwargs_cp.items()
            if k in self.get_plot_func("seaborn.histplot").__code__.co_varnames
        }
        kwargs = super().update_plot_kwargs(kwargs)
        return kwargs

    def call_plot_func(self: PlotFancyPlots, func_name: str, **kwargs) -> Any:
        plt.style.use(mplhep.style.CMS)
        plt.rcParams.update({"legend.facecolor": "white"})
        fig_obj = super().call_plot_func(func_name, **kwargs)
        return fig_obj, None

    def get_plot_parameters(self: PlotFancyPlots, variable_tuple) -> dict:
        x_inst, y_inst = self.get_variable_insts(variable_tuple)
        params = super().get_plot_parameters()
        params["hue"] = "selection_mask"
        params["height"] = 15
        params["xlim"] = (x_inst.x_min, x_inst.x_max)
        params["ylim"] = (y_inst.x_min, y_inst.x_max)
        params["binwidth"] = tuple(var.bin_width for var in (x_inst, y_inst))
        params["log_scale"] = tuple(var.log_x for var in (x_inst, y_inst))
        params.update(self.general_settings)
        return params

    def make_pretty(
        self: PlotFancyPlots,
        fig: plt.Figure,
        ax: plt.Axes,
        variable_tuple: tuple,
        plt_title: str = "",
        effeciency: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        ax = fig.ax_joint
        fig = fig.figure
        fig, ax = super().make_pretty(fig, ax, variable_tuple, plt_title, effeciency)
        return fig, ax


class PlotSelectionHist(PlotBaseHBT):

    plot_function = "matplotlib.pyplot.hist"

    stacked = luigi.BoolParameter(
        default=False,
        description="If True, the histograms will be stacked.",
    )

    def get_input_as_df(self, inp: DotDict) -> pd.DataFrame:
        # get needed columns
        variables = {val for vals in self.variable_tuples.values() for val in vals}
        columns = {v.expression for v in self.get_variable_insts(variables)}
        columns |= {"category_ids", "selection_mask"}

        # read the data
        events = inp["collection"][0]["columns"].load(columns=columns)

        # create a column for each category
        category_insts = [self.config_inst.get_category(cat) for cat in self.categories]
        events_dict = {}
        for cat in category_insts:
            mask = ak.any(events["category_ids"] == cat.id, axis=-1)
            events_dict[cat.name] = ak.without_field(events[mask], "category_ids")

        return events_dict

    def call_plot_func(self, func_name: str, data, **kwargs) -> Any:
        plt.style.use(mplhep.style.CMS)
        fig, ax = plt.subplots(figsize=(15, 15))
        kwargs = self.update_plot_kwargs(kwargs)
        for lab, d in data.items():
            ax.hist(d, label=lab, **kwargs)
        return fig, ax

    def get_data_args(self, array, route) -> dict:
        mask = array["selection_mask"]
        data = route.apply(array)
        return {"fail": data[~mask], "pass": data[mask]}

    def get_plot_parameters(self: PlotScatterPlots, variable: tuple) -> dict:
        var = self.get_variable_insts(variable)
        params = super().get_plot_parameters()
        params["bins"] = var.bin_edges
        params["log"] = var.log_x
        params["stacked"] = self.stacked
        params["histtype"] = "step"
        params.update(self.general_settings)
        return params

    def make_pretty(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        variable_tuple: tuple,
        plt_title: str = "",
        effeciency: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        x_inst = self.get_variable_insts(variable_tuple)
        fig.suptitle(plt_title, size=35, va="top", ha="center")
        ax.set_xlabel(x_inst.x_title, fontsize=ax.xaxis.label.get_size() + 4)
        ax.set_ylabel("Counts (Unweighted)", fontsize=ax.yaxis.label.get_size() + 4)
        legend = ax.legend(
            loc="upper right",
            fancybox=True,
            shadow=True,
            ncol=5,
        )
        for text in legend.get_texts():
            text.set_text(f"pass ({effeciency:.2f}%)" if text.get_text() == "pass" else "fail")
            text.set_fontsize(text.get_fontsize() + 4)
        # TODO add mean and std of the variables to the axes
        return fig, ax

    @law.decorator.log
    @view_output_plots
    def run(self):

        for dataset, inp in self.input().items():
            print(f"plotting dataset: {dataset}")
            events = self.get_input_as_df(inp)

            for category in self.categories:
                print(f"├── plotting in {category}")

                for variable in self.variables:
                    print(f"│   ├── Plotting variable: {variable}", end="  ", flush=True)
                    if all([f.complete() for f in self.output()[dataset][category][variable]]):
                        print("✅")
                        continue
                    route = Route(self.get_variable_insts(variable).expression)
                    sel_events = events[category]

                    # call the plot function
                    fig, ax = self.call_plot_func(
                        self.plot_function,
                        self.get_data_args(sel_events, route),
                        **self.get_plot_parameters(variable),
                    )
                    # make the plot prettier
                    plt_title = f"{dataset} ({category})"
                    fig, ax = self.make_pretty(fig, ax, variable, plt_title, ak.mean(sel_events["selection_mask"]))

                    # save the outputs
                    for outp in self.output()[dataset][category][variable]:
                        outp.dump(fig, formatter="mpl", dpi=150, bbox_inches="tight")
                    print("✅")

        print("└── Plotting completed.")


class PlotEfficiencyHist(PlotSelectionHist):

    def call_plot_func(self, func_name: str, data, **kwargs) -> Any:
        plt.style.use(mplhep.style.CMS)
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 15), gridspec_kw={"height_ratios": [3, 1]})
        kwargs = self.update_plot_kwargs(kwargs)
        weights = {
            lab: ak.full_like(d, 1 + ak.sum(d == -10) / ak.sum(d > 0))
            for lab, d in data.items()
        }
        hists = {lab: ax0.hist(d, weights=weights[lab], label=lab, **kwargs)[0] for lab, d in data.items()}
        eff = hists["pass"] / (hists["pass"] + hists["fail"])
        bin_edges = kwargs["bins"]
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2. for i in range(len(bin_edges) - 1)]
        ax1.step(bin_centers, eff, where="mid", color="black")
        ax1.set_ylabel("Efficiency")
        return fig, (ax0, ax1)

    def make_pretty(
        self,
        fig: plt.Figure,
        axs: tuple,
        variable_tuple: tuple,
        plt_title: str = "",
        effeciency: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        fig, ax0 = super().make_pretty(fig, axs[0], variable_tuple, plt_title, effeciency)
        ax1 = axs[1]
        ax1.set_xlim(ax0.get_xlim())
        return fig, (ax0, ax1)
