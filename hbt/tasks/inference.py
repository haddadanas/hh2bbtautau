from typing import Any
import law
import order as od

from columnflow.tasks.cms.inference import CreateDatacards
from columnflow.tasks.framework.base import AnalysisTask, wrapper_factory

from hbt.tasks.base import HBTTask


class CreateHBTDatacards(HBTTask, CreateDatacards):
    """
    Task to create HBT datacards for inference purposes.
    This task extends the base CreateDatacards task with HBT-specific functionality.
    """

    def invoke_hist_hooks(
        self,
        hists: dict[od.Config, dict[od.Process, Any]],
    ) -> dict[od.Config, dict[od.Process, Any]]:
        if not self.hist_hooks:
            return hists

        # get variable names
        first_hist = next(iter(hists[self.config_insts[0]].values()), None)
        variable_name = first_hist.axes[-1].name

        if "fine" not in variable_name:
            self.publish_message(
                f"skipping hist hooks for {variable_name} as it is not a fine binning variable"
            )
            return hists
        # apply hooks in order
        for hook in self.hist_hooks:
            if hook in {None, "", law.NO_STR}:
                continue

            # get the hook
            func = self._get_hist_hook(hook)

            # validate it
            if not callable(func):
                raise TypeError(f"hist hook '{hook}' is not callable: {func}")

            # invoke it
            self.publish_message(f"invoking hist hook '{hook}' for {variable_name}")
            hists = func(self, hists)

        return hists


CreateDatacardsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=CreateHBTDatacards,
    enable=["configs", "skip_configs"],
)
