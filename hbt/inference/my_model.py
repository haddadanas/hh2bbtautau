# coding: utf-8

"""
Default inference model.
"""

from __future__ import annotations

import law

from columnflow.inference import ParameterType, FlowStrategy

from hbt.inference.base import HBTInferenceModelBase

logger = law.logger.get_logger(__name__)


class ml_inference(HBTInferenceModelBase):

    def init_proc_map(self) -> None:
        # mapping of process names in the datacard ("combine name") to configs and process names in a dict
        name_map = dict([
            *[
                (f"ggHH_kl_{kl}_kt_1_13p6TeV_hbbhtt", f"hh_ggf_hbb_htt_kl{kl}_kt1")
                for kl in ["0", "1", "2p45", "5"]
            ],
            ("ttbar", "tt"),
            ("DY", "dy"),
        ])

        # insert into proc_map
        # (same process name for all configs for now)
        for combine_name, proc_name in name_map.items():
            # same process name for all configs for now
            for config_inst in self.config_insts:
                _combine_name = combine_name
                self.proc_map.setdefault(_combine_name, {})[config_inst] = proc_name

    def init_categories(self) -> None:
        self.add_category(
            "bin_dnn_signal",
            config_data={
                config_inst.name: self.category_config_spec(
                    # name of the analysis category in the config
                    category="incl",  # m1 rejected < 0.2
                    # name of the variable
                    variable="bin_dnn_signal_fine",  # m1 no flat s
                    # names (or patterns) of datasets with real data in the config
                    data_datasets=["data_*"],
                )
                for config_inst in self.config_insts
            },
            mc_stats=10.0,
            data_from_processes=["ttbar", "DY"],
            flow_strategy=FlowStrategy.move,
        )

    def init_processes(self) -> None:
        # processes
        for kl in ["0", "1", "2p45", "5"]:
            self.add_process(
                f"ggHH_kl_{kl}_kt_1_13p6TeV_hbbhtt",
                is_signal=True,
                config_data={
                    config_inst.name: self.process_config_spec(
                        process=f"hh_ggf_hbb_htt_kl{kl}_kt1",
                        mc_datasets=[f"hh_ggf_hbb_htt_kl{kl}_kt1_powheg"],
                    )
                    for config_inst in self.config_insts
                },
            )
        self.add_process(
            "ttbar",
            config_data={
                config_inst.name: self.process_config_spec(
                    process="tt",
                    mc_datasets=["^tt_sl_powheg$"],
                )
                for config_inst in self.config_insts
            },
        )
        self.add_process(
            "DY",
            config_data={
                config_inst.name: self.process_config_spec(
                    process="dy",
                    mc_datasets=["dy_m50toinf_amcatnlo"],
                )
                for config_inst in self.config_insts
            },
        )

    def init_parameters(self) -> None:
        # groups
        self.add_parameter_group("experiment")
        self.add_parameter_group("theory")

        # groups that contain parameters that solely affect the signal cross section and/or br
        self.add_parameter_group("signal_norm_xs")
        self.add_parameter_group("signal_norm_xsbr")

        # parameter that is added by the HH physics model, representing kl-dependent QCDscale + mtop
        # uncertainties on the ggHH cross section
        self.add_parameter_to_group("THU_HH", "theory")
        self.add_parameter_to_group("THU_HH", "signal_norm_xs")
        self.add_parameter_to_group("THU_HH", "signal_norm_xsbr")

        def inject_all_eras(*names: str) -> list[str]:
            gen = (
                {self.inject_era(config_inst, name) for config_inst in self.config_insts}
                for name in names
            )
            return list(set.union(*gen))

        # theory uncertainties
        self.add_parameter(
            "BR_hbb",
            type=ParameterType.rate_gauss,
            process=["*_hbb", "*_hbbhtt"],
            effect=(0.9874, 1.0124),
            group=["theory", "signal_norm_xsbr"],
        )
        self.add_parameter(
            "BR_htt",
            type=ParameterType.rate_gauss,
            process=["*_htt", "*_hbbhtt"],
            effect=(0.9837, 1.0165),
            group=["theory", "signal_norm_xsbr"],
        )
        self.add_parameter(
            "pdf_gg",  # contains alpha_s
            type=ParameterType.rate_gauss,
            process=inject_all_eras("ttbar"),
            effect=1.042,
            group=["theory"],
        )
        self.add_parameter(
            "pdf_Higgs_ggHH",  # contains alpha_s
            type=ParameterType.rate_gauss,
            process="ggHH_*",
            effect=1.023,
            group=["theory", "signal_norm_xs", "signal_norm_xsbr"],
        )
        self.add_parameter(
            "pdf_Higgs_qqHH",  # contains alpha_s
            type=ParameterType.rate_gauss,
            process="qqHH_*",
            effect=1.027,
            group=["theory", "signal_norm_xs", "signal_norm_xsbr"],
        )
        self.add_parameter(
            "QCDscale_ttbar",
            type=ParameterType.rate_gauss,
            process=inject_all_eras("ttbar"),
            effect=(0.965, 1.024),
            group=["theory"],
        )
        self.add_parameter(
            "QCDscale_qqHH",
            type=ParameterType.rate_gauss,
            process="qqHH_*",
            effect=(0.9997, 1.0005),
            group=["theory", "signal_norm_xs", "signal_norm_xsbr"],
        )

        # lumi
        for config_inst in self.config_insts:
            ckey = self.campaign_keys[config_inst]
            lumi = config_inst.x.luminosity
            for unc_name in lumi.uncertainties:
                self.add_parameter(
                    unc_name,
                    type=ParameterType.rate_gauss,
                    effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
                    process=[f"*{ckey}*", "!QCD*"],
                    process_match_mode=all,
                    group="experiment",
                )

        # electron uncertainty
        self.add_parameter(
            "CMS_eff_e",  # this is the name of the uncertainty as it will show in the datacard. Let's use some variant of the official naming # noqa
            process="*",
            type=ParameterType.shape,
            config_data={
                config_inst.name: self.parameter_config_spec(
                    shift_source="e",  # this is the name of the shift (alias) in the config
                )
                for config_inst in self.config_insts
            },
            group=["experiment"],
        )

        # # btag
        # for name in self.config_inst.x.btag_unc_names:
        #     self.add_parameter(
        #         f"CMS_btag_{name}",
        #         type=ParameterType.shape,
        #         config_shift_source=f"btag_{name}",
        #         group="experiment",
        #     )

        # pileup
        self.add_parameter(
            "CMS_pileup_2022",
            type=ParameterType.shape,
            config_data={
                config_inst.name: self.parameter_config_spec(
                    shift_source="minbias_xs",
                )
                for config_inst in self.config_insts
            },
            group="experiment",
        )


class ml_inference_all_lora_no_cut(ml_inference):
    """
    Inference model for the HBT classification task using a LoRA model.
    This model extends the base `ml_inference` model with LoRA-specific functionality.
    """

    def init_categories(self) -> None:
        self.add_category(
            "lora_dnn_signal",
            config_data={
                config_inst.name: self.category_config_spec(
                    # name of the analysis category in the config
                    category="incl",  # m1 selected > 0.2
                    # name of the variable
                    variable="lora_dnn_signal_fine",  # m2 variable with flat s
                    # names (or patterns) of datasets with real data in the config
                    data_datasets=["data_*"],
                )
                for config_inst in self.config_insts
            },
            mc_stats=10.0,
            data_from_processes=["ttbar", "DY"],
            flow_strategy=FlowStrategy.move,
        )


class ml_inference_all_lora_cut(ml_inference):
    """
    Inference model for the HBT classification task using a LoRA model.
    This model extends the base `ml_inference` model with LoRA-specific functionality.
    """

    def init_categories(self) -> None:
        self.add_category(
            "lora_dnn_signal",
            config_data={
                config_inst.name: self.category_config_spec(
                    # name of the analysis category in the config
                    category="ml_selected_25",  # m1 selected > 0.2
                    # name of the variable
                    variable="lora_dnn_signal_fine",  # m2 variable with flat s
                    # names (or patterns) of datasets with real data in the config
                    data_datasets=["data_*"],
                )
                for config_inst in self.config_insts
            },
            mc_stats=10.0,
            data_from_processes=["ttbar", "DY"],
            flow_strategy=FlowStrategy.move,
        )


class ml_inference_lora(ml_inference):
    """
    Inference model for the HBT classification task using a LoRA model.
    This model extends the base `ml_inference` model with LoRA-specific functionality.
    """

    def init_categories(self) -> None:
        self.add_category(
            "bin_dnn_signal",
            config_data={
                config_inst.name: self.category_config_spec(
                    # name of the analysis category in the config
                    category="ml_rejected_25",  # m1 rejected < 0.2
                    # name of the variable
                    variable="bin_dnn_signal",  # m1 no flat s
                    # names (or patterns) of datasets with real data in the config
                    data_datasets=["data_*"],
                )
                for config_inst in self.config_insts
            },
            mc_stats=10.0,
            data_from_processes=["ttbar", "DY"],
            flow_strategy=FlowStrategy.move,
        )
        self.add_category(
            "lora_dnn_signal",
            config_data={
                config_inst.name: self.category_config_spec(
                    # name of the analysis category in the config
                    category="ml_selected_25",  # m1 selected > 0.2
                    # name of the variable
                    variable="lora_dnn_signal_fine",  # m2 variable with flat s
                    # names (or patterns) of datasets with real data in the config
                    data_datasets=["data_*"],
                )
                for config_inst in self.config_insts
            },
            mc_stats=10.0,
            data_from_processes=["ttbar", "DY"],
            flow_strategy=FlowStrategy.move,
        )

# @inference_model(
#     used_category="ml_selected_50",
# )
# def ml_inference(self):
#     self.add_category(
#         self.used_category,
#         config_data={
#             config_inst.name: self.category_config_spec(
#                 # name of the analysis category in the config
#                 category=self.used_category,
#                 # name of the variable
#                 variable="bin_dnn_signal_fine",
#                 # names (or patterns) of datasets with real data in the config
#             )
#             for config_inst in self.config_insts
#         },
#         data_from_processes=["TT", "DY"],
#         mc_stats=True,
#     )

#     #
#     # processes
#     #
#     for kl in ["0", "1", "2p45", "5"]:
#         self.add_process(
#             f"ggHH_kl_{kl}_kt_1_13p6TeV_hbbhtt",
#             is_signal=True,
#             config_data={
#                 config_inst.name: self.process_config_spec(
#                     # names of processes in the config
#                     process=f"hh_ggf_hbb_htt_kl{kl}_kt1",
#                     # names of MC datasets in the config
#                     mc_datasets=[f"hh_ggf_hbb_htt_kl{kl}_kt1_powheg"],
#                 )
#                 for config_inst in self.config_insts
#             },
#         )
#     self.add_process(
#         "TT",
#         config_data={
#             config_inst.name: self.process_config_spec(
#                 # names of processes in the config
#                 process="tt",
#                 # names of MC datasets in the config
#                 mc_datasets=["^tt_sl_powheg$"],
#             )
#             for config_inst in self.config_insts
#         },
#     )
#     self.add_process(
#         "DY",
#         config_data={
#             config_inst.name: self.process_config_spec(
#                 # names of processes in the config
#                 process="dy",
#                 # names of MC datasets in the config
#                 mc_datasets=["dy_m50toinf_amcatnlo"],
#             )
#             for config_inst in self.config_insts
#         },
#     )

#     #
#     # parameters
#     #

#     # groups
#     self.add_parameter_group("experiment")
#     self.add_parameter_group("theory")

#     # groups that contain parameters that solely affect the signal cross section and/or br
#     self.add_parameter_group("signal_norm_xs")
#     self.add_parameter_group("signal_norm_xsbr")

#     # parameter that is added by the HH physics model, representing kl-dependent QCDscale + mtop
#     # uncertainties on the ggHH cross section
#     self.add_parameter_to_group("THU_HH", "theory")
#     self.add_parameter_to_group("THU_HH", "signal_norm_xs")
#     self.add_parameter_to_group("THU_HH", "signal_norm_xsbr")

#     # theory uncertainties
#     self.add_parameter(
#         "BR_hbb",
#         type=ParameterType.rate_gauss,
#         process=["*_hbb", "*_hbbhtt"],
#         effect=(0.9874, 1.0124),
#         group=["theory", "signal_norm_xsbr"],
#     )
#     self.add_parameter(
#         "BR_htt",
#         type=ParameterType.rate_gauss,
#         process=["*_htt", "*_hbbhtt"],
#         effect=(0.9837, 1.0165),
#         group=["theory", "signal_norm_xsbr"],
#     )
#     self.add_parameter(
#         "pdf_gg",  # contains alpha_s
#         type=ParameterType.rate_gauss,
#         process="TT",
#         effect=1.042,
#         group=["theory"],
#     )
#     self.add_parameter(
#         "pdf_Higgs_ggHH",  # contains alpha_s
#         type=ParameterType.rate_gauss,
#         process="ggHH_*",
#         effect=1.023,
#         group=["theory", "signal_norm_xs", "signal_norm_xsbr"],
#     )
#     self.add_parameter(
#         "pdf_Higgs_qqHH",  # contains alpha_s
#         type=ParameterType.rate_gauss,
#         process="qqHH_*",
#         effect=1.027,
#         group=["theory", "signal_norm_xs", "signal_norm_xsbr"],
#     )
#     self.add_parameter(
#         "QCDscale_ttbar",
#         type=ParameterType.rate_gauss,
#         process="TT",
#         effect=(0.965, 1.024),
#         group=["theory"],
#     )
#     self.add_parameter(
#         "QCDscale_qqHH",
#         type=ParameterType.rate_gauss,
#         process="qqHH_*",
#         effect=(0.9997, 1.0005),
#         group=["theory", "signal_norm_xs", "signal_norm_xsbr"],
#     )

#     # lumi
#     for config_inst in self.config_insts:
#         lumi = config_inst.x.luminosity
#         for unc_name in lumi.uncertainties:
#             self.add_parameter(
#                 unc_name,
#                 type=ParameterType.rate_gauss,
#                 effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
#                 group="experiment",
#             )

#     # electron uncertainty
#     self.add_parameter(
#         "CMS_eff_e",  # this is the name of the uncertainty as it will show in the datacard. Let's use some variant of the official naming # noqa
#         process="*",
#         type=ParameterType.shape,
#         config_data={
#             config_inst.name: self.parameter_config_spec(
#                 shift_source="e",  # this is the name of the shift (alias) in the config
#             )
#             for config_inst in self.config_insts
#         },
#         group=["experiment"],
#     )

#     # # btag
#     # for name in self.config_inst.x.btag_unc_names:
#     #     self.add_parameter(
#     #         f"CMS_btag_{name}",
#     #         type=ParameterType.shape,
#     #         config_shift_source=f"btag_{name}",
#     #         group="experiment",
#     #     )

#     # pileup
#     self.add_parameter(
#         "CMS_pileup_2022",
#         type=ParameterType.shape,
#         config_data={
#             config_inst.name: self.parameter_config_spec(
#                 shift_source="minbias_xs",
#             )
#             for config_inst in self.config_insts
#         },
#         group="experiment",
#     )

#     #
#     # cleanup
#     #

#     self.cleanup(keep_parameters="THU_HH")


# # @inference_model
# # def ml_inference_no_shifts(self):
# #     # same initialization as "default" above
# #     ml_inference.init_func.__get__(self, self.__class__)()

# #     #
# #     # remove all parameters that require a shift source other than nominal
# #     #

# #     for category_name, process_name, parameter in self.iter_parameters():
# #         if parameter.type.is_shape or any(trafo.from_shape for trafo in parameter.transformations):
# #             self.remove_parameter(parameter.name, process=process_name, category=category_name)

# #     #
# #     # cleanup
# #     #

# #     self.cleanup(keep_parameters="THU_HH")
