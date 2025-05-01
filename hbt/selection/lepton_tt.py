# coding: utf-8

"""
Lepton selection methods.
"""

from __future__ import annotations

import law

from operator import or_
from functools import reduce

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import (
    set_ak_column, sorted_indices_from_mask, flat_np_view, full_like,
)
from columnflow.util import maybe_import

from hbt.selection.lepton import (
    electron_selection, muon_selection, update_channel_ids, electron_trigger_matching,
    muon_trigger_matching, tau_trigger_matching
)
from hbt.config.util import Trigger

np = maybe_import("numpy")
ak = maybe_import("awkward")


logger = law.logger.get_logger(__name__)


@selector(
    uses={
        "Tau.{pt,eta,phi,dz,decayMode}",
        "{Electron,Muon,TrigObj}.{pt,eta,phi}",
    },
    # shifts are declared dynamically below in tau_tt_selection_init
    exposed=False,
)
def tau_tt_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    electron_mask: ak.Array | None,
    muon_mask: ak.Array | None,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Tau selection returning a masks for taus that are at least VVLoose isolated (vs jet)
    and a second mask to select isolated ones, eventually to separate normal and iso inverted taus
    for QCD estimations.
    """
    # return empty mask if no tagged taus exists in the chunk
    if ak.all(ak.num(events.Tau) == 0):
        logger.info("no taus found in event chunk")
        false_mask = full_like(events.Tau.pt, False, dtype=bool)
        return false_mask, false_mask

    is_single_e = trigger.has_tag("single_e")
    is_single_mu = trigger.has_tag("single_mu")
    is_cross_e = trigger.has_tag("cross_e_tau")
    is_cross_mu = trigger.has_tag("cross_mu_tau")
    is_cross_tau = trigger.has_tag("cross_tau_tau")
    is_cross_tau_vbf = trigger.has_tag("cross_tau_tau_vbf")
    is_cross_tau_jet = trigger.has_tag("cross_tau_tau_jet")
    is_2016 = self.config_inst.campaign.x.year == 2016
    is_run3 = self.config_inst.campaign.x.run == 3
    get_tau_tagger = lambda tag: f"id{self.config_inst.x.tau_tagger}VS{tag}"
    wp_config = self.config_inst.x.tau_id_working_points

    # determine minimum pt and maximum eta
    max_eta = 2.5
    base_pt = 20.0
    if is_single_e or is_single_mu:
        min_pt = 20.0
    elif is_cross_e:
        # only existing after 2016
        min_pt = 0.0 if is_2016 else 35.0
    elif is_cross_mu:
        min_pt = 25.0 if is_2016 else 32.0
    elif is_cross_tau:
        min_pt = 35.0
    elif is_cross_tau_vbf:
        # only existing after 2016
        min_pt = 0.0 if is_2016 else 25.0
    elif is_cross_tau_jet:
        min_pt = None if not is_run3 else 35.0

    # base tau mask for default and qcd sideband tau
    base_mask = (
        (abs(events.Tau.eta) < max_eta) &
        (events.Tau.pt > base_pt) &
        (abs(events.Tau.dz) < 0.2) &
        reduce(or_, [events.Tau.decayMode == mode for mode in (0, 1, 10, 11)]) &
        (events.Tau[get_tau_tagger("jet")] >= wp_config.tau_vs_jet.vvvloose)
        # vs e and mu cuts are channel dependent and thus applied in the overall lepton selection
    )

    # remove taus with too close spatial separation to previously selected leptons
    if electron_mask is not None:
        base_mask = base_mask & ak.all(events.Tau.metric_table(events.Electron[electron_mask]) > 0.5, axis=2)
    if muon_mask is not None:
        base_mask = base_mask & ak.all(events.Tau.metric_table(events.Muon[muon_mask]) > 0.5, axis=2)

    # trigger dependent cuts
    trigger_specific_mask = base_mask & (events.Tau.pt > min_pt)

    # compute the isolation mask separately as it is used to defined (qcd) categories later on
    iso_mask = events.Tau[get_tau_tagger("jet")] >= wp_config.tau_vs_jet.medium

    return base_mask, trigger_specific_mask, iso_mask


@tau_tt_selection.init
def tau_tt_selection_init(self: Selector) -> None:
    # register tec shifts
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag("tec")
    }

    # Add columns for the right tau tagger
    self.uses |= {
        f"Tau.id{self.config_inst.x.tau_tagger}VS{tag}"
        for tag in ("e", "mu", "jet")
    }


@selector(
    uses={"{Tau,TrigObj}.{pt,eta,phi}"},
    # shifts are declared dynamically below in tau_selection_init
    exposed=False,
)
def tau_trigger_matching(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    trigger_fired: ak.Array,
    leg_masks: dict[str, ak.Array],
    **kwargs,
) -> tuple[ak.Array]:
    """
    Tau trigger matching.
    """
    if ak.all(ak.num(events.Tau) == 0):
        logger.info("no taus found in event chunk")
        return full_like(events.Tau.pt, False, dtype=bool)

    is_cross_e = trigger.has_tag("cross_e_tau")
    is_cross_mu = trigger.has_tag("cross_mu_tau")
    is_cross_tau = trigger.has_tag("cross_tau_tau")
    is_cross_tau_vbf = trigger.has_tag("cross_tau_tau_vbf")
    is_cross_tau_jet = trigger.has_tag("cross_tau_tau_jet")
    is_any_cross_tau = is_cross_tau or is_cross_tau_vbf or is_cross_tau_jet
    assert is_cross_e or is_cross_mu or is_any_cross_tau

    # start per-tau mask with trigger object matching per leg
    if is_cross_e or is_cross_mu:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 2
        assert abs(trigger.legs["tau"].pdg_id) == 15
        # match leg 1
        return trigger_object_matching(
            events.Tau,
            events.TrigObj[leg_masks["tau"]],
            event_mask=trigger_fired,
        )

    # is_any_cross_tau
    # catch config errors
    assert trigger.n_legs == len(leg_masks) >= 2
    assert abs(trigger.legs["tau1"].pdg_id) == 15
    assert abs(trigger.legs["tau2"].pdg_id) == 15

    # match both legs
    matches_leg0 = trigger_object_matching(
        events.Tau,
        events.TrigObj[leg_masks["tau1"]],
        event_mask=trigger_fired,
    )
    matches_leg1 = trigger_object_matching(
        events.Tau,
        events.TrigObj[leg_masks["tau2"]],
        event_mask=trigger_fired,
    )

    # taus need to be matched to at least one leg, but as a side condition
    # each leg has to have at least one match to a tau
    matches = (
        (matches_leg0 | matches_leg1) &
        ak.any(matches_leg0, axis=1) &
        ak.any(matches_leg1, axis=1)
    )

    return matches


@selector(
    uses={
        electron_selection, electron_trigger_matching, muon_selection, muon_trigger_matching,
        tau_tt_selection, tau_trigger_matching,
        "event", "{Electron,Muon,Tau}.{charge,mass}",
    },
    produces={
        electron_selection, electron_trigger_matching, muon_selection, muon_trigger_matching,
        tau_tt_selection, tau_trigger_matching,
        # new columns
        "channel_id", "leptons_os", "tau2_isolated", "single_triggered", "cross_triggered",
        "matched_trigger_ids",
    },
)
def lepton_tt_selection(
    self: Selector,
    events: ak.Array,
    trigger_results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Combined lepton selection.
    """
    wp_config = self.config_inst.x.tau_id_working_points
    get_tau_tagger = lambda tag: f"id{self.config_inst.x.tau_tagger}VS{tag}"

    # get channels from the config
    ch_etau = self.config_inst.get_channel("etau")
    ch_mutau = self.config_inst.get_channel("mutau")
    ch_tautau = self.config_inst.get_channel("tautau")
    # ch_ee = self.config_inst.get_channel("ee")
    # ch_mumu = self.config_inst.get_channel("mumu")
    # ch_emu = self.config_inst.get_channel("emu")

    # prepare vectors for output vectors
    false_mask = (abs(events.event) < 0)
    channel_id = np.uint8(1) * false_mask
    tau2_isolated = false_mask
    leptons_os = false_mask
    single_triggered = false_mask
    cross_triggered = false_mask
    sel_electron_mask = full_like(events.Electron.pt, False, dtype=bool)
    sel_muon_mask = full_like(events.Muon.pt, False, dtype=bool)
    sel_tau_mask = full_like(events.Tau.pt, False, dtype=bool)
    leading_taus = events.Tau[:, :0]
    # store trigger ids for the fully matched triggers
    matched_trigger_ids = []
    # store trigger ids for the matched triggers that do have additional jet legs
    lepton_part_trigger_ids = []

    # indices for sorting taus first by isolation, then by pt
    # for this, combine iso and pt values, e.g. iso 255 and pt 32.3 -> 2550032.3
    f = 10**(np.ceil(np.log10(ak.max(events.Tau.pt))) + 2)
    tau_sorting_key = events.Tau[f"raw{self.config_inst.x.tau_tagger}VSjet"] * f + events.Tau.pt
    tau_sorting_indices = ak.argsort(tau_sorting_key, axis=-1, ascending=False)

    # perform each lepton election step separately per trigger, avoid caching
    sel_kwargs = {**kwargs, "call_force": True}
    for trigger, trigger_fired, leg_masks in trigger_results.x.trigger_data:
        is_single = trigger.has_tag("single_trigger")
        is_cross = trigger.has_tag("cross_trigger")

        # electron selection
        electron_mask, electron_control_mask, electron_veto_mask = self[electron_selection](
            events,
            trigger,
            **sel_kwargs,
        )

        # muon selection
        muon_mask, muon_control_mask, muon_veto_mask = self[muon_selection](
            events,
            trigger,
            **sel_kwargs,
        )

        # tau selection
        tau_base_mask, tau_trigger_specific_mask, tau_iso_mask = self[tau_tt_selection](
            events,
            trigger,
            electron_mask,
            muon_mask,
            **sel_kwargs,
        )

        # conditions potentially leading to etau channel
        if trigger.has_tag({"single_e", "cross_e_tau"}) and (
            self.dataset_inst.is_mc or
            self.dataset_inst.has_tag("etau")
        ):
            # channel dependent deeptau cuts vs e and mu
            ch_base_tau_mask = (
                tau_base_mask &
                (events.Tau[get_tau_tagger("e")] >= wp_config.tau_vs_e.vloose) &
                (events.Tau[get_tau_tagger("mu")] >= wp_config.tau_vs_mu.tight)
            )

            # fold trigger matching into the selection
            trig_electron_mask = (
                electron_mask &
                self[electron_trigger_matching](events, trigger, trigger_fired, leg_masks, **sel_kwargs)
            )
            trig_tau_mask = ch_base_tau_mask
            if trigger.has_tag("cross_e_tau"):
                trig_tau_mask = (
                    trig_tau_mask &
                    self[tau_trigger_matching](events, trigger, trigger_fired, leg_masks, **sel_kwargs)
                )

            # check if the taus fulfil the offline requirements for the trigger (pt cut)
            trig_tau_mask = trig_tau_mask & tau_trigger_specific_mask

            # check if the most isolated tau among the base selected ones is matched
            # and fulfils the additional trigger requirements
            # for that: take the most isolated tau among the base selected ones and check its entry
            # in the tau mask considering the matching and the additional trigger requirements
            first_tau_matched = ak.fill_none(
                ak.firsts(trig_tau_mask[tau_sorting_indices[ch_base_tau_mask[tau_sorting_indices]]], axis=1),
                False,
            )

            # expect 1 electron, 1 veto electron (the same one), 0 veto muons, and at least one tau
            # without and with trigger matching on the default objects
            is_etau = (
                trigger_fired &
                (ak.sum(electron_mask, axis=1) == 1) &
                (ak.sum(trig_electron_mask, axis=1) == 1) &
                (ak.sum(electron_veto_mask, axis=1) == 1) &
                (ak.sum(muon_veto_mask, axis=1) == 0) &
                first_tau_matched
            )

            # get selected taus and sort them
            # (this will be correct for events for which is_etau is actually True)
            sorted_sel_taus = events.Tau[tau_sorting_indices][trig_tau_mask[tau_sorting_indices]]
            # determine the relative charge and tau2 isolation
            e_charge = ak.firsts(events.Electron[trig_electron_mask].charge, axis=1)
            tau_charge = ak.firsts(sorted_sel_taus.charge, axis=1)
            is_os = e_charge == -tau_charge
            is_iso = ak.sum(tau_iso_mask[trig_tau_mask], axis=1) >= 1
            # store global variables
            channel_id = update_channel_ids(events, channel_id, ch_etau.id, is_etau)
            tau2_isolated = ak.where(is_etau, is_iso, tau2_isolated)
            leptons_os = ak.where(is_etau, is_os, leptons_os)
            single_triggered = ak.where(is_etau & is_single, True, single_triggered)
            cross_triggered = ak.where(is_etau & is_cross, True, cross_triggered)
            sel_electron_mask = ak.where(is_etau, trig_electron_mask, sel_electron_mask)
            sel_tau_mask = ak.where(is_etau, trig_tau_mask, sel_tau_mask)
            leading_taus = ak.where(is_etau, sorted_sel_taus[:, :1], leading_taus)

            # store the matched trigger id
            ids = ak.where(is_etau, np.float32(trigger.id), np.float32(np.nan))
            matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

        # mutau channel
        if trigger.has_tag({"single_mu", "cross_mu_tau"}) and (
            self.dataset_inst.is_mc or
            self.dataset_inst.has_tag("mutau")
        ):
            # channel dependent deeptau cuts vs e and mu
            ch_base_tau_mask = (
                tau_base_mask &
                (events.Tau[get_tau_tagger("e")] >= wp_config.tau_vs_e.vvloose) &
                (events.Tau[get_tau_tagger("mu")] >= wp_config.tau_vs_mu.tight)
            )

            # fold trigger matching into the selection
            trig_muon_mask = (
                muon_mask &
                self[muon_trigger_matching](events, trigger, trigger_fired, leg_masks, **sel_kwargs)
            )
            trig_tau_mask = ch_base_tau_mask
            if trigger.has_tag("cross_mu_tau"):
                trig_tau_mask = (
                    trig_tau_mask &
                    self[tau_trigger_matching](events, trigger, trigger_fired, leg_masks, **sel_kwargs)
                )

            # check if the taus fulfil the offline requirements for the trigger (pt cut)
            trig_tau_mask = trig_tau_mask & tau_trigger_specific_mask

            # check if the most isolated tau among the base selected ones is matched
            # and fulfils the additional trigger requirements
            # for that: take the most isolated tau among the base selected ones and check its entry
            # in the tau mask considering the matching and the additional trigger requirements
            first_tau_matched = ak.fill_none(
                ak.firsts(trig_tau_mask[tau_sorting_indices[ch_base_tau_mask[tau_sorting_indices]]], axis=1),
                False,
            )

            # expect 1 muon, 1 veto muon (the same one), 0 veto electrons, and at least one tau
            # without and with trigger matching on the default objects
            is_mutau = (
                trigger_fired &
                (ak.sum(muon_mask, axis=1) == 1) &
                (ak.sum(trig_muon_mask, axis=1) == 1) &
                (ak.sum(muon_veto_mask, axis=1) == 1) &
                (ak.sum(electron_veto_mask, axis=1) == 0) &
                first_tau_matched
            )

            # get selected, sorted taus to obtain quantities
            # (this will be correct for events for which is_mutau is actually True)
            sorted_sel_taus = events.Tau[tau_sorting_indices][trig_tau_mask[tau_sorting_indices]]
            # determine the relative charge and tau2 isolation
            mu_charge = ak.firsts(events.Muon[trig_muon_mask].charge, axis=1)
            tau_charge = ak.firsts(sorted_sel_taus.charge, axis=1)
            is_os = mu_charge == -tau_charge
            is_iso = ak.sum(tau_iso_mask[trig_tau_mask], axis=1) >= 1
            # store global variables
            channel_id = update_channel_ids(events, channel_id, ch_mutau.id, is_mutau)
            tau2_isolated = ak.where(is_mutau, is_iso, tau2_isolated)
            leptons_os = ak.where(is_mutau, is_os, leptons_os)
            single_triggered = ak.where(is_mutau & is_single, True, single_triggered)
            cross_triggered = ak.where(is_mutau & is_cross, True, cross_triggered)
            sel_muon_mask = ak.where(is_mutau, trig_muon_mask, sel_muon_mask)
            sel_tau_mask = ak.where(is_mutau, trig_tau_mask, sel_tau_mask)
            leading_taus = ak.where(is_mutau, sorted_sel_taus[:, :1], leading_taus)

            # store the matched trigger id
            ids = ak.where(is_mutau, np.float32(trigger.id), np.float32(np.nan))
            matched_trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

        # tautau channel
        if (
            trigger.has_tag({"cross_tau_tau", "cross_tau_tau_vbf", "cross_tau_tau_jet"}) and
            (self.dataset_inst.is_mc or self.dataset_inst.has_tag("tautau"))
        ):
            # channel dependent deeptau cuts vs e and mu
            ch_base_tau_mask = (
                tau_base_mask &
                (events.Tau[get_tau_tagger("e")] >= wp_config.tau_vs_e.vvloose) &
                (events.Tau[get_tau_tagger("mu")] >= wp_config.tau_vs_mu.vloose)
            )

            # fold trigger matching into the selection
            trig_tau_mask = (
                ch_base_tau_mask &
                self[tau_trigger_matching](events, trigger, trigger_fired, leg_masks, **sel_kwargs)
            )

            # check if the taus fulfil the offline requirements for the trigger (pt cut)
            trig_tau_mask = trig_tau_mask & tau_trigger_specific_mask

            # check if the two leading (most isolated) taus are matched
            leading_taus_matched = ak.fill_none(
                ak.firsts(trig_tau_mask[tau_sorting_indices[ch_base_tau_mask[tau_sorting_indices]]], axis=1) &
                ak.firsts(trig_tau_mask[tau_sorting_indices[ch_base_tau_mask[tau_sorting_indices]]][:, 1:], axis=1),
                False,
            )

            # expect 0 veto electrons, 0 veto muons and at least two taus of which one is isolated
            is_tautau = (
                trigger_fired &
                (ak.sum(electron_veto_mask, axis=1) == 0) &
                (ak.sum(muon_veto_mask, axis=1) == 0) &
                leading_taus_matched
            )

            # get selected, sorted taus to obtain quantities
            # (this will be correct for events for which is_tautau is actually True)
            sorted_sel_taus = events.Tau[tau_sorting_indices][trig_tau_mask[tau_sorting_indices]]
            # determine the relative charge and tau2 isolation
            tau1_charge = ak.firsts(sorted_sel_taus.charge, axis=1)
            tau2_charge = ak.firsts(sorted_sel_taus.charge[:, 1:], axis=1)
            is_os = tau1_charge == -tau2_charge
            is_iso = ak.sum(tau_iso_mask[trig_tau_mask], axis=1) >= 2
            # store global variables
            channel_id = update_channel_ids(events, channel_id, ch_tautau.id, is_tautau)
            tau2_isolated = ak.where(is_tautau, is_iso, tau2_isolated)
            leptons_os = ak.where(is_tautau, is_os, leptons_os)
            single_triggered = ak.where(is_tautau & is_single, True, single_triggered)
            cross_triggered = ak.where(is_tautau & is_cross, True, cross_triggered)
            sel_tau_mask = ak.where(is_tautau, trig_tau_mask, sel_tau_mask)
            leading_taus = ak.where(is_tautau, sorted_sel_taus[:, :2], leading_taus)

            # store the matched trigger id
            ids = ak.where(is_tautau, np.float32(trigger.id), np.float32(np.nan))
            ids = ak.singletons(ak.nan_to_none(ids))
            # we need to store the trigger id for the ttv and ttj triggers separately, as
            # we are not sure yet whether the matching is correct for the jet legs of the trigger
            if trigger.has_tag({"cross_tau_tau_vbf", "cross_tau_tau_jet"}):
                lepton_part_trigger_ids.append(ids)
            else:
                matched_trigger_ids.append(ids)

    # some final type conversions
    channel_id = ak.values_astype(channel_id, np.uint8)
    leptons_os = ak.fill_none(leptons_os, False)

    # concatenate matched trigger ids
    empty_ids = ak.singletons(full_like(events.event, 0, dtype=np.int32), axis=0)[:, :0]
    merge_ids = lambda ids: ak.values_astype(ak.concatenate(ids, axis=1), np.int32) if ids else empty_ids
    matched_trigger_ids = merge_ids(matched_trigger_ids)
    lepton_part_trigger_ids = merge_ids(lepton_part_trigger_ids)

    # save new columns
    events = set_ak_column(events, "channel_id", channel_id)
    events = set_ak_column(events, "leptons_os", leptons_os)
    events = set_ak_column(events, "tau2_isolated", tau2_isolated)
    events = set_ak_column(events, "single_triggered", single_triggered)
    events = set_ak_column(events, "cross_triggered", cross_triggered)
    events = set_ak_column(events, "matched_trigger_ids", matched_trigger_ids)

    # convert lepton masks to sorted indices (pt for e/mu, iso for tau)
    sel_electron_indices = sorted_indices_from_mask(sel_electron_mask, events.Electron.pt, ascending=False)
    sel_muon_indices = sorted_indices_from_mask(sel_muon_mask, events.Muon.pt, ascending=False)
    sel_tau_indices = sorted_indices_from_mask(sel_tau_mask, tau_sorting_key, ascending=False)

    return events, SelectionResult(
        steps={
            "lepton": channel_id != 0,
        },
        objects={
            "Electron": {
                "Electron": sel_electron_indices,
            },
            "Muon": {
                "Muon": sel_muon_indices,
            },
            "Tau": {
                "Tau": sel_tau_indices,
            },
        },
        aux={
            # save the selected lepton pair for the duration of the selection
            # multiplication of a coffea particle with 1 yields the lorentz vector
            "lepton_pair": ak.concatenate(
                [
                    events.Electron[sel_electron_indices] * 1,
                    events.Muon[sel_muon_indices] * 1,
                    events.Tau[sel_tau_indices] * 1,
                ],
                axis=1,
            )[:, :2],

            # save the matched trigger ids of the trigger with jet legs for the duration of the selection
            # these will be updated in the jet selection and then stored in the matched_trigger_ids column
            "lepton_part_trigger_ids": lepton_part_trigger_ids,

            # save the leading taus for the duration of the selection
            # exactly 1 for etau/mutau and exactly 2 for tautau
            "leading_taus": leading_taus,
        },
    )


@lepton_tt_selection.init
def lepton_tt_selection_init(self: Selector, **kwargs) -> None:
    # add column to load the raw tau tagger score
    self.uses.add(f"Tau.raw{self.config_inst.x.tau_tagger}VSjet")
