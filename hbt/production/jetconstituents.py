# coding: utf-8

"""
Column production methods related to higher-level features.
"""

import functools

from columnflow.production import Producer, producer
from columnflow.production.util import attach_coffea_behavior
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from hbt.production.default import default

np = maybe_import("numpy")
ak = maybe_import("awkward")

# helpers
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i8 = functools.partial(set_ak_column, value_type=np.int8)


@producer(
    uses={
        "Tau.*", "PFCandidate.*", "PFCandidateIndices.tau",
        attach_coffea_behavior,
    },
    produces={
        f"PFCandidatesSumTau.{name}" for name in [
            "charge", "pt", "eta", "phi", "mass", "abs_charge", "ncand", "ncand_charged",
            "pt_hsum", "dpt", "dpt_hsum", "decaymode_charge",
        ]
    },
)
def taucand(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    decaymodes = {
        0: 1,
        1: 1,
        2: 1,
        5: 2,
        6: 2,
        7: 2,
        10: 3,
        11: 3,
    }
    collections = {
        "PFCandidate": {
            "type_name": "PFCand",
        },
        "Tau": {
            "type_name": "Tau",
        },
    }

    events = self[attach_coffea_behavior](events, collections=collections, **kwargs)

    tau = events.Tau
    tau_indecies = ak.local_index(tau)
    pfcand_indices = events.PFCandidateIndices.tau
    pfcand = events.PFCandidate

    charge_results = ak.full_like(tau.charge, -1)
    ncand_results = ak.full_like(tau.charge, -1)
    ncand_charged_results = ak.full_like(tau.charge, -1)
    pt_results = ak.full_like(tau.charge, -1)
    eta_results = ak.full_like(tau.charge, -9)
    phi_results = ak.full_like(tau.charge, -4)
    mass_results = ak.full_like(tau.charge, -1)
    pt_hsum_results = ak.full_like(tau.charge, -1)
    decaymode_charge = ak.full_like(tau.charge, -1)
    abs_charge_sum_results = ak.full_like(tau.charge, -1)

    for i in range(ak.max(tau_indecies)):
        decay_prod = pfcand[pfcand_indices == i]
        decay_prod_sum = decay_prod.sum(axis=-1)
        charged_decay_prod = ak.num(decay_prod[decay_prod.charge != 0].pt)

        abs_charge_sum = ak.sum(abs(decay_prod.charge), axis=-1)
        hsum = ak.sum(decay_prod.pt, axis=-1)

        charge_results = ak.where(tau_indecies == i, decay_prod_sum.charge, charge_results)
        ncand_results = ak.where(tau_indecies == i, ak.num(decay_prod.pt), ncand_results)
        ncand_charged_results = ak.where(tau_indecies == i, charged_decay_prod, ncand_charged_results)
        pt_results = ak.where(tau_indecies == i, decay_prod_sum.pt, pt_results)
        eta_results = ak.where(tau_indecies == i, decay_prod_sum.eta, eta_results)
        phi_results = ak.where(tau_indecies == i, decay_prod_sum.phi, phi_results)
        mass_results = ak.where(tau_indecies == i, decay_prod_sum.mass, mass_results)
        pt_hsum_results = ak.where(tau_indecies == i, hsum, pt_hsum_results)
        abs_charge_sum_results = ak.where(tau_indecies == i, abs_charge_sum, abs_charge_sum_results)

    # Calculating the difference of pt
    dpt_results = abs(tau.pt - pt_results)
    dpt_hsum_results = abs(tau.pt - pt_hsum_results)

    # charge decay mode difference
    for mode, charge in decaymodes.items():
        decaymode_charge = ak.where(tau.decayMode == mode, charge, decaymode_charge)

    # Check for non-finite values and replace them with -1
    mass_results = ak.where(np.isfinite(mass_results), mass_results, -1)
    eta_results = ak.where(np.isfinite(eta_results), eta_results, -9)

    events = set_ak_column_i8(events, "PFCandidatesSumTau.charge", charge_results)
    events = set_ak_column_i8(events, "PFCandidatesSumTau.ncand", ncand_results)
    events = set_ak_column_f32(events, "PFCandidatesSumTau.pt", pt_results)
    events = set_ak_column_f32(events, "PFCandidatesSumTau.eta", eta_results)
    events = set_ak_column_f32(events, "PFCandidatesSumTau.phi", phi_results)
    events = set_ak_column_f32(events, "PFCandidatesSumTau.mass", mass_results)
    events = set_ak_column_f32(events, "PFCandidatesSumTau.pt_hsum", pt_hsum_results)
    events = set_ak_column_f32(events, "PFCandidatesSumTau.dpt", dpt_results)
    events = set_ak_column_f32(events, "PFCandidatesSumTau.dpt_hsum", dpt_hsum_results)
    events = set_ak_column_i8(events, "PFCandidatesSumTau.decaymode_charge", decaymode_charge)
    events = set_ak_column_i8(events, "PFCandidatesSumTau.abs_charge", abs_charge_sum_results)
    events = set_ak_column_i8(events, "PFCandidatesSumTau.ncand_charged", ncand_charged_results)

    return events


@producer(
    uses={
        "Jet.*", "PFCandidate.*", "PFCandidateIndices.jet", attach_coffea_behavior,
    },
    produces={
        f"PFCandidatesSumJet.{name}" for name in [
            "pt", "eta", "phi", "mass", "ncand", "ncand_charged", "pt_hsum", "dpt", "dpt_hsum",
        ]
    },
)
def jetcand(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    collections = {
        "PFCandidate": {
            "type_name": "PFCand",
        },
        "Jet": {
            "type_name": "Jet",
        },
    }

    events = self[attach_coffea_behavior](events, collections=collections, **kwargs)

    jet = events.Jet
    jet_indices = ak.local_index(jet)

    pfcand_indices = events.PFCandidateIndices.jet
    pfcand = events.PFCandidate

    pt_results = ak.full_like(jet.pt, -1)
    eta_results = ak.full_like(jet.pt, -9)
    phi_results = ak.full_like(jet.pt, -4)
    mass_results = ak.full_like(jet.pt, -1)
    pt_hsum_results = ak.full_like(jet.pt, -1)
    ncand_results = ak.full_like(jet.pt, -1)
    ncand_charged_results = ak.full_like(jet.pt, -1)

    for i in range(ak.max(jet_indices)):
        decay_prod = pfcand[pfcand_indices == i]
        decay_prod_sum = decay_prod.sum(axis=-1)
        charged_n_cand_prod = ak.num(decay_prod[decay_prod.charge != 0].pt)

        hsum = ak.sum(decay_prod.pt, axis=-1)

        ncand_results = ak.where(jet_indices == i, ak.num(decay_prod.pt), ncand_results)
        ncand_charged_results = ak.where(jet_indices == i, charged_n_cand_prod, ncand_charged_results)
        pt_results = ak.where(jet_indices == i, decay_prod_sum.pt, pt_results)
        eta_results = ak.where(jet_indices == i, decay_prod_sum.eta, eta_results)
        phi_results = ak.where(jet_indices == i, decay_prod_sum.phi, phi_results)
        mass_results = ak.where(jet_indices == i, decay_prod_sum.mass, mass_results)
        pt_hsum_results = ak.where(jet_indices == i, hsum, pt_hsum_results)

    # Calculating the difference of pt
    dpt_vsum_results = abs(jet.pt - pt_results)
    dpt_hsum_results = abs(jet.pt - pt_hsum_results)

    # Check for non-finite values and replace them with -1
    mass_results = ak.where(np.isfinite(mass_results), mass_results, -1)
    eta_results = ak.where(np.isfinite(eta_results), eta_results, -9)

    events = set_ak_column_f32(events, "PFCandidatesSumJet.pt", pt_results)
    events = set_ak_column_f32(events, "PFCandidatesSumJet.eta", eta_results)
    events = set_ak_column_f32(events, "PFCandidatesSumJet.phi", phi_results)
    events = set_ak_column_f32(events, "PFCandidatesSumJet.mass", mass_results)
    events = set_ak_column_f32(events, "PFCandidatesSumJet.pt_hsum", pt_hsum_results)
    events = set_ak_column_f32(events, "PFCandidatesSumJet.dpt", dpt_vsum_results)
    events = set_ak_column_f32(events, "PFCandidatesSumJet.dpt_hsum", dpt_hsum_results)
    events = set_ak_column_i8(events, "PFCandidatesSumJet.ncand", ncand_results)
    events = set_ak_column_i8(events, "PFCandidatesSumJet.ncand_charged", ncand_charged_results)

    return events


@producer(
    uses={
        default, taucand, jetcand,
    },
    produces={
        default, taucand, jetcand,
    },
)
def merged_producer(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    events = self[default](events, **kwargs)
    events = self[taucand](events, **kwargs)
    events = self[jetcand](events, **kwargs)

    return events
