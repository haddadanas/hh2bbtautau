# coding: utf-8

"""
Column production methods related to higher-level features.
"""

import functools

from columnflow.production import Producer, producer
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
    },
    produces={
        f"PFCandidatesTau.{name}" for name in [
            "charge_sum_pfcand", "ncand_per_part",
            "pt_vsum_pfcand", "pt_hsum_pfcand", "dpt_vsum", "dpt_hsum",
            "decaymode_charge", "abs_charge_sum_pfcand",
        ]
    },
)
def taucand(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    from coffea.nanoevents.methods import vector

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

    tau = events.Tau
    pfcand_indices = events.PFCandidateIndices.tau
    pfcand = events.PFCandidate
    pfcand = ak.zip(
        {
            "pt": pfcand.pt,
            "eta": pfcand.eta,
            "phi": pfcand.phi,
            "mass": pfcand.mass,
            "charge": pfcand.charge,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )

    charge_results = ak.local_index(tau)
    ncand_results = ak.local_index(tau)
    pt_vsum_results = ak.local_index(tau)
    pt_hsum_results = ak.local_index(tau)
    decaymode_charge = ak.local_index(tau)
    abs_charge_sum_pfcand = ak.local_index(tau)

    for i in range(ak.max(charge_results)):
        decay_prod = pfcand[pfcand_indices == i]
        charge_sum = ak.sum(decay_prod.charge, axis=-1)
        abs_chatge_sum = ak.sum(abs(decay_prod.charge), axis=-1)
        charge_results = ak.where(charge_results == i, charge_sum, charge_results)
        abs_charge_sum_pfcand = ak.where(abs_charge_sum_pfcand == i, abs_chatge_sum, abs_charge_sum_pfcand)
        ncand_results = ak.where(ncand_results == i, ak.num(decay_prod), ncand_results)
        vec_sum = ak.zip(
            {
                "x": ak.sum(decay_prod.x, axis=-1),
                "y": ak.sum(decay_prod.y, axis=-1),
                "z": ak.sum(decay_prod.z, axis=-1),
                "t": ak.sum(decay_prod.t, axis=-1),
            },
            with_name="LorentzVector",
            behavior=vector.behavior,
        )
        hsum = ak.sum(decay_prod.pt, axis=-1)
        pt_vsum_results = ak.where(pt_vsum_results == i, vec_sum.pt, pt_vsum_results)
        pt_hsum_results = ak.where(pt_hsum_results == i, hsum, pt_hsum_results)

    # Calculating the difference of pt
    dpt_vsum_results = abs(tau.pt - pt_vsum_results)
    dpt_hsum_results = abs(tau.pt - pt_hsum_results)

    # charge decay mode difference
    for mode, charge in decaymodes.items():
        decaymode_charge = ak.where(tau.decayMode == mode, charge, decaymode_charge)

    events = set_ak_column_i8(events, "PFCandidatesTau.charge_sum_pfcand", charge_results)
    events = set_ak_column_i8(events, "PFCandidatesTau.ncand_per_part", ncand_results)
    events = set_ak_column_f32(events, "PFCandidatesTau.pt_vsum_pfcand", pt_vsum_results)
    events = set_ak_column_f32(events, "PFCandidatesTau.pt_hsum_pfcand", pt_hsum_results)
    events = set_ak_column_f32(events, "PFCandidatesTau.dpt_vsum", dpt_vsum_results)
    events = set_ak_column_f32(events, "PFCandidatesTau.dpt_hsum", dpt_hsum_results)
    events = set_ak_column_i8(events, "PFCandidatesTau.decaymode_charge", decaymode_charge)
    events = set_ak_column_i8(events, "PFCandidatesTau.abs_charge_sum_pfcand", abs_charge_sum_pfcand)

    return events


@producer(
    uses={
        "Jet.*", "PFCandidate.*", "PFCandidateIndices.jet",
    },
    produces={
        f"PFCandidatesJet.{name}" for name in [
            "pt_vsum_pfcand", "pt_hsum_pfcand", "dpt_vsum", "dpt_hsum", "ncand_per_part",
        ]
    },
)
def jetcand(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    from coffea.nanoevents.methods import vector

    jet = events.Jet
    pfcand_indices = events.PFCandidateIndices.jet
    pfcand = events.PFCandidate
    pfcand = ak.zip(
        {
            "pt": pfcand.pt,
            "eta": pfcand.eta,
            "phi": pfcand.phi,
            "mass": pfcand.mass,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )
    pt_vsum_results = ak.local_index(jet)
    pt_hsum_results = ak.local_index(jet)
    ncand_results = ak.local_index(jet)

    for i in range(ak.max(pt_vsum_results)):
        decay_prod = pfcand[pfcand_indices == i]
        vec_sum = ak.zip(
            {
                "x": ak.sum(decay_prod.x, axis=-1),
                "y": ak.sum(decay_prod.y, axis=-1),
                "z": ak.sum(decay_prod.z, axis=-1),
                "t": ak.sum(decay_prod.t, axis=-1),
            },
            with_name="LorentzVector",
            behavior=vector.behavior,
        )
        hsum = ak.sum(decay_prod.pt, axis=-1)
        pt_vsum_results = ak.where(pt_vsum_results == i, vec_sum.pt, pt_vsum_results)
        pt_hsum_results = ak.where(pt_hsum_results == i, hsum, pt_hsum_results)
        ncand_results = ak.where(ncand_results == i, ak.num(decay_prod), ncand_results)

    # Calculating the difference of pt
    dpt_vsum_results = abs(jet.pt - pt_vsum_results)
    dpt_hsum_results = abs(jet.pt - pt_hsum_results)

    events = set_ak_column_f32(events, "PFCandidatesJet.pt_vsum_pfcand", pt_vsum_results)
    events = set_ak_column_f32(events, "PFCandidatesJet.pt_hsum_pfcand", pt_hsum_results)
    events = set_ak_column_i8(events, "PFCandidatesJet.ncand_per_part", ncand_results)
    events = set_ak_column_f32(events, "PFCandidatesJet.dpt_vsum", dpt_vsum_results)
    events = set_ak_column_f32(events, "PFCandidatesJet.dpt_hsum", dpt_hsum_results)

    return events


@producer(
    uses={
        default, jetcand, taucand,
    },
    produces={
        default, jetcand, taucand,
    },
)
def merged_producer(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    events = self[default](events, **kwargs)
    events = self[jetcand](events, **kwargs)
    events = self[taucand](events, **kwargs)

    return events
