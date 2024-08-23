# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import set_ak_column


np = maybe_import("numpy")
ak = maybe_import("awkward")


set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


# Invariant Mass Producers
@producer(
    uses={
        "gen_*",
        attach_coffea_behavior,
    },
    produces={
        f"{y}_{x}" for x in ["bb", "tautau"] for y in ["pt", "dR"]
    },
)
def hbb_mjj(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    collections = {x: {"type_name": "GenParticle"} for x in ["gen_b_from_h", "gen_tau_from_h"]}
    events = self[attach_coffea_behavior](events, collections=collections, **kwargs)
    b = events.gen_b_from_h
    tau = events.gen_tau_from_h

    # Calculate invariant mass of b-jets
    pt_bb = b.sum(axis=-1).pt
    pt_tautau = tau.sum(axis=-1).pt

    # calculate dR between jets
    dR_bb = b[:, 0].delta_r(b[:, 1])
    dR_tautau = tau[:, 0].delta_r(tau[:, 1])

    # add columns
    events = set_ak_column_f32(events, "pt_bb", pt_bb)
    events = set_ak_column_f32(events, "pt_tautau", pt_tautau)
    events = set_ak_column_f32(events, "dR_bb", dR_bb)
    events = set_ak_column_f32(events, "dR_tautau", dR_tautau)

    return events


# Invariant Mass Producers
@producer(
    uses={
        "gen_h_to_*", attach_coffea_behavior,
    },
    produces={
        f"gen_hh.{var}"
        for var in ["pt", "eta", "phi", "mass", "cos_theta"]
    },
)
def get_h_gen_columns(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    collections = {x: {"type_name": "GenParticle"} for x in ["gen_h_to_b", "gen_h_to_tau"]}
    events = self[attach_coffea_behavior](events, collections=collections, **kwargs)

    h_bb = events.gen_h_to_b[:, 0]
    h_tautau = events.gen_h_to_tau[:, 0]

    hh = h_bb + h_tautau
    theta = abs(h_bb.theta - h_tautau.theta)
    theta = ak.where(theta > np.pi / 2, theta - np.pi / 2, theta)
    cos_theta = np.cos(theta)

    # add columns
    for var in ["pt", "eta", "phi", "mass"]:
        events = set_ak_column_f32(events, f"gen_hh.{var}", getattr(hh, var))
    events = set_ak_column_f32(events, "gen_hh.cos_theta", cos_theta)

    return events
