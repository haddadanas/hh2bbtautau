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
        "pt_bb", "pt_tautau",
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

    # add columns
    events = set_ak_column_f32(events, "pt_bb", pt_bb)
    events = set_ak_column_f32(events, "pt_tautau", pt_tautau)

    return events
