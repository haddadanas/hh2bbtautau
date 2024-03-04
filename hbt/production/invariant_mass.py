# coding: utf-8

import functools
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, set_ak_column
from columnflow.production.util import attach_coffea_behavior

np = maybe_import("numpy")
ak = maybe_import("awkward")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses=(
        {
            f"{field}.{var}"
            for field in ["Muon"]
            for var in ["pt", "mass", "eta", "phi", "charge"]
        } | {
            attach_coffea_behavior,
        }
    ),
    produces={
        "m2mu",
    },
)
def muons_invariant_mass(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Construct the invariant mass of the muons in each event.
    """

    # attach coffea behavior for four-vector arithmetic
    events = self[attach_coffea_behavior](
        events,
        collections=["Muon"],
        **kwargs,
    )

    # sort the muons by pt
    pt_sort_mask = ak.argsort(events.Muon.pt, ascending=False, axis=-1)
    sorted_muons = events.Muon[pt_sort_mask]

    # sum the four-vectors of the first leading muons
    di_muon = sorted_muons[:, :2].sum(axis=-1)

    # Select only events with at least two muons
    mask = ak.num(sorted_muons, axis=-1)

    di_muon_mass = ak.where(mask == 2, di_muon.mass, EMPTY_FLOAT)
    di_muon_mass = ak.nan_to_num(di_muon_mass, nan=EMPTY_FLOAT)

    # write the invariant mass to the events
    events = set_ak_column_f32(events, "m2mu", di_muon_mass)

    return events


@producer(
    uses=(
        {
            f"{field}.{var}"
            for field in ["Tau"]
            for var in ["pt", "mass", "eta", "phi", "charge"]
        } | {
            attach_coffea_behavior,
        }
    ),
    produces={
        "m2tau",
    },
)
def taus_invariant_mass(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Construct the invariant mass of the taus in each event.
    """

    # attach coffea behavior for four-vector arithmetic
    events = self[attach_coffea_behavior](
        events,
        collections=["Tau"],
        **kwargs,
    )

    # sort the taus by pt
    pt_sort_mask = ak.argsort(events.Tau.pt, ascending=False, axis=-1)
    sorted_taus = events.Tau[pt_sort_mask]

    # sum the four-vectors of the first leading taus
    di_tau = sorted_taus[:, :2].sum(axis=-1)

    # Select only events with at least two taus
    mask = ak.num(sorted_taus, axis=-1)

    di_tau_mass = ak.where(mask == 2, di_tau.mass, EMPTY_FLOAT)

    # write the invariant mass to the events
    events = set_ak_column_f32(events, "m2tau", di_tau_mass)

    return events
