# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

import functools

from columnflow.production import Producer, producer
from columnflow.production.normalization import stitched_normalization_weights
from columnflow.production.categories import category_ids
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, set_ak_column, optional_column as optional
from columnflow.production.util import attach_coffea_behavior

from hbt.production.features import features
from hbt.production.weights import (
    normalized_pu_weight, normalized_pdf_weight, normalized_murmuf_weight,
)
from hbt.production.btag import normalized_btag_weights
from hbt.production.tau import tau_weights, trigger_weights
from hbt.util import IF_DATASET_HAS_LHE_WEIGHTS

ak = maybe_import("awkward")
np = maybe_import("numpy")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        category_ids, features, stitched_normalization_weights, normalized_pu_weight,
        normalized_btag_weights, tau_weights, electron_weights, muon_weights, trigger_weights,
        IF_DATASET_HAS_LHE_WEIGHTS(normalized_pdf_weight, normalized_murmuf_weight),
    },
    produces={
        category_ids, features, stitched_normalization_weights, normalized_pu_weight,
        normalized_btag_weights, tau_weights, electron_weights, muon_weights, trigger_weights,
        IF_DATASET_HAS_LHE_WEIGHTS(normalized_pdf_weight, normalized_murmuf_weight),
    },
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # category ids
    events = self[category_ids](events, **kwargs)

    # features
    events = self[features](events, **kwargs)

    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[stitched_normalization_weights](events, **kwargs)

        # normalized pdf weight
        if self.has_dep(normalized_pdf_weight):
            events = self[normalized_pdf_weight](events, **kwargs)

        # normalized renorm./fact. weight
        if self.has_dep(normalized_murmuf_weight):
            events = self[normalized_murmuf_weight](events, **kwargs)

        # normalized pu weights
        events = self[normalized_pu_weight](events, **kwargs)

        # btag weights
        events = self[normalized_btag_weights](events, **kwargs)

        # tau weights
        events = self[tau_weights](events, **kwargs)

        # electron weights
        events = self[electron_weights](events, **kwargs)

        # muon weights
        events = self[muon_weights](events, **kwargs)

        # trigger weights
        events = self[trigger_weights](events, **kwargs)

    return events


@producer(
    uses=(
        {
            f"{field}.{var}"
            for field in ["Muon"]
            for var in ["pt", "mass", "eta", "phi", "charge"]
        } | {
            attach_coffea_behavior, "MET.pt", optional("gen*.*"), optional("LHE.Vpt"), optional("old_*.*"),
        }
    ),
    produces={
        "pt2mu", "plot_pt2mu", "plot_ptV",
    },
)
def muons_pt(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
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

    # Select only events with at least two muons
    num_mask = ak.num(sorted_muons, axis=-1) >= 2
    sorted_muons = ak.mask(sorted_muons, num_mask)

    # sum the four-vectors of the first leading muons
    di_muon = sorted_muons[:, :2].sum(axis=-1)

    cut = num_mask & (di_muon.mass > 50) & (events.MET.pt < 45) & (sorted_muons[:, 0].delta_r(sorted_muons[:, 1]) > 0.1)
    di_muon_pt = ak.where(cut, di_muon.pt, EMPTY_FLOAT)
    di_muon_pt = ak.fill_none(di_muon_pt, EMPTY_FLOAT)
    di_muon_pt = ak.nan_to_num(di_muon_pt, nan=EMPTY_FLOAT)
    if self.dataset_inst.has_tag("is_dy"):
        plot_pt2mu = ak.sum(events.gen_z_to_mu.pt[:, :1], axis=-1)
        plot_pt2mu = ak.where(plot_pt2mu > 0, plot_pt2mu, EMPTY_FLOAT)
        events = set_ak_column_f32(events, "plot_pt2mu", plot_pt2mu)
        events = set_ak_column_f32(events, "plot_ptV", events.LHE.Vpt)
        
    else:
        events = set_ak_column_f32(events, "plot_pt2mu", di_muon_pt)
        events = set_ak_column_f32(events, "plot_ptV", di_muon_pt)
    # write the invariant mass to the events
    events = set_ak_column_f32(events, "pt2mu", di_muon_pt)

    return events
