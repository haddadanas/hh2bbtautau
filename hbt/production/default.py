# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.normalization import stitched_normalization_weights
from columnflow.production.categories import category_ids
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights
from columnflow.util import maybe_import

from hbt.production.features import features
from hbt.production.weights import (
    normalized_pu_weight, normalized_pdf_weight, normalized_murmuf_weight,
)
from hbt.production.btag import normalized_btag_weights
from hbt.production.tau import tau_weights, trigger_weights
from hbt.production.gen_column_producer import hbb_mjj, get_h_gen_columns
from hbt.production.reco_higgs_decay import reco_higgs
from hbt.util import IF_DATASET_HAS_LHE_WEIGHTS

ak = maybe_import("awkward")


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
    uses={
        default, hbb_mjj, reco_higgs, get_h_gen_columns,
    },
    produces={
        default, hbb_mjj, "selection_mask", "process_id", reco_higgs, get_h_gen_columns,
    },
)
def gen_default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # default
    events = self[default](events, **kwargs)

    # hbb mjj
    if "hh_" in self.dataset_inst.name or "h_" in self.dataset_inst.name:
        events = self[hbb_mjj](events, **kwargs)
        events = self[get_h_gen_columns](events, **kwargs)

    # reco higgs decay
    events = self[reco_higgs](events, **kwargs)

    return events
