from collections import defaultdict
from functools import partial

from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
from columnflow.production import Producer, producer
from columnflow.production.normalization import normalization_weights
from columnflow.production.processes import process_ids
from columnflow.util import DotDict, maybe_import

from hbt.production.hh_mass import hh_mass


ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        "channel_id"
    },
    produces={
        "channel_id"
    },
)
def pp_channel_id(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events["channel_id"] = events["channel_id"] - 1
    return events


@producer(
    uses={
        f"Electron.{var}" for var in ["pt", "eta", "dz", "dxy", "mvaIso_WP80", "mvaIso"]
    } | {
        f"Muon.{var}" for var in ["pt", "eta", "dz", "dxy", "tightId", "pfRelIso04_all"]
    } | {
        f"Tau.{var}" for var in [
            "pt", "eta", "dz", "dxy", "idDeepTau2018v2p5VSe", "idDeepTau2018v2p5VSmu", "idDeepTau2018v2p5VSjet"
        ]
    } | {
        "channel_id"
    },
    produces={
        f"l1.{var}" for var in ["pt", "eta", "dz", "dxy", "tauVSjet", "tauVSe", "tauVSmu", "is_iso", "iso_score"]
    } | {
        f"l2.{var}" for var in ["pt", "eta", "dz", "dxy", "tauVSjet", "tauVSe", "tauVSmu", "is_iso"]
    },
)
def pp_leptons(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # helper functions
    # get the used DeepTau wp
    tau_iso_wp = DotDict.wrap({
        k: self.config_inst.x.tau_id_working_points[f"tau_vs_{k}"] for k in ["e", "mu", "jet"]
    })
    tau_matcher = lambda tag, wp: events.Tau[f"idDeepTau2018v2p5VS{tag}"] >= tau_iso_wp[tag][wp]
    channel_matcher = lambda ch: events.channel_id == self.config_inst.get_channel(ch).id

    # Create the Tau is Iso column
    tau_is_iso = 1 * ((tau_matcher("jet", "loose")) & (
        (
            (events.channel_id == 3) & (tau_matcher("e", "vvloose") | tau_matcher("mu", "vloose"))
        ) | (
            (events.channel_id != 3) & (tau_matcher("e", "vloose") | tau_matcher("mu", "tight"))
        )
    ))

    # set the columns
    result = defaultdict(lambda: np.full(len(events), EMPTY_FLOAT))

    # Take care of the different channels seperately
    # etau and mutau channel
    for ch in ["etau", "mutau"]:
        ch_mask = channel_matcher(ch)
        l1_route = Route("Electron[:, 0]") if ch == "etau" else Route("Muon[:, 0]")
        l1 = l1_route.apply(events, None)
        l2 = Route("Tau[:, 0]").apply(events, None)
        iso_tag = "mvaIso" if ch == "etau" else "pfRelIso04_all"
        is_iso_tag = "mvaIso_WP80" if ch == "etau" else "tightId"
        for f in ["pt", "eta", "dz", "dxy"]:
            result[f"l1.{f}"] = ak.where(ch_mask, l1[f], result[f"l1.{f}"])
            result[f"l2.{f}"] = ak.where(ch_mask, l2[f], result[f"l2.{f}"])
        iso_events = l1[iso_tag] / ak.max(l1[iso_tag])
        result["l1.iso_score"] = ak.where(ch_mask, iso_events, result["l1.iso_score"])
        result["l1.is_iso"] = ak.where(ch_mask, 1 * l1[is_iso_tag], result["l1.is_iso"])
        result["l2.is_iso"] = ak.where(ch_mask, tau_is_iso[:, 0], result["l2.is_iso"])
        result["l2.tauVSjet"] = ak.where(ch_mask, l2["idDeepTau2018v2p5VSjet"], result["l2.tauVSjet"])
        result["l2.tauVSe"] = ak.where(ch_mask, l2["idDeepTau2018v2p5VSe"], result["l2.tauVSe"])
        result["l2.tauVSmu"] = ak.where(ch_mask, l2["idDeepTau2018v2p5VSmu"], result["l2.tauVSmu"])

    # tautau channel
    is_tautau = channel_matcher("tautau")
    l1 = Route("Tau[:, 0]").apply(events, None)
    l2 = Route("Tau[:, 1]").apply(events, None)
    get_tau_iso = lambda ind: Route(f"[:, {ind}]").apply(tau_is_iso, None)
    for f in ["pt", "eta", "dz", "dxy"]:
        result[f"l1.{f}"] = ak.where(is_tautau, l1[f], result[f"l1.{f}"])
        result[f"l2.{f}"] = ak.where(is_tautau, l2[f], result[f"l2.{f}"])
    for f in ["jet", "e", "mu"]:
        result[f"l1.tauVS{f}"] = ak.where(is_tautau, l1[f"idDeepTau2018v2p5VS{f}"], result[f"l1.tauVS{f}"])
        result[f"l2.tauVS{f}"] = ak.where(is_tautau, l2[f"idDeepTau2018v2p5VS{f}"], result[f"l2.tauVS{f}"])

    result["l1.is_iso"] = ak.where(is_tautau, get_tau_iso(0), result["l1.is_iso"])
    result["l2.is_iso"] = ak.where(is_tautau, get_tau_iso(1), result["l2.is_iso"])

    for field in self.produces:
        events = set_ak_column_f32(events, field, result[field])
    return events


@producer(
    uses={
        f"Jet.{var}" for var in ["pt", "eta", "mass", "hhbtag", "btagPNetB"]
    },
    produces={
        f"jet{i}.{var}" for i in range(4) for var in ["pt", "eta", "mass", "hhbtag", "btagPNetB"]
    },
)
def pp_jets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for i in range(4):
        jets = Route(f"Jet[:, {i}]").apply(events, None)
        for f in ["pt", "eta", "mass", "hhbtag", "btagPNetB"]:
            fill_value = ak.fill_none(getattr(jets, f), EMPTY_FLOAT)
            events = set_ak_column_f32(events, f"jet{i}.{f}", fill_value)

    return events


@producer(
    uses={
        pp_jets, pp_leptons, pp_channel_id, hh_mass, normalization_weights, process_ids
    },
    produces={
        pp_jets, pp_leptons, pp_channel_id, hh_mass, normalization_weights, process_ids
    },
)
def preprocess(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[normalization_weights](events, **kwargs)

    events = self[hh_mass](events)

    events = self[pp_jets](events)

    events = self[pp_leptons](events)

    events = self[pp_channel_id](events)

    return events
