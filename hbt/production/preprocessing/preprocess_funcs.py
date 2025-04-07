from collections import defaultdict
from functools import partial

from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights, stitched_normalization_weights
from columnflow.production.processes import process_ids
from columnflow.util import DotDict, maybe_import

from hbt.production.hh_mass import hh_mass


ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        "channel_id",
    },
    produces={
        "channel_id",
    },
)
def pp_channel_id(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events["channel_id"] = events["channel_id"] - 1
    return events


@producer(
    uses={
        "channel_id",
    },
    produces={
        "channel_id",
    },
)
def channel_id_mask(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    ch_mask = events.channel_id <= 3
    return events[ch_mask]


@producer(
    uses={
        f"Electron.{var}" for var in ["pt", "eta", "dz", "dxy", "mvaIso_WP80", "mvaIso"]
    } | {
        f"Muon.{var}" for var in ["pt", "eta", "dz", "dxy", "tightId", "pfRelIso04_all"]
    } | {
        f"Tau.{var}" for var in [
            "pt", "eta", "dz", "dxy", "idDeepTau2018v2p5VSe", "idDeepTau2018v2p5VSmu", "idDeepTau2018v2p5VSjet",
        ]
    } | {
        "channel_id",
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
    channel_matcher = lambda ch: events.channel_id == self.config_inst.get_channel(ch).id - 1  # Adjust if ch changes

    # Create the Tau is Iso column
    tau_is_iso = 1 * ((tau_matcher("jet", "loose")) & (
        (
            (events.channel_id == 2) & (tau_matcher("e", "vvloose") | tau_matcher("mu", "vloose"))  # Adjust
        ) | (
            (events.channel_id != 2) & (tau_matcher("e", "vloose") | tau_matcher("mu", "tight"))  # Adjust
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

    # # delta r between leptons
    # delta_r = np.sqrt(
    #     (events.l1.eta - events.l2.eta) ** 2 + (events.l1.phi - events.l2.phi) ** 2
    # )
    # events = set_ak_column_f32(events, "delta_r_leps", delta_r)

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
        f"HHBJet.{var}" for var in ["pt", "phi", "eta", "mass", "hhbtag", "btagPNetB"]
    },
    produces={
        f"bjet{i}.{var}" for i in range(2) for var in ["pt", "phi", "eta", "mass", "hhbtag", "btagPNetB"]
    },
)
def pp_bjets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for i in range(2):
        bjets = Route(f"HHBJet[:, {i}]").apply(events, None)
        for f in ["pt", "phi", "eta", "mass", "hhbtag", "btagPNetB"]:
            fill_value = ak.fill_none(getattr(bjets, f), EMPTY_FLOAT)
            events = set_ak_column_f32(events, f"bjet{i}.{f}", fill_value)

    # delta r between bjets
    bjet0 = Route("HHBJet[:, 0]").apply(events, None)
    bjet1 = Route("HHBJet[:, 1]").apply(events, None)
    delta_r = np.sqrt(
        (bjet0.eta - bjet1.eta) ** 2 + (bjet0.phi - bjet1.phi) ** 2,
    )
    events = set_ak_column_f32(events, "delta_r_bjets", delta_r)

    return events


@producer(
    uses={"category_ids"},
    produces={"tau_pt{35,36,37,38,40,45,50,80}"},
)
def tau_selection_cuts(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    cats = [f"tau_pt{i}" for i in self.config_inst.x.selection_cuts]
    for cat in cats:
        cat_inst = self.config_inst.get_category(f"tautau__{cat}")
        cat_id = cat_inst.id
        mask = ak.any(events.category_ids == cat_id, axis=-1)
        events = set_ak_column(events, cat, mask, value_type=np.bool_)

    return events


category_ids_only_tau = category_ids.derive("category_ids_only_tau", cls_dict={
    "skip_category": (lambda self, task, category_inst: not category_inst.name.startswith("tautau__tau_pt"))
})


@producer(
    uses={
        pp_bjets, pp_jets, pp_leptons, pp_channel_id, hh_mass, process_ids, channel_id_mask,
        tau_selection_cuts,
    },
    produces={
        pp_bjets, pp_jets, pp_leptons, pp_channel_id, hh_mass, process_ids, "n_jets", "n_bjets",
        "n_taus", "normalization_weight", "channel_id", tau_selection_cuts,
    },
)
def preprocess(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # add categories
    # events = self[category_ids](events)
    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        if self.dataset_inst.name == "dy_m50toinf_amcatnloooo":
            events = self[stitched_normalization_weights](events, **kwargs)
        else:
            events = self[normalization_weights](events, **kwargs)

    if "normalization_weight_inclusive" in events.fields:
        events["normalization_weight"] = events["normalization_weight_inclusive"]

    events = self[hh_mass](events)

    events = self[channel_id_mask](events)


    events = self[tau_selection_cuts](events)

    events = self[pp_channel_id](events)

    # events = self[pp_jets](events)

    events = self[pp_bjets](events)

    events = self[pp_leptons](events)

    # only get tautau channel
    tautau_mask = events.channel_id == 2
    events = events[tautau_mask]

    n_jets = ak.num(events.Jet, axis=1)
    n_bjets = ak.num(events.HHBJet, axis=1)
    n_taus = ak.num(events.Tau, axis=1)

    events = set_ak_column_f32(events, "n_jets", n_jets)
    events = set_ak_column_f32(events, "n_bjets", n_bjets)
    events = set_ak_column_f32(events, "n_taus", n_taus)

    return events


@preprocess.init
def preprocess_init(self: Producer) -> None:
    if getattr(self, "dataset_inst", None) is None:
        return
    if self.dataset_inst.name == "dy_m50toinf_amcatnlo":
        self.uses.add(stitched_normalization_weights)
    else:
        self.uses.add(normalization_weights)
