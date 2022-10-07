# coding: utf-8

"""
Jet selection methods.
"""

from operator import or_
from functools import reduce

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from hbt.production.hhbtag import hhbtag

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        hhbtag,
        "trigger_ids",
        "nJet", "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.jetId", "Jet.puId",
        "Jet.btagDeepFlavB",
        "nFatJet", "FatJet.pt", "FatJet.eta", "FatJet.phi", "FatJet.mass", "FatJet.msoftdrop",
        "FatJet.jetId", "FatJet.subJetIdx1", "FatJet.subJetIdx2",
        "nSubJet", "SubJet.pt", "SubJet.eta", "SubJet.phi", "SubJet.mass", "SubJet.btagDeepB",
    },
    produces={
        "Jet.hhbtag",
    },
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    trigger_results: SelectionResult,
    lepton_results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Jet selection based on ultra-legacy recommendations.

    Resources:
    https://twiki.cern.ch/twiki/bin/view/CMS/JetID?rev=107#nanoAOD_Flags
    https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVUL?rev=15#Recommendations_for_the_13_T_AN1
    https://twiki.cern.ch/twiki/bin/view/CMS/PileupJetIDUL?rev=17
    https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD?rev=100#Jets
    """
    is_2016 = self.config_inst.campaign.x.year == 2016

    # local jet index
    li = ak.local_index(events.Jet)

    # common ak4 jet mask for normal and vbf jets
    ak4_mask = (
        (events.Jet.jetId == 6) &  # tight plus lepton veto
        ((events.Jet.pt >= 50.0) | (events.Jet.puId == (1 if is_2016 else 4))) &  # flipped in 2016
        ak.all(events.Jet.metric_table(lepton_results.x.lepton_pair) > 0.5, axis=2)
    )

    # default jets
    default_mask = (
        ak4_mask &
        (events.Jet.pt > 20.0) &
        (abs(events.Jet.eta) < 2.4)
    )

    # get the scores of the hhbtag and per event get the two indices corresponding to the best pick
    hhbtag_scores = self[hhbtag](events, default_mask, lepton_results.x.lepton_pair, **kwargs)
    score_indices = ak.argsort(hhbtag_scores, axis=1, ascending=False)
    # pad the indices to simplify creating the bjet mask
    padded_bjet_indices = ak.pad_none(score_indices, 2, axis=1)[..., :2][..., :2]
    bjet_mask = ((li == padded_bjet_indices[..., [0]]) | (li == padded_bjet_indices[..., [1]]))
    # get indices for actual book keeping only for events with both lepton candidates and where at
    # least two jets pass the default mask (bjet candidates)
    valid_score_mask = (
        default_mask &
        (ak.sum(default_mask, axis=1) >= 2) &
        (ak.num(lepton_results.x.lepton_pair, axis=1) == 2)
    )
    bjet_indices = score_indices[valid_score_mask[score_indices]][..., :2]

    # vbf jets
    vbf_mask = (
        ak4_mask &
        (events.Jet.pt > 30.0) &
        (abs(events.Jet.eta) < 4.7) &
        (~bjet_mask)
    )

    # build vectors of vbf jets representing all combinations and apply selections
    vbf1, vbf2 = ak.unzip(ak.combinations(events.Jet[vbf_mask], 2, axis=1))
    vbf_pair = ak.concatenate([vbf1[..., None], vbf2[..., None]], axis=2)
    vbfjj = vbf1 + vbf2
    vbf_pair_mask = (
        (vbfjj.mass > 500.0) &
        (abs(vbf1.eta - vbf2.eta) > 3.0)
    )

    # extra requirements for events for which only the tau tau vbf cross trigger fired
    cross_vbf_ids = [t.id for t in self.config_inst.x.triggers if t.has_tag("cross_tau_tau_vbf")]
    if not cross_vbf_ids:
        cross_vbf_mask = ak.full_like(1 * events.event, False, dtype=bool)
    else:
        cross_vbf_masks = [events.trigger_ids == tid for tid in cross_vbf_ids]
        cross_vbf_mask = ak.all(reduce(or_, cross_vbf_masks), axis=1)
    vbf_pair_mask = vbf_pair_mask & (
        (~cross_vbf_mask) | (
            (vbfjj.mass > 800) &
            (ak.max(vbf_pair.pt, axis=2) > 140.0) &
            (ak.min(vbf_pair.pt, axis=2) > 60.0)
        )
    )

    # get the index to the pair with the highest pass
    vbf_mass_indices = ak.argsort(vbfjj.mass, axis=1, ascending=False)
    vbf_pair_index = vbf_mass_indices[vbf_pair_mask[vbf_mass_indices]][..., :1]

    # get the two indices referring to jets passing vbf_mask
    # and change them so that they point to jets in the full set, sorted by pt
    vbf_indices_local = ak.concatenate(
        [
            ak.singletons(idx) for idx in
            ak.unzip(ak.firsts(ak.argcombinations(events.Jet[vbf_mask], 2, axis=1)[vbf_pair_index]))
        ],
        axis=1,
    )
    vbfjet_indices = li[vbf_mask][vbf_indices_local]
    vbfjet_indices = vbfjet_indices[ak.argsort(events.Jet[vbfjet_indices].pt, axis=1, ascending=False)]

    # check whether the two bjets were matched by fatjet subjets to mark it as boosted
    fatjet_mask = (
        (events.FatJet.jetId == 6) &  # tight plus lepton veto
        (events.FatJet.msoftdrop > 30.0) &
        (abs(events.FatJet.eta) < 2.4) &
        ak.all(events.FatJet.metric_table(lepton_results.x.lepton_pair) > 0.5, axis=2) &
        (events.FatJet.subJetIdx1 >= 0) &
        (events.FatJet.subJetIdx2 >= 0)
    )

    # unique subjet matching
    metrics = events.FatJet.subjets.metric_table(events.Jet[bjet_indices])
    subjets_match = (
        ak.all(ak.sum(metrics < 0.4, axis=3) == 1, axis=2) &
        (ak.num(bjet_indices, axis=1) == 2)
    )
    fatjet_mask = fatjet_mask & subjets_match

    # store fatjet and subjet indices
    fatjet_indices = ak.local_index(events.FatJet.pt)[fatjet_mask]
    subjet_indices = ak.concatenate(
        [
            events.FatJet[fatjet_mask].subJetIdx1[..., None],
            events.FatJet[fatjet_mask].subJetIdx2[..., None],
        ],
        axis=2,
    )

    # discard the event in case the (first) fatjet with matching subjets is found
    # but they are not b-tagged (TODO: move to deepjet when available for subjets)
    wp = self.config_inst.x.btag_working_points.deepcsv.loose
    subjets_btagged = ak.all(events.SubJet[ak.firsts(subjet_indices)].btagDeepB > wp, axis=1)

    # for central jets (bjetcandidates) remove bjets to avoid amiguities later on
    default_mask = default_mask & (~bjet_mask)

    # pt sorted indices to convert mask
    sorted_indices = ak.argsort(events.Jet.pt, axis=-1, ascending=False)
    jet_indices = sorted_indices[default_mask[sorted_indices]]

    # final event selection
    jet_sel = (
        (ak.sum(default_mask, axis=1) >= 2) &
        ak.fill_none(subjets_btagged, True)  # was none for events with not matched fatjet
    )

    # some final type conversions
    jet_indices = ak.values_astype(ak.fill_none(jet_indices, 0), np.int32)
    bjet_indices = ak.values_astype(bjet_indices, np.int32)
    fatjet_indices = ak.values_astype(fatjet_indices, np.int32)
    vbfjet_indices = ak.values_astype(ak.fill_none(vbfjet_indices, 0), np.int32)

    # store some columns
    events = set_ak_column(events, "Jet.hhbtag", hhbtag_scores)

    # build and return selection results plus new columns (src -> dst -> indices)
    return events, SelectionResult(
        steps={
            "jet": jet_sel,
        },
        objects={
            "Jet": {
                "Jet": jet_indices,
                "BJet": bjet_indices,
                "FatJet": fatjet_indices,
                "SubJet1": subjet_indices[..., 0],
                "SubJet2": subjet_indices[..., 1],
                "VBFJet": vbfjet_indices,
            },
        },
        aux={
            "n_central_jets": (
                ak.num(jet_indices, axis=1) +
                ak.num(bjet_indices, axis=1)
            ),
        },
    )
