# coding: utf-8

"""
Producers that determine the generator-level particles related to a top quark decay.
"""
from __future__ import annotations

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import set_ak_column


ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")


def get_sorting_mask(tau: ak.Array) -> ak.Array:
    """
    Get the sorting mask for the tau collection. Sorts that the first two taus have opposite charge and highest tau pt.
    """
    num_mask = ak.num(tau) > 2
    sorting_mask = ak.argsort(tau.pt, axis=-1, ascending=False)

    # Get the indices of all possible tau pairs
    ind1, ind2 = ak.unzip(ak.argcombinations(tau.charge, 2, axis=1))
    charge_mask = tau.charge[ind1] + tau.charge[ind2] == 0
    ind1, ind2 = ind1[charge_mask][:, :1], ind2[charge_mask][:, :1]

    return ak.where(num_mask, ak.concatenate([ind1, ind2], axis=-1), sorting_mask)


def get_nu(array: ak.Array, mask: ak.Array) -> ak.Array:
    """
    Get the neutrino 4-momentum from the dilepton 4-momentum with the constraints m_nu = 0 and m_H = 125 GeV.
    """
    nu_factor = ak.where(mask, (125.0**2 - array.mass2) / (2 * array.p * (array.energy - array.p)), 0)
    return ak.zip(
        {
            "pt": nu_factor * array.pt,
            "eta": ak.nan_to_num(array.eta, nan=-10),
            "phi": ak.nan_to_num(array.phi, nan=-10),
            "energy": nu_factor * array.p,
        },
        with_name="PtEtaPhiELorentzVector",
        behavior=coffea.nanoevents.methods.vector.behavior,
    )


@producer(
    uses={
        f"{part}.{var}"
        for part in ["Tau", "Electron", "Muon"]
        for var in ["pt", "eta", "phi", "mass", "charge"]
    } | {
        f"{part}.{var}"
        for part in ["HHBJet"]
        for var in ["pt", "eta", "phi", "mass"]
    } | {attach_coffea_behavior, "category_ids"},
    produces={
        f"reco_{higgs}.{var}"
        for var in ["pt", "eta", "phi", "mass"]
        for higgs in ["hh", "hh_tau", "hh_e", "hh_mu", "h_tau", "h_e", "h_mu", "h_jet"]
    },
)
def reco_higgs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    collections = {"Tau": {"type_name": "Tau"}, "HHBJet": {"type_name": "Jet"},
        "Jet": {"type_name": "Jet"}, "Electron": {"type_name": "Electron"}, "Muon": {"type_name": "Muon"}}
    events = self[attach_coffea_behavior](events, collections=collections, **kwargs)

    tau = events.Tau
    sorting_mask = get_sorting_mask(tau)
    tau = tau[sorting_mask]
    get_first_tau = lambda mask: ak.mask(tau, mask)[:, 0]

    # e_tau channel
    etau_mask = ak.any(events.category_ids == 140, axis=-1) & (ak.num(tau) >= 1) & (ak.num(events.Electron) >= 1)
    e = ak.mask(events.Electron, etau_mask)[:, 0]
    first_tau = get_first_tau(etau_mask)
    etau = e + first_tau
    nu = get_nu(etau, etau_mask)
    charge_mask = e.charge + first_tau.charge == 0
    h_e = ak.mask(etau + nu, etau_mask & charge_mask)

    # mu_tau channel
    mutau_mask = ak.any(events.category_ids == 150, axis=-1) & (ak.num(tau) >= 1) & (ak.num(events.Muon) >= 1)
    mu = ak.mask(events.Muon, mutau_mask)[:, 0]
    first_tau = get_first_tau(mutau_mask)
    mutau = mu + first_tau
    nu = get_nu(mutau, mutau_mask)
    charge_mask = mu.charge + first_tau.charge == 0
    h_mu = ak.mask(mutau + nu, mutau_mask & charge_mask)

    # tautau channel
    tautau_mask = ak.any(events.category_ids == 160, axis=-1) & (ak.num(tau) == 2) & (tau.sum(axis=-1).charge == 0)
    di_tau = tau.sum(axis=-1)
    nu = get_nu(di_tau, tautau_mask)
    h_tau = ak.mask(di_tau + nu, tautau_mask)

    # Select the two highest-pt jets
    jet = events.HHBJet
    jet_mask = ak.num(jet) == 2
    h_jet = ak.mask(jet.sum(axis=-1), jet_mask)

    hh_tau = ak.mask(h_tau + h_jet, tautau_mask & jet_mask)
    hh_e = ak.mask(h_e + h_jet, etau_mask & jet_mask)
    hh_mu = ak.mask(h_mu + h_jet, mutau_mask & jet_mask)

    for higgs, h in zip(
        ["hh_tau", "hh_e", "hh_mu", "h_tau", "h_e", "h_mu", "h_jet"],
        [hh_tau, hh_e, hh_mu, h_tau, h_e, h_mu, h_jet],
    ):
        for var in ["pt", "eta", "phi", "mass"]:
            array = ak.fill_none(getattr(h, var), -10)
            events = set_ak_column(events, f"reco_{higgs}.{var}", array)

    for var in ["pt", "eta", "phi", "mass"]:
        array = ak.full_like(tautau_mask, -10.0, dtype=np.float64)
        for hh, mask in zip([hh_tau, hh_e, hh_mu], [tautau_mask, etau_mask, mutau_mask]):
            array = ak.where(mask & jet_mask, getattr(hh, var), array)
        events = set_ak_column(events, f"reco_hh.{var}", array)

    return events
