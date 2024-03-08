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


@producer(
    uses=(
        {
            f"{field}.{var}"
            for field in ["Muon"]
            for var in ["pt", "mass", "eta", "phi"]
        } | {
            "MET.pt", "MET.phi", attach_coffea_behavior,
        }
    ),
    produces={
        "mT_W",
    },
)
def transverse_mass_W(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Construct the transverse mass of the W boson in each event.
    """
    from coffea.nanoevents.methods import vector
    # attach coffea behavior for four-vector arithmetic
    events = self[attach_coffea_behavior](
        events,
        collections=["Muon"],
        **kwargs,
    )

    muons = events.Muon
    met = ak.zip(
        {"r": events.MET.pt, "phi": events.MET.phi},
        with_name="PolarTwoVector",
        behavior=vector.behavior,
    )
    met = ak.unflatten(
        ak.zip(
            {"x": met.x, "y": met.y},
            with_name="TwoVector",
            behavior=vector.behavior,
        ),
        1,
    )
    # Calculate the W candidates
    wCand = np.sqrt((abs(met.pt) + abs(muons.pt))**2 - ((met + muons).pt)**2)
    wCand = ak.nan_to_num(wCand)

    # Sort the W candidates by the distance to the W boson mass
    sorting_mask = ak.argsort(abs(wCand - 80.377), axis=-1)
    wCand = wCand[sorting_mask]
    muons = muons[sorting_mask]

    # Apply a mask to select only the best W candidate
    mask = (met.pt > 30) & (abs(muons[:, :1].eta) < 0.5)

    # Define the best W candidate and apply the mask
    w = wCand[:, :1][mask]
    w = ak.where(ak.num(w, axis=-1) == 0, [[EMPTY_FLOAT]], w)

    # write the transverse mass to the events
    events = set_ak_column_f32(events, "mT_W", ak.flatten(w))

    return events


@producer(
    uses=(
        {
            f"{field}.{var}"
            for field in ["Muon"]
            for var in ["pt", "mass", "eta", "phi"]
        } | {
            "MET.pt", "MET.phi", attach_coffea_behavior,
        }
    ),
    produces={
        "m4mu",
    },
)
def four_lepton_mass(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Construct the invariant mass of the four muons in each event.
    """
    # attach coffea behavior for four-vector arithmetic
    events = self[attach_coffea_behavior](
        events,
        collections=["Muon"],
        **kwargs,
    )

    muons = events.Muon

    # Calculate the Z candidates
    combinations = ak.combinations(muons, 2, axis=-1)
    m1, m2 = ak.unzip(combinations)

    # Apply a mask to select only valid Z candidates
    selection_mask = (
        (m1.charge != m2.charge) &
        ((m1 + m2).mass > 12) &
        ((m1 + m2).mass < 170)
    )

    z_cand = m1 + m2
    z_cand = z_cand[selection_mask]

    # Define the best Z candidate (most on-shell)
    sort_mask = ak.argsort(abs(z_cand.mass - 91.2), axis=-1)
    z_cand = z_cand[sort_mask]

    combinations = ak.cartesian([z_cand[:, :1], z_cand[:, 1:]])
    z1, z2 = ak.unzip(combinations)

    m4mu = (z1 + z2).mass
    sort_mask = ak.argsort(abs(m4mu - 125), axis=-1)
    m4mu = m4mu[sort_mask][:, :1]
    m4mu = ak.where(ak.num(m4mu, axis=-1) == 0, [[EMPTY_FLOAT]], m4mu)

    # write the mass to the events
    events = set_ak_column_f32(events, "m4mu", ak.flatten(m4mu))

    return events
