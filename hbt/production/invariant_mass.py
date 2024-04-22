# coding: utf-8

import functools
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, set_ak_column
from columnflow.production.util import attach_coffea_behavior

np = maybe_import("numpy")
ak = maybe_import("awkward")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


def delta_R(vector1, vector2):
    """
    Calculate the delta R between two vectors.
    """
    deta = vector1.eta - vector2.eta
    dphi = np.abs(vector1.phi - vector2.phi)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)

    return np.sqrt(deta ** 2 + dphi ** 2)


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
    # sorting_mask = ak.argsort(abs(wCand - 80.377), axis=-1)
    sorting_mask = ak.argsort(wCand, axis=-1, ascending=True)
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

    combinations = ak.argcombinations(muons, 2, axis=1)
    m1_ind, m2_ind = ak.unzip(combinations)
    m1 = muons[m1_ind]
    m2 = muons[m2_ind]

    # Apply a mask to select only valid Z candidates
    selection_mask = (
        (m1.charge != m2.charge) &
        ((m1 + m2).mass > 12)
    )

    m1_ind = m1_ind[selection_mask]
    m2_ind = m2_ind[selection_mask]
    m1 = m1[selection_mask]
    m2 = m2[selection_mask]

    # Define the best Z candidate (most on-shell)
    sort_mask = ak.argsort(abs((m1 + m2).mass - 91.2), axis=-1)
    m1_ind = m1_ind[sort_mask]
    m2_ind = m2_ind[sort_mask]

    # Define best Z candidates
    z1 = m1[sort_mask][:, :1] + m2[sort_mask][:, :1]

    m1_mask = m1_ind != ak.flatten(ak.where(ak.num(m1_ind) > 0, m1_ind[:, :1], [[0]]))
    m2_mask = m2_ind != ak.flatten(ak.where(ak.num(m2_ind) > 0, m2_ind[:, :1], [[0]]))

    # Define remaining Muons
    m1 = m1[m1_mask]
    m2 = m2[m2_mask]

    # Define z2
    combinations = ak.cartesian([m1, m2])
    m1, m2 = ak.unzip(combinations)
    z2 = m1 + m2

    combinations = ak.cartesian([z1, z2])
    z1, z2 = ak.unzip(combinations)

    m4mu = (z1 + z2).mass
    sort_mask = ak.argsort(m4mu, axis=-1, ascending=False)
    m4mu = m4mu[sort_mask][:, :1]
    m4mu = ak.where(ak.num(m4mu, axis=-1) == 0, [[EMPTY_FLOAT]], m4mu)
    m4mu = ak.where(np.isfinite(m4mu), m4mu, [[EMPTY_FLOAT]])

    # write the mass to the events
    events = set_ak_column_f32(events, "m4mu", ak.flatten(m4mu))

    return events


@producer(
    uses=(
        {
            f"{field}.{var}"
            for field in ["Muon"]
            for var in ["pt", "mass", "eta", "phi", "charge"]
        } | {
            "MET.pt", "MET.phi", attach_coffea_behavior,
        }
    ),
    produces={
        "mT_Z",
    },
)
def transverse_mass_Z(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Construct the transverse mass of the Z boson in each event.
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

    combination = ak.combinations(muons, 2, axis=1)
    m1, m2 = ak.unzip(combination)

    # Some selection
    mass_sel = ak.argsort((m1 + m2).mass, axis=-1, ascending=False)

    m1 = m1[mass_sel][:, :1]
    m2 = m2[mass_sel][:, :1]

    charge_sel = m1.charge != m2.charge

    m1 = m1[charge_sel]
    m2 = m2[charge_sel]

    # create np array containing 1 and 0 randomly
    random_mask = ak.unflatten(np.random.randint(2, size=len(ak.flatten(m1.pt))), ak.num(m1.pt)) == 1

    killed_muons = ak.where(random_mask, m1, m2)
    survived_muons = ak.where(random_mask, m2, m1)

    new_met = met + killed_muons

    mT_Z = np.sqrt((abs(new_met.pt) + abs(survived_muons.pt))**2 - ((new_met + survived_muons).pt)**2)
    mT_Z = ak.where(ak.num(mT_Z, axis=-1) == 0, [[EMPTY_FLOAT]], mT_Z)
    mT_Z = ak.where(np.isfinite(mT_Z), mT_Z, [[EMPTY_FLOAT]])

    # write the transverse mass to the events
    events = set_ak_column_f32(events, "mT_Z", ak.flatten(mT_Z))

    return events


@producer(
    uses=(
        {
            f"{field}.{var}"
            for field in ["Muon"]
            for var in ["pt", "mass", "eta", "phi", "charge"]
        } | {
            f"{field}.{var}"
            for field in ["Jet"]
            for var in ["pt", "mass", "eta", "phi", "btagDeepFlavB"]
        } | {
            "MET.pt", "MET.phi", "MET.pz", attach_coffea_behavior,
        }
    ),
    produces={
        f"Top.{var}"
        for var in ["pt", "mass", "eta", "phi"]
    },
)
def top_invariant_mass(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Construct the invariant mass of the muons in each event.
    """
    from coffea.nanoevents.methods import vector

    # attach coffea behavior for four-vector arithmetic
    events = self[attach_coffea_behavior](
        events,
        collections=["Muon", "Jet"],
        **kwargs,
    )
    muons = events.Muon
    jet = events.Jet

    # Calculate the MET four-vector
    met = ak.zip(
        {
            "pt": events.MET.pt,
            "eta": np.arcsinh(events.MET.pz / np.sqrt(events.MET.pt ** 2 + events.MET.pz ** 2)),
            "phi": events.MET.phi,
            "mass": np.full(len(events.MET.pt), 0.),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )

    # select the muon with the smallest delta R to the MET
    muon_mask = np.argsort(delta_R(muons, met), axis=-1)[:, :1]
    muon = muons[muon_mask]
    met = met[muon_mask]

    # reconstruct the 4 momentum of the W boson
    w = muon + met

    # select the jet with the smallest delta R to the W boson
    btag_mask = ak.argsort(jet.btagDeepFlavB, ascending=False, axis=-1)[:, :1]
    jet_mask = btag_mask[w.pt != -42]
    jet = jet[jet_mask]

    # reconstruct the 4 momentum of the top quark\
    top = w + jet

    # write the invariant mass to the events
    for var in ["pt", "mass", "eta", "phi"]:
        events = set_ak_column(
            events,
            f"Top.{var}",
            getattr(top, var),
        )

    return events


@producer(
    uses=(
        {
            f"{field}.{var}"
            for field in ["Muon"]
            for var in ["pt", "mass", "eta", "phi", "charge"]
        } | {
            "MET.pt", "MET.eta", "" "MET.phi", "MET.fit_methode", attach_coffea_behavior,
        }
    ),
    produces={
        "mW", "mW_ana", "mW_kin", "mW_corr", "mW_corr_ana", "mW_corr_kin",
    },
)
def W_mass(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Construct the transverse mass of the Z boson in each event.
    """
    from coffea.nanoevents.methods import vector
    # attach coffea behavior for four-vector arithmetic
    events = self[attach_coffea_behavior](
        events,
        collections=["Muon"],
        **kwargs,
    )

    muons = events.Muon
    met = events.MET
    met = ak.zip(
        {
            "pt": events.MET.pt,
            "eta": events.MET.eta,
            "phi": events.MET.phi,
            "mass": np.full(len(events.MET.pt), 0.),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )
    met_corr = ak.zip(
        {
            "pt": np.sqrt(events.MET.fit_px ** 2 + events.MET.fit_py ** 2),
            "eta": events.MET.eta,
            "phi": np.arctan2(events.MET.fit_py, events.MET.fit_px),
            "mass": np.full(len(events.MET.pt), 0.),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )

    # W 4 momentum
    w = met + muons

    w_m = w.mass
    mW_ana = ak.where(events.MET.fit_methode == 1, w_m, EMPTY_FLOAT)
    mW_kin = ak.where(events.MET.fit_methode == -1, w_m, EMPTY_FLOAT)

    events = set_ak_column_f32(events, "mW", w_m)
    events = set_ak_column_f32(events, "mW_ana", mW_ana)
    events = set_ak_column_f32(events, "mW_kin", mW_kin)

    # W 4 momentum corrected
    w_corr = met_corr + muons

    w_corr_m = w_corr.mass
    mW_corr_ana = ak.where(events.MET.fit_methode == 1, w_corr_m, EMPTY_FLOAT)
    mW_corr_kin = ak.where(events.MET.fit_methode == -1, w_corr_m, EMPTY_FLOAT)

    events = set_ak_column_f32(events, "mW_corr", w_corr_m)
    events = set_ak_column_f32(events, "mW_corr_ana", mW_corr_ana)
    events = set_ak_column_f32(events, "mW_corr_kin", mW_corr_kin)

    return events
