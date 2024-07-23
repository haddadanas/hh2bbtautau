# coding: utf-8

"""
Column production methods related to kinematic fits to reconstruct the z component of the missing energy.
"""

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
            for var in ["pt", "mass", "eta", "phi"]
        } | {
            "MET.pt", "MET.phi", "MET.covXX", "MET.covYY", "MET.covXY",
            attach_coffea_behavior,
        }
    ),
    produces={
        "MET.pz", "MET.eta", "MET.fit_methode", "MET.fit_px", "MET.fit_py",
    },
)
def met_z_component(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Reconstruct the z component of the missing energy.
    """
    from coffea.nanoevents.methods import vector

    # attach coffea behavior for four-vector arithmetic
    events = self[attach_coffea_behavior](
        events,
        collections=["Muon"],
        **kwargs,
    )

    # define some useful variables
    m_w = 80.3

    muons, met = ak.broadcast_arrays(events.Muon, events.MET)
    muons = ak.Array(ak.flatten(muons), with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
    met = ak.flatten(met)

    met_px = ak.to_numpy(met.pt * np.cos(met.phi))
    met_py = ak.to_numpy(met.pt * np.sin(met.phi))
    k = np.square(m_w) / 2 + met_px * muons.px + met_py * muons.py
    h, h_sign_mask = get_h(met, muons, k)
    pz = ak.full_like(h, EMPTY_FLOAT)
    fit_px = ak.full_like(h, EMPTY_FLOAT)
    fit_py = ak.full_like(h, EMPTY_FLOAT)

    # h == 1
    pz = ak.where(h_sign_mask == 0, get_pz(muons, k, 0), pz)
    fit_px = ak.where(h_sign_mask == 0, met_px, fit_px)
    fit_py = ak.where(h_sign_mask == 0, met_py, fit_py)

    # h < 1
    sqrt_sign = 1  # ak.where(muons.z > 0, -1.0, 1.0)
    pz = ak.where(h_sign_mask == 1, get_pz(muons, k, sqrt_sign * np.sqrt(abs(h))), pz)
    fit_px = ak.where(h_sign_mask == 1, met_px, fit_px)
    fit_py = ak.where(h_sign_mask == 1, met_py, fit_py)

    # h > 1
    # define the rotation coeffitients
    xi = np.sqrt(1 / (1 + ak.to_numpy(muons.y / muons.x) ** 2))
    zeta = ak.to_numpy(muons.y / muons.x) * xi

    trafo_mat = get_trafo_matrix(zeta, xi)
    met_vec = concat_columns(met_px, met_py)[:, :, np.newaxis]

    px_rot = cubic_solve(*get_params(muons, trafo_mat @ met_vec, m_w))

    met_rot = concat_columns(
        px_rot,
        ak.to_numpy(muons.x / (m_w**2 * xi) * px_rot ** 2 - m_w ** 2 / (4 * muons.x) * xi),
    )[:, :, np.newaxis]

    px_prime, py_prime = (np.transpose(trafo_mat, axes=(0, 2, 1)) @ met_rot).T[0]
    k_prime = (m_w ** 2) / 2 + px_prime * muons.x + py_prime * muons.y
    pz = ak.where(h_sign_mask == -1, get_pz(muons, k_prime, 0), pz)
    fit_px = ak.where(h_sign_mask == -1, px_prime, fit_px)
    fit_py = ak.where(h_sign_mask == -1, py_prime, fit_py)

    # Get original shape
    pz = ak.unflatten(pz, ak.num(events.Muon))
    h_sign_mask = ak.unflatten(h_sign_mask, ak.num(events.Muon))
    fit_px = ak.unflatten(fit_px, ak.num(events.Muon))
    fit_py = ak.unflatten(fit_py, ak.num(events.Muon))

    for field in ["pz", "fit_px", "fit_py"]:
        events = set_ak_column_f32(events, f"MET.{field}", eval(field))
    events = set_ak_column_f32(
        events,
        "MET.eta",
        np.arcsinh(pz / events.MET.pt),
    )
    events = set_ak_column(events, "MET.fit_methode", h_sign_mask)

    return events


def get_h(met, muons, k):
    """
    calculates the h value which is defined as (met_pt * muon_pt / k)^2

    :param met: ak.Array, met array
    :param muons: ak.Array, muon array
    :param k: ak.Array, k value
    :return: ak.Array, h value
    """
    h = np.square(k) - np.square(met.pt) * (np.square(muons.E) - np.square(muons.pz))
    sign_mask = ak.zeros_like(h)

    sign_mask = ak.where(h > 0, 1, sign_mask)
    sign_mask = ak.where(h < 0, -1, sign_mask)

    return h, sign_mask


def get_pz(muons, k, pm_term):
    """
    calculates the pz value which is defined as k / muon_pt^2 * (muon_z + muon_E * sqrt(1 - h^2))

    :param muons: ak.Array, muon array
    :param k: ak.Array, k value
    :param pm_term: ak.Array, muon_E * sqrt(1 - h**2) value
    :return: ak.Array, pz value
    """

    return ((muons. pz * k) + muons.E * pm_term) / (np.square(muons.E) - np.square(muons.pz))


def get_params(in_muons, transformed_met, m_w):
    """
    calculates the parameters for the cubic equation

    :param in_muons: ak.Array, muon array
    :param transformed_met: ak.Array, transformed met array
    :param m_w: float, W boson mass
    :return: tuple, a, c, d values
    """

    x_prime, y_prime = transformed_met.T[0]

    a = ak.to_numpy((4 * in_muons.pt ** 2))
    c = m_w ** 4 - ak.to_numpy((4 * np.sign(in_muons.x) * in_muons.pt * y_prime * m_w ** 2))
    d = - 2 * m_w ** 4 * x_prime
    return a, c, d


def cubic_solve(a, c, d):
    """
    helper function to find the roots of a cubic equation with the form: a * x ** 3 + c * x + d = 0
    """

    # Helper function to return float value of f.
    def findF(a, c):
        return (3.0 * c / a) / 3.0

    # Helper function to return float value of g.
    def findG(a, d):
        return (27.0 * d / a) / 27.0

    # Helper function to return float value of h.
    def findH(g, f):
        return ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)

    # counts = ak.num(a)
    # a = np.asarray(ak.flatten(a))
    # c = np.asarray(ak.flatten(c))
    # d = np.asarray(ak.flatten(d))

    sol = np.zeros_like(a)
    a_mask = a == 0

    sol = np.where(a_mask, (-d * 1.0) / c, sol)

    f = findF(a, c)                          # Helper Temporary Variable
    g = findG(a, d)                       # Helper Temporary Variable
    h = findH(g, f)                             # Helper Temporary Variable

    h0_mask = (f == 0) & (g == 0) & (h == 0) & a_mask
    if np.any(h0_mask):
        x1 = np.cbrt(d[h0_mask] / (1.0 * a[h0_mask])) * -1
        x2 = np.cbrt(-d[h0_mask] / (1.0 * a[h0_mask]))
        sol[h0_mask] = np.where((d / a)[h0_mask] >= 0, x1, x2)

    h_le0_mask = (h <= 0) & (~a_mask)
    if np.any(h_le0_mask):
        i = np.sqrt(((g[h_le0_mask] ** 2.0) / 4.0) - h[h_le0_mask])   # Helper Temporary Variable
        j = i ** (1 / 3.0)                      # Helper Temporary Variable
        k = np.arccos(-(g[h_le0_mask] / (2 * i)))           # Helper Temporary Variable
        # L = j[h_le0_mask] * -1                              # Helper Temporary Variable
        # M = np.cos(k[h_le0_mask] / 3.0)                   # Helper Temporary Variable
        # N = np.sqrt(3) * np.sin(k[h_le0_mask] / 3.0)    # Helper Temporary Variable
        # P = (b[h_le0_mask] / (3.0 * a[h_le0_mask])) * -1                # Helper Temporary Variable

        sol[h_le0_mask] = 2 * j * np.cos(k / 3.0)
        # x2 = L * (M + N) + P
        # x3 = L * (M - N) + P

    h_g0_mask = (~a_mask) & (h > 0)
    if np.any(h_g0_mask):
        R = -(g[h_g0_mask] / 2.0) + np.sqrt(h[h_g0_mask])           # Helper Temporary Variable
        S = np.where(R >= 0, np.cbrt(R), np.cbrt(-R) * -1)
        T = -(g[h_g0_mask] / 2.0) - np.sqrt(h[h_g0_mask])
        U = np.where(T >= 0, (np.cbrt(T)), np.cbrt(-T) * -1)

        sol[h_g0_mask] = (S + U)

    return sol


def get_trafo_matrix(zeta, xi):
    """
    calculates the rotation matrix for the x and y components

    :param zeta: np.ndarray, zeta value
    :param xi: np.ndarray, xi value
    :return: np.ndarray, rotation matrix
    """

    return np.concatenate(
        [
            concat_columns(zeta, -xi)[:, np.newaxis],
            concat_columns(xi, zeta)[:, np.newaxis],
        ],
        axis=1,
    )


def concat_columns(col1, col2):
    """
    concatenates two columns to a vector

    :param col1: ak.Array, first column
    :param col2: ak.Array, second column
    :return: ak.Array, concatenated column
    """
    return np.concatenate([col1[:, np.newaxis], col2[:, np.newaxis]], axis=-1)