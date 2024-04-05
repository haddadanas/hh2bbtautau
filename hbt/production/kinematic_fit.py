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
            "MET.pt", "MET.phi", attach_coffea_behavior,
        }
    ),
    produces={
        "MET.pz",
    },
)
def met_z_component(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Reconstruct the z component of the missing energy.
    """

    # attach coffea behavior for four-vector arithmetic
    events = self[attach_coffea_behavior](
        events,
        collections=["Muon"],
        **kwargs,
    )

    # define some useful variables
    m_w = 80.3
    muons = events.Muon
    met = events.MET
    met_px = met.pt * np.cos(met.phi)
    met_py = met.pt * np.sin(met.phi)
    k = (m_w ** 2) / 2 + met_px * muons.x + met_py * muons.y
    h, h_sign_mask = get_h(met, muons, k)
    pz = ak.full_like(h, EMPTY_FLOAT)

    # h == 1
    pz = ak.where(h_sign_mask == 0, get_pz(muons, k, 0), pz)

    # h < 1
    pm_term = muons.E * np.sqrt(np.abs(1 - h ** 2))
    sqrt_sign = ak.where(muons.z > 0, -1, 1)
    pz = ak.where(h_sign_mask == 1, get_pz(muons, k, sqrt_sign * pm_term), pz)

    # h > 1
    # define the rotation coeffitients
    xi = np.sqrt(1 / (1 + (muons.y / muons.x) ** 2))
    zeta = (muons.y / muons.x) * xi
    px_rot = cubic_solve(*get_params(muons, met_px, met_py, xi, zeta, m_w))
    py_rot = muons.x / (m_w**2 * xi) * px_rot ** 2 - m_w ** 2 / (4 * muons.x) * xi
    px_prime = zeta * px_rot + xi * py_rot
    py_prime = -xi * px_rot + zeta * px_prime
    k_prime = (m_w ** 2) / 2 + px_prime * muons.x + py_prime * muons.y
    pz = ak.where(h_sign_mask == -1, get_pz(muons, k_prime, 0), pz)

    events = set_ak_column_f32(events, "MET.pz", pz)

    return events


def get_h(met, muons, k):
    """
    calculates the h value which is defined as (met_pt * muon_pt / k)^2

    :param met: ak.Array, met array
    :param muons: ak.Array, muon array
    :param k: ak.Array, k value
    :return: ak.Array, h value
    """
    h = ((met.pt * muons.pt) / k) ** 2
    sign_mask = ak.ones_like(h)

    sign_mask = ak.where(h == 1, 0, sign_mask)
    sign_mask = ak.where(h > 1, -1, sign_mask)

    return h, sign_mask


def get_pz(muons, k, pm_term):
    """
    calculates the pz value which is defined as k / muon_pt^2 * (muon_z + muon_E * sqrt(1 - h^2))

    :param muons: ak.Array, muon array
    :param k: ak.Array, k value
    :param pm_term: ak.Array, muon_E * sqrt(1 - h**2) value
    :return: ak.Array, pz value
    """

    return (k / (muons.pt ** 2)) * (muons.z + pm_term)


def get_params(in_muons, met_x, met_y, xi, zeta, m_w):
    """
    calculates the parameters for the cubic equation

    :param in_muons: ak.Array, muon array
    :param met_x: ak.Array, met x component
    :param met_y: ak.Array, met y component
    :param xi: ak.Array, xi value
    :param zeta: ak.Array, zeta value
    :param m_w: float, W boson mass
    :return: tuple, a, c, d values
    """

    x_prime = met_x * zeta - met_y * xi
    y_prime = met_x * xi + met_y * zeta

    a = (4 * in_muons.pt ** 2)
    c = m_w ** 4 - (4 * np.sign(in_muons.x) * in_muons.pt * y_prime * m_w ** 2)
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

    counts = ak.num(a)
    a = np.asarray(ak.flatten(a))
    c = np.asarray(ak.flatten(c))
    d = np.asarray(ak.flatten(d))

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

    return ak.unflatten(sol, counts)
