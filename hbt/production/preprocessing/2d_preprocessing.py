from functools import partial

from columnflow.columnar_util import Route, set_ak_column
from columnflow.production import Producer, producer
from columnflow.util import maybe_import


ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        f"Electron.{var}" for var in ["pt", "eta", "phi"]
    } | {
        f"Muon.{var}" for var in ["pt", "eta", "phi"]
    } | {
        f"Tau.{var}" for var in ["pt", "eta", "phi"]
    } | {
        f"Jet.{var}" for var in ["pt", "eta", "phi"]
    },
    produces={"event_map"},
)
def event_maps(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # create bins for the eta-phi space
    eta_bins = np.linspace(-5, 5, 50)
    phi_bins = np.linspace(-3.1416016, 3.1416016, 50)

    for part in ["Electron", "Muon", "Tau", "Jet"]:
        # Create the event map
        event_map = np.zeros((len(events), len(eta_bins) - 1, len(phi_bins) - 1), dtype=np.float32)
        arr = getattr(events, part)

        pt0_route = Route("pt[:, 0]")
        pt1_route = Route("pt[:, 1]")
        eta0_route = Route("eta[:, 0]")
        eta1_route = Route("eta[:, 1]")
        phi0_route = Route("phi[:, 0]")
        phi1_route = Route("phi[:, 1]")

        pt0 = pt0_route.apply(arr, 0)
        pt1 = pt1_route.apply(arr, 0)
        eta0 = np.searchsorted(eta_bins, eta0_route.apply(arr, 0)) - 1
        eta1 = np.searchsorted(eta_bins, eta1_route.apply(arr, 0)) - 1
        phi0 = np.searchsorted(phi_bins, phi0_route.apply(arr, 0)) - 1
        phi1 = np.searchsorted(phi_bins, phi1_route.apply(arr, 0)) - 1

        # fill the event map
        for i, (pt0_i, pt1_i, eta0_i, eta1_i, phi0_i, phi1_i) in enumerate(zip(pt0, pt1, eta0, eta1, phi0, phi1)):
            event_map[i, eta0_i, phi0_i] += pt0_i
            event_map[i, eta1_i, phi1_i] += pt1_i

        # set the event map
        events = set_ak_column_f32(events, f"{part}_event_map", event_map)

    return events
