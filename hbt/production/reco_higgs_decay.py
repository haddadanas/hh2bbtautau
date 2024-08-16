# coding: utf-8

"""
Producers that determine the generator-level particles related to a top quark decay.
"""
from __future__ import annotations
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import (
    set_ak_column, optional_column as optional,
)


ak = maybe_import("awkward")


@producer(
    uses={"Tau.*", "Jet.*", "Electron.*", "Muon.*", attach_coffea_behavior},
    produces={
        "something_something",
    },
)
def reco_higgs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections=["Tau", "Jet"], **kwargs)

    # Select the two highest-pt taus
    tau = events.Tau
    sort_mask = ak.argsort(tau.pt, axis=-1, ascending=False)
    tau = tau[sort_mask]
    tau = tau[:, :2]

    # Select the two highest-pt jets
    jet = events.Jet
    sort_mask = ak.argsort(jet.btagPNetB, axis=-1, ascending=False)
    jet = jet[sort_mask]
    jet = jet[:, :2]

    di_tau = tau.sum(axis=-1)
    di_jet = jet.sum(axis=-1)

    # Select the two highest-pt electrons
    from IPython import embed; embed(header="")
    pass
