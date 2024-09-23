# coding: utf-8

"""
Producers that determine the generator-level particles related to a top quark decay.
"""
from __future__ import annotations
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import (
    set_ak_column, optional_column as optional,
)


ak = maybe_import("awkward")


@producer(
    uses={"GenPart.*"},
    produces={
        optional(f"old_gen_{mother}_to_{child}.{var}")
        for mother in ("z",)
        for child in ("mu",)
        for var in ("pt", "eta", "phi", "mass", "pdgId")
    } |
    {
        optional(f"gen_{mother}_to_{child}.{var}")
        for mother in ("z",)
        for child in ("mu",)
        for var in ("pt", "eta", "phi", "mass", "pdgId")
    } |
    {
        optional(f"gen_{child}_from_{mother}.{var}")
        for mother in ("z",)
        for child in ("mu",)
        for var in ("pt", "eta", "phi", "mass", "pdgId")
    },
)
def gen_z_decay_products(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new ragged column "gen_higgs_decay" with one element per hard higgs boson. Each element is
    a GenParticleArray with five or more objects in a distinct order: higgs boson, bottom quark, tau lepton,
    W boson, down-type quark or charged lepton, up-type quark or neutrino, and any additional decay
    produces of the W boson (if any, then most likly photon radiations). Per event, the structure
    will be similar to:

    .. code-block:: python

        [
            # event 1
            [
                # top 1
                [t1, b1, W1, q1/l, q2/n(, additional_w_decay_products)],
                # top 2
                [...],
            ],
            # event 2
            ...
        ]
    """
    abs_id = abs(events.GenPart.pdgId)
    def get_decay_idx(
        events: ak.Array,
        mother_id: int,
        children_id: int,
        children_output_name: str | None = None,
        mother_output_name: str | None = None,
        mother_gen_flags: list | None = None,
        children_gen_flags: list | None = None,
    ):
        if not mother_gen_flags:
            mother_gen_flags = ["isLastCopy", "fromHardProcess"]
        if not children_gen_flags:
            children_gen_flags = ["isFirstCopy", "fromHardProcess"]
        # obtain indices and GenParticles for mothers that match input 'mother_id'
        mask_mother_id = abs_id == mother_id
        mask_gen_stati = ak.mask(events.GenPart, mask_mother_id).hasFlags(*mother_gen_flags)
        mask_gen_stati = ak.fill_none(mask_gen_stati, False, axis=-1)
        mother_idx = ak.local_index(events.GenPart.pt, axis=-1)[mask_gen_stati]
        mothers = events.GenPart[mother_idx]

        # sort mothers for pt
        sorted_mothers_idx = ak.argsort(mothers.pt, axis=-1, ascending=False)
        mother_idx = mother_idx[sorted_mothers_idx]
        mothers = mothers[sorted_mothers_idx]

        # get corresponding decay products, aka children
        children = mothers.distinctChildrenDeep
        # make sure you only consider real children
        children_gen_stati_mask = children.hasFlags(*children_gen_flags)
        children = children[children_gen_stati_mask]
        abs_children_id = abs(children.pdgId)

        children_mask = abs_children_id == children_id
        any_relevant_children_mask = ak.any(children_mask, axis=-1)
        relevant_mother_idx = mother_idx[any_relevant_children_mask]
        relevant_mothers = events.GenPart[relevant_mother_idx]
        # update children masks
        children = children[any_relevant_children_mask]
        children_mask = children_mask[any_relevant_children_mask]
        relevant_children = children[children_mask]
        relevant_children_idx = ak.local_index(relevant_children.pt, axis=-1)

        sorted_children_idx = ak.argsort(relevant_children.pt, axis=-1, ascending=False)
        relevant_children_idx = relevant_children_idx[sorted_children_idx]

        relevant_children = relevant_children[sorted_children_idx]
        relevant_children = ak.flatten(relevant_children, axis=2)

        if children_output_name:
            for var in ("pt", "eta", "phi", "mass", "pdgId"):
                events = set_ak_column(events, f"{children_output_name}.{var}", getattr(relevant_children, var))
        if mother_output_name:
            for var in ("pt", "eta", "phi", "mass", "pdgId"):
                events = set_ak_column(events, f"{mother_output_name}.{var}", getattr(relevant_mothers, var))
        return events, relevant_mother_idx, relevant_children_idx

    events, z_mu_idx, z_mu_particles = get_decay_idx(
        events,
        mother_id=23,
        children_id=13,
        children_output_name="old_gen_mu_from_z",
        mother_output_name="old_gen_z_to_mu",
    )

    events, z_mu_idx, z_mu_particles = get_decay_idx(
        events,
        mother_id=23,
        children_id=13,
        children_output_name="gen_mu_from_z",
        mother_output_name="gen_z_to_mu",
        mother_gen_flags=["isFirstCopy", "fromHardProcess"],
    )

    return events
