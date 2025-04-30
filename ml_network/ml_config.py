from functools import partial

import torch

from ml_network.models.losses import NLL_Focal_Loss

CONFIG = {
    "target_nodes": ["signal_node"],
    "optimizer": partial(torch.optim.Adam, lr=0.001, eps=1e-06),  # TODO AdamW + L2 Weight
    "loss_func": partial(NLL_Focal_Loss, reduction="mean"),
    "val_loss_func": partial(NLL_Focal_Loss, reduction="sum"),
}


NUM_FIELDS = sorted([
    *[f"bjet0.{field}" for field in ["btagPNetB", "eta", "hhbtag", "mass", "phi", "pt"]],
    *[f"bjet1.{field}" for field in ["btagPNetB", "eta", "hhbtag", "mass", "phi", "pt"]],
    *[f"diBJet.{field}" for field in ["eta", "mass", "pt"]],
    *[f"diTau.{field}" for field in ["eta", "mass", "pt"]],
    *[f"hh.{field}" for field in ["eta", "mass", "pt"]],
    *[f"l1.{field}" for field in ["dxy", "dz", "eta", "is_iso", "pt"]],  # "iso_score",
    *[f"l2.{field}" for field in ["dxy", "dz", "eta", "is_iso", "pt"]],
    "n_bjets",
    "n_jets",
    "n_taus"
])

EMBED_FIELDS = sorted([
    "channel_id",
    *[f"l1.tauVS{field}" for field in ["e", "jet", "mu"]],
    *[f"l2.tauVS{field}" for field in ["e", "jet", "mu"]],
])
