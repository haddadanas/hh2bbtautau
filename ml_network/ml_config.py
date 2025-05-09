from functools import partial
from typing import Any

import torch

from ml_network.models.losses import NLL_Focal_Loss

cfg = {
    "optimizer": {
        "func": torch.optim.AdamW,
        "params": {
            "lr": 0.001,
            "weight_decay": 0.01,
            "eps": 1e-06,
        },
    },
    "scheduler": {
        "func": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "params": {
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "verbose": True,
        },
    },
}

losses = {
    "loss_func": {
        "func": NLL_Focal_Loss,
        "params": {
            "alpha": 0.25,
            "gamma": 2,
            "reduction": "mean",
        },
    },
    "val_loss_func": {
        "func": NLL_Focal_Loss,
        "params": {
            "alpha": 0.25,
            "gamma": 2,
            "reduction": "sum",
        },
    },
}

fitting_comps = {
    "early_stopping": {
        "mode": "min",
        "patience": 7,
        "delta": 0,
        "metric": "val_loss",
    },

}
CONFIG: dict[str, Any] = {}

# add optimizer and scheduler to config
CONFIG.update({
    key: partial(value["func"], **value["params"])
    for key, value in cfg.items()
})
# add loss functions to config
CONFIG.update({
    key: partial(torch.jit.script(value["func"]), **value["params"])
    for key, value in losses.items()
})

# add fitting components to config
CONFIG["fitting_components"] = fitting_comps

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
