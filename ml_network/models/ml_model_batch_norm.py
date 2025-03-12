from functools import partial

import torch
from torch import nn

from ml_network.models.losses import NLL_Focal_Loss


CONFIG = {
    "target_nodes": ["signal_node"],
    "optimizer": partial(torch.optim.Adam, lr=0.001, eps=1e-06),
    "loss": partial(NLL_Focal_Loss, reduction="mean"),
}
NUM_FIELDS = sorted([
    *[f"bjet0.{field}" for field in ["btagPNetB", "eta", "hhbtag", "mass", "phi", "pt"]],
    *[f"bjet1.{field}" for field in ["btagPNetB", "eta", "hhbtag", "mass", "phi", "pt"]],
    *[f"diBJet.{field}" for field in ["eta", "mass", "pt"]],
    *[f"diTau.{field}" for field in ["eta", "mass", "pt"]],
    *[f"hh.{field}" for field in ["eta", "mass", "pt"]],
    *[f"l1.{field}" for field in ["dxy", "dz", "eta", "is_iso", "iso_score", "pt"]],
    *[f"l2.{field}" for field in ["dxy", "dz", "eta", "is_iso", "pt"]],
    "n_bjets",
    "n_jets",
    "n_taus"
])
EMBED_FIELDS = sorted([
    "channel_id",
    *[f"l1.tauVS{field}" for field in ["e", "jet", "mu"]],
    *[f"l2.tauVS{field}" for field in ["e", "jet", "mu"]]
])


class CustomModel(nn.Module):
    def __init__(self, name, save_path="./models", num_fields=NUM_FIELDS, embed_fields=EMBED_FIELDS):
        super(CustomModel, self).__init__()
        self.model_name = name
        self.save_path = save_path
        self.input_length = len(num_fields)
        embedding_out = {
            "channel_id": (3, 2),
            "tauVSjet": (10, 6),
            "tauVSe": (10, 6),
            "tauVSmu": (5, 3),
        }
        self.input_dim = (
            self.input_length + 2 * sum(v[1] for v in embedding_out.values()) - embedding_out["channel_id"][1]
        )

        # embedding layers
        self.embed = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Embedding(*embedding_out[feat.split(".")[-1]]),
                    nn.Flatten(),
                )
                for feat in embed_fields
            ]
        )

        # define the layers with batch normalization
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Linear(32, 2),
            nn.Softmax(1),
        )

    def forward(self, X_embed, X_num):
        x = [X_num]
        for layer, data in zip(self.embed, X_embed):
            x.append(layer(data.clamp(0, 9)))
        x = torch.cat(x, dim=1)
        x = x.float()

        logits = self.linear_relu_stack(x)

        return logits
