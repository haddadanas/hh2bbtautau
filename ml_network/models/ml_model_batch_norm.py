from functools import partial

import torch
from torch import nn

CONFIG = {
    "target_nodes": ["signal_node"],
    "embedding_fields": ["channel_id"] + [f"l{n}.tauVS{var}" for n in [1, 2] for var in ["jet", "e", "mu"]],
    "optimizer": partial(torch.optim.Adam, lr=0.001, eps=1e-06),
    "loss": nn.BCELoss(reduction="none"),
    "epochs": 30,
    "batch_size": 265,
}


class CustomModel(nn.Module):
    def __init__(self, name, input_features, save_path="./models"):
        super(CustomModel, self).__init__()
        self.model_name = name
        self.save_path = save_path
        self.input_length = (
            len(input_features["num_fields"]) if isinstance(input_features, dict) else len(input_features)
        )
        self.embedding_out = {
            "channel_id": 2,
            "tauVSjet": 6,
            "tauVSe": 6,
            "tauVSmu": 3,
        }
        self.input_dim = self.input_length + 2 * sum(self.embedding_out.values()) - self.embedding_out["channel_id"]

        # embedding layers
        self.embed = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Embedding(10, self.embedding_out["tauVSe"]),
                    nn.Flatten(),
                ),
                nn.Sequential(
                    nn.Embedding(10, self.embedding_out["tauVSjet"]),
                    nn.Flatten(),
                ),
                nn.Sequential(
                    nn.Embedding(5, self.embedding_out["tauVSmu"]),
                    nn.Flatten(),
                ),
                nn.Sequential(
                    nn.Embedding(10, self.embedding_out["tauVSe"]),
                    nn.Flatten(),
                ),
                nn.Sequential(
                    nn.Embedding(10, self.embedding_out["tauVSjet"]),
                    nn.Flatten(),
                ),
                nn.Sequential(
                    nn.Embedding(5, self.embedding_out["tauVSmu"]),
                    nn.Flatten(),
                ),
                nn.Sequential(
                    nn.Embedding(3, self.embedding_out["channel_id"]),
                    nn.Flatten(),
                ),
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
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, X_embed, X_num):
        x = [X_num]
        for layer, data in zip(self.embed, X_embed):
            x.append(layer(data))
        x = torch.cat(x, dim=1)
        x = x.float()

        logits = self.linear_relu_stack(x)

        return logits
