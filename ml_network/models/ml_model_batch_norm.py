from __future__ import annotations

import torch
from torch import nn

from ml_network.ml_config import NUM_FIELDS, EMBED_FIELDS


class CustomModel(nn.Module):
    def __init__(self, model_name, save_path="./models", num_fields=NUM_FIELDS, embed_fields=EMBED_FIELDS):
        super(CustomModel, self).__init__()
        self.name = model_name
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
        self.embed: nn.ModuleDict = nn.ModuleDict(
            {
                feat.replace(".", "_"): nn.Sequential(
                    nn.Embedding(*embedding_out[feat.split(".")[-1]]),
                    nn.Flatten(),
                )
                for feat in embed_fields
            }
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

    @torch.jit.export
    def forward(self: CustomModel, X_embed: dict[str, torch.Tensor], X_num: dict[str, torch.Tensor]) -> torch.Tensor:
        x = [layer(X_embed[key]) for key, layer in self.embed.items()]
        x.append(X_num["num"])
        # for key, layer in self.embed.items():
        #     x.append(layer(X_embed[key]))
        # x.extend(map(lambda item: self.embed[item[0]](item[1]), X_embed.items()))
        x = torch.cat(x, dim=1)
        x = x.float()

        logits = self.linear_relu_stack(x)

        return logits
