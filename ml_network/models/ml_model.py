from functools import partial

import torch
from torch import nn

CONFIG = {
    "target": ["signal_node"],
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
        self.input_length = len(input_features)
        self.embedding_out = 2
        self.input_dim = self.input_length + self.embedding_out

        # embedding layers
        self.embed = nn.Sequential(
            nn.Embedding(3, self.embedding_out),
            nn.Flatten()
        )

        # define the layers
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.PReLU(),
            nn.Linear(64, 256),
            nn.PReLU(),
            nn.Linear(256, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.Linear(128, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, X_embed, X_num):
        x0 = self.embed(X_embed[0])
        x = torch.cat([x0, X_num], dim=1)
        x = x.float()

        logits = self.linear_relu_stack(x)

        return logits
