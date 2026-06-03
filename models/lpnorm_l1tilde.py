import time
import math
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from models.basemodel import BaseModel


class LpNormL1Tilde(BaseModel):
    def __init__(self, p=2, node_attributes=None, l1tilde_r=1, l1tilde_s=1):
        """
        Initializes the LpNorm model.

        Args:
            p (int or float): The order of the norm (e.g., 2 for Euclidean distance, 1 for Manhattan distance).
        """
        super().__init__()
        self.p = p
        self.l1tilde_r = l1tilde_r
        self.l1tilde_s = l1tilde_s

        ## Define layers
        # Embedding layer
        node_features = torch.from_numpy(node_attributes).float()
        # Convert lat/lon from degrees to approximate meters
        mean_lat_rad = node_features[:, 1].mean().item() * math.pi / 180.0
        lon_scale = 111320.0 * math.cos(mean_lat_rad)
        lat_scale = 111320.0
        node_features[:, 0] *= lon_scale
        node_features[:, 1] *= lat_scale
        self.embedding = nn.Embedding.from_pretrained(node_features, freeze=True)

    def forward(self, x1, x2):
        """
        Computes the Lp norm between node embeddings.

        Args:
            x1 (torch.Tensor): Node indices for the first set of nodes.
            x2 (torch.Tensor): Node indices for the second set of nodes.

        Returns:
            torch.Tensor: Lp norm between the embeddings.
        """
        # Embedding layer
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)

        r, s = self.l1tilde_r, self.l1tilde_s
        diff = x2 - x1
        sym = torch.abs(diff[:, :r]).sum(dim=1, keepdim=True)
        asym = diff[:, r:r+s].sum(dim=1, keepdim=True)
        x = sym + asym

        return x

    def fit(self, **kwargs):
        """
        Dummy fit function to match the API of other models.
        Since no training is required, this function does nothing.
        """

        return {
            "loss_epoch_history": [],
            "loss_iter_history": [],
            "val_mre_epoch_history": [],
            "time_history": [],
        }
