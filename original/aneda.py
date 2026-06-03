"""
References:
    [1] Paper: ANEDA: Adaptable Node Embeddings for Shortest Path Distance Approximation (HPEC 2023)
    [2] Original implementation: https://github.com/frankpacini/ANEDA/blob/path_search/src/aneda.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basemodel import BaseModel


class ANEDA(BaseModel):
    def __init__(self, num_nodes, embed_size, init_embeddings=None, max_distance=1.0, distance_measure="inv_dotproduct", p=1):
        """
        Initializes the ANEDA model.

        Args:
            p (int or float): The order of the norm (e.g., 2 for Euclidean distance, 1 for Manhattan distance).
        """
        super().__init__()
        self.max_distance = max_distance
        self.distance_measure = distance_measure
        self.p = p
        print(f"Initializing ANEDA with distance measure: {self.distance_measure}, p={self.p}, max_distance={self.max_distance}")

        ## Define layers
        # Embedding layer
        if init_embeddings is None:
            self.embedding = nn.Embedding(num_nodes, embed_size)
            print("Initializing node embeddings randomly. Shape:", self.embedding.weight.shape)
        else:
            init_embeddings = torch.from_numpy(init_embeddings).float()  # Convert node attributes to tensor
            assert init_embeddings.shape[0] == num_nodes, f"Expected {num_nodes} nodes, but got {init_embeddings.shape[0]}."
            assert init_embeddings.shape[1] == embed_size, f"Expected embedding size {embed_size}, but got {init_embeddings.shape[1]}."
            self.embedding = nn.Embedding.from_pretrained(init_embeddings, freeze=False)  # Make it trainable
            print("Initializing node embeddings from provided attributes. Shape:", self.embedding.weight.shape)

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

        # Compute distance based on distance measure
        if self.distance_measure == "inv_dotproduct":
            # Equivalent to: x = (1 - dotproduct(u, v)/(norm(u)*norm(v)))*self.max_distance/2
            x = (1 - F.cosine_similarity(x1, x2, dim=1, eps=1e-8)).unsqueeze(-1) * self.max_distance/2  # Shape: (B, 1)
        elif self.distance_measure == "norm":
            x = torch.norm(x1 - x2, p=self.p, dim=1, keepdim=True)
        elif self.distance_measure == "dotproduct":
            x = (1 + F.cosine_similarity(x1, x2, dim=1, eps=1e-8).unsqueeze(-1)) * self.max_distance/2  # Shape: (B, 1)
        return x
