"""
References:
    [1] Paper: Shortest path distance approximation using deep learning techniques (ASONAM 2018)
    [2] Original implementation: https://github.com/fatemehsrz/Shortest_Distance/blob/master/feedforward.py


TODO:
    [ ] Evaluate model performance without dropout layers
    [ ] Test with lower dropout rates (0.1, 0.2)
    [ ] Experiment with SGD optimizer instead of default Adam
"""

import numpy as np

import torch
import torch.nn as nn

from models.basemodel import BaseModel


# Define the DistanceNN model
class DistanceNN(BaseModel):
    def __init__(self, num_nodes, embed_size=64, init_embeddings=None, aggregation_method="mean", normalize=True):
        super().__init__()
        self.aggregation_method = aggregation_method  # enum: ["hadamard", "subtraction", "mean", "concat"]
        print(f"Initializing DistanceNN with aggregation method: {self.aggregation_method} and embedding size: {embed_size}")

        ## Define layers
        # Embedding layer
        if init_embeddings is None:
            init_embeddings = np.random.rand(num_nodes, embed_size).astype(np.float32)
            print("Initializing node embeddings randomly. Shape:", init_embeddings.shape)
        else:
            print("Initializing node embeddings from provided attributes. Shape:", init_embeddings.shape)
        if normalize:
            init_embeddings = (init_embeddings - init_embeddings.mean(axis=0)) / init_embeddings.std(axis=0)
        assert init_embeddings.shape[0] == num_nodes, f"Expected {num_nodes} nodes, but got {init_embeddings.shape[0]}."
        assert init_embeddings.shape[1] == embed_size, f"Expected embedding size {embed_size}, but got {init_embeddings.shape[1]}."
        init_embeddings = torch.from_numpy(init_embeddings).float()  # Convert node attributes to tensor
        self.embedding = nn.Embedding.from_pretrained(init_embeddings, freeze=True)

        # Define the layers
        dense_size = int(0.2 * embed_size)
        self.feed_forward = nn.Sequential(
            # Input layer
            nn.Linear(2 * embed_size if aggregation_method == "concat" else embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Hidden layer
            nn.Linear(embed_size, dense_size),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Output Layer
            nn.Linear(dense_size, 1),
            nn.Softplus()
        )

    def forward(self, x1, x2):
        # Embedding layer
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)

        # Aggregate embeddings
        if self.aggregation_method == "hadamard":
            x = x1 * x2
        elif self.aggregation_method == "subtract":
            x = x1 - x2
        elif self.aggregation_method == "mean":
            x = (x1 + x2) / 2
        elif self.aggregation_method == "concat":
            x = torch.cat((x1, x2), dim=1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        # Pass through the sequential model
        x = self.feed_forward(x)

        return x
