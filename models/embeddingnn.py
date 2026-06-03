"""
References:
    [1] Paper: Learning to Predict Shortest Path Distance (ADMA 2023)
    [2] Original implementation: (None found)

TODO:
    [ ] Logical error in randomly initializing embeddings with freeze=True? Because ranomly initialized embeddings won't be updated.
"""

import numpy as np

import torch
import torch.nn as nn

from models.basemodel import BaseModel


# Define the EmbeddingNN model
class EmbeddingNN(BaseModel):
    def __init__(self, num_nodes, embed_size=64, n_hidden_1=500, n_output=1, init_embeddings=None, aggregation_method="mean", normalize=True, max_distance=1.0):
        super().__init__()
        self.max_distance = max_distance
        self.aggregation_method = aggregation_method  # enum: ["hadamard", "subtraction", "mean", "concat"]
        print(f"Initializing EmbeddingNN with aggregation method: {self.aggregation_method} and embedding size: {embed_size}")

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

        # Fully connected layers
        embed_size = embed_size*2 if aggregation_method == "concat" else embed_size  # Adjust embed_size for concatenation
        self.fc1 = nn.Linear(embed_size, n_hidden_1)         # d --> 500
        self.fc2 = nn.Linear(n_hidden_1, n_output)          # 500 --> 1

        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # Embedding layer
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)

        # Concatenate embeddings
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

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x)) * self.max_distance  # Scale output to max_distance

        return x

    def _train_step(self, x1, x2, y, criterion, optimizer):
        optimizer.zero_grad()           # Clear gradients
        y_pred = self.forward(x1, x2)   # Forward pass

        # Normalize predictions and targets
        y_pred = y_pred / self.max_distance
        y = y / self.max_distance

        loss = criterion(y_pred, y)     # Compute Loss
        loss.backward()                 # Backward pass (gradient computation)
        optimizer.step()                # Update weights
        return loss
