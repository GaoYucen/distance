"""
References:
    [1] Paper: A Learning Based Approach to Predict Shortest-Path Distances (EDBT 2020)
    [2] Original implementation: https://github.com/alvinzhaowei/vdist2vec/blob/main/model/vdist2vec.py

TODO:
    [ ] Experiment with scaled vs unscaled loss computation (y and y_pred vs y/max_distance and y_pred/max_distance)
    [ ] Add scheduler (STEP LR) as given in paper.
"""

import torch
import torch.nn as nn

from models.basemodel import BaseModel


# Define the Vdist2vec model
class Vdist2vec(BaseModel):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output, max_distance=1.0):
        super().__init__()
        self.max_distance = max_distance

        ## Define layers
        # Embedding layer
        self.embedding = nn.Embedding(n_input, n_hidden_1)

        # Fully connected layers
        self.fc1 = nn.Linear(n_hidden_1 * 2, n_hidden_2)
        self.fc2 = nn.Linear(n_hidden_2, n_hidden_3)
        self.fc3 = nn.Linear(n_hidden_3, n_output)

        # Activation function
        self.sigmoid = nn.Sigmoid()

        ## Initialize weights
        # nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.trunc_normal_(self.embedding.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc1.bias, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc2.bias, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc3.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc3.bias, mean=0.0, std=0.01)

    def forward(self, x1, x2):
        # Embedding layer
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)

        # Concatenate embeddings
        x = torch.cat((x1, x2), dim=1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x)) * self.max_distance  # Scale output to [0, max_distance]

        return x
