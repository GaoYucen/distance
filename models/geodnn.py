"""
References:
    [1] Paper: A Unified Neural Network Approach for Estimating Travel Time and Distance for a Taxi Trip (ArXiv 2017)
    [2] Original implementation: (None found)
"""

import torch
import torch.nn as nn

from models.basemodel import BaseModel


# Define the GeoDNN model
class GeoDNN(BaseModel):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output, node_attributes, max_distance=1.0):
        super().__init__()
        self.max_distance = max_distance

        ## Define layers
        # Embedding layer
        node_features = torch.from_numpy(node_attributes).float()  # Convert node attributes to tensor
        # node_features = (node_features - node_features.min(dim=0)) / (node_features.max(dim=0) - node_features.min(dim=0))
        node_features = (node_features - node_features.mean(dim=0)) / node_features.std(dim=0)
        self.embedding = nn.Embedding.from_pretrained(node_features, freeze=True)

        # Fully connected layers
        self.fc1 = nn.Linear(n_input, n_hidden_1)       # 4 --> 20
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)    # 20 --> 100
        self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)    # 100 --> 20
        self.fc4 = nn.Linear(n_hidden_3, n_output)      # 20 --> 1

        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # Embedding layer
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)

        # Concatenate embeddings
        x = torch.cat((x1, x2), dim=1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x)) * self.max_distance

        return x
