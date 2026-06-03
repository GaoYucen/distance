"""
References:
    [1] Paper: Ndist2vec: Node with Landmark and New Distance to Vector Method for Predicting Shortest Path Distance along Road Networks (ISPRS 2022)
    [2] Original implementation: https://figshare.com/articles/dataset/ndist2vec/20238813/1
"""

import torch
import torch.nn as nn

from models.basemodel import BaseModel


# Define the Ndist2vec model
class Ndist2vec(BaseModel):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output, max_distance):
        super().__init__()

        ## Define layers
        # Shared embedding layer
        self.embedding = nn.Embedding(n_input, n_hidden_1)

        # Branch 1
        self.fc1_branch1 = nn.Linear(n_hidden_1 * 2, n_hidden_2)
        self.fc2_branch1 = nn.Linear(n_hidden_2, n_hidden_3)
        self.out_branch1 = nn.Linear(n_hidden_3, n_output)

        # Branch 2
        self.fc1_branch2 = nn.Linear(n_hidden_1 * 2, n_hidden_2)
        self.fc2_branch2 = nn.Linear(n_hidden_2, n_hidden_3)
        self.out_branch2 = nn.Linear(n_hidden_3, n_output)

        # Branch 3
        self.fc1_branch3 = nn.Linear(n_hidden_1 * 2, n_hidden_2)
        self.fc2_branch3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.out_branch3 = nn.Linear(n_hidden_3, n_output)

        # Branch 4
        self.fc1_branch4 = nn.Linear(n_hidden_1 * 2, n_hidden_2)
        self.fc2_branch4 = nn.Linear(n_hidden_2, n_hidden_3)
        self.out_branch4 = nn.Linear(n_hidden_3, n_output)

        # Learned weights for combining branch outputs.
        # Deterministic initialization at midpoints of the original uniform ranges
        # to avoid seed-sensitive training divergence (see EXPERIMENT_LOG.md #10).
        assert max_distance > 10000, f"max_distance={max_distance} should be greater than 10000"
        self.v1 = nn.Parameter(torch.tensor([50.]))              # midpoint of [0, 100]
        self.v2 = nn.Parameter(torch.tensor([550.]))             # midpoint of [100, 1000]
        self.v3 = nn.Parameter(torch.tensor([5500.]))            # midpoint of [1000, 10000]
        self.v4 = nn.Parameter(torch.tensor([max_distance / 2.]))  # midpoint of [10000, max-10000]

        # Activation function
        self.sigmoid = nn.Sigmoid()

        ## Initialize weights
        # nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.trunc_normal_(self.embedding.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc1_branch1.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc1_branch1.bias, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc2_branch1.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc2_branch1.bias, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.out_branch1.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.out_branch1.bias, mean=0.0, std=0.01)

        nn.init.trunc_normal_(self.fc1_branch2.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc1_branch2.bias, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc2_branch2.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc2_branch2.bias, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.out_branch2.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.out_branch2.bias, mean=0.0, std=0.01)

        nn.init.trunc_normal_(self.fc1_branch3.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc1_branch3.bias, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc2_branch3.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc2_branch3.bias, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.out_branch3.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.out_branch3.bias, mean=0.0, std=0.01)

        nn.init.trunc_normal_(self.fc1_branch4.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc1_branch4.bias, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc2_branch4.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc2_branch4.bias, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.out_branch4.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.out_branch4.bias, mean=0.0, std=0.01)

    def forward(self, x1, x2):
        # Embedding layer
        x1_embed = self.embedding(x1)
        x2_embed = self.embedding(x2)

        # Concatenate embeddings
        x = torch.cat((x1_embed, x2_embed), dim=1)

        # Branch 1
        x_branch1 = torch.relu(self.fc1_branch1(x))
        x_branch1 = torch.relu(self.fc2_branch1(x_branch1))
        x_branch1 = self.sigmoid(self.out_branch1(x_branch1))

        # Branch 2
        x_branch2 = torch.relu(self.fc1_branch2(x))
        x_branch2 = torch.relu(self.fc2_branch2(x_branch2))
        x_branch2 = self.sigmoid(self.out_branch2(x_branch2))

        # Branch 3
        x_branch3 = torch.relu(self.fc1_branch3(x))
        x_branch3 = torch.relu(self.fc2_branch3(x_branch3))
        x_branch3 = self.sigmoid(self.out_branch3(x_branch3))

        # Branch 4
        x_branch4 = torch.relu(self.fc1_branch4(x))
        x_branch4 = torch.relu(self.fc2_branch4(x_branch4))
        x_branch4 = self.sigmoid(self.out_branch4(x_branch4))

        # Weighted combination of branch outputs
        out = self.v1 * x_branch1 + self.v2 * x_branch2 \
                + self.v3 * x_branch3 + self.v4 * x_branch4

        return out
