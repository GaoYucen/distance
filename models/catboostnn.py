"""
Adapted version of CatBoost model by replacing CatBoost regressor with a neural network.
"""

import numpy as np

import torch
import torch.nn as nn

from models.basemodel import BaseModel


# Define the CatBoost model
class CatBoostNN(BaseModel):
    def __init__(self, num_nodes, coordinate_embs, landmark_embs, max_distance=1.0):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_landmarks = landmark_embs.shape[1]
        self.max_distance = max_distance

        # Normalize embeddings
        coordinate_embs = (coordinate_embs - np.mean(coordinate_embs, axis=0)) / np.std(coordinate_embs, axis=0)
        landmark_embs = (landmark_embs - np.mean(landmark_embs, axis=0)) / np.std(landmark_embs, axis=0)

        ## Define layers
        # Embedding layers
        self.coordinate_embs = nn.Embedding.from_pretrained(torch.from_numpy(coordinate_embs).float(), freeze=True)
        self.landmark_embs = nn.Embedding.from_pretrained(torch.from_numpy(landmark_embs).float(), freeze=True)

        # Fully connected layers
        # self.bn1 = nn.BatchNorm1d(2*self.num_landmarks + 2*self.coordinate_embs.embedding_dim + 2)
        self.fc1 = nn.Linear(2*self.num_landmarks + 2*self.coordinate_embs.embedding_dim + 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.5)
        # self.bn2 = nn.BatchNorm1d(512)

    def encode(self, x1, x2):
        # Convert x1 and x2 to embeddings
        x1_coord_emb = self.coordinate_embs(x1)
        x1_landmark_emb = self.landmark_embs(x1)
        x2_coord_emb = self.coordinate_embs(x2)
        x2_landmark_emb = self.landmark_embs(x2)

        # Compute cosine similarity between landmark embeddings
        cosine_sim = nn.functional.cosine_similarity(x1_landmark_emb, x2_landmark_emb, dim=-1).unsqueeze(-1)

        # Compute euclidean distance between coordinate embeddings
        euclidean_dist = torch.norm(x1_coord_emb - x2_coord_emb, p=2, dim=-1, keepdim=True)

        # Concatenate features, cast to CPU and convert to numpy
        features = torch.cat([x1_landmark_emb, x2_landmark_emb, x1_coord_emb, x2_coord_emb, cosine_sim, euclidean_dist], dim=-1)

        return features

    def forward(self, x1, x2):
        # Compute features
        features = self.encode(x1, x2)

        x = features.float()  # Convert features to torch tensor
        # x = self.bn1(x)
        x = self.relu(self.fc1(x))
        # x = self.bn2(x)
        # x = self.dropout(x)
        x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = torch.sigmoid(self.fc3(x))
        x = self.fc3(x)

        # Scale output to max_distance
        x = x * self.max_distance

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


# Example usage:
# python train.py --device cuda --data_dir W_NewYork --model_class catboostnn --learning_rate 0.0003 --epochs 20
#   result: MRE ~3% for W_NewYork (with 1024, 512, 1 architecture)
#  for W_Beijing, ~7% for 50 epochs and learning rate 0.0003
