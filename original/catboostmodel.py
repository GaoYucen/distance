"""
References:
    [1] Paper: Shortest Path Distance Prediction Based on CatBoost (WISA 2021)
    [2] Original implementation: (None found)

TODO:
    [ ] Add one more catboost model in serial fashion as given in the paper.
    [ ] Try with landmark chosen by k-Means (as suggested in the paper) and with subset of train nodes.
    [ ] Use 2 serial catboost models as mentioned in the paper.
"""

import os

import numpy as np

from catboost import CatBoostRegressor, Pool
from catboost.utils import get_gpu_device_count

import torch
import torch.nn as nn

from models.basemodel import BaseModel


# Define the CatBoost model
class CatBoostModel(BaseModel):
    def __init__(self, num_nodes, coordinate_embs, landmark_embs, seed=42):
        super().__init__()

        self.seed = seed

        # Embedding layers
        print(f"Coordinate embeddings shape: {coordinate_embs.shape}")
        print(f"Landmark embeddings shape: {landmark_embs.shape}")
        self.coordinate_embs = nn.Embedding.from_pretrained(torch.from_numpy(coordinate_embs).float(), freeze=True)
        self.landmark_embs = nn.Embedding.from_pretrained(torch.from_numpy(landmark_embs).float(), freeze=True)

        # CatBoost model
        self.catboost_model = None

    def encode(self, x1, x2):
        # Convert x1 and x2 to embeddings
        if self.coordinate_embs is not None:
            x1_coord_emb = self.coordinate_embs(x1)
            x2_coord_emb = self.coordinate_embs(x2)
        if self.landmark_embs is not None:
            x1_landmark_emb = self.landmark_embs(x1)
            x2_landmark_emb = self.landmark_embs(x2)

        # Compute cosine similarity between landmark embeddings
        if self.landmark_embs is not None:
            cosine_sim = nn.functional.cosine_similarity(x1_landmark_emb, x2_landmark_emb, dim=-1).unsqueeze(-1)
        elif self.coordinate_embs is not None:
            cosine_sim = nn.functional.cosine_similarity(x1_coord_emb, x2_coord_emb, dim=-1).unsqueeze(-1)
        else:
            cosine_sim = None

        # Compute euclidean distance between coordinate embeddings
        if self.coordinate_embs is not None:
            euclidean_dist = torch.norm(x1_coord_emb - x2_coord_emb, p=1, dim=-1, keepdim=True)
        elif self.landmark_embs is not None:
            euclidean_dist = torch.norm(x1_landmark_emb - x2_landmark_emb, p=1, dim=-1, keepdim=True)
        else:
            euclidean_dist = None

        # Concatenate features, cast to CPU and convert to numpy
        features = []
        if self.landmark_embs is not None:
            features.append(x1_landmark_emb)
            features.append(x2_landmark_emb)
        if self.coordinate_embs is not None:
            features.append(x1_coord_emb)
            features.append(x2_coord_emb)
        if cosine_sim is not None:
            features.append(cosine_sim)
        if euclidean_dist is not None:
            features.append(euclidean_dist)
        features = torch.cat(features, dim=-1).numpy()

        return features

    def forward(self, x1, x2):
        # Compute features
        features = self.encode(x1, x2)

        # Use catboost model to predict distances
        predictions = self.catboost_model.predict(features)

        # Convert predictions to tensor
        predictions = torch.from_numpy(predictions).float().unsqueeze(-1)

        return predictions

    def fit(self, dataloader=None, val_dataloader=None, epochs=1, learning_rate=0.01, device="cpu", **kwargs):
        # Initialize CatBoost regressor
        # NOTE: epochs from train.py CLI is for NN training (e.g. 20 passes).
        # For GBDT, "iterations" = number of trees — needs thousands, not dozens.
        catboost_iterations = 3000
        # NOTE: CatBoost needs higher LR than NN optimizers (0.1 vs 0.01)
        catboost_lr = 0.1
        self.catboost_model = CatBoostRegressor(
            iterations=catboost_iterations,  ## Total number of trees (paper: 3000)
            learning_rate=catboost_lr,       ## Paper uses 0.1, not NN-default 0.01
            # depth=6,
            random_seed=self.seed,
            loss_function='RMSE',
            task_type='CPU',                ## Incremental training with init_model requires CPU
            # task_type='GPU',              ## BUG: GPU training increases MRE
            # devices='0',
            verbose=1,
            train_dir='/tmp/catboost_info',  # Temporary directory for CatBoost info
            eval_metric='MAPE',
        )

        # Initialize lists to store features and targets
        features = []
        targets = []

        # Extract features and targets from dataloader
        for idx, (i, j, d_ij) in enumerate(dataloader):
            i, j = i.to(device), j.to(device)   ## Move data to device
            features_batch = self.encode(i, j)
            features.append(features_batch)
            targets_batch = d_ij.cpu().numpy()  ## Move targets to CPU
            targets.append(targets_batch)

        # Stack features and targets
        features = np.vstack(features)          ## Vertical Stacking
        targets = np.hstack(targets)            ## Horizontal Stacking
        num_samples = features.shape[0]

        # Create Pool object for CatBoost
        train_pool = Pool(data=features, label=targets)

        # Validation dataloader
        val_pool = None
        if val_dataloader is not None:
            features_val = []
            targets_val = []
            for idx, (i, j, d_ij) in enumerate(val_dataloader):
                i, j = i.to(device), j.to(device)
                features_batch = self.encode(i, j)
                features_val.append(features_batch)
                targets_batch = d_ij.cpu().numpy()
                targets_val.append(targets_batch)
            features_val = np.vstack(features_val)
            targets_val = np.hstack(targets_val)
            val_pool = Pool(data=features_val, label=targets_val)

        # Train catboost model in mini-batches
        print("Training CatBoost model...")
        # NOTE: This batch size is for incremental training of CatBoost model, independent of PyTorch batch size
        batch_size = 1_000_000  # You can adjust this based on system's memory capacity

        # Temporary model file for incremental training
        model_file = "/tmp/catboost_info/catboost_model.cbm"

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            print(f"Training on samples {start_idx} to {end_idx}...")

            # Slice the Pool for the current batch
            batch_pool = train_pool.slice(np.arange(start_idx, end_idx))

            # Set the init_model based on whether it's the first batch
            current_init_model = model_file if start_idx > 0 else None

            # Train the model
            self.catboost_model.fit(batch_pool, verbose=100, init_model=current_init_model, eval_set=val_pool)

            # Save the updated model for the next iteration
            self.catboost_model.save_model(model_file)
        print("CatBoost model training completed.")

        # Print model size
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"CatBoost model size: {size_mb:.2f} MB")
        else:
            print(f"Model file `{model_file}` not found.")

        loss_epoch_history = []  # Store the loss values per epoch
        loss_iter_history = []  # Store the loss values per iteration
        val_mre_epoch_history = []  # Store the validation mre values per epoch
        return {
            "loss_epoch_history": loss_epoch_history,
            "loss_iter_history": loss_iter_history,
            "val_mre_epoch_history": val_mre_epoch_history,
            "time_history": [],
        }

# Example usage:
# python train.py --device cpu --data_dir W_Chicago --model_class catboost
#    epochs=3000, learning_rate=0.1
#    result: MRE ~0.56% for W_Chicago
