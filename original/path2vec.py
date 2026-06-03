"""
References:
    [1] Paper: Making Fast Graph-based Algorithms with Graph Metric Embeddings (ACL 2019)
    [2] Original implementation: https://github.com/uhh-lt/path2vec
"""

import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basemodel import BaseModel


class Path2vec(BaseModel):
    def __init__(self, G, embed_size=64, max_distance=1.0,
                 regularize=True, l1factor=1e-10,
                 use_neighbors=True, neighbor_count=1, alpha=0.01):
        """
        Initializes the Path2vec model.
        """
        super().__init__()
        self.max_distance = max_distance
        self.regularize = regularize
        self.l1factor = l1factor  # L1 regularization coefficient
        self.use_neighbors = use_neighbors
        self.neighbor_count = neighbor_count
        self.alpha = alpha  # Regularization coefficient for neighbors

        num_nodes = G.number_of_nodes()
        self.neighbor_map = {i: list(G.successors(i)) for i in G.nodes()}

        ## Define layers
        # Embedding layer
        self.embedding = nn.Embedding(num_nodes, embed_size)

    def forward(self, x1, x2, distance_measure="inv_dotproduct"):
        """
        Compute the normalized dot product between two sets of node embeddings.

        Args:
            x1 (torch.Tensor): Node indices for the first set of nodes.
            x2 (torch.Tensor): Node indices for the second set of nodes.

        Returns:
            torch.Tensor: Normalized dot product between the embeddings.
        """
        # Embedding layer
        x1 = self.embedding(x1)  # Shape: (B, D)
        x2 = self.embedding(x2)  # Shape: (B, D)

        # Compute normalized dot product (cosine similarity)
        if distance_measure == "inv_dotproduct":
            x = (1 - F.cosine_similarity(x1, x2, dim=1, eps=1e-8)).unsqueeze(-1) * self.max_distance/2  # Shape: (B, 1)
        elif distance_measure == "dotproduct":
            x = (1 + F.cosine_similarity(x1, x2, dim=1, eps=1e-8).unsqueeze(-1)) * self.max_distance/2  # Shape: (B, 1)

        return x

    def _train_step(self, x1, x2, y, criterion, optimizer, sampled_neighbors):
        optimizer.zero_grad()           # Clear gradients
        y_pred = self.forward(x1, x2)   # Forward pass
        loss = criterion(y_pred, y)     # Compute Loss

        if self.use_neighbors:
            # Get all nodes
            src_nodes = x1.cpu().numpy()
            dst_nodes = x2.cpu().numpy()

            # # Get all neighbors
            # src_neighbors = [self.neighbor_map[int(node)] for node in src_nodes]
            # dst_neighbors = [self.neighbor_map[int(node)] for node in dst_nodes]

            # # Sample k neighbors if needed
            # if self.neighbor_count > 0:
            #     src_neighbors = [np.random.choice(n, min(len(n), self.neighbor_count)) for n in src_neighbors]
            #     dst_neighbors = [np.random.choice(n, min(len(n), self.neighbor_count)) for n in dst_neighbors]
            src_neighbors = [sampled_neighbors[int(node)] for node in src_nodes]
            dst_neighbors = [sampled_neighbors[int(node)] for node in dst_nodes]

            # For every src node, compute normalized dot product with its src_neighbors
            nodes, neighbors = [], []
            for node_i, neighbors_i in zip(src_nodes, src_neighbors):
                nodes.extend([node_i] * len(neighbors_i))
                neighbors.extend(neighbors_i)
            for node_j, neighbors_j in zip(dst_nodes, dst_neighbors):
                nodes.extend([node_j] * len(neighbors_j))
                neighbors.extend(neighbors_j)

            # Convert to tensors
            nodes = torch.tensor(nodes, device=x1.device)
            neighbors = torch.tensor(neighbors, device=x1.device)

            # Compute normalized dot product (cosine similarity)
            neighbor_similarity = self.forward(nodes, neighbors, distance_measure="dotproduct")  # Shape: (total_neighbors, 1)

            # Apply neighbor regularization
            # NOTE: this can lead to negative losses if the neighbor similarities are very high
            neighbor_loss = torch.sum(neighbor_similarity)/len(neighbor_similarity)
            loss = (1-self.alpha) * loss - self.alpha * neighbor_loss

        if self.regularize:
            # Apply l1 regularization on the embeddings
            l1_reg = torch.norm(self.embedding.weight, p=1)
            ## Alternative cheap computation, apply l1 only on batch
            # l1_reg = torch.norm(self.embedding(x1), p=1) + torch.norm(self.embedding(x2), p=1)
            loss = loss + self.l1factor * l1_reg

        loss.backward()                 # Backward pass (gradient computation)
        optimizer.step()                # Update weights
        return loss

    def fit(self,
            dataloader,
            criterion,
            optimizer,
            val_dataloader=None,
            epochs=1,
            display_step=10,
            device="cpu",
            time_limit=None,
            fast_dev_run=False,
            **kwargs
        ):
        """
        Trains the model using the provided dataloader.
        Optionally evaluates on validation data.
        """
        # Set the model to training mode
        self.train()

        # Move model and criterion to device
        self.to(device)
        criterion.to(device)

        # Initialize history lists for tracking training progress
        loss_epoch_history = []      # Average train loss per epoch
        loss_iter_history = []       # Train loss per batch
        val_mre_epoch_history = []   # Validation MRE per epoch
        time_history = []            # Time elapsed per epoch

        # Calculate how often to display progress
        # NOTE: max handles cases where dataloader is small
        display_step = max(1, len(dataloader) // display_step)

        # Start time for training
        start_time = time.perf_counter()  # Start time for training
        time_limit_checkpoint = None  # 5-minute checkpoint

        for epoch in range(epochs):  # Loop over epochs
            # Initialize running loss
            running_loss = 0.0

            # Pre-sample neighbors for all nodes to speed up training
            sampled_neighbors = {node: np.random.choice(neighbors, min(len(neighbors), self.neighbor_count), replace=False).tolist()
                                 for node, neighbors in self.neighbor_map.items()}

            for idx, batch in enumerate(dataloader):  # Loop over batches
                i, j, d_ij = batch          # Unpack batch
                d_ij = d_ij.unsqueeze(-1)   # Ensure target shape is (batch_size, 1)

                # Move data to device
                i = i.to(device)
                j = j.to(device)
                d_ij = d_ij.to(device)

                # Perform a training step and update running loss
                loss = self._train_step(i, j, d_ij, criterion, optimizer, sampled_neighbors)
                running_loss += loss.item()
                loss_iter_history.append(loss.item())

                # Display progress at intervals
                if (idx + 1) % display_step == 0:  # Print every display_step
                    if loss.item() > 1.0:
                        loss_str = f"{loss.item():.2f}"
                    else:
                        loss_str = f"{loss.item():.8f}"
                    print(
                        f"Epoch: {epoch + 1:>2}/{epochs}, "
                        f"Batch: {idx + 1:>4} ({len(d_ij):>4} samples), "
                        f"Train Loss: {loss_str:>12}"
                    )

                # For quick debugging
                if fast_dev_run:
                    break

            # If validation data is provided, evaluate model
            val_str = ""
            if val_dataloader is not None:
                val_predictions, val_targets, _ = self.evaluate(
                    val_dataloader, device=device, verbose=False, profile_time=False
                )
                # Compute MRE on validation set
                val_mre = np.mean(np.abs(val_predictions - val_targets) / np.maximum(val_targets, 1e-6))
                val_mre_epoch_history.append(val_mre)

                val_str = f", Val MRE: {val_mre:.2%}"

                # Switch back to training mode after validation
                self.train()

            # Compute average loss
            avg_loss = running_loss / len(dataloader)
            loss_epoch_history.append(avg_loss)

            # Compute time elapsed
            time_elapsed = (time.perf_counter() - start_time) / 60  # (in minutes)
            time_remaining = (time_elapsed / (epoch + 1)) * (epochs - epoch - 1)
            time_history.append(time_elapsed)

            if avg_loss > 1.0:
                avg_loss_str = f"{avg_loss:.2f}"
            else:
                avg_loss_str = f"{avg_loss:.8f}"
            print(
                f"Epoch: {epoch + 1:>2}/{epochs}, "
                f"Time elapsed/remaining/total: {time_elapsed:.2f}/{time_remaining:.2f}/{(time_elapsed + time_remaining):.2f} min, "
                f"Train Loss: {avg_loss_str:>12}{val_str}"
            )

            # Record 5-minute checkpoint (but don't stop)
            if time_limit is not None and time_elapsed >= time_limit and time_limit_checkpoint is None:
                time_limit_checkpoint = {
                    "epoch": epoch + 1,
                    "time_min": time_elapsed,
                    "val_mre": val_mre_epoch_history[-1] if val_mre_epoch_history else None,
                }
                print(f"[5min checkpoint] epoch={epoch+1}, val_mre={time_limit_checkpoint['val_mre']}")

        result = {
            "loss_epoch_history": loss_epoch_history,
            "loss_iter_history": loss_iter_history,
            "val_mre_epoch_history": val_mre_epoch_history,
            "time_history": time_history,
        }
        if time_limit_checkpoint is not None:
            result["time_limit_checkpoint"] = time_limit_checkpoint
        return result
