"""
References:
    [1] Paper: RNE: computing shortest paths using road network embedding (VLDB Journal 2022)
    [2] Original implementation: https://github.com/LennyHuang15/RNE-notebook/blob/master/RNE.ipynb
"""

import time
import numpy as np

import torch
import torch.nn as nn

from models.basemodel import BaseModel


class RNE(BaseModel):
    def __init__(self, num_nodes, embed_size, max_distance=1.0, parts=None):
        """
        Initializes the RNE model.
        """
        super().__init__()
        self.embed_size = embed_size
        self.max_distance = max_distance
        self.parts = torch.from_numpy(parts) if parts is not None else None
        print(f"Initializing RNE...")
        print(f"  - Number of nodes: {num_nodes}")
        print(f"  - Embedding size: {embed_size}")
        print(f"  - Max distance: {max_distance}")

        ## Define layers
        # Embedding layer
        self.embedding = nn.Embedding(num_nodes, embed_size)
        nn.init.uniform_(self.embedding.weight, -3/2, 3/2)

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

        x = torch.abs(x1 - x2)                  # Shape: (B, embed_size)
        x = torch.mean(x, dim=1, keepdim=True)  # Shape: (B, 1)
        x = torch.clamp(x, min=0.0)
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

        ## Hierarchical training
        if self.parts is not None:
            num_levels = self.parts.shape[1]
            print(f"Number of hierarchical levels: {num_levels}")
            h_epochs = [5]*(num_levels-1) + [10]  # Train 10 epochs on the last level, 5 epochs on all other levels, TODO: make this as a parameter
            print(f"Hierarchical epochs per level: {h_epochs}")
            self.parts = self.parts.to(device)

            prev_embeddings = None
            print(f"Starting hierarchical training with {self.parts.shape[1]} levels")
            for level in range(self.parts.shape[1]):
                print(f"Training on hierarchical level {level} with {h_epochs[level]} epochs")
                part_indices = self.parts[:, level]

                # Transfer embeddings from previous level
                if level > 0:
                    with torch.no_grad():
                        prev_part_indices = self.parts[:, level-1]
                        self.embedding.weight.data[part_indices] = prev_embeddings[prev_part_indices]

                # Train current level
                for epoch in range(h_epochs[level]):  # Loop over epochs
                    for idx, batch in enumerate(dataloader):  # Loop through the dataloader
                        i, j, d_ij = batch          # Unpack batch
                        d_ij = d_ij.unsqueeze(-1)   # Ensure target shape is (batch_size, 1)

                        # Map to part indices
                        i = part_indices[i]
                        j = part_indices[j]

                        # Move data to device
                        i = i.to(device)
                        j = j.to(device)
                        d_ij = d_ij.to(device)

                        # Perform a training step
                        loss = self._train_step(i, j, d_ij, criterion, optimizer)

                        # Display progress at intervals
                        if (idx + 1) % display_step == 0:  # Print every display_step
                            if loss.item() > 1.0:
                                loss_str = f"{loss.item():.2f}"
                            else:
                                loss_str = f"{loss.item():.8f}"
                            print(
                                f"[Level {level}] Epoch: {epoch + 1:>2}/{h_epochs[level]}, "
                                f"Batch: {idx + 1:>4} ({len(d_ij):>4} samples), "
                                f"Loss: {loss_str:>{12}}"
                            )

                # Store current embeddings for next level
                prev_embeddings = self.embedding.weight.data.clone()
            print("Completed hierarchical training.")

        ## Standard training
        print(f"Starting standard training for {epochs} epochs")
        time_limit_checkpoint = None  # 5-minute checkpoint (record but don't stop)
        for epoch in range(epochs):  # Loop over epochs
            # Initialize running loss
            running_loss = 0.0

            for idx, batch in enumerate(dataloader):  # Loop over batches
                i, j, d_ij = batch          # Unpack batch
                d_ij = d_ij.unsqueeze(-1)   # Ensure target shape is (batch_size, 1)

                # Move data to device
                i = i.to(device)
                j = j.to(device)
                d_ij = d_ij.to(device)

                # Perform a training step and update running loss
                loss = self._train_step(i, j, d_ij, criterion, optimizer)
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

            # Record 5-minute checkpoint (but don't stop training)
            if time_limit is not None and time_elapsed >= time_limit and time_limit_checkpoint is None:
                time_limit_checkpoint = {
                    "epoch": epoch + 1,
                    "time_min": time_elapsed,
                    "train_loss": avg_loss,
                    "val_mre": val_mre_epoch_history[-1] if val_mre_epoch_history else None,
                }
                print(f"[5min checkpoint] epoch={epoch+1}, train_loss={avg_loss:.6f}, val_mre={time_limit_checkpoint['val_mre']}")

        # Return training and validation history for analysis
        result = {
            "loss_epoch_history": loss_epoch_history,
            "loss_iter_history": loss_iter_history,
            "val_mre_epoch_history": val_mre_epoch_history,
            "time_history": time_history,
        }
        if time_limit_checkpoint is not None:
            result["time_limit_checkpoint"] = time_limit_checkpoint
        return result
