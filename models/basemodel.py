import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn


# Define the BaseModel model
class BaseModel(nn.Module):
    """
    Base model for distance prediction.
    Provides training and evaluation utilities.
    """

    def __init__(self):
        """
        Initializes the BaseModel.
        """
        super().__init__()

    def forward(self):
        """
        Forward pass. Should be implemented by subclasses.

        Args:
            x1 (torch.Tensor): First input tensor.
            x2 (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pass

    def _train_step(self, x1, x2, y, criterion, optimizer):
        """
        Performs a single training step:
        - Clears gradients
        - Computes predictions
        - Calculates loss
        - Backpropagates and updates weights

        Args:
            x1 (torch.Tensor): First input tensor (src nodes).
            x2 (torch.Tensor): Second input tensor (dst nodes).
            y (torch.Tensor): Target tensor (ground truth distances).
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.

        Returns:
            torch.Tensor: Loss value for the batch.
        """
        optimizer.zero_grad()           # Clear gradients
        y_pred = self.forward(x1, x2)   # Forward pass
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
            fast_dev_run=False,
            time_limit=None,
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
        time_limit_checkpoint = None  # 5-minute checkpoint (record but don't stop)

        for epoch in range(epochs):  # Loop over epochs
            # Initialize running loss
            running_loss = 0.0

            for idx, batch in enumerate(dataloader):  # Loop over batches
                if len(batch) == 4:
                    i, j, d_ij, _ = batch   # Unpack batch (directed, ignore reverse)
                else:
                    i, j, d_ij = batch      # Unpack batch (undirected)
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

            # Record 5-minute checkpoint (but don't stop)
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

    def evaluate(self, dataloader, verbose=True, profile_time=True, device="cpu", **kwargs):
        """
        Evaluates the model on the provided dataloader.
        Returns predictions, targets, and average query latency.
        """
        # Set model to evaluation mode
        self.eval()

        # Move model to device
        self.to(device)

        # Initialize lists to store predictions and targets
        predictions = []
        targets = []

        # For profiling time (if enabled)
        total_time = 0.0

        with torch.no_grad():
            if profile_time:
                # Profile time for each batch
                for idx, (i, j, d_ij) in enumerate(dataloader):
                    targets.append(d_ij.cpu().numpy())      # Store ground truth
                    start_time = time.perf_counter()        # Start timing
                    i, j = i.to(device), j.to(device)       # Move data to device
                    outputs = self.forward(i, j)            # Model inference
                    outputs = outputs.cpu().numpy()[:, 0]   # Convert to numpy
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()            # Ensure CUDA ops are finished
                    end_time = time.perf_counter()          # End timing
                    total_time += end_time - start_time     # Accumulate time
                    predictions.append(outputs)             # Store predictions
            else:
                # Standard evaluation without timing
                for idx, (i, j, d_ij) in enumerate(dataloader):
                    targets.append(d_ij.cpu().numpy())
                    i, j = i.to(device), j.to(device)
                    outputs = self.forward(i, j)
                    outputs = outputs.cpu().numpy()[:, 0]
                    predictions.append(outputs)

        # Concatenate all batch predictions and targets
        predictions = np.hstack(predictions)
        targets = np.hstack(targets)

        # Compute average query latency (if enabled)
        query_latency = total_time / len(targets)

        return predictions, targets, query_latency
