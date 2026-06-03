"""
References:
    [1] Paper: RGCNdist2vec: Using Graph Convolutional Networks and Distance2Vector to Estimate Shortest Path Distance Along Road Networks (SDI 2024)
    [2] Original implementation: (None found)
"""

import time
import numpy as np

import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.transforms import AddSelfLoops, ToUndirected
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import k_hop_subgraph


# Define the RGNNdist2vec model
class RGNNdist2vec(nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, layer_type, node_attributes, edge_attributes, max_distance=1.0, disable_edge_weight=True):
        super().__init__()
        self.max_distance = max_distance

        # Define GNN layers
        self.layer_type = layer_type
        if layer_type.lower() == "gcn":
            # Cached is explicitly False because of subgraph extraction optimization,
            # as it makes the (sub-)graph non-static and dynamically changing in each iteration
            # and thus caching is not applicable.
            self.layer1 = GCNConv(n_input, n_hidden_1, add_self_loops=True, cached=False)
            self.layer2 = GCNConv(n_hidden_1, n_hidden_2, add_self_loops=True, cached=False)
        elif layer_type.lower() == "gat":
            self.layer1 = GATConv(n_input, n_hidden_1, add_self_loops=True, fill_value='mean')
            self.layer2 = GATConv(n_hidden_1, n_hidden_2, add_self_loops=True, fill_value='mean')
        elif layer_type.lower() == "sage":
            self.layer1 = SAGEConv(n_input, n_hidden_1)
            self.layer2 = SAGEConv(n_hidden_1, n_hidden_2)

        # Activation function
        self.leaky_relu = nn.LeakyReLU()

        # Number of GNN layers
        self.num_layers = 2

        # Build geometric data
        self.geometric_data = self.build_geometric_data(node_attributes, edge_attributes, layer_type, disable_edge_weight)

        # Register geometric data as buffers (to save with model state_dict)
        self.register_buffer('node_features', self.geometric_data.x)
        self.register_buffer('edge_index', self.geometric_data.edge_index)
        self.register_buffer('edge_weight', self.geometric_data.edge_weight)

        # Cache for full graph embeddings
        self.cached_embeddings = None  # Cache for full graph embeddings

    def build_geometric_data(self, node_attributes, edge_attributes, layer_type, disable_edge_weight):
        # Prepare node features
        node_features = torch.from_numpy(node_attributes).float()
        # Normalize node features
        node_features = (node_features - node_features.mean(dim=0)) / node_features.std(dim=0)

        # Prepare edge index and weights
        edge_index = torch.from_numpy(edge_attributes[:, :2]).long().t().contiguous()
        edge_weight = torch.from_numpy(edge_attributes[:, 2]).float()
        if disable_edge_weight or layer_type.lower() == 'sage':  # SAGE does not use edge weights
            print(f"Disabling edge weights...")
            edge_weight = None

        # Save everything in a torch_geometric Data object
        print(f"Building geometric data object...")
        geometric_data = Data(
                            x=node_features,
                            edge_index=edge_index,
                            edge_weight=edge_weight)
        print(f"  - Node Features shape: {geometric_data.x.shape}")
        print(f"  - Edge Index shape: {geometric_data.edge_index.shape}")
        print(f"  - Edge Weight shape: {geometric_data.edge_weight.shape if geometric_data.edge_weight is not None else 'None'}")

        ## Making graph undirected
        # Since edge_index in pyg is considered directed by default,
        # we need to ensure that the graph is undirected if required.
        # TODO: The following workaround will not work when the edge_index is
        # directed already and will need to be handled differently.
        # Currently, we assume the input edge_index is undirected right from the start.
        geometric_data = ToUndirected()(geometric_data)
        print(f"Converting to undirected...")
        print(f"  - Edge Index shape: {geometric_data.edge_index.shape}")
        print(f"  - Edge Weight shape: {geometric_data.edge_weight.shape if geometric_data.edge_weight is not None else 'None'}")

        # Return the prepared geometric data
        return geometric_data

    def encode(self, node_features, edge_index, edge_weight=None):
        # GCN layers
        x = self.leaky_relu(self.layer1(node_features, edge_index, edge_weight))
        x = self.leaky_relu(self.layer2(x, edge_index, edge_weight))
        return x

    def forward(self, x1, x2, embeddings=None):
        if embeddings is None:
            if self.cached_embeddings is not None:
                # Use cached embeddings if available
                embeddings = self.cached_embeddings
                print("Using cached embeddings.")
            else:
                # Compute embeddings for the entire graph
                self.cached_embeddings = self.encode(self.geometric_data.x, self.geometric_data.edge_index, self.geometric_data.edge_weight).detach().clone()
                embeddings = self.cached_embeddings
                print("Computed embeddings and cached them.")

        # Extract embeddings for the given node indices
        emb1 = embeddings[x1]
        emb2 = embeddings[x2]

        # Compute L1 norm
        distances = torch.norm(emb1 - emb2, p=1, dim=1, keepdim=True)

        # Scale distances back to original range
        distances = distances * self.max_distance

        return distances

    def subgraph_extraction(self, x1, x2, geometric_data, subgraph_node_map, num_layers):
        # Get unique nodes in the batch
        batch_nodes = torch.cat([x1, x2]).unique()

        # Extract k-hop subgraph around the nodes in x1 and x2
        subgraph_nodes, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=batch_nodes, num_hops=num_layers,
            edge_index=geometric_data.edge_index,
            num_nodes=geometric_data.num_nodes,
            relabel_nodes=True
        )
        # Now, `subgraph_nodes` contains unique nodes in `x1`, `x2`, and their num_hops neighbors

        # Map x1 and x2 to their corresponding indices in the subgraph
        subgraph_node_map[subgraph_nodes] = torch.arange(len(subgraph_nodes), device=subgraph_nodes.device)
        x1_sub = subgraph_node_map[x1]
        x2_sub = subgraph_node_map[x2]

        # Construct the subgraph data (features, edges and edge weights)
        subgraph_features = geometric_data.x[subgraph_nodes]
        subgraph_edge_weight = None
        if geometric_data.edge_weight is not None:
            subgraph_edge_weight = geometric_data.edge_weight[edge_mask]

        return (subgraph_features, subgraph_edge_index, subgraph_edge_weight), x1_sub, x2_sub

    def _train_step(self, geometric_data, x1, x2, y, criterion, optimizer, num_layers, subgraph_node_map):
        optimizer.zero_grad()           # Clear gradients

        # Extract subgraph for the current batch
        subgraph, x1_sub, x2_sub = self.subgraph_extraction(x1, x2, geometric_data, subgraph_node_map, num_layers)
        subgraph_features, subgraph_edge_index, subgraph_edge_weight = subgraph

        # Compute embeddings for the subgraph and perform forward pass
        embeddings = self.encode(subgraph_features, subgraph_edge_index, subgraph_edge_weight)
        y_pred = self.forward(x1_sub, x2_sub, embeddings)

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
        self.geometric_data.to(device)

        # Initialize history lists for tracking training progress
        loss_epoch_history = []      # Average train loss per epoch
        loss_iter_history = []       # Train loss per batch
        val_mre_epoch_history = []   # Validation MRE per epoch
        time_history = []            # Time elapsed per epoch

        # Calculate how often to display progress
        # NOTE: max handles cases where dataloader is small
        display_step = max(1, len(dataloader) // display_step)

        # Initialize subgraph node map
        subgraph_node_map = torch.full((self.geometric_data.num_nodes,), -1, dtype=torch.long, device=device)

        # Start time for training
        start_time = time.perf_counter()  # Start time for training
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
                loss = self._train_step(self.geometric_data, i, j, d_ij, criterion, optimizer, self.num_layers, subgraph_node_map)
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

            # Check for time limit
            if time_limit is not None:
                if time_elapsed >= time_limit:
                    print(f"Time limit of {time_limit} minutes reached. Stopping training.")
                    break

        # Return training and validation history for analysis
        return {
            "loss_epoch_history": loss_epoch_history,
            "loss_iter_history": loss_iter_history,
            "val_mre_epoch_history": val_mre_epoch_history,
            "time_history": time_history,
        }

    def evaluate(self, dataloader, verbose=True, profile_time=True, device="cpu", **kwargs):
        """
        Evaluates the model on the provided dataloader.
        Returns predictions, targets, and average query latency.
        """
        # Set model to evaluation mode
        self.eval()

        # Move model to device
        self.to(device)

        # Move geometric data to device
        self.geometric_data.to(device)

        # Initialize lists to store predictions and targets
        predictions = []
        targets = []

        # For profiling time (if enabled)
        total_time = 0.0

        with torch.no_grad():
            ## Compute embeddings one time for the entire graph
            # This is more efficient than computing embeddings for each batch
            # as the embeddings stay the same for the entire graph.
            start_time = time.perf_counter()
            embeddings = self.encode(self.geometric_data.x, self.geometric_data.edge_index, self.geometric_data.edge_weight)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            embedding_time_per_sample = (end_time - start_time) / self.geometric_data.num_nodes
            if verbose:
                print(f"Embedding time per sample: {embedding_time_per_sample * 1_000_000:.3f} microseconds")

            if profile_time:
                # Profile time for each batch
                for idx, (i, j, d_ij) in enumerate(dataloader):
                    targets.append(d_ij.cpu().numpy())      # Store ground truth
                    start_time = time.perf_counter()        # Start timing
                    i, j = i.to(device), j.to(device)       # Move data to device
                    outputs = self.forward(i, j, embeddings)    # Model inference
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
                    outputs = self.forward(i, j, embeddings)
                    outputs = outputs.cpu().numpy()[:, 0]
                    predictions.append(outputs)

        # Concatenate all batch predictions and targets
        predictions = np.hstack(predictions)
        targets = np.hstack(targets)

        # Compute average query latency (if enabled)
        query_latency = total_time / len(targets)

        return predictions, targets, query_latency
