# import libraries
import os
import glob
from tqdm import tqdm
import networkx as nx

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from utils.data_utils import (
    print_green,
    print_warning,
    read_query_file,
    write_query_file,
)


## Function to get the available device (MPS or GPU or CPU)
def get_available_device():
    device = None
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device

## Function to print device information
def print_device_info(device):
    if device == 'cuda':
        print(f"Device Info: {torch.cuda.get_device_name(0)}")
    elif device == 'mps':
        print(f"Device Info: MPS (Apple Silicon GPU)")
    else:
        print(f"Device Info: CPU")

## Function to get optimizer
def get_optimizer(optimizer_type, model, learning_rate):
    optimizer = None
    has_trainable_params = any(p.requires_grad for p in model.parameters())

    if not has_trainable_params:
        print("Model has no trainable parameters. Skipping optimizer initialization.")
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    return optimizer

## Function to get criterion
def get_criterion(loss_function, model):
    criterion = None
    has_trainable_params = any(p.requires_grad for p in model.parameters())

    if not has_trainable_params:
        print("Model has no trainable parameters. Skipping criterion initialization.")
    elif loss_function == 'mse':
        criterion = nn.MSELoss()
    elif loss_function == 'mae':
        criterion = nn.L1Loss()
    elif loss_function == 'smoothl1':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")
    return criterion


#################################################
# Torch I/O Utilities
#################################################

## Function to save a dataset to a file
def save_dataset(train_dataset, test_dataset, dir_name, data_name, delimiter=',', comment='#'):
    """
    Save the train and test datasets to a file.
    """
    # Create the directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Save the train dataset to a file
    file_name = os.path.join(dir_name, f"{data_name}_train.queries")
    write_query_file(file_name, train_dataset, delimiter=delimiter, comment=comment)

    # Save the test dataset to a file
    file_name = os.path.join(dir_name, f"{data_name}_test.queries")
    write_query_file(file_name, test_dataset, delimiter=delimiter, comment=comment)

## Function to load a dataset from a file
def load_dataset(dir_name, delimiter=None, comment='#', test_size=0.2, seed=42, replicate_test=False, target_test_size=1_000_000, drop_duplicates=False, force_shift=None, return_reverse=False):
    """
    Load the train and test datasets from a file.
    """
    file_names = glob.glob(os.path.join(dir_name, "*.queries"))
    file_names = [os.path.normpath(fn) for fn in file_names]
    assert len(file_names) > 0, f"No `.queries` file found in {dir_name}, expected a file ending with `.queries`"

    if len(file_names) == 1:
        if not file_names[0].endswith(".queries"):
            raise ValueError("Expected a single `.queries` file")
        data_file = file_names[0]
        full_data = read_query_file(data_file, delimiter=delimiter, comment=comment, drop_duplicates=drop_duplicates, force_shift=force_shift)
        train_data, test_data = train_test_split(full_data, test_size=test_size, random_state=seed)
    elif len(file_names) == 2:
        train_file = None
        test_file = None
        for f in file_names:
            if f.endswith("_train.queries"):
                train_file = f
            elif f.endswith("_test.queries"):
                test_file = f
            else:
                raise ValueError("Expected `*_train.queries` and `*_test.queries` files")

        train_data = read_query_file(train_file, delimiter=delimiter, comment=comment, drop_duplicates=drop_duplicates, force_shift=force_shift)
        test_data = read_query_file(test_file, delimiter=delimiter, comment=comment, drop_duplicates=drop_duplicates, force_shift=force_shift)
    else:
        raise NotImplementedError("Loading datasets from more than two .queries files is not supported.")

    # Create torch datasets
    train_dataset = WorkloadDataset(train_data, return_reverse=return_reverse)
    test_dataset = WorkloadDataset(test_data, replicate=replicate_test, target_size=target_test_size)

    return train_dataset, test_dataset

## Function to save a model to a file
def save_model(model, model_name, data_name, data_strategy, dir_name, metadata=None):
    """
    Save the trained model to a file.
    """
    # Create the directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Save the model
    file_name = os.path.join(dir_name, f"{model_name}_{data_name}_{data_strategy}.pt")
    print_green(f"Saving model: {file_name}")

    # Build the dictionary to save
    dict_to_save = {'model_state_dict': model.state_dict()}
    if metadata is not None:
        dict_to_save['metadata'] = metadata
    torch.save(dict_to_save, file_name)

    # Print size in MB of each parameter
    print("  - Model parameters:")
    total_params = 0
    for name, param in model.state_dict().items():
        param_size = param.numel() * param.element_size() / (1024 * 1024)  # Convert to MB
        total_params += param.numel()
        print(f"    - {name:<32}: {param_size:>8.4f} MB")
    print(f"  - Total parameters: {total_params}")

    # Print model size
    # Torch stores model in best possible format, so the size on disk is equivalent to the in-memory size
    model_size = os.path.getsize(file_name) / (1024 * 1024)  # Convert to MB
    print(f"  - Model size: {model_size:.2f} MB")

## Function to load a model from a file
def load_model(model_name, data_name, data_strategy, dir_name, device='cpu'):
    """
    Load a trained model from a file.

    Args:
        model_name (str): Name of the model.
        data_name (str): Name of the dataset.
        data_strategy (str): Strategy used for the dataset.
        dir_name (str): Directory where the model is saved.
        device (str): Device to load the model on ('cpu', 'cuda', 'mps').

    Returns:
        model: The loaded model.
        metadata: Metadata associated with the model, if any.
    """
    # Load the model
    file_name = os.path.join(dir_name, f"{model_name}_{data_name}_{data_strategy}.pt")
    print_green(f"Reading model: {file_name}")
    checkpoint = torch.load(file_name, map_location=device)
    return checkpoint

## Function to save json data to a file
def save_dictionary(data, model_name, data_name, data_strategy, dir_name):
    """
    Save data dictionary to a torch file.
    """
    # Create the directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Save the data to a torch file
    file_name = os.path.join(dir_name, f"{model_name}_{data_name}_{data_strategy}_debug_info.pt")
    print_green(f"Saving debug information: {file_name}")
    torch.save(data, file_name)

    # Print model size
    file_size = os.path.getsize(file_name) / (1024 * 1024)  # Convert to MB
    print(f"File size: {file_size:.2f} MB")

## Function to load json data from a file
def load_dictionary(model_name, data_name, data_strategy, dir_name):
    """
    Load data dictionary from a torch file.
    """
    # Load the data from a torch file
    file_name = os.path.join(dir_name, f"{model_name}_{data_name}_{data_strategy}_debug_info.pt")
    print_green(f"Reading debug information: {file_name}")
    data = torch.load(file_name, map_location='cpu')
    return data

#################################################
# Torch Dataset Classes
#################################################

## Function to create a dataset for the all-pairs strategy
class AllPairsDataset(Dataset):
    def __init__(self, G, weight_key="weight"):
        self.n = len(G.nodes())
        print(f"Using all pairs strategy with n={self.n} nodes")

        # Generate all pairs (i, j) and their distances d_ij
        self.queries = []
        for i in tqdm(range(self.n)):
            # TODO: use non-ml index for larger graphs
            lengths = nx.single_source_dijkstra_path_length(G, i, weight=weight_key)
            for j, d in lengths.items():
                if i != j:
                    self.queries.append((i, j, d))
        self.queries = np.array(self.queries)
        self.D = self.queries[:, 2].reshape(-1, 1)

        # print stats
        print(f"Distance matrix of size {self.D.shape} created successfully")
        print(f"  - No. of samples: {len(self)}")
        print(f"  - Min/Max distance: {self.D.min():.2f}/{self.D.max():.2f}")
        print(f"  - Mean/Std distance: {self.D.mean():.2f}/{self.D.std():.2f}")

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        i = self.queries[idx, 0]
        j = self.queries[idx, 1]
        d_ij = self.queries[idx, 2]
        return np.int32(i), np.int32(j), np.float32(d_ij)

## Function to create a dataset for the landmark pairs strategy
class LandmarkPairsDataset(Dataset):
    def __init__(self, G, l=100, seed=42, weight_key="weight", subset_nodes=None):
        if subset_nodes is not None:
            self.n = len(subset_nodes)
            subset_nodes = [int(i) for i in subset_nodes]
            subset_nodes = set(subset_nodes)
        else:
            self.n = len(G.nodes())
        # checks for l
        if l < 0:
            raise ValueError(f"Landmarks l={l} is negative; it should be a positive number")
        if l == 0:
            raise ValueError(f"Landmarks l={l} is zero; it should be a positive number")
        if l < 1:
            print(f"Landmarks l={l} is a fraction; converting it to an absolute number")
            l = int(l * self.n)
        if l > self.n:
            print(f"Warning: Landmarks l={l} is greater than the number of nodes {self.n}")
            l = self.n
            print(f"Setting l to {self.n}")
        self.l = int(l)
        print(f"Using landmark-based strategy with n={self.n} nodes and l={l} landmarks")

        # select landmarks randomly
        np.random.seed(seed)
        if subset_nodes:
            self.landmarks = np.random.choice(list(subset_nodes), self.l, replace=False)
        else:
            self.landmarks = np.random.choice(self.n, self.l, replace=False)

        # create distance matrix using landmarks
        # self.D = np.zeros((self.l, self.n), dtype=np.float32)

        # self.src_nodes = []
        # self.dst_nodes = []
        # self.distances = []
        self.queries = []
        for i in self.landmarks:
            lengths = nx.single_source_dijkstra_path_length(G, i, weight=weight_key)
            for j in lengths.keys():
                i = int(i)
                j = int(j)
                if (subset_nodes is None) or (j in subset_nodes):
                    if i != j:
                        # self.src_nodes.append(i)
                        # self.dst_nodes.append(j)
                        # self.distances.append(lengths[j])
                        self.queries.append((i, j, lengths[j]))
        # self.src_nodes = np.array(self.src_nodes, dtype=np.int32)
        # self.dst_nodes = np.array(self.dst_nodes, dtype=np.int32)
        # self.distances = np.array(self.distances, dtype=np.float32)
        self.queries = np.array(self.queries, dtype=object)

        # print stats
        # print(f"Distance matrix of size {self.D.shape} created successfully")
        print(f"  - No. of samples: {len(self)}")
        print(f"  - Min/Max distance: {self.queries[:, 2].min():.2f}/{self.queries[:, 2].max():.2f}")
        print(f"  - Mean/Std distance: {self.queries[:, 2].mean():.2f}/{self.queries[:, 2].std():.2f}")

    def __len__(self):
        # return len(self.distances)
        return len(self.queries)

    # def get_indices(self, idx):
    #     i = idx // (self.n - 1)
    #     j = idx % (self.n - 1)
    #     # skip the landmark node
    #     if j >= self.landmarks[i]:
    #         j += 1
    #     return i, j

    def __getitem__(self, idx):
        # i, j = self.get_indices(idx)
        # landmark_i = self.landmarks[i]
        # range of int32 is -2B to 2B
        # return np.int32(landmark_i), np.int32(j), self.D[i, j]
        # return np.int32(self.src_nodes[idx]), np.int32(self.dst_nodes[idx]), np.float32(self.distances[idx])
        i, j, d_ij = self.queries[idx]
        return np.int32(i), np.int32(j), np.float32(d_ij)

# create a torch dataset of (i, j, d_ij) pairs
class RandomPairsDataset(Dataset):
    def __init__(self, G, k=1000, seed=42, weight_key="weight"):
        self.n = len(G.nodes())
        self.k = k
        if k > self.n * (self.n - 1):
            print(f"Warning: k={k} is greater than the maximum number of pairs {self.n * (self.n - 1)}")
            self.k = self.n * (self.n - 1)
            print(f"Setting k to {self.k}")
        print(f"Using random pairs strategy with n={self.n} nodes and k={k} pairs")

        # create distance matrix
        D_full = np.zeros((self.n, self.n), dtype=np.float32)
        for i in range(self.n):
            lengths = nx.single_source_dijkstra_path_length(G, i, weight=weight_key)
            D_full[i, [*lengths.keys()]] = [*lengths.values()]

        # select k random pairs
        np.random.seed(seed)
        self.random_indices = np.random.choice(self.n*(self.n-1), self.k, replace=False).astype(np.int32)

        # compute the distance for each pair
        self.D = np.zeros(self.k, dtype=np.float32)
        for idx in range(self.k):
            i, j = self.get_indices(self.random_indices[idx])
            self.D[idx] = D_full[i, j]
        print(f"Distance matrix of size {self.D.shape} created successfully")
        print(f"  - No. of samples: {len(self)}")
        print(f"  - Min/Max distance: {self.D.min():.2f}/{self.D.max():.2f}")
        print(f"  - Mean/Std distance: {self.D.mean():.2f}/{self.D.std():.2f}")

    def __len__(self):
        return self.k

    def get_indices(self, idx):
        i = idx // (self.n - 1)
        j = idx % (self.n - 1)
        # skip the node itself
        if j >= i:
            j += 1
        return i, j

    def __getitem__(self, idx):
        i, j = self.get_indices(idx)
        # range of int32 is -2B to 2B
        return np.int32(i), np.int32(j), self.D[idx]

class WorkloadDataset(Dataset):
    def __init__(self, queries, replicate=False, target_size=1_000_000, return_reverse=False):
        if replicate:
            print("Replicating queries to reach target size")
            print(f"  - Original size: {len(queries)}")
            num_copies = max(1, target_size // len(queries))
            queries = np.tile(queries, (num_copies, 1))
            print(f"  - New size:      {len(queries)}")
        self.queries = np.array(queries)
        self.D = self.queries[:, 2].astype(np.float32)
        self.has_reverse = self.queries.shape[1] > 3
        self.return_reverse = return_reverse and self.has_reverse

        print(f"Distance matrix of size {self.D.shape} created successfully")
        if self.has_reverse:
            print(f"  Directed graph: reverse distances available (return_reverse={return_reverse})")

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        u, v, d_uv = self.queries[idx, 0], self.queries[idx, 1], self.queries[idx, 2]
        if self.return_reverse:
            return np.int32(u), np.int32(v), np.float32(d_uv), np.float32(self.queries[idx, 3])
        return np.int32(u), np.int32(v), np.float32(d_uv)
