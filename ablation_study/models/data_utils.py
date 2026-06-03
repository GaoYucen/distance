# import libraries
import os
import sys
import json
import csv
import gzip
import glob
import random
import time
import requests
from zipfile import ZipFile

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


#################################################
# General Utilities
#################################################

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

## Function to print text in green color
def print_green(text):
    """Print text in green color."""
    print(f"{bcolors.OKGREEN}{text}{bcolors.ENDC}")

## Function to print text in yellow color
def print_warning(text):
    """Print warning text in yellow color."""
    print(f"{bcolors.WARNING}{text}{bcolors.ENDC}")

def print_error(text):
    """Print error text in red color."""
    print(f"{bcolors.FAIL}{text}{bcolors.ENDC}")

def detect_delimiter(sample_data):
    """
    Detect the delimiter used in a CSV formatted string.
    """
    try:
        dialect = csv.Sniffer().sniff(sample_data)
        print(f"Detected delimiter: `{dialect.delimiter}`")
        return dialect.delimiter
    except:
        raise ValueError("Could not detect delimiter.")

def detect_delimiter_file(file_path):
    # Detect delimiter by reading first 128 lines
    with open(file_path, 'r') as f:
        sample_data = ""
        for i, line in enumerate(f):
            if i < 128:
                sample_data += line
            else:
                break
    delimiter = detect_delimiter(sample_data)
    return delimiter

## Function to set seeds for reproducibility
def seed_everything(seed):
    """Set seeds for reproducibility.

    Args:
        seed (int): Seed value for random number generators
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set seeds for NumPy if available
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass  # Skip if NumPy is not available

    # Set seeds for PyTorch if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # Skip if PyTorch is not available

    print(f"Seed set to {seed}")

## Function to get the number of workers for pytorch dataloaders
def get_num_workers():
    # Get number of cores available to this process
    if hasattr(os, 'sched_getaffinity'):
        # For SLURM jobs, use sched_getaffinity to get the number of available cores
        num_workers = len(os.sched_getaffinity(0))
    elif sys.platform == "darwin":
        # For macOS, mps backend has some limitations with multiprocessing
        # and may not work well with multiple workers.
        # Set num_workers to default value (0) to avoid issues
        num_workers = 0
    else:
        # Fallback to total CPU count
        num_workers = os.cpu_count()
    # Cap num_workers to avoid excessive I/O contention on high-core-count machines
    num_workers = min(num_workers, 16)
    # num_workers = num_workers // 2
    # NOTE: sometimes increasing the number of workers may decrease performance, so be careful!!
    return num_workers

def read_parts_file(dir_name, data_name, delimiter=None, comment='#'):
    file_name = os.path.join(dir_name, f'{data_name}.parts')

    # Check if parts file exists
    if not os.path.exists(file_name):
        print_green(f"Parts file not found: {file_name}. Proceeding without parts information.")
        return None

    # Detect delimiter (if not provided)
    if delimiter is None:
        delimiter = detect_delimiter_file(file_name)

    # Read parts file
    print_green(f"Reading parts file: {file_name}")
    parts = np.array(pd.read_csv(file_name, sep=delimiter, header=None, comment=comment, dtype=np.int32))

    # (Optional) Print summary statistics of parts
    print(f"Parts shape: {parts.shape}")
    for i in range(parts.shape[1]):
        print(f"Parts column {i}: min={parts[:, i].min()}, max={parts[:, i].max()}, unique={len(np.unique(parts[:, i]))}")

    return parts

def read_embedding_file(file_name, delimiter=None, comment='#'):
    if file_name is None:
        print_warning("Warning: No embedding file provided.")
        return None
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Embedding file `{file_name}` not found.")
    node_embeds = []
    if delimiter is None:
        delimiter = detect_delimiter_file(file_name)
    print_green(f"Reading embeddings: {file_name}")
    print_warning("Warning: The node ids are left-shifted by 1 (i.e., node ids start from `0 to n-1` instead of `1 to n`) in the loaded graph.")
    with open(file_name, 'r') as f:
        for line in f:
            if line.startswith(comment):
                continue
            parts = line.strip().split(delimiter)
            node = int(parts[0]) - 1  # Left-shift node id by 1
            features = list(map(float, parts[1:]))
            node_embeds.append([node] + features)
    # Sort by node ID
    node_embeds = sorted(node_embeds, key=lambda x: x[0])

    # Assert that node ids are 0 to n-1 (0-indexed)
    for idx, i in enumerate(node_embeds):
        assert idx == i[0], f"Mismatch {idx} != {i[0]}."

    # Convert to numpy array
    node_embeds = np.array(node_embeds).astype(np.float32)

    # Exclude first column which contains node IDs
    node_embeds = node_embeds[:, 1:]

    # (Optional) Print shape of embeddings
    print(f"  - Embeddings shape: {node_embeds.shape}")

    return node_embeds

def ensure_landmark_embeddings(data_dir, num_landmarks=61):
    """Get landmark distance embeddings, auto-generating if needed.

    Checks for `landmark_dim{num_landmarks}.embeddings` in data_dir.
    If missing, computes shortest-path distances from random landmarks
    and saves the file.  Returns (num_nodes, num_landmarks) float32 array.

    Args:
        data_dir: Path to directory with *.nodes and *.edges files.
        num_landmarks: Number of landmark nodes to select.

    Returns:
        np.ndarray of shape (num_nodes, num_landmarks), or None on failure.
    """
    embedding_path = os.path.join(data_dir, f"landmark_dim{num_landmarks}.embeddings")
    if os.path.exists(embedding_path):
        return read_embedding_file(embedding_path)
    # Auto-generate
    print_warning(f"Landmark embeddings not found: {embedding_path}")
    print_warning("Auto-generating... (this runs once per city)")
    try:
        # Import here to avoid circular dependency at module load
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_landmark_distances",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scripts", "generate_landmark_distances.py"))
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load generate_landmark_distances.py")
        gen_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_module)
        return gen_module.generate_landmark_distances(data_dir, num_landmarks)
    except Exception as e:
        print_error(f"Failed to auto-generate landmark embeddings: {e}")
        raise

def ensure_parts_file(data_dir, data_name):
    """Get .parts file for RNE, auto-generating with METIS if needed.

    Args:
        data_dir: Path to directory with *.nodes and *.edges files.
        data_name: Dataset name (basename of data_dir).

    Returns:
        np.ndarray of shape (num_nodes, num_levels), or None on failure.
    """
    parts = read_parts_file(data_dir, data_name)
    if parts is not None:
        return parts
    # Auto-generate
    print_warning(f"Parts file not found for {data_name}. Auto-generating with METIS...")
    print_warning("This runs once per city and requires pymetis.")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_parts_file_rne",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scripts", "generate_parts_file_rne.py"))
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load generate_parts_file_rne.py")
        gen_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_module)
        gen_module.generate_parts(data_name, data_dir)
        return read_parts_file(data_dir, data_name)
    except ImportError as e:
        print_error(f"Cannot auto-generate .parts: {e}")
        print_error("Install pymetis:  pip install pymetis")
        print_error("Or generate manually: python scripts/generate_parts_file_rne.py --data_dir DATA_DIR")
        return None
    except Exception as e:
        print_error(f"Failed to auto-generate .parts: {e}")
        return None

def write_query_file(file_name, dataset, delimiter=',', comment='#'):
    print_green(f"Saving dataset: {file_name}")
    print_warning("Warning: The node ids are right-shifted by 1 (i.e., node ids start from `1 to n` instead of `0 to n-1`) in the saved files.")
    with open(file_name, "w") as f:
        f.write(f"{comment} Format: source{delimiter}target{delimiter}distance\n")
        for idx in range(len(dataset)):
            u, v, d = dataset[idx]
            f.write(f"{u+1}{delimiter}{v+1}{delimiter}{d}\n")  # Right-shift node id by 1

def read_query_file(file_name, delimiter=None, comment='#', drop_duplicates=False, force_shift=None):
    print_green(f"Reading dataset: {file_name}")
    file_data = []
    if delimiter is None:
        delimiter = detect_delimiter_file(file_name)
    if force_shift is not None:
        shift = force_shift
        print(f"  Node index shift: {shift} (forced)")
    else:
        min_id = None
        with open(file_name, "r") as f:
            for line in f:
                if line.startswith(comment):
                    continue
                parts = line.strip().split(delimiter)
                u_raw, v_raw = int(parts[0]), int(parts[1])
                if min_id is None or u_raw < min_id:
                    min_id = u_raw
                if v_raw < min_id:
                    min_id = v_raw
                if min_id == 0:
                    break
        shift = min_id if min_id is not None else 1
        print(f"  Auto-detected node index shift: {shift} (min node id = {min_id})")
    print_warning(f"Node ids are shifted by {shift} (0-indexed after shift).")
    with open(file_name, "r") as f:
        for idx, line in enumerate(f):
            if line.startswith(comment):
                continue
            parts = line.strip().split(delimiter)
            u = int(parts[0]) - shift
            v = int(parts[1]) - shift
            distance = float(parts[2])
            if distance == 0:
                print_warning(f"Warning: Zero distance found in query file (Line {idx + 1}: {line.strip()}). Skipping entry.")
                continue
            if len(parts) >= 4:
                d_vu = float(parts[3])
                file_data.append((u, v, distance, d_vu))
            else:
                file_data.append((u, v, distance))
    print(f"  - Total number of queries: {len(file_data)}")
    # Exclude duplicate queries
    if drop_duplicates:
        print("  - Removing duplicate queries...")
        file_data = list(set(file_data))  # Remove exact duplicates
        print(f"  - Number of queries after removing duplicates: {len(file_data)}")
    return file_data

def download_file(file_name, url, referer, max_retries=10):
    """
    Downloads a file from the given URL and saves it to the specified file name using requests.
    Retries up to max_retries times if an error is encountered.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": referer
    }
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {url} (Attempt {attempt})")
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            with open(file_name, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded: {file_name}")
            return True
        except Exception as e:
            print_warning(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries + 1:
                wait_time = random.randint(1, 10)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print_warning(f"Failed to download file from {url} after {max_retries} attempts.")
                sys.exit(1)

## Function to read workload data from files
def read_workload_data(data_name, dir_name, convert_to_int=True):
    # Define file paths
    node_file = os.path.join(dir_name, f"Workload_{data_name}", f"{data_name}.node")
    edge_file = os.path.join(dir_name, f"Workload_{data_name}", f"{data_name}.edgelist")
    groundtruth_file = os.path.join(dir_name, f"Workload_{data_name}", f"{data_name}.groundtruth")

    # Read edge data
    print_green(f"Reading edge data: {edge_file}")
    edgelist = pd.read_csv(edge_file, header=None, names=["START_NODE", "END_NODE", "LENGTH"]) \
                    .drop_duplicates(subset=['START_NODE', 'END_NODE']) \
                    .sort_values(by=['START_NODE', 'END_NODE', 'LENGTH']) \
                    .reset_index(drop=True)
    print(f"Number of edges found: {len(edgelist)}")

    # Display the first few rows of the edgelist
    print(edgelist.head())

    # Create a graph from the edge list
    G = nx.from_pandas_edgelist(edgelist,
                                source="START_NODE",
                                target="END_NODE",
                                edge_attr="LENGTH",
                                create_using=nx.Graph())  # nx.Graph() creates an undirected graph

    # Convert edge weight (LENGTH) to int type for a fair comparison with non-ML indexes
    # (Since non-ML indexes can't process floats)
    if convert_to_int:
        print_warning("Warning: Converting edge lengths to integers.")
        print("Before conversion:")
        print(edgelist.head())
        edgelist['LENGTH'] = edgelist['LENGTH'].astype(int)
        print("After conversion:")
        print(edgelist.head())

    # Check if the graph is undirected
    # TODO: removing this check for now, as we assume the graph is undirected
    # assert _has_reverse_edges(edgelist) == False, "Edgelist is has some reverse edges; We assume the graph is undirected"
    # print("Finished checking: graph is undirected")

    # Check if the graph has self loops
    assert _has_self_loops(edgelist) == False, "Edgelist has self loops; We assume the graph has no self loops"
    print("Finished checking: graph has no self loops")

    # Add metadata to the graph
    G.graph['data_name'] = data_name

    # Read node data
    print_green(f"Reading node data: {node_file}")
    node_attr = pd.read_csv(node_file, header=None, names=["START_NODE", "XCoord", "YCoord"]) \
                    .drop_duplicates(subset=["START_NODE"]) \
                    .set_index("START_NODE") \
                    .to_dict('index')
    print(f"Number of nodes attributes found: {len(node_attr)}")

    # Set node attributes in the graph
    nx.set_node_attributes(G, node_attr)

    # Read ground truth queries
    print_green(f"Reading queries data: {groundtruth_file}")
    groundtruth_df = pd.read_csv(groundtruth_file, header=None, names=["START_NODE", "END_NODE", "distance"])
    print(f"Number of ground truth queries found: {len(groundtruth_df)}")

    return G, groundtruth_df

#################################################
# DIMACS Data I/O Utilities
#################################################

## Function to download data from DIMACS
def download_dimacs_data(dir_name, data_name):
    """
    Downloads data from DIMACS to the specified directory and returns the path to the downloaded zip file.
    """
    # Create the directory if it doesn't exist
    dir_name = os.path.join(dir_name, data_name, "raw")
    os.makedirs(dir_name, exist_ok=True)

    # Load the URLs from the JSON file
    all_urls = json.load(open('./utils/dimacs_urls.json', 'r'))
    referer = "https://www.diag.uniroma1.it/challenge9/download.shtml"
    graph_file_name = os.path.join(dir_name, f"USA-road-d.{data_name}.gr.gz")
    coordinates_file_name = os.path.join(dir_name, f"USA-road-d.{data_name}.co.gz")

    # Download the graph file
    url = all_urls.get(data_name).get("graph_url")
    download_file(graph_file_name, url, referer)

    # Download the coordinates file
    url = all_urls.get(data_name).get("coordinates_url")
    download_file(coordinates_file_name, url, referer)

    return graph_file_name, coordinates_file_name

## Function to read dimacs data from files
def read_dimacs_data(edge_file_name, node_file_name):
    """
    Reads DIMACS .co and .gr files and returns a graph and edge DataFrame.
    """
    # Get data_name from the edge/node file name, e.g., "USA-road-d.NY.gr.gz" --> "NY"
    data_name = os.path.basename(edge_file_name).split('.')[2]

    # Read edge data
    print_green(f"Reading edge data: {edge_file_name}")
    edges = []
    with gzip.open(edge_file_name, 'rt') as f:
        for line in f:
            if line.startswith('a'):
                _, source, target, weight = line.strip().split()
                edges.append((int(source), int(target), int(weight)))  # Convert to int
    edgelist = pd.DataFrame(edges, columns=['START_NODE', 'END_NODE', 'weight'])
    print(f"Number of edges found: {len(edgelist)}")
    print(edgelist.dtypes)

    # Display the first few rows of the edgelist
    print(edgelist.head())

    # Create a graph from the edge list
    G = nx.from_pandas_edgelist(edgelist,
                                source="START_NODE",
                                target="END_NODE",
                                edge_attr="weight",
                                create_using=nx.Graph())  # nx.Graph() creates an undirected graph

    # Read node data
    print_green(f"Reading node data: {node_file_name}")
    node_attr = {}
    with gzip.open(node_file_name, 'rt') as f:
        for line in f:
            if line.startswith('v'):
                _, node_id, longitude, latitude = line.strip().split()
                node_attr[int(node_id)] = {'XCoord': float(longitude), 'YCoord': float(latitude)}
    print(f"Number of nodes attributes found: {len(node_attr)}")

    # Set node attributes in the graph
    for node in node_attr:
        data = node_attr[node]
        G.nodes[node]['feature'] = [data['XCoord'], data['YCoord']]

    # Add metadata to the graph
    G.graph['data_name'] = data_name

    return G

#################################################
# Figshare Data I/O Utilities
#################################################

## Function to download data from Figshare
def download_figshare_data(dir_name, data_name):
    """
    Downloads data from Figshare to the specified directory and returns the path to the downloaded zip file.
    """
    # Create the directory if it doesn't exist
    dir_name = os.path.join(dir_name, data_name, "raw")
    os.makedirs(dir_name, exist_ok=True)

    # Load the URLs from the JSON file
    all_urls = json.load(open('./utils/figshare_urls.json', 'r'))
    zip_file_name = os.path.join(dir_name, f"{data_name}.zip")

    # Download the zip file
    url = all_urls.get(f"{data_name}.zip")
    referer = "https://figshare.com/"
    download_file(zip_file_name, url, referer)

    return zip_file_name

## Function to read the Figshare data
def read_figshare_data(zip_file_name, convert_to_int=True):
    # Define the path to the zip file and the CSV file inside it
    data_name = os.path.basename(zip_file_name).replace('.zip', '')
    csv_file_name = f"{data_name}_Edgelist.csv"
    print_green(f"Reading data: {zip_file_name}")

    # Handle discrepancy in the data name for "Shanghai" and "Guangzhou"
    if data_name == "Shanghai":
        csv_file_name = "Shangai_Edgelist.csv"
    elif data_name == "Guangzhou":
        csv_file_name = "Guangzhou_dgelist.csv"

    # Read the zip file and extract the CSV file
    with ZipFile(zip_file_name, 'r') as zip:
        # Remove duplicates edges and keep the shortest edge
        edgelist = pd.read_csv(zip.open(csv_file_name)) \
                    .drop_duplicates(subset=['START_NODE', 'END_NODE']) \
                    .rename(columns={"LENGTH": "weight"}) \
                    .sort_values(by=['START_NODE', 'END_NODE', 'weight']) \
                    .reset_index(drop=True)
    print(f"Finished reading data: {data_name}")

    # Convert edge weight to int type for a fair comparison with non-ML indexes
    # (Since non-ML indexes can't process floats)
    if convert_to_int:
        print_warning("Warning: Converting edge weights to integers.")
        print("Before conversion:")
        print(edgelist.head())
        edgelist['weight'] = edgelist['weight'].astype(int)
        print("After conversion:")
        print(edgelist.head())

    # Set 'START_NODE' and 'END_NODE' as source and target nodes
    # Set 'weight' as edge attribute
    G = nx.from_pandas_edgelist(edgelist,
                            source='START_NODE',
                            target='END_NODE',
                            edge_attr='weight',
                            create_using=nx.Graph)  # nx.Graph() creates an undirected graph

    # Create a dictionary of node attributes (XCoord and YCoord) from the edgelist
    node_attr = edgelist[['START_NODE', 'XCoord', 'YCoord']] \
                    .drop_duplicates() \
                    .set_index('START_NODE') \
                    .to_dict('index')

    # Set node attributes in the graph
    for node in node_attr:
        data = node_attr[node]
        G.nodes[node]['feature'] = [data['XCoord'], data['YCoord']]

    # Add metadata to the graph
    G.graph['data_name'] = data_name

    return G

#################################################
# Graph I/O Utilities
#################################################

## Function to save graph data to files
def save_graph(G, dir_name, delimiter=',', comment='#'):
    """
    Saves the graph data to .edges and .nodes files in the specified directory.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Get data_name
    data_name = os.path.basename(os.path.normpath(dir_name))

    # Save the edge list
    edge_list_path = os.path.join(dir_name, f"{data_name}.edges")
    print_green(f"Saving edges: {edge_list_path}")
    print_warning("Warning: The node ids are right-shifted by 1 (i.e., node ids start from `1 to n` instead of `0 to n-1`) in the saved files.")
    with open(edge_list_path, 'w') as f:
        f.write(f"{comment} Format: source target weight\n")
        for u, v, data in G.edges(data=True):
            f.write(f"{u+1}{delimiter}{v+1}{delimiter}{data['weight']}\n")  # Right-shift node id by 1

    # Save the node attributes
    node_attr_path = os.path.join(dir_name, f"{data_name}.nodes")
    print_green(f"Saving nodes: {node_attr_path}")
    print_warning("Warning: The node ids are right-shifted by 1 (i.e., node ids start from `1 to n` instead of `0 to n-1`) in the saved files.")
    with open(node_attr_path, 'w') as f:
        f.write(f"{comment} Format: node_id features\n")
        for node, data in G.nodes(data=True):
            f.write(f"{node+1}{delimiter}{delimiter.join(map(str, data['feature']))}\n")  # Right-shift node id by 1

## Function to load graph data from files
def load_graph(dir_name, delimiter=None, comment='#', force_shift=None):
    """
    Loads the graph data from .edges and .nodes files in the specified directory.
    """
    if force_shift is not None:
        shift = force_shift
    else:
        edge_list_files = glob.glob(os.path.join(dir_name, "*.edges"))
        assert len(edge_list_files) > 0, f"Expected an `.edges` file in {dir_name}, found none."
        edge_list_path = edge_list_files[0]
        if delimiter is None:
            auto_delimiter = detect_delimiter_file(edge_list_path)
        min_id = None
        with open(edge_list_path, 'r') as f:
            for line in f:
                if line.startswith(comment):
                    continue
                parts = line.strip().split(auto_delimiter)
                u_raw, v_raw = int(parts[0]), int(parts[1])
                if min_id is None or u_raw < min_id:
                    min_id = u_raw
                if v_raw < min_id:
                    min_id = v_raw
                if min_id == 0:
                    break
        shift = min_id if min_id is not None else 1
    print(f"  Node index shift: {shift}")
    
    edge_list_files = glob.glob(os.path.join(dir_name, "*.edges"))
    assert len(edge_list_files) > 0, f"Expected an `.edges` file in {dir_name}, found none."
    edge_list_path = edge_list_files[0]
    print_green(f"Reading edges: {edge_list_path}")
    G = nx.DiGraph()  # Directed graph for OSM road networks
    if delimiter is None:
        auto_delimiter = detect_delimiter_file(edge_list_path)
    with open(edge_list_path, 'r') as f:
        for line in f:
            if line.startswith(comment):
                continue
            parts = line.strip().split(auto_delimiter)
            u, v = int(parts[0]) - shift, int(parts[1]) - shift
            weight = float(parts[2])
            G.add_edge(u, v, weight=weight)

    node_attr_files = glob.glob(os.path.join(dir_name, "*.nodes"))
    assert len(node_attr_files) > 0, f"Expected a `.nodes` file in {dir_name}, found none."
    node_attr_path = node_attr_files[0]
    print_green(f"Reading nodes: {node_attr_path}")
    if delimiter is None:
        auto_delimiter = detect_delimiter_file(node_attr_path)
    with open(node_attr_path, 'r') as f:
        for line in f:
            if line.startswith(comment):
                continue
            parts = line.strip().split(auto_delimiter)
            node = int(parts[0]) - shift
            features = list(map(float, parts[1:]))
            G.nodes[node]['feature'] = features

    data_name = os.path.basename(os.path.normpath(dir_name))
    G.graph['data_name'] = data_name

    n = G.number_of_nodes()
    min_label = min([i for i in G.nodes()])
    max_label = max([i for i in G.nodes()])
    assert (min_label == 0) and (max_label == n - 1), f"Node labels must be from 0 to n-1. However, found {n} labels from {min_label} to {max_label}."

    return G

#################################################
# Graph Preprocessing Utilities
#################################################

## Function to augment queries by perturbing source and target nodes
def augment_queries(G, queries, perturb_k=4, k_hop=1, distance_func=None, drop_duplicates=False):
    augmented_queries = []
    if distance_func is None:
        # Use a dummy distance function that returns 1 for all pairs
        distance_func = lambda u, v: 1

    print(f"Augmenting queries with perturb_k={perturb_k} and k_hop={k_hop}...")
    print("Before augmentation:")
    print("  - Number of unique queries: ", len(set([(u, v) for u, v, d in queries])))
    print("  - Number of unique nodes:   ", len(set([u for u, v, d in queries] + [v for u, v, d in queries])))
    print("  - Total number of queries:  ", len(queries))

    # Compute k-hop neighbors for unique nodes in the queries
    unique_nodes = set([u for u, v, d in queries] + [v for u, v, d in queries])
    k_hop_neighbors = {}
    k_hop_probabilities = {}
    for node in unique_nodes:
        k_hop_distances = nx.single_source_shortest_path_length(G, node, cutoff=k_hop)
        k_hop_neighbors[node] = {n: d for n, d in k_hop_distances.items() if d > 0}
        probabilities = [1 / distance for _, distance in k_hop_neighbors[node].items()]
        total = sum(np.exp(probabilities))
        probabilities = [np.exp(p) / total for p in probabilities]
        k_hop_probabilities[node] = probabilities
        k_hop_neighbors[node] = list(k_hop_neighbors[node].keys())

    for src, dst, dist in queries:
        src = int(src)
        dst = int(dst)
        # Add the original query
        augmented_queries.append((src, dst, distance_func(src, dst)))

        # For each query, perturb source and target by replacing with a random neighbor (if exists)
        for _ in range(perturb_k):
            new_src = src
            new_dst = dst
            # Perturb source
            if k_hop_neighbors[src]:
                new_src = np.random.choice(k_hop_neighbors[src], p=k_hop_probabilities[src])
            # Perturb target
            if k_hop_neighbors[dst]:
                new_dst = np.random.choice(k_hop_neighbors[dst], p=k_hop_probabilities[dst])
            # Only add if not the same as original and not self-loop
            if (new_src != src or new_dst != dst) and (new_src != new_dst):
                augmented_queries.append((new_src, new_dst, distance_func(new_src, new_dst)))

    print("After augmentation:")
    print("  - Number of unique queries: ", len(set([(u, v) for u, v, d in augmented_queries])))
    print("  - Number of unique nodes:   ", len(set([u for u, v, d in augmented_queries] + [v for u, v, d in augmented_queries])))
    print("  - Total number of queries:  ", len(augmented_queries))

    # Remove duplicates
    if drop_duplicates:
        print("Removing duplicate queries...")
        augmented_queries = list(set(augmented_queries))

    return augmented_queries

## Function to augment nodelist by adding k-perturbed neighbors
def augment_nodes(G, seed_nodes, k):
    neighbor_nodes = []
    unique_set = set(seed_nodes)
    for node in seed_nodes:
        added = set()
        visited = set([node])
        queue = [node]
        depth = 0
        while len(added) < k and queue:
            next_queue = []
            for current in queue:
                neighbors = list(G.neighbors(current))
                for neighbor in neighbors:
                    if neighbor not in unique_set and neighbor not in added:
                        neighbor_nodes.append(neighbor)
                        added.add(neighbor)
                        if len(added) == k:
                            break
                    if neighbor not in visited:
                        next_queue.append(neighbor)
                        visited.add(neighbor)
                if len(added) == k:
                    break
            queue = next_queue
            depth += 1
    return np.unique(np.concatenate([seed_nodes, neighbor_nodes]))

## Function to fuse a node with its two neighbors by summing their edge weights
def _fuse_2_degree_node(G, node):
    # Get the neighbors of the node (there will be exactly 2 neighbors)
    neighbors = list(G.neighbors(node))
    # Check if there is a direct connection between the neighbors
    if G.has_edge(neighbors[0], neighbors[1]):
        # TODO: Handle the case where there is a direct connection between the neighbors
        # Shall we keep the shortest edge or stop here? Currently, we stop here.
        return G
    # Get the edge attributes of the neighbors
    edge1 = G.get_edge_data(neighbors[0], node)
    edge2 = G.get_edge_data(neighbors[1], node)
    # Add the edge weights
    new_edge = edge1['LENGTH'] + edge2['LENGTH']
    # Add the combined edge to the graph
    G.add_edge(neighbors[0], neighbors[1], LENGTH=new_edge)
    # Remove the node from the graph
    G.remove_node(node)
    return G

# Function to remove 2-degree nodes iteratively
def remove_2_degree_nodes(G):
    G = G.copy()
    initial_nodes = G.number_of_nodes()
    inital_node_set = set(G.nodes())
    nodes_to_remove = [node for node in G.nodes() if G.degree(node) == 2]
    print(f"Number of 2-degree nodes: {len(nodes_to_remove)}")
    for node in nodes_to_remove:
        G = _fuse_2_degree_node(G, node)
    final_nodes = G.number_of_nodes()
    final_node_set = set(G.nodes())
    print(f"Number of nodes removed: {initial_nodes - final_nodes}")
    print(f"Nodes removed: {list(inital_node_set - final_node_set)[:3]} ...")
    return G

## Function to remove 1-degree nodes iteratively
def remove_1_degree_nodes(G):
    G = G.copy()
    count = 0
    removed_nodes = []
    while True:
        nodes_to_remove = [node for node in G.nodes() if G.degree(node) == 1]
        removed_nodes.extend(nodes_to_remove)  ## Same as removed_nodes += nodes_to_remove
        count += len(nodes_to_remove)
        G.remove_nodes_from(nodes_to_remove)
        if len(nodes_to_remove) == 0:
            break
    print(f"Number of nodes removed: {count}")
    print(f"Nodes removed: {removed_nodes[:3]} ...")
    return G

## Function to check if the edgelist is directed
def _is_directed(edgelist):
    ## TODO: Need to check if this implementation is correct!
    # Create a set of tuples of (source, target) nodes
    edgelist_set = set(zip(edgelist['START_NODE'], edgelist['END_NODE']))
    # Create a set of tuples of (target, source) nodes
    edgelist_set_rev = set(zip(edgelist['END_NODE'], edgelist['START_NODE']))
    # Check if the two sets are equal
    return edgelist_set != edgelist_set_rev

## Function to check if the edgelist has reverse edges
def _has_reverse_edges(edgelist):
    # Create a set of tuples of (source, target) nodes
    edgelist_set = set(zip(edgelist['START_NODE'], edgelist['END_NODE']))
    # Create a set of tuples of (target, source) nodes
    edgelist_set_rev = set(zip(edgelist['END_NODE'], edgelist['START_NODE']))
    # Check if there are any reverse edges
    return len(edgelist_set.intersection(edgelist_set_rev)) > 0

## Function to check if the edgelist has self loops
def _has_self_loops(edgelist):
    # Check if there are any self loops
    return edgelist.query('START_NODE == END_NODE').shape[0] > 0

## Function to preprocess the graph
def preprocess_graph(G, remove_1_degree=False, remove_2_degree=False):
    G = G.copy()
    print("Original graph:")
    print_summary_stats(G)

    # Take the largest connected component
    if not nx.is_connected(G):
        print("Taking the largest connected component...")
        G = G.subgraph(max(nx.connected_components(G), key=len))
        print_summary_stats(G)

    # Iteratively remove 1-degree nodes
    if remove_1_degree:
        print("Removing 1-degree nodes...")
        G = remove_1_degree_nodes(G)
        print_summary_stats(G)

    # Iteratively remove 2-degree nodes from the graph G
    if remove_2_degree:
        print("Removing 2-degree nodes...")
        G = remove_2_degree_nodes(G)
        print_summary_stats(G)

    # Check if one-degree nodes are still present
    if len([node for node in G.nodes() if G.degree(node) == 1]) > 0:
        print_warning("Warning: One-degree nodes are still present in the graph.")

    # Check if two-degree nodes are still present
    if len([node for node in G.nodes() if G.degree(node) == 2]) > 0:
        print_warning("Warning: Two-degree nodes are still present in the graph.")

    # Re-index the nodes
    # Check if re-index is required
    if min(G.nodes()) != 0 or max(G.nodes()) != G.number_of_nodes() - 1:
        print_warning("Warning: Node labels are not in the range [0, n-1]. Re-indexing is required.")
        print("Re-indexing the nodes...")
        print(f"Min/Max node labels before re-indexing: {min(G.nodes())}/{max(G.nodes())}")
        G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
        print(f"Min/Max node labels after re-indexing: {min(G.nodes())}/{max(G.nodes())}")

    print("Preprocessing complete.")
    print("Processed graph:")
    print_summary_stats(G)

    return G

## Function to get the edge attributes of a graph
def get_edge_attributes(G):
    return nx.to_pandas_edgelist(G).values.astype(np.float32)

## Function to get node attributes from a graph
def get_node_attributes(G):
    node_attrs = []
    for node, data in G.nodes(data=True):
        node_attrs.append([node] + data['feature'])
    # Sort by node ID
    node_attrs = sorted(node_attrs, key=lambda x: x[0])
    # Convert to numpy array
    node_df = np.array(node_attrs, dtype=np.float32)[:, 1:]  # Exclude node ID
    return node_df

## Function to print summary statistics of a graph
def print_summary_stats(G):
    print(f"  - Node count: {G.number_of_nodes()}")
    print(f"  - Edge count: {G.number_of_edges()}")
    degrees = dict(G.degree()).values()
    print(f"  - Min/Max/Avg degree: {min(degrees)}/{max(degrees)}/{np.mean(list(degrees)):.2f}")
    weights = list(nx.get_edge_attributes(G, 'weight').values())
    print(f"  - Min/Max/Avg weight: {min(weights):.2f}/{max(weights):.2f}/{np.mean(weights):.2f}")
    if G.is_directed():
        print(f"  - No. of weakly connected components: {nx.number_weakly_connected_components(G)}")
    else:
        print(f"  - No. of connected components: {nx.number_connected_components(G)}")

#################################################
# Landmark Selection Utilities
#################################################

def select_landmarks(graph, num_landmarks, strategy="random", weight_key="weight", seed=42, subset=None, node_features=None):
    """Select landmarks based on the chosen strategy."""
    print(f"Selecting {num_landmarks} landmarks using strategy: {strategy} (weight_key='{weight_key}', seed={seed})")
    num_nodes = graph.number_of_nodes()
    random.seed(seed)

    # Basic Checks for num_landmarks
    if num_landmarks < 0:
        raise ValueError(f"Landmarks num_landmarks={num_landmarks} is negative; it should be a positive integer")
    elif num_landmarks == 0:
        raise ValueError(f"Landmarks num_landmarks={num_landmarks} is zero; it should be a positive integer")
    elif num_landmarks < 1:
        print(f"Landmarks num_landmarks={num_landmarks} is a fraction; converting it to a positive integer")
        num_landmarks = int(num_landmarks * num_nodes)
    elif num_landmarks > num_nodes:
        print(f"Warning: Landmarks num_landmarks={num_landmarks} is greater than the number of nodes {num_nodes}")
        num_landmarks = num_nodes
        print(f"Setting num_landmarks to {num_nodes}")
    elif num_landmarks is None:
        raise ValueError(f"Landmarks num_landmarks={num_landmarks} is None; it should be a positive integer")
    if subset is not None and len(subset) < num_landmarks:
        raise ValueError(f"Subset size {len(subset)} is smaller than the number of landmarks {num_landmarks} to select.")

    # Select landmarks based on the strategy
    if strategy == "degree":
        # Sort nodes by descending degree
        landmarks_sorted = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    elif strategy.startswith("betweenness"):
        # Sort nodes by betweenness centrality
        centrality = nx.betweenness_centrality(graph, k=min(100, graph.number_of_nodes()), weight=weight_key, seed=seed)
        reverse = strategy.endswith("high")  # [high|low] strategy
        landmarks_sorted = sorted(centrality.items(), key=lambda x: x[1], reverse=reverse)
    elif strategy.startswith("closeness"):
        # Sort nodes by closeness centrality
        centrality = nx.closeness_centrality(graph, distance=weight_key)
        reverse = strategy.endswith("high")  # [high|low] strategy
        landmarks_sorted = sorted(centrality.items(), key=lambda x: x[1], reverse=reverse)
    elif strategy == "eigenvector":
        # Sort nodes by descending eigenvector centrality
        centrality = nx.eigenvector_centrality(graph, weight=weight_key)
        landmarks_sorted = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    elif strategy == "pagerank":
        # Sort nodes by descending PageRank score
        centrality = nx.pagerank(graph, alpha=0.85, max_iter=100, weight=weight_key)
        landmarks_sorted = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    elif strategy == "katz":
        # Sort nodes by descending Katz centrality (using numpy-based method)
        centrality = nx.katz_centrality_numpy(graph, weight=weight_key)
        landmarks_sorted = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    elif strategy == "harmonic":
        # Sort nodes by descending harmonic centrality
        centrality = nx.harmonic_centrality(graph, distance=weight_key)
        landmarks_sorted = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    elif strategy == "random":
        # Randomly select landmarks from the nodes without replacement.
        # Exclude nodes with out_degree == 0: in a directed graph they cannot
        # reach any other node, which breaks the min-aggregation formula.
        all_nodes = [node for node in graph.nodes() if graph.out_degree(node) > 0]
        if len(all_nodes) < num_landmarks:
            print(f"Warning: only {len(all_nodes)} nodes with out_degree>0 available for {num_landmarks} landmarks")
            all_nodes = list(graph.nodes())  # fallback
        landmarks_sorted = [(node, 0) for node in all_nodes]  # Dummy centrality value (0) for uniformity with other strategies
        random.shuffle(landmarks_sorted)
    elif strategy == "kmeans":
        # Use KMeans clustering to select landmarks
        if node_features is None:
            raise ValueError("Node features must be provided for kmeans strategy.")
        print(f"Node features provided with shape: {node_features.shape}")
        # Filter out nodes with out_degree == 0 (same as random strategy):
        # in a directed graph they cannot reach any other node, which breaks
        # the min-aggregation formula (distance matrix stays 0).
        # Filter out nodes with out_degree == 0 (same reason as random strategy):
        # in a directed graph they cannot reach other nodes, making distance matrix = 0.
        if subset is not None:
            candidate_nodes = subset
            candidate_features = np.array(node_features)[subset]
        else:
            candidate_nodes = [n for n in range(len(node_features)) if graph.out_degree(n) > 0]
            if len(candidate_nodes) < num_landmarks:
                print(f"Warning: only {len(candidate_nodes)} nodes with out_degree>0 for {num_landmarks} landmarks, using all nodes")
                candidate_nodes = list(range(len(node_features)))
            candidate_features = np.array(node_features)[candidate_nodes]
        kmeans = KMeans(n_clusters=num_landmarks, random_state=seed)
        kmeans.fit(candidate_features)
        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, candidate_features)
        landmarks_sorted = [(candidate_nodes[i], 0) for i in closest_indices]  # Dummy centrality value (0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # TODO: there may be some bug with subset logic in this function
    # If subset is provided, filter landmarks to be within the subset
    if subset is not None:
        subset = set(subset)
        print(f"Filtering landmarks to be within the provided subset of size {len(subset)}")
        landmarks_sorted = [(lm, _) for lm, _ in landmarks_sorted if lm in subset]

    # Select top-K landmarks
    landmarks = [node for node, _ in landmarks_sorted[:num_landmarks]]

    return landmarks

def compute_landmark_distances(graph, landmarks, weight_key="weight"):
    """
    Construct a matrix of shape (n_nodes, n_landmarks) such that the (i,j)-th entry is the
    shortest path distance from node i to the j-th landmark.
    """
    num_nodes = len(graph.nodes())
    num_landmarks = len(landmarks)
    matrix = np.zeros((num_nodes, num_landmarks), dtype=np.float32)

    for j, landmark in enumerate(landmarks):
        # Compute shortest path distances from the current landmark.
        lengths = nx.single_source_dijkstra_path_length(graph, landmark, weight=weight_key)
        for node, dist in lengths.items():
            # Distance of `node` to `landmark` (and vice versa) is `dist`
            matrix[node, j] = dist

    return matrix

#################################################
# Miscellaneous Utilities
#################################################

## Function to print the size of the distance matrix
def size_of_distance_matrix(n):
    size = n*n*4/(1024*1024)
    print(f"Size of distance matrix ({n}x{n}): {size:.2f} MB (assuming float32, i.e., 4 bytes for each element)")
