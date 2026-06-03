# Usage:
#   python scripts/generate_parts_file_rne.py --data_dir data/OSM_Beijing
#   python scripts/generate_parts_file_rne.py --data_dir data/OSM_Harbin
#
# Requires: pip install pymetis

import os
import sys
import argparse

# Auto-detect project root (scripts/ -> l1tilde-metric-study/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

import numpy as np
import pandas as pd
import networkx as nx
try:
    import pymetis
except ImportError:
    print("Error: pymetis is required but not installed.")
    print("Install it with: pip install pymetis")
    print("Note: pymetis requires METIS (http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)")
    sys.exit(1)

from utils.data_utils import load_graph
from utils.data_utils import print_green


def nx_to_pymetis_format(G, weight_key=None):
    n = G.number_of_nodes()
    xadj = [0]
    adjncy = []
    eweights = []
    if weight_key is None:
        print("Using uniform weights (w=1) for partitioning")
    else:
        print(f"Using edge weights with key='{weight_key}' for partitioning")

    for i in range(n):
        neighbors = list(G[i].keys())  # adjacent nodes
        weights = [int(G[i][j].get(weight_key, 1)) for j in neighbors]  # default weight = 1

        adjncy.extend(neighbors)
        eweights.extend(weights)
        xadj.append(len(adjncy))

    xadj = np.array(xadj, dtype=np.int32)
    adjncy = np.array(adjncy, dtype=np.int32)
    eweights = np.array(eweights, dtype=np.int32)
    return xadj, adjncy, eweights


def generate_hierarchical_partitions(xadj, adjncy, eweights, nlevels, recursive=False):
    n = xadj.shape[0] - 1
    parts = np.ones((n, len(nlevels)), dtype=np.int32)  # all nodes in one partition
    print(f"Graph has {n} nodes")
    print(f"Using recursive bisection: {recursive}")
    print(f"nlevels: {nlevels}")
    for _ in range(len(nlevels)):
        print(f"Partitioning level {_} with {nlevels[_]} partitions")
        assert nlevels[_] <= n, f"Number of partitions {nlevels[_]} cannot exceed number of nodes {n}"

        if nlevels[_] == n:
            # Last level, no partitioning
            print("Last level reached, assigning each node to its own partition")
            parts[:, _] = np.arange(0, n)
            continue

        # Use recursive bisection for better partitioning
        edgecuts, membership = pymetis.part_graph(nlevels[_], xadj=xadj, adjncy=adjncy, eweights=eweights, recursive=recursive)

        parts[:, _] = np.array(membership).reshape(-1)

    return parts


def generate_parts(data_name, data_dir):
    # Load graph
    G = load_graph(data_dir)

    # Convert to pymetis format
    xadj, adjncy, eweights = nx_to_pymetis_format(G, weight_key=None)
    print(f"Graph has {xadj.shape[0]-1} nodes and {len(adjncy)//2} edges")
    print(f"xadj.shape: {xadj.shape}, adjncy.shape: {adjncy.shape}, eweights.shape: {eweights.shape}")

    # Create levels
    nlevels = [4**i for i in range(2, 20) if 4**i <= xadj.shape[0] - 1]
    nlevels.append(xadj.shape[0] - 1)  # Ensure the last level is the number of nodes
    print(f"Using hierarchical levels: {nlevels}")

    # Generate hierarchical partitions
    parts_ = generate_hierarchical_partitions(xadj, adjncy, eweights, nlevels, recursive=False)

    # (Optional) Print some statistics about the partitions
    for level_i in range(len(nlevels)):
        membership = parts_[:, level_i]
        print(f"Level {level_i}: min index={np.min(membership)}, max index={np.max(membership)}, unique partitions={len(np.unique(membership))}")
    print(f"Parts shape: {parts_.shape}")

    # (Optional) Check containment property: each node in level i+1 maps to only one node in level i
    for level in range(len(nlevels) - 1, 0, -1):
        mapping = {}
        for node_idx in range(parts_.shape[0]):
            higher = parts_[node_idx, level]
            lower = parts_[node_idx, level - 1]
            if higher not in mapping:
                mapping[higher] = set()
            elif lower not in mapping[higher]:
                print("Containment property violated:")
                print(f"Idx: {node_idx}, Level: {level}, Lower part: {lower}, Higher part: {higher}")
                print(f"Node id: {higher} maps to multiple parents: {mapping[higher]} and {lower}")
                print("Moving on to next level check...")
                break
            mapping[higher].add(lower)

    # Save the parts to a file
    file_name = os.path.join(data_dir, data_name + ".parts")
    print_green(f"Saving parts: {file_name}")
    np.savetxt(file_name, parts_, fmt='%d')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate hierarchical partitions for RNE datasets')
    parser.add_argument('--data_dir', type=str, default=os.path.join(PROJECT_DIR, 'data', 'OSM_Harbin_Small'),
                        help='Path to directory containing *.nodes and *.edges files')

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Data directory '{args.data_dir}' does not exist.")
        print(f"Run 'python scripts/prepare_data.py' first to download data.")
        sys.exit(1)

    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  - {arg}: {value}")

    data_name = os.path.basename(os.path.normpath(args.data_dir))
    data_dir = args.data_dir

    generate_parts(data_name, data_dir)
