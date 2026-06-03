# Usage:
#   python scripts/generate_landmark_distances.py --data_dir data/OSM_Beijing --num_landmarks 61
#   python scripts/generate_landmark_distances.py --data_dir data/OSM_Harbin_Small --num_landmarks 61

import os
import sys
import argparse

# Auto-detect project root (scripts/ -> l1tilde-metric-study/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from utils.data_utils import (
    load_graph,
    print_summary_stats,
    print_green,
    print_warning,
    select_landmarks,
    compute_landmark_distances,
)

def generate_landmark_distances(data_dir, num_landmarks=61):
    """Generate landmark distance embeddings for a dataset.

    Args:
        data_dir: Path to directory containing *.nodes and *.edges files
        num_landmarks: Number of landmarks to select

    Returns:
        numpy array of shape (num_nodes, num_landmarks) — the embeddings
    """
    print(f"  Auto-generating landmark embeddings for: {data_dir}")
    G = load_graph(data_dir)

    landmarks = select_landmarks(G, num_landmarks, strategy="random")
    dist_matrix = compute_landmark_distances(G, landmarks)
    print(f"  Computed landmark distance matrix: {dist_matrix.shape}")

    # Save landmark distance embeddings
    node_attr_path = os.path.join(data_dir, f"landmark_dim{num_landmarks}.embeddings")
    print_green(f"  Saving: {node_attr_path}")
    comment = "#"
    delimiter = " "
    with open(node_attr_path, 'w') as f:
        f.write(f"{comment} Format: node_id features\n")
        for node, data in enumerate(dist_matrix):
            f.write(f"{node+1}{delimiter}{delimiter.join(map(str, data))}\n")

    # Return the embeddings in the same format as read_embedding_file
    import numpy as np
    return dist_matrix.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate landmark distance embeddings for shortest-distance datasets.")
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to directory containing *.nodes and *.edges files '
                             '(e.g., data/OSM_Harbin_Small)')
    parser.add_argument('--num_landmarks', type=int, default=61,
                        help='Number of landmarks to select')
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(PROJECT_DIR, 'data', 'OSM_Harbin_Small')
        print(f"No --data_dir provided, using default: {args.data_dir}")

    if not os.path.exists(args.data_dir):
        print(f"Data directory '{args.data_dir}' does not exist.")
        print(f"Run 'python scripts/prepare_data.py' first to download data.")
        sys.exit(1)

    generate_landmark_distances(args.data_dir, args.num_landmarks)
    print_green(f"Done!")
