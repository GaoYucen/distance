import os
import sys
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
import multiprocessing
from unittest.mock import MagicMock
from scipy import sparse

# --- 1. Fix for SciPy compatibility (Environment Patching) ---
# Error: "C function scipy.spatial._qhull._barycentric_coordinates has wrong signature"
try:
    import scipy.interpolate
except Exception as e:
    print(f"⚠️ Warning: Detected broken scipy.interpolate ({e}).")
    sys.modules['scipy.interpolate'] = MagicMock()

# --- 2. Fix for SciPy linalg compatibility ---
import scipy.linalg
try:
    if not hasattr(scipy.linalg, 'triu'):
        scipy.linalg.triu = np.triu
    if not hasattr(scipy.linalg, 'tril'):
        scipy.linalg.tril = np.tril
except Exception as e:
    logging.warning(f"Failed to patch scipy.linalg: {e}")

# --- 3. Fix for NumPy/PecanPy compatibility (CRITICAL FIX) ---
# Error: "ValueError: The truth value of an array with more than one element is ambiguous"
# Cause: PecanPy calls np.where(sparse_matrix != 0). Newer NumPy versions reject this usage.
# Fix: Intercept np.where calls. If the input is a sparse matrix, call .nonzero() instead.
_orig_where = np.where
def patched_where(condition, *args, **kwargs):
    if sparse.issparse(condition):
        # For a sparse matrix, .nonzero() returns the (row, col) indices of non-zero elements,
        # which is exactly what np.where(condition) used to do for booleans.
        return condition.nonzero()
    return _orig_where(condition, *args, **kwargs)

# Apply the patch
np.where = patched_where
print("✅ Patched numpy.where to handle sparse matrices for PecanPy compatibility.")

# Import libraries AFTER patching
from gensim.models import Word2Vec
from pecanpy import pecanpy

# Configure Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)

def run_node2vec_optimized(
    data_dir, 
    city, 
    output_dir, 
    dimensions=64, 
    walk_length=80, 
    num_walks=10, 
    window_size=10, 
    p=1.0, 
    q=1.0, 
    n_jobs=-1
):
    """
    Optimized Node2Vec pipeline using PecanPy (SparseOTF)
    """
    if output_dir is None:
        output_dir = os.path.join(data_dir, city)
    
    os.makedirs(output_dir, exist_ok=True)
    
    nodes_file = os.path.join(data_dir, f"{city}", f"nodes_lscc.txt")
    edges_file = os.path.join(data_dir, f"{city}", f"edges_lscc.txt")
    
    # 1. Load Data
    # ---------------------------------------------------------
    print(f"📖 Loading nodes from {os.path.basename(nodes_file)}...")
    nodes_df = pd.read_csv(nodes_file)
    node_list = nodes_df['Node'].astype(str).tolist()
    num_nodes = len(node_list)
    print(f"✅ Nodes: {num_nodes}")

    print(f"📖 Loading edges from {os.path.basename(edges_file)}...")
    edges_df = pd.read_csv(edges_file)
    
    # 2. Build Sparse CSR Matrix
    # ---------------------------------------------------------
    print("⚡ Building Sparse CSR Matrix directly...")
    
    # Map original IDs -> 0..N-1 indices
    index_map = {node: i for i, node in enumerate(node_list)}
    
    # Filter edges to ensure they connect existing nodes
    valid_edges = edges_df[
        edges_df['Node_Start'].astype(str).isin(index_map) & 
        edges_df['Node_End'].astype(str).isin(index_map)
    ].copy()
    
    if len(valid_edges) < len(edges_df):
        print(f"⚠️ Warning: Ignored {len(edges_df) - len(valid_edges)} edges containing unknown nodes.")

    # Prepare CSR data
    row_indices = valid_edges['Node_Start'].astype(str).map(index_map).values
    col_indices = valid_edges['Node_End'].astype(str).map(index_map).values
    data = valid_edges['Length'].values 
    
    adj_mat = sparse.csr_matrix(
        (data, (row_indices, col_indices)), 
        shape=(num_nodes, num_nodes)
    )
    # Sorting indices is often required by C++ extensions in graph libs
    adj_mat.sort_indices()
    
    # 3. Initialize PecanPy
    # ---------------------------------------------------------
    workers = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    print(f"🧠 Initializing PecanPy (SparseOTF, p={p}, q={q})...")
    
    # Use the classmethod from_mat directly.
    # The 'node_list' provides the ID mapping for the matrix rows.
    # The 'np.where' patch ensures this call doesn't crash internally.
    g = pecanpy.SparseOTF.from_mat(
        adj_mat, 
        node_list, 
        p=p, 
        q=q, 
        workers=workers, 
        verbose=True
    )
    
    # 4. Generate Walks & Train
    # ---------------------------------------------------------
    print(f"🚶 Generating walks (L={walk_length}, N={num_walks})...")
    walks = g.simulate_walks(num_walks=num_walks, walk_length=walk_length)
    
    print("🔥 Training Word2Vec model...")
    model = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,  # Skip-gram
        workers=workers,
        epochs=5
    )
    
    # 5. Save Results
    # ---------------------------------------------------------
    output_model_file = os.path.join(output_dir, f"{city}_node2vec.model")
    output_embed_file = os.path.join(output_dir, f"{city}_node2vec.emb")
    
    print(f"💾 Saving model to {output_model_file}...")
    model.save(output_model_file)
    
    print("💾 Extracting and saving embeddings...")
    embeddings = {}
    
    count = 0
    # Retrieve vectors using original IDs (PecanPy handles the mapping back to strings)
    for node_id in node_list:
        if node_id in model.wv:
            embeddings[node_id] = model.wv[node_id]
            count += 1
            
    print(f"📊 Extracted {count}/{num_nodes} embeddings.")
            
    with open(output_embed_file, 'wb') as f:
        pickle.dump(embeddings, f)
        
    print(f"✅ Done! Embeddings saved to {output_embed_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument("--city", default="beijing", type=str)
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("--n-jobs", default=-1, type=int)
    
    args = parser.parse_args()
    
    print(f"🚀 [Optimized] Processing {args.city} with {args.n_jobs} workers...")
    
    run_node2vec_optimized(
        data_dir=args.data_dir,
        city=args.city,
        output_dir=args.output_dir,
        dimensions=args.dim,
        n_jobs=args.n_jobs
    )