# DistNet: Road Network Distance Estimation (Branch: bigmap)

DistNet is a deep learning framework designed for efficient and accurate shortest-path distance estimation in road networks. It combines structural node embeddings (Node2Vec/Dist2Vec) with geographic coordinates and utilizes an improved MLP architecture to predict distances, with specific optimizations for directed graphs and large-scale networks.

## Project Structure

- `code/`: Core implementation of the model, training loops, and preprocessing logic.
  - `preprocess/`: Scripts for generating node embeddings (Node2Vec variants).
- `scripts/`: Utility scripts for data format conversion.
- `data/`: Storage for raw datasets and preprocessed outputs (organized by city).
- `param/`: Directory for saved model checkpoints (`.ckpt`).
- `SDM_OPTIMIZATION_README.md`: Detailed guide on optimizing preprocessing for large-scale graphs.

## Workflow Pipeline

### 1. Data Format Conversion & Cleaning
Convert raw node and edge data from external formats (Excel/CSV) into project-standard text files.
- **Script**: `scripts/convert_graphs.py`
- **Input**: Files containing node coordinates (lon, lat, osmid) and edge details (u, v, length).
- **Output**: 
  - `nodes.txt`, `edges.txt`: Standardized raw graph.
  - `nodes_lscc.txt`, `edges_lscc.txt`: **Largest Strongly Connected Component** (generated automatically if the raw graph is not strongly connected).

### 2. Node Embedding Generation
Generate vector representations of nodes to capture the network topology.
- **Script**: `code/preprocess/Node2Vec.py`
- **Logic**: Automatically detects and uses `edges_lscc.txt` (if available) or `edges.txt` to ensure embeddings align with the training graph.
- **Variant**: `code/preprocess/Node2Vec_haversine.py`
  - Generates embeddings where edge transition probabilities are weighted by geographic (Haversine) distance, encouraging walks along geographically shorter paths.

### 3. Comprehensive Preprocessing (SDM & Landmarks)
Compute the ground-truth distance matrix (SDM) and prepare training tensors.
- **Script**: `code/preprocess_all.py`
- **Key Features**:
  - **Parallel Computing**: Multi-process Dijkstra for speed.
  - **Large Scale Support**: Automatic **Sparse Matrix** storage and **Batch Processing** for massive graphs (e.g., >50k nodes) to prevent OOM errors.
  - Selects **Landmarks** using Farthest Point Sampling.
  - Normalizes geographic coordinates and packages data into `.npy` files in `data/<city>/pre/`.
  - **Note**: See `SDM_OPTIMIZATION_README.md` for performance tuning on large datasets.

### 4. Model Training
Train the `ImprovedMultiLayerPerceptron` model.
- **Script**: `code/distnet_train-cuda.py`
- **Highlights**:
  - Supports **CUDA** and **Mac MPS** acceleration.
  - **tilde-L1 Loss**: A specialized loss function for directed graphs (Type 2).
  - **Landmark Augmentation**: Includes landmark-to-landmark pairs in the training set.
  - **Fine-tuning**: Optional phase to re-train on high-error samples.

### 5. Evaluation
Test the trained model on sampled node pairs to calculate accuracy metrics.
- **Script**: `code/distnet_test_sample.py`
- **Metrics**: Mean Square Error (MSE), Mean Absolute Error (MAE), and Mean Relative Error (MRE).

## Quick Start

```bash
# Ensure you are on the 'bigmap' branch
# git checkout bigmap

# 1. Convert raw data and extract LSCC
python scripts/convert_graphs.py --data-dir ./data --folders chengdu

# 2. Generate node embeddings (reads *_lscc.txt automatically)
# Option A: Standard Node2Vec
python code/preprocess/Node2Vec.py --cities chengdu
# Option B: Haversine Weighted Node2Vec
# python code/preprocess/Node2Vec_haversine.py

# 3. Run full preprocessing (SDM, Landmarks, Normalization)
# For large cities, optimizations (sparse/batch) are applied automatically.
python code/preprocess_all.py --data-dir ./data --cities chengdu --parallel

# 4. Train the model (using tilde-L1 loss by default)
python code/distnet_train-cuda.py --cities chengdu --num_epoch 10

# 5. Evaluate the model
python code/distnet_test_sample.py --city chengdu --eval-samples 10000
```

## Configuration

Global settings and hyperparameters are managed in `code/config.py`. Key parameters include:
- `--type`: Loss function selection (1: MSE, 2: tilde-L1, 3: L1).
- `--embed_dim`: Dimension of the node embeddings (default: 128).
- `--batch_size` & `--learning_rate`: Training hyperparameters.

## Requirements

- Python 3.8+
- PyTorch, NetworkX, Pandas, NumPy, Scipy, Gensim, PecanPy, tqdm.