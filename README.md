# DistNet: Road Network Distance Estimation (Branch: bigmap)

DistNet is a deep learning framework designed for efficient and accurate shortest-path distance estimation in road networks. It combines structural node embeddings (Node2Vec) with geographic coordinates and utilizes an improved MLP architecture to predict distances.

## Project Structure

- `code/`: Core implementation.
  - `preprocess/`: Embedding generation scripts.
- `scripts/`: Data conversion utilities.
- `data/`: Dataset storage.
- `param/`: Model checkpoints.

## Workflow Selection

We provide two distinct workflows depending on your dataset size and resource constraints.

### 1. Standard Workflow (Recommended for Small/Medium Cities)
Best for datasets where the full distance matrix (N*N) fits in memory and disk space is not a bottleneck.
- **Preprocessing**: `preprocess_all.py` (Uses NetworkX/Multi-processing)
- **Training**: `distnet_train-cuda.py` (Static dataset loading, high memory usage)
- **Testing**: `distnet_test_sample.py`

### 2. Optimized Workflow (Recommended for Large Cities)
Designed for massive graphs (e.g., >50k nodes) where memory efficiency and speed are critical.
- **Preprocessing**: `preprocess_optimized.py` (Uses SciPy/Sparse Matrices, faster SDM calculation)
- **Embeddings**: `Node2Vec_optimized.py` (Uses PecanPy for faster random walks)
- **Training**: `distnet_train_optimized.py` (Dynamic sampling, avoids generating massive index files)
- **Testing**: `distnet_test_optimized.py`

---

## Execution Guide

### Step 0: Data Preparation (Common)
Convert raw data to standard format.
```bash
# git checkout bigmap
python scripts/convert_graphs.py --data-dir ./data --folders chengdu
```

### Option A: Standard Workflow Execution

**1. Generate Embeddings**
```bash
python code/preprocess/Node2Vec.py --cities chengdu
```

**2. Full Preprocessing**
Computes SDM, landmarks, and normalizes data.
```bash
python code/preprocess_all.py --data-dir ./data --cities chengdu --parallel
```

**3. Train Model**
Uses static data loading.
```bash
python code/distnet_train-cuda.py --cities chengdu --num_epoch 50
```

**4. Evaluate**
```bash
python code/distnet_test_sample.py --city chengdu --eval-samples 10000
```

### Option B: Optimized Workflow Execution

**1. Optimized Embeddings**
Uses PecanPy for faster random walks on large graphs.
```bash
python code/preprocess/Node2Vec_optimized.py --data-dir ./data --city chengdu
```

**2. Optimized Preprocessing**
Uses SciPy sparse matrices for SDM calculation and skips generating massive index files.
```bash
python code/preprocess_optimized.py --data-dir ./data --city chengdu --n-jobs 16
```

**3. Dynamic Training**
Uses on-the-fly sampling to avoid loading massive edge lists into memory.
```bash
python code/distnet_train_optimized.py --cities chengdu --samples-per-epoch 1000000
```

**4. Optimized Evaluation**
```bash
python code/distnet_test_optimized.py --city chengdu --eval-samples 10000
```

## Configuration

Global settings and hyperparameters are managed in `code/config.py`. Key parameters include:
- `--type`: Loss function selection (1: MSE, 2: tilde-L1, 3: L1).
- `--embed_dim`: Dimension of the node embeddings (default: 128).
- `--batch_size` & `--learning_rate`: Training hyperparameters.

## Requirements

- Python 3.8+
- PyTorch, NetworkX, Pandas, NumPy, Scipy, Gensim, PecanPy, tqdm.