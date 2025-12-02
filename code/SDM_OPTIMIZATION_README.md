# SDM 计算优化说明

## 问题
对于大规模图（如 city-india），计算最短距离矩阵（SDM）时可能遇到：
- 内存不足
- 计算时间过长
- 进程被系统杀死

## 优化方案

### 1. 并行计算（默认启用）
使用多进程并行计算不同源节点的最短路径，大幅提升速度。

```bash
# 使用默认并行设置（自动检测CPU核心数，最多8-16个）
python preprocess_all.py --data-dir data --cities city-india

# 指定并行worker数量
python preprocess_all.py --data-dir data --cities city-india --n-jobs 16

# 禁用并行（仅用于调试）
python preprocess_all.py --data-dir data --cities city-india --no-parallel
```

### 2. 分批处理（自动启用）
对于大规模图，自动将节点分批处理，避免内存溢出。

```bash
# 自动批次大小（推荐）
python preprocess_all.py --data-dir data --cities city-india

# 手动指定批次大小（每个批次处理的节点数）
python preprocess_all.py --data-dir data --cities city-india --batch-size 1000
```

### 3. 稀疏矩阵存储（大规模图自动启用）
对于节点数 > 50000 的图，自动使用稀疏矩阵存储，节省内存。

```bash
# 自动启用（n > 50000时）
python preprocess_all.py --data-dir data --cities city-india

# 手动启用稀疏存储
python preprocess_all.py --data-dir data --cities city-india --use-sparse
```

## 推荐配置

### 对于 city-india（超大规模图）

```bash
# 方案1：使用所有优化（推荐）
python preprocess_all.py \
    --data-dir data \
    --cities city-india \
    --n-jobs 16 \
    --batch-size 1000 \
    --use-sparse

# 方案2：如果内存充足，可以不用稀疏矩阵
python preprocess_all.py \
    --data-dir data \
    --cities city-india \
    --n-jobs 16 \
    --batch-size 2000
```

## 性能优化建议

1. **内存优化**：
   - 使用 `--use-sparse` 可以节省约 50-90% 的内存
   - 使用 `--batch-size` 控制内存峰值

2. **速度优化**：
   - 增加 `--n-jobs` 可以提升速度（但会增加内存使用）
   - 建议 `n_jobs` 不超过 CPU 核心数

3. **大规模图（n > 100000）**：
   ```bash
   python preprocess_all.py \
       --data-dir data \
       --cities city-india \
       --n-jobs 16 \
       --batch-size 500 \
       --use-sparse
   ```

## 监控进度

优化后的代码会显示：
- 图构建进度
- 批次处理进度（使用 tqdm）
- 内存使用情况

## 故障排除

### 内存不足
- 减小 `--batch-size`（如 500 或更小）
- 启用 `--use-sparse`
- 减少 `--n-jobs`

### 计算太慢
- 增加 `--n-jobs`（但要注意内存）
- 增大 `--batch-size`（如果内存允许）

### 进程被杀死
- 检查系统内存限制
- 使用 `--use-sparse` 和更小的 `--batch-size`
- 考虑使用更大内存的机器

## 技术细节

1. **并行计算**：每个worker独立构建图并计算单源最短路径
2. **分批处理**：避免一次性加载所有结果到内存
3. **稀疏存储**：只存储非无穷大的距离值
4. **自动优化**：对于 n > 50000 的图，自动启用优化选项

