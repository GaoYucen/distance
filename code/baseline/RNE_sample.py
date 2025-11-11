import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import time
import os
import random

class RNEModel(nn.Module):
    """Road Network Embedding Model - 将路网节点嵌入到低维空间"""
    def __init__(self, num_nodes, embedding_dim=64):
        """
        :param num_nodes: 路网中节点的总数
        :param embedding_dim: 嵌入向量的维度 (论文推荐64)
        """
        super(RNEModel, self).__init__()
        # 创建嵌入层 - 每个节点映射到d维向量
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
        # 使用小的正态分布初始化 (论文中使用)
        nn.init.normal_(self.embeddings.weight, std=0.01)
        
    def forward(self, source_nodes, target_nodes):
        """
        :param source_nodes: 源节点ID的张量
        :param target_nodes: 目标节点ID的张量
        :return: 源节点到目标节点的预测距离 (使用L1距离)
        """
        # 获取节点嵌入
        source_embeds = self.embeddings(source_nodes)
        target_embeds = self.embeddings(target_nodes)
        
        # 计算L1距离 (曼哈顿距离) - 论文推荐的度量方式
        distances = torch.sum(torch.abs(source_embeds - target_embeds), dim=1)
        return distances
    
    def get_embeddings(self):
        """获取所有节点的嵌入向量"""
        return self.embeddings.weight.data.cpu().numpy()


class PathDistanceDataset(Dataset):
    """数据集类，用于加载最短路径距离样本"""
    def __init__(self, distance_matrix, min_distance=10.0, max_distance=None):
        """
        :param distance_matrix: 真实的最短路径距离矩阵
        :param min_distance: 忽略小于该值的距离 (过滤太近的点对)
        :param max_distance: 忽略大于该值的距离 (可选，用于平衡样本)
        """
        self.distance_matrix = distance_matrix
        self.num_nodes = distance_matrix.shape[0]
        
        # 创建样本列表: (source, target, distance)
        self.samples = []
        
        # 遍历所有节点对
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                # 跳过自身到自身的距离
                if i == j:
                    continue
                
                dist = distance_matrix[i, j]
                
                # 跳过无效或不符合条件的距离
                if not np.isfinite(dist) or dist < min_distance:
                    continue
                    
                if max_distance is not None and dist > max_distance:
                    continue
                    
                self.samples.append((i, j, dist))
        
        print(f"Created dataset with {len(self.samples)} samples from {self.num_nodes} nodes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        i, j, dist = self.samples[idx]
        return torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long), torch.tensor(dist, dtype=torch.float)


def train_rne(distance_matrix, 
              embedding_dim=64,
              num_epochs=100,
              batch_size=4096,
              learning_rate=0.001,
              weight_decay=1e-5,
              min_distance=10.0,
              val_split=0.1,
              patience=15,
              save_dir="param/baseline/rne_checkpoints"):
    """
    训练RNE模型
    
    :param distance_matrix: 真实的最短路径距离矩阵
    :param embedding_dim: 嵌入维度 (论文推荐64)
    :param num_epochs: 最大训练轮数
    :param batch_size: 批量大小
    :param learning_rate: 学习率
    :param weight_decay: 权重衰减(L2正则化)
    :param min_distance: 最小距离阈值
    :param val_split: 验证集比例
    :param patience: 早停耐心值
    :param save_dir: 模型保存目录
    :return: 训练好的模型
    """
    num_nodes = distance_matrix.shape[0]
    print(f"Training RNE model for {num_nodes} nodes with embedding dimension {embedding_dim}")
    
    # 创建数据集
    dataset = PathDistanceDataset(distance_matrix, min_distance=min_distance)
    
    # 划分训练集和验证集
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # 设置随机种子确保可重现性
    generator = torch.Generator()
    generator.manual_seed(42)
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # 初始化模型
    # 使用mps作为device
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = RNEModel(num_nodes, embedding_dim).to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 论文中使用MSE损失
    criterion = nn.MSELoss()
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练循环
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        for source, target, true_distance in train_loader:
            source, target, true_distance = source.to(device), target.to(device), true_distance.to(device)
            
            # 前向传播
            predicted_distance = model(source, target)
            
            # 计算损失
            loss = criterion(predicted_distance, true_distance)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for source, target, true_distance in val_loader:
                source, target, true_distance = source.to(device), target.to(device), true_distance.to(device)
                predicted_distance = model(source, target)
                loss = criterion(predicted_distance, true_distance)
                val_loss += loss.item()
        
        # 计算平均损失
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, f"{save_dir}/best_rne_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # 打印进度
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1:3d}/{num_epochs}], "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    # 加载最佳模型
    checkpoint = torch.load(f"{save_dir}/best_rne_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def evaluate_rne(model, distance_matrix, sample_size=10000, max_distance=None, max_distance_value=1.0):
    """
    评估RNE模型的性能
    
    :param model: 训练好的RNE模型
    :param distance_matrix: 真实的最短路径距离矩阵
    :param sample_size: 评估样本数量
    :param max_distance: 只评估距离小于该值的样本 (用于分析不同距离范围的性能)
    :param max_distance_value: 归一化时使用的最大距离值
    :return: 平均绝对误差和平均相对误差
    """
    num_nodes = distance_matrix.shape[0]
    device = next(model.parameters()).device
    model.eval()
    
    # 随机选择评估样本
    np.random.seed(42)
    samples = []
    
    # 尝试采样直到达到sample_size或尝试太多次
    max_attempts = sample_size * 5
    attempts = 0
    
    while len(samples) < sample_size and attempts < max_attempts:
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)
        attempts += 1
        
        if i == j:
            continue
            
        true_dist = distance_matrix[i, j]
        if not np.isfinite(true_dist) or true_dist <= 0:
            continue
            
        if max_distance is not None and true_dist > max_distance:
            continue
            
        samples.append((i, j, true_dist))
    
    if len(samples) == 0:
        print("No valid samples for evaluation!")
        return float('inf'), float('inf')
    
    print(f"Evaluating on {len(samples)} samples (out of {sample_size} requested)")
    
    # 计算误差
    total_abs_error = 0
    total_rel_error = 0
    max_abs_error = 0
    max_rel_error = 0
    min_abs_error = float('inf')
    min_rel_error = float('inf')
    total_sq_error = 0

    with torch.no_grad():
        for i, j, true_dist in samples:
            source = torch.tensor([i], dtype=torch.long).to(device)
            target = torch.tensor([j], dtype=torch.long).to(device)
            predicted_dist = model(source, target).item()
            # 还原为真实距离
            predicted_dist_real = predicted_dist * max_distance_value
            true_dist_real = true_dist * max_distance_value
            abs_error = abs(predicted_dist_real - true_dist_real)
            rel_error = abs_error / true_dist_real
            sq_error = (predicted_dist_real - true_dist_real) ** 2

            total_abs_error += abs_error
            total_rel_error += rel_error
            total_sq_error += sq_error

            if abs_error > max_abs_error:
                max_abs_error = abs_error
            if rel_error > max_rel_error:
                max_rel_error = rel_error
            if abs_error < min_abs_error:
                min_abs_error = abs_error
            if rel_error < min_rel_error:
                min_rel_error = rel_error

    avg_abs_error = total_abs_error / len(samples)
    avg_rel_error = total_rel_error / len(samples)
    mean_sq_error = total_sq_error / len(samples)

    print(f"\nEvaluation on {len(samples)} random samples:")
    print(f"Mean Square Error: {mean_sq_error:.4f}")
    print(f"Average Absolute Error: {avg_abs_error:.4f}")
    print(f"Max Absolute Error: {max_abs_error:.4f}")
    print(f"Min Absolute Error: {min_abs_error:.4f}")
    print(f"Average Relative Error: {avg_rel_error:.8f} ({avg_rel_error*100:.2f}%)")
    print(f"Max Relative Error: {max_rel_error:.5f} ({max_rel_error*100:.2f}%)")
    print(f"Min Relative Error: {min_rel_error:.5f} ({min_rel_error*100:.2f}%)")

    return avg_abs_error, avg_rel_error, mean_sq_error, max_abs_error, min_abs_error, max_rel_error, min_rel_error


def save_embeddings(model, output_file="node_embeddings.npy"):
    """保存节点嵌入向量"""
    embeddings = model.get_embeddings()
    np.save(output_file, embeddings)
    print(f"Embeddings saved to {output_file}")
    return embeddings


def main():
    # 设置随机种子以确保可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("Loading shortest path distance matrix...")
    distance_matrix = np.load("data/chengdu_directed_shortest_distance_matrix.npy")
    max_distance_value = np.nanmax(distance_matrix[np.isfinite(distance_matrix)])
    print(f"Max distance for normalization: {max_distance_value}")
    
    # 归一化距离矩阵
    norm_distance_matrix = distance_matrix / max_distance_value

    # 归一化min_distance
    min_distance_norm = 10.0 / max_distance_value

    # print("\nTraining RNE model...")
    # model = train_rne(
    #     norm_distance_matrix,
    #     embedding_dim=64,  # 论文推荐的维度
    #     num_epochs=20,
    #     batch_size=4096,
    #     learning_rate=0.001,
    #     weight_decay=1e-5,
    #     min_distance=min_distance_norm,  # 用归一化后的阈值
    #     val_split=0.1,
    #     patience=15
    # )

    #读取模型后进行评估
    num_nodes = distance_matrix.shape[0]
    embedding_dim = 64  # 必须与训练时一致
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = RNEModel(num_nodes, embedding_dim).to(device)
    checkpoint = torch.load("param/baseline/rne_checkpoints/best_rne_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\nEvaluating model...")
    # 全局评估
    global_abs, global_rel, mse, max_abs, min_abs, max_rel, min_rel = evaluate_rne(
        model, norm_distance_matrix, max_distance_value=max_distance_value
    )

    # 按距离范围评估 (论文Figure 17的做法)
    distance_ranges = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    for max_dist in distance_ranges:
        print(f"\n--- Evaluation for distances <= {max_dist} km ---")
        # 归一化阈值
        max_dist_norm = (max_dist * 1000) / max_distance_value
        evaluate_rne(model, norm_distance_matrix, max_distance=max_dist_norm, max_distance_value=max_distance_value)
    
    print("\nSaving embeddings...")
    save_embeddings(model, "param/baseline/chengdu_node_embeddings.npy")

    # 保存评估结果
    with open("log/rne_sample.txt", "w") as f:
        f.write(f"Mean Square Error: {mse:.5f}\n")
        f.write(f"Mean Absolute Error: {global_abs:.5f}\n")
        f.write(f"Max Absolute Error: {max_abs:.5f}\n")
        f.write(f"Min Absolute Error: {min_abs:.5f}\n")
        f.write(f"Mean Relative Error: {global_rel:.8f}\n")
        f.write(f"Max Relative Error: {max_rel:.5f}\n")
        f.write(f"Min Relative Error: {min_rel:.5f}\n")

    print("\nRNE training and evaluation completed!")


if __name__ == "__main__":
    main()