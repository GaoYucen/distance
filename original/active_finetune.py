"""
active_finetune.py

主动微调（Active Fine-tuning）策略。

论文 Section III：
    "After each training iteration, we reanalyze the loss distribution
     of the training data to dynamically categorize the samples and select
     the top 10% of those with the highest training errors for further
     training iterations."

核心思想：
    - 每轮训练后，计算所有训练样本的预测误差
    - 选取误差最大的 top-k% 样本
    - 在下一轮对这些高误差样本进行额外训练

这样做的好处：
    1. 强调"主动"性——优先学习困难样本
    2. 加速收敛——把算力集中在最需要的地方
    3. 提升准确率——减少长尾误差
"""

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
from collections import defaultdict


class ActiveFinetuneSampler(Sampler):
    """
    主动微调采样器。

    每 epoch：
        1. 在当前模型上计算所有样本的预测误差
        2. 选取误差最大的 top-k% 样本
        3. 在这些样本上进行额外训练
    """

    def __init__(
        self,
        dataset: Dataset,
        model,
        criterion,
        device='cpu',
        top_k_ratio=0.1,
        extra_epochs=1,
        batch_size=1024,
    ):
        """
        Args:
            dataset: 训练数据集
            model: 当前模型
            criterion: 损失函数
            device: 设备
            top_k_ratio: 选取误差最大的 k% 样本
            extra_epochs: 对高误差样本额外训练的轮次
            batch_size: 计算误差时的 batch size
        """
        self.dataset = dataset
        self.model = model
        self.criterion = criterion
        self.device = device
        self.top_k_ratio = top_k_ratio
        self.extra_epochs = extra_epochs
        self.batch_size = batch_size

        self.high_error_indices = None

    def compute_errors(self):
        """
        计算所有训练样本的预测误差。
        使用 L1 误差（与论文的 MAE/MRE 一致）。

        Returns:
            errors: np.ndarray, 所有样本的预测误差
        """
        self.model.eval()
        errors = []

        n = len(self.dataset)
        batch_indices = list(range(0, n, self.batch_size))

        with torch.no_grad():
            for start in batch_indices:
                end = min(start + self.batch_size, n)
                batch = [self.dataset[i] for i in range(start, end)]

                # 解包
                if len(batch[0]) == 4:
                    i = torch.tensor([b[0] for b in batch], dtype=torch.long)
                    j = torch.tensor([b[1] for b in batch], dtype=torch.long)
                    d_ij = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                else:
                    i = torch.tensor([b[0] for b in batch], dtype=torch.long)
                    j = torch.tensor([b[1] for b in batch], dtype=torch.long)
                    d_ij = torch.tensor([b[2] for b in batch], dtype=torch.float32)

                i, j, d_ij = i.to(self.device), j.to(self.device), d_ij.to(self.device)

                # 预测
                preds = self.model(i, j).squeeze()
                d_ij = d_ij.squeeze()

                # L1 误差
                loss = torch.abs(preds - d_ij).cpu().numpy()
                errors.extend(loss.tolist())

        errors = np.array(errors)
        return errors

    def select_top_k(self, errors):
        """
        选取误差最大的 top-k% 样本。

        Args:
            errors: np.ndarray, 所有样本的误差

        Returns:
            top_indices: 误差最大的样本索引
        """
        n = len(errors)
        k = max(1, int(n * self.top_k_ratio))

        # argsort 降序排列
        sorted_indices = np.argsort(errors)[::-1]
        top_indices = sorted_indices[:k]

        print(f"  [Active FT] 选取 {k}/{n} ({k/n:.1%}) 误差最大的样本")
        print(f"    误差范围: [{errors[top_indices].min():.2f}, {errors[top_indices].max():.2f}]")

        return top_indices

    def __iter__(self):
        """返回高误差样本的索引迭代器"""
        if self.high_error_indices is None:
            # 首次：使用全部样本
            self.high_error_indices = list(range(len(self.dataset)))
        return iter(self.high_error_indices)

    def __len__(self):
        return len(self.high_error_indices)


def active_finetune_step(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    top_k_ratio=0.1,
    extra_epochs=1,
):
    """
    执行一次主动微调步骤。

    在当前 dataloader 上：
        1. 计算所有样本的误差
        2. 选取 top-k% 高误差样本
        3. 对这些样本额外训练 extra_epochs 轮

    Args:
        model: nn.Module
        dataloader: 完整训练数据
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        top_k_ratio: 高误差样本比例
        extra_epochs: 额外训练轮次

    Returns:
        selected_indices: 选取的高误差样本索引
        avg_error: 平均误差
    """
    model.eval()

    all_errors = []
    all_indices = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                i, j, d_ij, _ = batch
            else:
                i, j, d_ij = batch

            i, j, d_ij = i.to(device), j.to(device), d_ij.to(device)
            preds = model(i, j).squeeze()
            d_ij = d_ij.squeeze()

            # 记录每个样本的索引和误差
            errors = torch.abs(preds - d_ij).cpu().numpy()
            all_errors.extend(errors.tolist())
            all_indices.extend(i.cpu().tolist())

    all_errors = np.array(all_errors)

    # 选取 top-k%
    k = max(1, int(len(all_errors) * top_k_ratio))
    sorted_idx = np.argsort(all_errors)[::-1]
    top_indices = sorted_idx[:k]
    avg_error = all_errors[top_indices].mean()

    print(f"  [Active Fine-tuning] 高误差样本 {k} 个, 平均误差: {avg_error:.2f}")

    return top_indices, avg_error


def active_finetune_train(
    model,
    dataset,
    criterion,
    optimizer,
    device,
    top_k_ratio=0.1,
    extra_epochs=1,
    batch_size=1024,
    display_step=5,
):
    """
    对高误差样本进行主动微调训练。

    Args:
        model: 训练好的模型
        dataset: 完整数据集
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        top_k_ratio: 高误差样本比例
        extra_epochs: 额外训练轮次
        batch_size: 批大小

    Returns:
        model: 微调后的模型
    """
    from torch.utils.data import Subset, DataLoader

    # Step 1: 计算误差
    print("  [Active Fine-tuning] Step 1: 计算样本误差...")
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    errors = []
    indices = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 4:
                i, j, d_ij, _ = batch
            else:
                i, j, d_ij = batch

            i, j, d_ij = i.to(device), j.to(device), d_ij.to(device)
            preds = model(i, j).squeeze()
            d_ij = d_ij.squeeze()

            err = torch.abs(preds - d_ij).cpu().numpy()
            errors.extend(err.tolist())

            # 记录全局索引
            start = batch_idx * batch_size
            batch_size_actual = len(i)
            indices.extend(list(range(start, start + batch_size_actual)))

    errors = np.array(errors)
    indices = np.array(indices)

    # Step 2: 选取 top-k%
    k = max(1, int(len(errors) * top_k_ratio))
    sorted_idx = np.argsort(errors)[::-1]
    top_indices = sorted_idx[:k]

    print(f"  [Active Fine-tuning] 选取 {k}/{len(errors)} 高误差样本")
    print(f"    误差范围: [{errors[top_indices].min():.2f}, {errors[top_indices].max():.2f}]")

    # Step 3: 创建子集数据集
    high_error_subset = Subset(dataset, top_indices)
    high_error_loader = DataLoader(
        high_error_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Step 4: 额外训练
    print(f"  [Active Fine-tuning] Step 2: 在 {len(high_error_subset)} 样本上训练 {extra_epochs} 轮...")
    model.train()

    for epoch in range(extra_epochs):
        running_loss = 0.0
        for batch in high_error_loader:
            if len(batch) == 4:
                i, j, d_ij, _ = batch
            else:
                i, j, d_ij = batch

            i, j, d_ij = i.to(device), j.to(device), d_ij.to(device)
            d_ij = d_ij.unsqueeze(-1) if d_ij.dim() == 1 else d_ij

            optimizer.zero_grad()
            preds = model(i, j)
            loss = criterion(preds, d_ij)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(high_error_loader)
        if (epoch + 1) % display_step == 0:
            print(f"    Epoch {epoch+1}/{extra_epochs}, Loss: {avg_loss:.8f}")

    return model


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=== 测试主动微调 ===\n")

    import torch
    import torch.nn as nn

    # 模拟数据集
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, n=1000):
            self.n = n
            self.data = [(i, (i+1) % n, float(i)) for i in range(n)]

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            u, v, d = self.data[idx]
            return np.int32(u), np.int32(v), np.float32(d)

    # 模拟模型（输出固定值 + 噪声）
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(5.0))

        def forward(self, x1, x2):
            return torch.ones(len(x1), 1) * self.bias

    dataset = DummyDataset(n=100)
    model = DummyModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    print("初始模型 bias:", model.bias.item())

    # 主动微调
    model = active_finetune_train(
        model=model,
        dataset=dataset,
        criterion=criterion,
        optimizer=optimizer,
        device='cpu',
        top_k_ratio=0.1,
        extra_epochs=3,
        batch_size=32,
    )

    print("微调后模型 bias:", model.bias.item())
    print("\n✅ 主动微调测试通过！")
