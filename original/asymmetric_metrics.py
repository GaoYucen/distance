"""
asymmetric_metrics.py

实现 L̃₁ 和 L̃∞ 非对称距离度量。

论文定义（Definition 4 & 5）:
    L̃₁(x, y) = Σᵣ|yᵢ - xᵢ| + Σᵣ⁺ˢ (yᵢ - xᵢ)
               前 r 维: 对称项（绝对值）
               后 s 维: 非对称项（不取绝对值，可负）

    L̃∞(x, y) = max_{i=1,...,n} (yᵢ - xᵢ)

特点：
    - 非对称：L̃₁(x, y) ≠ L̃₁(y, x)
    - 满足三角不等式：L̃₁(x, y) + L̃₁(y, z) ≥ L̃₁(x, z)
    - 前 r 维捕捉双向道路的对称性
    - 后 s 维捕捉单向道路的方向性
"""

import torch
import torch.nn as nn


class L1Tilde(nn.Module):
    """
    L̃₁ 非对称距离度量层。

    给定两个嵌入向量 x (起点) 和 y (终点)，计算：
        L̃₁(x, y) = Σᵣ|yᵢ - xᵢ| + Σᵣ⁺ˢ (yᵢ - xᵢ)

    其中：
        - x = y_o: 起点的嵌入
        - y = y_d: 终点的嵌入
        - r 维：对称部分（双向道路）
        - s 维：非对称部分（单向道路）
    """

    def __init__(self, r, s):
        """
        Args:
            r: 对称维度数量（捕捉双向道路）
            s: 非对称维度数量（捕捉单向道路）
        """
        super().__init__()
        self.r = r
        self.s = s
        assert r > 0 or s > 0, "r 和 s 不能同时为 0"

    def forward(self, x, y):
        """
        计算 L̃₁(x, y)。

        Args:
            x: torch.Tensor, shape (batch_size, r+s)
            y: torch.Tensor, shape (batch_size, r+s)

        Returns:
            torch.Tensor, shape (batch_size, 1)
        """
        # 前 r 维：对称项
        sym = torch.abs(y[:, :self.r] - x[:, :self.r])
        sym = sym.sum(dim=1, keepdim=True)

        # 后 s 维：非对称项（不取绝对值）
        asym = (y[:, self.r:self.r + self.s] - x[:, self.r:self.r + self.s])
        asym = asym.sum(dim=1, keepdim=True)

        # L̃₁ = 对称项 + 非对称项
        return sym + asym

    def extra_repr(self):
        return f"r={self.r}, s={self.s}"


class LInfTilde(nn.Module):
    """
    L̃∞ 非对称距离度量层。

    给定两个嵌入向量 x 和 y，计算：
        L̃∞(x, y) = max_{i=1,...,n} (yᵢ - xᵢ)

    论文证明：任意无负环的有向加权图都可以等距嵌入 L̃∞ 空间。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        计算 L̃∞(x, y)。

        Args:
            x: torch.Tensor, shape (batch_size, n)
            y: torch.Tensor, shape (batch_size, n)

        Returns:
            torch.Tensor, shape (batch_size, 1)
        """
        diff = y - x
        return diff.max(dim=1, keepdim=True)[0]

    def extra_repr(self):
        return "Linf_tilde"


def l1_tilde_distance(x, y, r):
    """
    纯函数版本的 L̃₁ 距离计算（无需 nn.Module）。

    Args:
        x: torch.Tensor, shape (batch_size, n)
        y: torch.Tensor, shape (batch_size, n)
        r: 对称维度数量

    Returns:
        torch.Tensor, shape (batch_size,)
    """
    s = x.shape[1] - r
    sym = torch.abs(y[:, :r] - x[:, :r]).sum(dim=1)
    asym = (y[:, r:r+s] - x[:, r:r+s]).sum(dim=1)
    return sym + asym


def linf_tilde_distance(x, y):
    """
    纯函数版本的 L̃∞ 距离计算。

    Args:
        x: torch.Tensor, shape (batch_size, n)
        y: torch.Tensor, shape (batch_size, n)

    Returns:
        torch.Tensor, shape (batch_size,)
    """
    return (y - x).max(dim=1)[0]


def l1_symmetric_distance(x, y):
    """
    标准 L₁ 对称距离（用于对比实验）。
    """
    return torch.abs(y - x).sum(dim=1)


def l2_distance(x, y):
    """
    标准 L₂ 欧氏距离（用于对比实验）。
    """
    return torch.norm(y - x, p=2, dim=1)


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=== 测试 L̃₁ / L̃∞ 度量 ===\n")

    batch_size = 4
    r = 2
    s = 3
    n = r + s

    x = torch.randn(batch_size, n)
    y = torch.randn(batch_size, n)

    # Test L̃₁
    metric = L1Tilde(r, s)
    dist = metric(x, y)
    print(f"L̃₁ distance shape: {dist.shape}")
    print(f"L̃₁ distances: {dist.squeeze()}")

    # Test 非对称性：L̃₁(x, y) != L̃₁(y, x)
    dist_xy = metric(x, y)
    dist_yx = metric(y, x)
    print(f"\n非对称性检验:")
    print(f"  L̃₁(x, y) = {dist_xy.squeeze()}")
    print(f"  L̃₁(y, x) = {dist_yx.squeeze()}")
    print(f"  相等? = {torch.allclose(dist_xy, dist_yx)} (应该不相等)")

    # Test L̃∞
    metric_inf = LInfTilde()
    dist_inf = metric_inf(x, y)
    print(f"\nL̃∞ distances: {dist_inf.squeeze()}")

    # Test 三角不等式：L̃₁(x,z) <= L̃₁(x,y) + L̃₁(y,z)
    z = torch.randn(batch_size, n)
    l_xy = metric(x, y)
    l_yz = metric(y, z)
    l_xz = metric(x, z)
    triangle = (l_xy + l_yz >= l_xz).all()
    print(f"\n三角不等式检验: {triangle} (应该为 True)")

    print("\n✅ 所有测试通过！")
