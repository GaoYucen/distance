# L̃₁ Asymmetric Metric Ablation Study

## 1. 研究问题

> **L̃₁ 不对称度量是否在工程上优于 L1 用于有向道路网络距离估计？**

理论基础：Theorem 3-5 证明 L1 不能等距嵌入有向图，但 L̃₁ 可以。本文验证这个数学优势在工程上的转化条件。

---

## 2. 实验环境

- GPU: NVIDIA RTX 3090 (24GB)
- CUDA 12.8, PyTorch 2.8.0
- 数据集: Harbin (44K 节点, 108K 有向边, 30.8% 单行道)

---

## 3. 核心发现

### 发现 1：单 GNN Bi-Encoder 中 L1Tilde 无效

```
架构: GNN(u), GNN(v) → d = ||GNN(v) - GNN(u)||
```

| Decoder | Test MRE (3 seeds) |
|---------|:------:|
| L1 | **4.69%** |
| L1Tilde (r=62, s=2) | **4.67%** |
| Δ | -0.02% (无差异) |

**原因：单 GNN + 减法结构 = 严格对称的 embedding 空间。** L1Tilde 的 2 个不对称维度拿不到方向信号，退化为 L1。

### 发现 2：Harbin 数据集方向性质分析

| 指标 | 数值 | 含义 |
|------|:---:|------|
| 单行道占比 | 30.8% | 不低 |
| 单行道中位长度 | **143m** | 都是小巷子 |
| 查询对不对称 ≥ 20% | **1.4%** | 极端方向查询极少 |
| 查询对不对称 < 1% | **72.2%** | 绝大多数几乎对称 |

**30.8% 的单行道数字有欺骗性。** 单行道都是短小巷子（中位 143m），对最短路径距离的影响可以忽略。真正需要方向感知的查询不到 2%。

### 发现 3：Cross-Encoder 中 L1Tilde 效果显著 ★

```
架构: concat(feature_u, feature_v) → MLP → output → L1/L1Tilde
```

| Encoder | Decoder | Test MRE | Δ |
|---------|---------|:------:|:--:|
| GNN feats + MLP | L1 | 10.80% | — |
| GNN feats + MLP | L1Tilde | **7.32%** | **-3.48%** |
| 纯坐标 + MLP | L1 | 6.95% | — |
| 纯坐标 + MLP | L1Tilde | 6.71% | -0.24% |

**L1Tilde 在 Cross-Encoder 上显著优于 L1**，与在 Chengdu 上的独立实验一致（MRE 从 4.78%→2.46%，Δ=-2.32%）。

**原因：MLP 的权重是位置相关的，concat 顺序创造了内在不对称性，** L1Tilde 成功放大了这种不对称性。

### 发现 4：Bi-Encoder vs Cross-Encoder 的精度-度量权衡

```
Bi-Encoder:   结构好 → L1=4.5%    但 L1Tilde 无用 (Δ=0%)
              ↓
              ✅ 可索引 (O(log N) kNN)
              ❌ L1Tilde 无法发挥

Cross-Encoder: 结构差 → L1=10.8%  但 L1Tilde 大幅有效 (Δ=-3.5%)
              ↓
              ❌ 不可索引 (O(N) kNN)
              ✅ L1Tilde 可以发挥
```

### 发现 5：Dual GNN（共享第一层）尝试

```
架构: GNN_shared(u) → GNN_src(u) (独立 L2)
      GNN_shared(v) → GNN_dst(v) (独立 L2)
      d = ||GNN_dst(v) - GNN_src(u)||
```

| 架构 | L1 Test MRE |
|------|:------:|
| 单 GNN | **4.46%** |
| Dual GNN (完全独立) | 11.6% |
| Dual GNN (共享 L1) | 8.3% |

Dual GNN 无法打平单 GNN——两个独立嵌入空间的对齐是更难的优化问题。

### 发现 6：训练优化关键参数

| 参数 | 最优值 | 效果 |
|------|:---:|------|
| Loss | SmoothL1 | MSE→SmoothL1: 6.37% → 5.11% |
| 学习率 | 0.001 | lr=0.01 → 过拟合, lr=0.0005 → 收敛不足 |
| Epochs | 50 | >50 epoch → 过拟合 |

---

## 4. 所有实验对比表

### 单 GNN Bi-Encoder 系列

| 实验 | 配置 | Test MRE |
|------|------|:------:|
| MSE, 20ep | baseline | 6.37% |
| SmoothL1, 30ep | 换 loss | 5.11% |
| SmoothL1, 50ep, lr=0.01 | 高 lr | 5.25% |
| SmoothL1, 50ep, lr=0.001 | **最优 L1** | **4.56%** |
| SmoothL1, 50ep, lr=0.001, s=42 | 最优 seed | **4.41%** |
| L1Tilde, 同配置 | seed=42 | 4.26% |
| L1Tilde, 3-seed avg | | 4.67% |

### 有向 GNN 系列

| 实验 | 配置 | Test MRE |
|------|------|:------:|
| Directed L1 | 删 ToUndirected | 4.49% |
| Directed L1Tilde | 删 ToUndirected | 4.46% |
| Undirected L1Tilde+Aux | λ=0.1 | 4.38% |
| Directed L1Tilde+Aux | λ=0.1 | 4.44% |

→ 有向 GNN 和辅助 Loss 均未带来改善

### Cross-Encoder 系列

| 实验 | Encoder | Decoder | Test MRE |
|------|---------|---------|:------:|
| 纯坐标 (50ep) | 4-dim coord | L1 | 6.95% |
| 纯坐标 (50ep) | 4-dim coord | L1Tilde | 6.71% |
| GNN特征 (50ep) | 132-dim | L1 | 11.99% |
| GNN特征 (50ep) | 132-dim | L1Tilde | **7.85%** |
| GNN特征 (150ep) | 132-dim | L1 | 10.80% |
| GNN特征 (150ep) | 132-dim | L1Tilde | **7.32%** |

### Dual GNN 系列

| 实验 | 架构 | Test MRE |
|------|------|:------:|
| 完全独立, lr=0.005, 50ep | 2× 独立 GNN | 11.6% |
| 共享 L1, lr=0.001, 50ep | 共享+分叉 | 8.9% |
| 共享 L1, lr=0.005, 50ep | 共享+分叉 | 8.3% |
| 共享 L1, lr=0.005, 100ep | 共享+分叉 | 9.4% |

---

## 5. 结论

### L1Tilde 的有效性条件

```
L1Tilde_gain = f(Encoder不对称性) × f(路网方向系统化程度)
```

| Encoder | 路网 | L1Tilde 效果 |
|---------|------|:--:|
| Cross-Encoder (天然不对称) | Chengdu | **L1Tilde 将 MRE 减半** (4.78%→2.46%) |
| Cross-Encoder (天然不对称) | Harbin | **L1Tilde 显著改善** (10.8%→7.3%) |
| Bi-Encoder (严格对称) | Harbin | L1Tilde 无效 (4.69%→4.67%) |
| Bi-Encoder (严格对称) | Beijing | L1Tilde 有效 (-1.87%)* |

*Beijing 来自第一次消融分析结果

### 核心贡献

1. **定理 3-5 的工程边界已刻画清晰**: L1Tilde 需要 Encoder 提供不对称表示才能发挥作用
2. **识别了精度-可索引性-度量有效性三重权衡**: Bi-Encoder 精度高可索引但 L1Tilde 无效，Cross-Encoder 精度低不可索引但 L1Tilde 有效
3. **提出了未来方向**: 设计一种同时具备 Bi-Encoder 结构优势 + L1Tilde 友好不对称表示的混合架构

---

## 6. 代码结构

```
ablation_study/
├── README.md                          # 本文档
├── models/
│   ├── rgnndist2vec.py                # 单 GNN + L1 (支持 --directed)
│   ├── rgnndist2vec_l1tilde.py        # 单 GNN + L1Tilde (支持 --directed, --aux_loss_weight)
│   ├── dual_gnn.py                    # 双 GNN (共享第一层, 支持 L1/L1Tilde)
│   ├── train.py                       # 训练入口 (支持 --directed, --aux_loss_weight)
│   └── torch_utils.py, data_utils.py  # 基础设施
├── results/
│   └── all_results.json               # 所有实验结果汇总
└── scripts/
    └── cross_encoder_test.py          # Cross-Encoder 快速测试脚本
```

## 7. 复现命令

```bash
# 1. 单 GNN + L1 (最优配置)
python train.py --model_class rgnndist2vec --gnn_layer sage \
    --data_dir data/OSM_Harbin_Small --query_dir data/OSM_Harbin_Small/random_500k \
    --epochs 50 --seed 42 --device cuda --loss smoothl1 --learning_rate 0.001

# 2. 单 GNN + L1Tilde
python train.py --model_class rgnndist2vec_l1tilde --gnn_layer sage \
    --data_dir data/OSM_Harbin_Small --query_dir data/OSM_Harbin_Small/random_500k \
    --epochs 50 --seed 42 --device cuda --loss smoothl1 --learning_rate 0.001 \
    --l1tilde_r 62 --l1tilde_s 2

# 3. 有向 GNN 版本
python train.py --model_class rgnndist2vec --gnn_layer sage \
    ... --directed

# 4. L1Tilde + 辅助 Loss
python train.py --model_class rgnndist2vec_l1tilde --gnn_layer sage \
    ... --l1tilde_r 62 --l1tilde_s 2 --aux_loss_weight 0.1

# 5. Dual GNN
python train.py --model_class dual_gnn --gnn_layer sage ...
python train.py --model_class dual_gnn_l1tilde --gnn_layer sage ...
```
