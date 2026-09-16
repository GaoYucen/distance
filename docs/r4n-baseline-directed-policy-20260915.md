# R4N｜有向路网 Baseline 归类与公平比较口径

日期：2026-09-15。

## 1. 三种版本必须分开

1. **Original / direct transfer**：保持原方法与官方/Survey 实现结构不变，只把预测目标换成 native-directed shortest-path labels。它回答“已有无向路网方法能否零结构修改迁移到 directed task”。
2. **Directed-target adaptation**：同上；若为了运行只修改数据加载/ordered OD，不增加方向建模能力，仍归入 direct transfer，不宣称原论文原生支持 directed graph。
3. **Directed-feature/model adaptation (ours)**：加入 forward/reverse landmarks、directed graph embedding、source/target 双表或改 asymmetric decoder。统一命名 `Dir-X (our adaptation)`；这是本工作的强化对照，不属于原 X 方法本身的能力。

主表至少保留 Original/direct-transfer；为避免“弱化 baseline”的质疑，再给最强合理 `Dir-X (ours)`。论文叙述不能把后者的收益归给原论文。

## 2. RNE / ANEDA 的结构限制

Pinned Survey 实现：

- RNE：`mean(abs(e_u-e_v))*scale`，严格满足 `pred(u,v)=pred(v,u)`；
- ANEDA：Lp norm / cosine / dot-product 三种解码均交换对称；
- Vdist2vec：按顺序拼接 `[e_u,e_v]` 后过 MLP，不具有上述对称限制，可直接建模 ordered OD。

因此，RNE/ANEDA 在 directed target 上的退化应表述为：

> 原模型的对称 inductive bias 与一般 directed shortest-path metric 存在结构性失配；有向任务暴露了该限制。

不应写成“15% 全部证明有向图天然更难”，因为其中同时包含模型假设失配。Vdist2vec 的 directed-vs-undirected 配对退化更适合说明在没有硬对称障碍时任务本身仍发生难度变化。

## 3. 对称模型的 paired-MRE oracle 下界

对真实双向距离 `a=d(u,v)>0`, `b=d(v,u)>0`，任意对称预测器必须给同一个 `p`。其双向平均 MRE 为

`0.5*(|p-a|/a + |p-b|/b)`。

若 `a<=b`，加权绝对值的最优点为 `p=a`，因此最小值为

`|a-b|/(2*max(a,b))`。

于是对一批无序 OD 对，任何对称模型的 paired-MRE 至少为上述量的平均值。该下界与训练算法、维数无关。

注意：Survey workload 主表通常只评价给定 ordered OD，不一定同时包含 reverse query；因此这个 oracle 是**结构诊断/双向扩展评估下界**，不能偷换为单方向 workload 主表的直接下界。

## 4. 论文建议

- 主精度表：Original/direct-transfer baselines + R4M；
- 强化对照表或同表附列：`Dir-RNE (ours)`, `Dir-ANEDA (ours)`, directed-feature LandmarkNN/CatBoost；
- challenge 表：同 OD 的 undirected→directed label shift、模型 MRE 退化、paired symmetric oracle；
- 对 adaptation 的代码和开销单独记录，不能把我们的改动计入 baseline 原论文能力。
