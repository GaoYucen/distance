# R4-1｜问题定义与最接近工作对照

日期：2026-09-14。性质：执行后的文献核验与研究定位，不是模型性能报告。
研究基线：GaoYucen/distance 的 research/l1tilde-audit-20260913 分支，R3 提交 2fb7dd971c794da0e9f4add283beb593639fe028。

## 1. 本轮锁定的问题

已知静态有向正权路网 G=(V,E,w)，在距离有限的节点域上，为每个节点存储固定表示。查询返回最短路长度估计，不恢复路径。本阶段不加入动态边权、跨城市零样本迁移或查询相关编码器。

目标是给出精度—实际存储字节—查询计算量之间有竞争力的取舍。训练图已知不等于所有距离标签免费：最短路监督、landmark 教师坐标、模型生成、认证扫描均计入离线成本。查询未被用来选择模型，才可以作为确认测试；R1–R3 已查看的查询只作开发材料。

用户原稿依据：Definition 4 的 tilde-L1、Definition 5 的 tilde-L-infinity、Theorem 4 的距离坐标构造。原稿 Equation (6) 是 OD 联合网络；本轮明确研究共享固定节点表示，不把其理论自动用于该联合网络。综述 Section 2 明确使用无向图；其排名不直接证明有向模型优劣。

## 2. 统一符号

令 Δz=z(v)-z(u)，Δh=h(v)-h(u)。

- 原单分量：q(u,v)=||Δz||₁+Δh；其正部为 [q]₊。
- 纯最大势差：P_M(u,v)=max(0,max_j[f_j(v)-f_j(u)])。
- 当前候选：D_K(u,v)=max(0,max_k[||Δz_k||₁+Δh_k])。
- 必须增加的强对照：S_K(u,v)=||Δz||₁+max(0,max_kΔh_k)，即共享对称部分的 MRN 风格 L1 变体。它不是原论文默认的 L2 实现，必须标明。

独立分量预算 B=Σ_k(r_k+1)。共享对称部分预算为 r+K。均缓存时按 float32 每节点 4B 字节核算；模型/索引元数据、量化尺度另列。

## 3. 文献对照与不能再当作新贡献的事项

| 工作 | 本次核验的相关点 | 与当前候选的关系／必须避免的误述 |
|---|---|---|
| 用户初稿 tilde-L1 / tilde-L-infinity [U1] | 有符号和与最大差；n 维距离坐标构造 | 原单势差是受限部分；n 维精确表示不是新低维保证 |
| Deep/Wide Norm，Pitis 等，2020 [S1] | 非对称距离结构和保持三角的组合；Proposition 8 包含 max | 当前分块形式属于结构化的凸、正齐次距离构造；不能把 max 保三角声称为新发明 |
| MRN，Liu 等，2023 版本 [S2] | 对称度量加最大势差；Proposition 1 与 Theorem 2 | 必须比较共享对称部分。当前版本使用 L2，不应拿旧平方 L2 实现制造弱对照 |
| PQE，Wang–Isola，2022 [S3] | 可训练准度量及有限空间近似理论 | 需区分其失真型理论与精确表示，不能混称统一64维精确 |
| IQE，Wang–Isola，2022 [S4] | 区间并长度分量；sum/maxmean 两种聚合；maxmean 有通用表示分析 | 必须分别标识配置及计入分量尺寸；不能仅比较一个任意差配置 |
| QRL，Wang 等，ICML 2023 [S5] | 局部转移约束结合准度量三角性质，以及最大化分离的恢复论证 | “所有边约束推出全局下界，再尽量增大预测”已有直接近邻，不宜独立包装为算法创新 |
| Subset Selection of Search Heuristics，Rayner 等，IJCAI 2013 [S6] | 最大聚合启发式的预算子集选择、次模贪心及抽样 | 固定下界库上的 1−1/e 选择保证是已有工具，不是新的核心理论 |
| AAC，Le–Ngo，2026-04 预印本 [S7] | 可微 landmark 压缩、架构可采纳性、部署时选择 ALT 子集，含覆盖半径分析 | 仅做“学 landmark + 下界 + 覆盖界”已有很近工作；其搜索目标不同于直接距离估计，应独立复现而不照搬结果 |
| planar directed ℓ1 embedding，Kawarabayashi–Sidiropoulos，FOCS 2021 [S8] | 平面准度量嵌入和失真分析 | directed ℓ1 不能只凭名称等同于用户定义的 tilde-L1；理论迁移前必须核对定义和图类 |
| 路网 learned-index survey，Choudhary 等，2026 [U2] | encoder–decoder、工作负载与时间/存储/精度共同评估 | 采用其问题维度和可用原生数据，不沿用测试集选模型或 Python 改写精确基线的协议 |

这里只确认“已检索到的近邻与冲突”，不是已证明论文创新完备或不存在其他更近工作。MRN、IQE、QRL 的任务背景与路网不同，不意味着它们可忽略。

## 4. 本轮新增的研究定位

推导发现，每个 L1+势差分量可精确展开为最多 2^r 个符号组合势函数的最大值。于是当前候选并非在无限维时超越纯最大势差，而是把许多相互依赖的势方向压缩在少量生成坐标中。

更值得争取的核心是：在有向路网和实际查询分布下，哪些距离下界函数可以这样分组、共享生成元，并在严格预算下保留精度？多组对称几何是否比一个共享对称几何有效？完整符号组合带来的额外分量如何避免越过真实距离？

这些仍是研究问题。R4-2 给出代数等价、定量构造、条件压缩界及反例；并没有给出适用于所有路网的低维小误差保证。

## 5. 已排除的三个过早结论

1. 不再把“多个 tilde-L1 取 max”本身作为充分的新颖性。
2. 不把局部边约束、通用表示和标准次模选择独立列成三个原创主要定理。
3. 不把构造图上相对纯 max-potential 的压缩优势外推为超过 MRN/IQE。构造的积图事实上也被更简单的共享 L1+max 势差精确表示；这是必须保留的强对照。

## 6. 核验来源

[U1] 用户提供的 Learning-Based Shortest Path Distance Estimation on Road Network Using Asymmetric Metric，Definition 4/5、Equation (6)、Theorem 4；研究仓库 https://github.com/GaoYucen/distance 。

[U2] 用户提供的 An Empirical Survey and Benchmark of Learned Distance Indexes for Road Networks，arXiv:2602.04068v1，Sections 2–4；https://github.com/purduedb/shortest-distance-survey 。

[S1] https://arxiv.org/html/2002.05825v3

[S2] https://arxiv.org/html/2208.08133v4

[S3] https://arxiv.org/html/2206.15478v4

[S4] https://arxiv.org/html/2211.15120v2

[S5] https://proceedings.mlr.press/v202/wang23al.html ；https://arxiv.org/html/2304.01203v7

[S6] https://webdocs.cs.ualberta.ca/~bowling/papers/13ijcai-hsubset.pdf

[S7] https://arxiv.org/html/2604.20744v1

[S8] https://arxiv.org/abs/2111.07974

书目版本固定供复核。本文没有复用外部论文图表、性能排名或大段原文。
