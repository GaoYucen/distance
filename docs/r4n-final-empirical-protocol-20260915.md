# R4N｜最终实证审查协议（冻结版）

日期：2026-09-15

## 1. 冻结点

Jinan 已作为 development benchmark 反复查看，不再用于结构/超参数选择。主模型冻结为 `scripts/r4m_bounded_landmarknn.py`（提交 `8431ddeb48c24e86c431a4303116212e11e3a7a8` 所含版本）：

- 32 个 random directed landmarks；每节点 forward/reverse 共 64 float32 scalar（256 B/node，不含坐标与模型参数）；
- 由有向三角不等式得到认证区间 `[L,U]`；
- 139 维输入；隐藏宽度 1024→512；输出 `sigmoid(alpha)`；
- `d_hat = L + alpha (U-L)`，因此预测始终落在认证区间；
- relative SmoothL1，beta=0.02；AdamW lr=1e-3, weight_decay=1e-5；batch=16384；
- 300 秒/seed；seeds 42/99/1234；验证 MRE 选择 checkpoint；
- Jinan development test：1.87927 / 1.87749 / 1.86418%，mean 1.87365 ± 0.00825%。

从本协议起，任何新城市结果都不能反向改变这些主配置。若未来形成新版本，必须命名为独立方法并使用新的 confirmation 数据。

## 2. 多数据集层级

优先级 A（workload-matched native-directed）：
1. Jinan：development only，已有 Survey 500K workload 的同 OD directed labels。
2. Shenzhen：fresh confirmation。优先取得 Survey `W_Shenzhen/real_workload_perturb_500k`，保持原 OD/split，只重算 native-directed labels。

优先级 B（额外 native-directed city benchmark）：
- 从公开原始路网中选至少 2–3 个不同规模城市，直接保留原始有向边；
- 若没有 trajectory workload，则预先固定 500K ordered OD workload，400K/50K/50K train/val/test；
- workload 生成规则、seed、LSCC 过滤和标签哈希必须在任何模型运行前落盘；
- 所有 baseline 和本方法使用完全相同的 directed labels/splits。

最终主表至少 4 个 native-directed datasets；Jinan 只标 development，至少一个城市必须是完全未见 confirmation。

## 3. Baseline 矩阵

Survey 代码固定到 `purduedb/shortest-distance-survey@dcaa89d38300bfb823eda84ccdfc85c42edbeae8`。

必须包含：
- CatBoost；
- LandmarkNN (`catboostnn.py`)；
- RNE；
- ANEDA；
- Vdist2vec；
- ALT32 lower bound（结构基线）；
- 非学习 Manhattan/Euclidean 或 landmark index 作为弱/解释性参考。

可选补充：Path2vec、Ndist2vec、SAGE。

Directed 适配原则：
- 首先替换监督标签为同 OD 的 directed shortest distance；
- 对本身允许 ordered pair 的模型保持其原架构；
- Survey 原 loader 强制 `directed=False`，因此依赖图预处理的原方法必须明确标成 “directed-target adaptation”，不能声称原论文原生支持 directed graph；
- LandmarkNN/CatBoost 可使用 forward+reverse directed landmark 特征时，必须同时保留 Survey-style 原版适配作为对照，避免只强化某一 baseline 或本方法。

所有随机学习 baseline 至少 3 seeds；若官方 5-minute protocol 为其主结果，则优先复现 5-minute/seed，验证集选择，test 最后一次读取。

## 4. 效率协议

与 Survey 的 inference protocol 对齐：
- CPU + GPU；
- batch_size = 100000, 1000000；
- eval_runs = 10；
- 报告最后 5 次的 mean/std latency (µs/query) 与 throughput (M queries/s)；
- GPU 计时包含 H2D + forward + D2H；
- CatBoost CPU-only；
- 同一服务器：2× RTX 4090；CPU Intel Xeon Silver 4309Y, 2 sockets × 8 cores/socket × 2 threads/core。

同时增加真实在线场景：batch_size = 1, 32, 1024。该扩展不替代 Survey 的 100K/1M 主协议。

本方法必须使用 end-to-end node-ID→distance 模块计时，计入：
- node index gather；
- 32-landmark L/U reduction；
- pair feature calculation；
- MLP forward；
- output decode。
不能只计 MLP forward。

## 5. 空间与离线成本

主表同时报告：
- per-node index bytes；
- learned model bytes；
- total index/model footprint；
- landmark/Dijkstra preprocessing time；
- model training time；
- query latency/throughput。

R4M 当前结构的学习网络参数量应按实际 checkpoint 统计；理论值约 668,673 个 float 参数（约 2.67 MB FP32）。节点索引 64 float32 = 256 B/node，坐标和标准化统计另计，不能漏报。

## 6. 结果收数门槛

进入理论/论文全面审查前需满足：
1. 至少 4 个 native-directed datasets；
2. 至少 CatBoost / LandmarkNN / RNE / ANEDA / Vdist2vec 五个有代表性的 learned baselines；
3. 主方法在大多数数据集的 MRE 显著优于 strongest baseline，不能只优于平均 baseline；
4. fresh confirmation dataset 上也成立；
5. query time / throughput / storage 至少处于可接受 Pareto 前沿；
6. 若精度依赖明显更大的网络或更高存储，必须在同预算消融中展示代价。

满足后停止工程调参，转入理论审查：重新核对表示定理、认证区间定理、projection 不增误差、训练目标与模型结构的关系，以及与 MRN/IQE/PQE/QRL/landmark heuristic 文献的原创性边界。
