# R4E｜综述强基线复现 + Survey-matched Directed Workload

日期：2026-09-14。性质：开发阶段，不是最终独立确认集。

## 目标

1. 固定 `purduedb/shortest-distance-survey@dcaa89d38300bfb823eda84ccdfc85c42edbeae8`，在其原始无向 W_Jinan workload 上复现强 baseline（优先 CatBoost / LandmarkNN / RNE / Vdist2vec）。
2. 保持同一批 W_Jinan train/val/test OD 行与频次，仅将节点映射回作者发布的原生济南有向路网，在最大强连通分量内重新计算 directed shortest-path label，形成 mirror benchmark。
3. 在该 directed workload 上，以 32 个训练节点中随机 landmark 的双向距离索引为 64-float/node 基础表示，比较 ALT lower bound、全局插值和轻量方向感知 residual decoder。

## 强 baseline 门槛

综述 Table 5 的 W_Jinan 报告值：Vdist2vec 4.30%、LandmarkNN 4.05%、ANEDA 2.86%、RNE 3.82%、CatBoost 2.33%。这些值来自无向图，不能与原生有向图结果直接宣称优劣；但最终方法若仍 >5% MRE，除非在 directed capability / storage / latency 上有非常明确的 Pareto 优势，否则不视为主结果达标。

## Directed mirror 数据

- 输入 OD：综述仓库 `data/W_Jinan/real_workload_perturb_500k/W_Jinan_{train,val,test}.queries.npz` 的 src/dst 行，保持顺序、重复次数与 split，不使用其中 undirected dist 作为标签。
- 节点映射：survey 节点 i（1-based）对应 Figshare Jinan `NodeID=i-1`；该对应关系此前已通过投影坐标逐点核验。仅保留 R2 已冻结的最大强连通分量中的节点。
- 标签：在原生 `Origin -> Destination` 十进制 Length 有向图上重新运行 SciPy Dijkstra；抽样用 NetworkX 独立核验。
- mirror protocol 为了与综述 workload 对齐，不清除综述 split 中既存的重复 OD；因此它是 benchmark reproduction/development protocol，不替代以后需要冻结的 group-clean confirmation split。

## 64-scalar directed landmark residual

32 个 landmark 仅从 mirror train 查询出现的节点中，用固定 seed=20260914 随机选择。每个节点保存：

- `F_l(v)=d(l,v)`，32 个；
- `T_l(v)=d(v,l)`，32 个。

对有序查询 (u,v)：

`L=max(0, max_l[F_l(v)-F_l(u)], max_l[T_l(u)-T_l(v)])`

`U=min_l[T_l(u)+F_l(v)]`

由有向三角不等式，在距离有限的 SCC 中 `L <= d(u,v) <= U`。

固定比较：

- ALT32-LB：直接输出 L；
- GlobalAlpha：`L + alpha(U-L)`，alpha 只由 validation MRE 选择；
- LinearAlpha：基于冻结 landmark/coordinate pair features 输出 sigmoid alpha；
- MLPAlpha：两层 128/64 MLP 输出 sigmoid alpha。

学习方法输出始终限制在 `[L,U]`。训练只使用 train labels；checkpoint 只按 validation MRE 选择；测试标签仅在全部 checkpoint 冻结后读取。三种子 42/99/1234；固定 1500 updates，batch=16384，AdamW lr=1e-3，relative SmoothL1 loss。当前是预注册开发协议，不根据中间 test 结果改宽度、loss、landmark 或预算。

## 报告

必须同时报告 MRE、MAE、P95 relative error、短距离切片、高非对称切片、index bytes/node、decoder parameter count。若 MLP 仍不能进入约 2–5% 区间，则下一步优先加入 CatBoost/LandmarkNN 风格的 directed features，而不是继续扩大随机 embedding。
