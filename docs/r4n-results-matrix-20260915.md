# R4N 最终结果矩阵与公平比较规则

日期：2026-09-15。该表定义收数格式，不包含未完成实验的占位成绩。

## 数据集主表

最终至少：Jinan（development）、Shenzhen（fresh confirmation）、Chengdu、DIMACS-FLA。若资源允许再增加一张原生 directed road graph。

每个数据集统一记录：
- 节点数 / directed arcs / largest SCC 节点与弧数；
- 单向弧比例、距离非对称统计；
- workload 来源与 query 分布；
- train/val/test 条数、ordered-pair overlap；
- 原始数据与冻结 workload SHA256；
- 是否 development / confirmation。

## 精度主表

每个 dataset × method 至少记录：
- MRE mean ± sd（随机模型 3 seeds）；
- MAE / RMSE；
- short-Q1 MRE；
- high-asymmetry MRE；
- validation-selected checkpoint / time budget；
- 是否原生 directed、directed-target adaptation、或 directed-feature adaptation。

必选方法：
1. ALT32 lower bound；
2. CatBoost；
3. LandmarkNN；
4. Vdist2vec；
5. ANEDA；
6. RNE；
7. R4M frozen bounded model。

可选补充：Path2vec、Ndist2vec、SAGE、exact index（只作速度/零误差参照）。

## Baseline directed 公平性

Survey pinned code 的 `load_graph` 强制 `directed=False`。因此：

- 仅将监督标签替换为 directed shortest distance 的版本，统一命名为 **directed-target adaptation**；
- 对本来可自然接受有向信息的方法，再实现 **directed-feature/graph adaptation**；
- CatBoost/LandmarkNN 已有 forward+reverse directed landmark 特征版本；
- ANEDA 若使用 node2vec 初始化，需补一个在 directed graph 上生成初始化的强版本；
- RNE 的 graph partition / part features 需补合理 directed 版本；若原算法定义本身要求无向结构，则明确说明并保留最强可运行适配，不人为削弱；
- Vdist2vec 若不依赖图结构，只需保证 ordered OD 与 directed labels 正确。

论文主比较必须和每个 baseline 的**最强公平适配**比；原 Survey 配方适配可作为复现/消融，不作为弱化对手。

## 效率主表

同一台服务器、同一数据集 query IDs：

| Method | Device | Batch | latency µs/query mean±sd | throughput Mq/s | node-index bytes | model bytes | total bytes | preprocessing s | training s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

Batch 固定：1、32、1024、100000、1000000。Survey 对齐主指标为 100K / 1M、10 runs、最后5次 mean/std；1/32/1024 作为在线延迟补充。

R4M forward 必须从 `(src,dst)` 开始，包含：原始64-float landmark gather、坐标 gather、在线 normalization、1-ULP outward rounding、L/U reduction、139维特征、MLP、decode。禁止只计 MLP，也禁止为加速额外缓存 normalized/lo/hi 三份节点表而仍按256 B/node报空间。

## 收数停止条件

当以下同时满足，就停止工程性调参并进入理论审查：
1. 至少4张 native-directed road graphs；
2. 五类 learned strong baselines 均完成；
3. R4M 在大多数数据集优于 strongest baseline，且 fresh confirmation 成立；
4. query-time/storage/preprocessing 不出现不可接受的数量级劣势，形成清晰 Pareto；
5. 所有 workload、模型配置、计时协议已经冻结并可重放。

若第3项失败，则先定位是某类路网/查询分布失效，不能通过继续查看 confirmation test 后调参来修补；需要新开发集形成方法版本后再另设 confirmation。
