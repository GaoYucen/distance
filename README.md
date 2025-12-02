###### environment
- python 3.11
- pytorch 2.4.0+cu124
- numpy
- pandas
- scikit-learn
- networkx
- tqdm
- matplotlib

###### 模型创新点
- 引入基于路网结构的节点嵌入（dist2vec）+经纬度，有效融合空间与结构信息。
- 设计多tildeL1输出层，适应有向图。
- 训练集中特别加入landmark节点对，提高模型对关键节点的预测能力。
<!-- - 后续计划加入finetune训练高误差训练样本 -->

###### data
- chengdu_node-mod.txt: 修正过的节点数据
- chengdu_link-mod.txt: 修正过的边数据
- graph_sc.pkl: 有向路网
- chengdu_directed_shortest_distance_matrix.npy: 有向图最短路径距离矩阵

###### code：训练时9:1训练集和验证集，测试时在全集上采样
- preprocess
  - generate.py：从地图数据计算出最短路径距离矩阵，注意需要将nodes重排到以0作为起始的连续列表
  - Node2Vec.py：计算Node2Vec计算得到的路网节点嵌入
  - Node2Vec_haversine.py: 计算Node2Vec计算得到的路网节点嵌入，使用haversine距离，本文的dist2vec嵌入
- config.py：模型参数；type=1、2、3对应1、tildeL1和L1
- dist_model.py：定义MLP模型，使用type选择output_layer
- distnet_train-cuda.py：
  - load_and_preprocess_data(): obtain the train_loader & valid_loader
    ###### environment
    - Python 3.11
    - PyTorch (建议与 CUDA 匹配，例如 `pytorch 2.4.0+cu124`)
    - numpy, pandas, scikit-learn, networkx, tqdm, matplotlib, gensim (可选，用于 .emb 加载)

    ###### 简要说明
    - 本目录包含 DistNet 的预处理、训练与测试代码，用于基于路网节点嵌入与经纬度预测路网距离。
    - 训练/测试流程现在支持按城市（city）组织的数据目录，预处理输出保存在 `data/<city>/pre/` 下，训练可一次对多个城市顺序训练并为每个城市保存独立 checkpoint 与日志。

    **运行说明**
    - 数据组织（约定）:
      - 原始城市数据可以是 `data/<city>/` 子目录，或以文件形式存在 `data/` 下（例如 `chengdu_node-mod.txt` + `chengdu_link-mod.txt`）。
      - 预处理输出目录（由 `distance/code/preprocess_all.py` 生成）为 `data/<city>/pre/`，其中包含：
        - `preprocessed_sdm.npy`：n x n 的最短距离矩阵
        - `preprocessed_embed.npy`：n x d 的节点嵌入（若存在）
        - `preprocessed_node_long_lat.npy`、`preprocessed_node_long_lat_origin.npy`：归一化与原始经纬度
        - `preprocessed_indices.npy`：训练/验证中用到的 i,j 对
        - `preprocessed_LM_indices.npy`：landmark 对（可选）

    **预处理（将原始 city 数据转换为 `pre/`）**
      - 运行示例（处理指定城市或全部城市）：
    ```bash
    python distance/code/preprocess_all.py --data-dir /home/you/distance/data --cities harbin,porto
    ```
      - 说明：脚本会尝试自动识别节点/边文件名（多种分隔符、列名兼容），并在 `data/<city>/pre/` 中保存标准化输出。

    **训练（支持一次训练多个城市，顺序训练）**
      - 运行示例（顺序训练两个城市）：
    ```bash
    python distance/code/distnet_train-cuda.py --cities harbin porto --data-root /home/you/distance/data --out-dir param --save-prefix distnet_best
    ```
      - 关键参数：
        - `--cities`：空格分隔的城市名列表（例如 `--cities harbin porto`）。
        - `--data-root`：包含 city 子目录的根路径（默认 `/home/lizhuoran/distance/data`）。
        - `--out-dir`：保存 checkpoint 的目录（默认为 `param`）。
        - `--save-prefix`：checkpoint 名称前缀（默认 `distnet_best`）。
      - 输出（每个城市）：
        - checkpoint：`{out-dir}/{save-prefix}_{city}_*.ckpt`（根据 `config.type` 会有后缀 `_1` / `_tilde_L1` / `_L1`）
        - 训练日志：`log/{save-prefix}_{city}_train.log`（每次保存最优模型时追加一行）

    **测试 / 评估（按城市）**
      - 运行示例（从默认位置加载 checkpoint）：
    ```bash
    python distance/code/distnet_test_sample.py --city harbin --data-root /home/you/distance/data --model-dir param --save-prefix distnet_best --eval-samples 10000
    ```
      - 可直接指定完整 checkpoint 路径：
    ```bash
    python distance/code/distnet_test_sample.py --city harbin --model-path /full/path/to/param/distnet_best_harbin_1.ckpt
    ```
      - 结果输出：`log/{save-prefix}_{city}_results_sample*.txt`，文件名包含 city 与 model 类型信息。

    **预处理 / 嵌入（Node2Vec）**
    - Node2Vec 可在 `distance/code/preprocess/Node2Vec.py` 中运行以生成 `.emb` 或 `.pkl` 嵌入，然后 `preprocess_all.py` 会尝试自动加载这些嵌入文件（需安装 `gensim`）。

    **注意事项与建议**
    - 计算完整最短距离矩阵（SDM）对大规模城市会非常耗时与耗内存（可能触发 OOM 或被系统 kill）。对大图建议：
      - 在更大内存的机器上运行；或
      - 修改 `preprocess_all.py`，只计算子集或稀疏化距离矩阵；或
      - 使用并行化/分块 Dijkstra 实现。
    - 若遇到 robosuite 写 `/tmp/robosuite.log` 的 PermissionError，可将 `lerobot_new/lerobot/sitecustomize.py` 放入 Python path（或安装到 site-packages），该文件已包含对 FileHandler 的防护。
    - 训练会尝试使用 CUDA（若可用），训练结束后会清空 CUDA 缓存以便下一个城市训练。

    **快速排查**
    - 若训练/测试报找不到文件，先确认 `data/<city>/pre/` 下存在 `preprocessed_sdm.npy` 与其他预处理文件；或者使用 `--data-root` 指定正确路径。
    - 模型加载：测试脚本优先使用 `--model-path`，否则根据 `--model-dir` 与 `--save-prefix` + city 名推断 checkpoint 文件名。

    更多细节请查看 `distance/code/` 下的脚本（`preprocess_all.py`、`distnet_train-cuda.py`、`distnet_test_sample.py`、`preprocess/Node2Vec.py`）。

    ---
    更新：已支持按城市顺序训练与评估，并在 README 中加入示例命令与输出说明。
