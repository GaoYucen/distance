###### environment
- python 3.12.3
- pytorch 2.8.0+cu128
- numpy
- pandas
- scikit-learn
- networkx
- tqdm
- pickle
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
    - 加载distance_matrix矩阵，归一化
    - 加载特征，包括Node2Vec学到的128维特征，和经纬度2维特征，都归一化到(0,1)
    - 定义训练集、验证集，比例9：1，在训练集中增加landmark节点对
  - train_model(): 进行训练
    - 为了加速，仅选择selected_ratio比例的数据进行训练，目前还是全量
    <!-- - 对高误差节点对进行fine-tune -->
    - 保存验证集上效果最好的model
- distnet_test_sample.py: 测试
  - 在全集中采样10000个点对进行评估→后续得考虑和训练联动，以训练、验证、测试的方式进行评估，但目前看全集评估也很好，因为这个是最终的应用场景
- baseline
  - RNE_sample.py: RNE模型采样评估
    - 加载训练好的RNE模型，在全集所有合法节点对中随机采样若干对进行距离预测和误差评估，输出均方误差、绝对误差、相对误差等指标
  - SARN_sample.py: SARN模型采样评估
    - 加载训练好的SARN模型，在全集所有合法节点对中随机采样若干对进行距离预测和误差评估，输出均方误差、绝对误差、相对误差等指标
  - vdist2vec_sample.py: vdist2vec模型采样评估
    - 加载训练好的vdist2vec模型，在全集所有合法节点对中随机采样若干对进行距离预测和误差评估，输出均方误差、绝对误差、相对误差等指标
- ablation
  - distnet_base_train-cuda.py: no landmark
  - distnet_base_test_sample.py: test for distnet_base model

###### param: 模型参数
- node2vec.emb: Node2Vec学习到的节点嵌入模型参数
- node2vec_embed.pkl: Node2Vec学习到的节点嵌入，保存为pkl文件
- node2vec_haversine.emb: Node2Vec学习到的节点嵌入模型参数，使用haversine距离
- node2vec_haversine_embed.pkl: Node2Vec学习到的节点嵌入，使用haversine距离，保存为pkl文件

###### log: 记录实验结果
