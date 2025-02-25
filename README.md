###### environment
- python 3.11
- pytorch 2.11

###### code
- preprocess
  - generate.py：从地图数据计算出最短路径距离矩阵，注意需要将nodes重排到以0作为起始的连续列表
  - Node2Vec.py：计算Node2Vec计算得到的路网节点嵌入
  - Node2Vec_haversine.py: 计算Node2Vec计算得到的路网节点嵌入，使用haversine距离
- config.py：模型参数
- dist_model.py：定义MLP模型
- distnet_train.py：
  - load_and_preprocess_data(): obtain the train_loader & valid_loader
    - 加载distance_matrix矩阵，归一化
    - 加载特征，包括Node2Vec学到的128维特征，和经纬度2维特征，都归一化到(0,1)
    - 定义训练集、验证集、和测试集，比例8：1：1，在训练集中增加landmark节点对
  - train_model(): 进行训练
    - 为了加速，仅选择selected_ratio比例的数据进行训练
    - 对高误差节点对进行fine-tune
    - 保存验证集上效果最好的model

###### data
- chengdu_node-mod.txt: 修正过的节点数据
- chengdu_link-mod.txt: 修正过的边数据
- graph_sc.pkl: 有向路网
- chengdu_directed_shortest_distance_matrix.npy: 有向图最短路径距离矩阵

###### param
- node2vec.emb: Node2Vec学习到的节点嵌入
- node2vec_embed.pkl: Node2Vec学习到的节点嵌入，保存为pkl文件
- node2vec_haversine.emb: Node2Vec学习到的节点嵌入，使用haversine距离
- node2vec_haversine_embed.pkl: Node2Vec学习到的节点嵌入，使用haversine距离，保存为pkl文件
