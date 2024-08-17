###### code

- config.py：模型参数
- generate.py：从地图数据计算出最短路径距离矩阵，注意需要将nodes重排到以0作为起始的连续列表
- Node2Vec.py：计算Node2Vec计算得到的路网节点嵌入
- distnet_1.py：
  - 加载distance_matrix矩阵，归一化
  - 加载特征，包括Node2Vec学到的128维特征，和经纬度2维特征，都归一化到(0,1)
  - 定义训练集、验证集、和测试集，比例8：1：1
  - 定义MLP model
  - 进行训练，每轮训练要打乱train_indices，同时保存验证集上效果最好的model
- distnet_1_test.py：在测试集上进行测试

###### data

- chengdu_node-mod.txt: 修正过的节点数据
- chengdu_link-mod.txt: 修正过的边数据