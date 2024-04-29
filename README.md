进展：

- 针对成都小规模路网复现EBDT 2020的vdist2vec模型



文件：

- code
  - config.py: 模型参数
  - generate.py: 从地图数据计算出最短路径距离矩阵
  - vdist2vec_chengdu.py: 使用最短路径距离矩阵训练vidst2vec模型
- data: 下载链接https://pan.sjtu.edu.cn/web/share/2e3b43e83c96389ad779d325a5d90462
  - chengdu_node-mod.txt: 修正过的节点数据
  - chengdu_link-mod.txt: 修正过的边数据
  - chengdu_directed_shortest_distance_matrix.npy: 有向最短路径距离矩阵