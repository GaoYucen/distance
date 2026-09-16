"""Render R2 status only from completed, verified experimental artifacts."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'reports/audit-r2-20260913'


def main():
    data=json.loads((R/'data_audit.json').read_text())
    pilot=json.loads((R/'pilot.json').read_text())
    assert data['passed'] and pilot['status']=='completed' and pilot['completed_runs']==30
    assert pilot['all_30_cpu_replays_passed'] and pilot['same_od_control_passed']
    d=json.loads((R/'Jinan_native_directed_diagnostic.json').read_text())['split_metrics']['test']
    s=json.loads((R/'Shenzhen_native_directed_uniform_diagnostic.json').read_text())['split_metrics']['test']
    lines=['# 路网距离项目 R2：综述数据接入与保留方向的配对实验','',
     '## 已完成','',
     '4090 `/workspace/distance`；基于 VLDB 的审计分支 `research/l1tilde-audit-20260913`。原始 main/VLDB 不改动，PR #1 不自动合并。',
     '下载作者公开数据、核验文件哈希、核验节点映射和最短路标签、完成济南有向/无向配对的 30 组 GPU 试跑；全部检查点经过独立 NumPy 解码公式与 CPU 重放核验。',
     '本轮是固定预算机制探索，不是综述排行榜复现，不是已确认的 SOTA，不意味着所有模型已经收敛。','',
     '## 数据来源与边界','',
     '综述代码快照：`purduedb/shortest-distance-survey@dcaa89d38300bfb823eda84ccdfc85c42edbeae8`。',
     '当前仓库列出 13 份数据，但仓库随附的是 Surat 和 W_Jinan；其他预处理数据通过作者公开下载入口提供。本轮获取了 W_Jinan 的图和查询、Surat 图，以及原始济南/深圳的四个道路 CSV，没有下载全部 13 个数据集。',
     '综述预处理图是无向图。原始济南/深圳数据来自 Scientific Data 2023，DOI `10.1038/s41597-023-02589-y` 及作者 Figshare 发布记录；使用其 Origin→Destination 原始方向，不从无向边的存储顺序虚构单行道。',
     '新实验保留原始十进制米制边长，在最大强连通分量中重新计算两个方向的最短路。无向配对图由同一批原始节点和边去方向生成，并取平行边最小长度。',
     f"济南：{data['jinan_native']['raw_nodes']:,} 个原始节点，{data['jinan_native']['raw_edge_rows']:,} 条原始弧；保留 {data['jinan_native']['largest_strong_component_nodes']:,} 个节点、{data['jinan_native']['directed_arcs_used']:,} 条有向弧。",
     f"深圳：{data['shenzhen_native']['raw_nodes']:,} 个原始节点，{data['shenzhen_native']['raw_edge_rows']:,} 条原始弧；保留 {data['shenzhen_native']['largest_strong_component_nodes']:,} 个节点、{data['shenzhen_native']['directed_arcs_used']:,} 条有向弧。",
     '济南 OD 来自综述发布的扰动轨迹工作负载；深圳本轮仅做均匀随机 OD 的方向性诊断，没有训练深圳模型。两者的诊断比例不能直接解释为城市固有差异。',
     '济南节点 ID 映射由原始经纬度投影与综述坐标逐节点核验，最大误差小于 3e-9 米。新标签用 SciPy 批量 Dijkstra 生成，各图再抽取 128 个方向用 NetworkX 独立核验。','',
     '## 两个数据协议细节','',
     '综述发布的济南训练/验证/测试文件是 400,000/50,000/50,000 行。按无序 OD 分组后，训练与测试有 1,746 组重复，训练与验证 1,590 组，验证与测试 217 组。本轮采用 test > validation > train 的优先顺序去除跨集合重叠；原始数据文件不修改。每个无序 OD 组等权，并同时监督正反向，因此不再等同于原始单向工作负载的频次权重。',
     '512 条综述标签抽检中，原始 .edges 的零权边导致 15 条查询相差 1 米。把存储边长下限设为 1 米后，512 条全部吻合。这是抽样验证得到的一致约定，不是全部 50 万标签的证明。本轮原生有向/无向实验另用原始十进制边长重新标注，没有直接套用无向标签。','',
     '## 势差结构诊断（不是学习模型精度）','',
     '方向差 A=(d_uv-d_vu)/2，对称分量 S=(d_uv+d_vu)/2；仅用训练 OD 拟合 h(v)-h(u)。非对称程度 alpha=abs(d_uv-d_vu)/S。',
     '| 诊断 | 济南：轨迹派生 OD | 深圳：均匀 OD |','|---|---:|---:|',
     f"| 可辨识测试 OD / 总测试 OD | {d['identifiable_pair_count']:,} / {d['input_count']:,} | {s['identifiable_pair_count']:,} / {s['input_count']:,} |",
     f"| alpha ≥ 20% 的比例 | {100*d['asymmetry_ge_20pct_fraction']:.3f}% | {100*s['asymmetry_ge_20pct_fraction']:.3f}% |",
     f"| 解释的方向差平方能量 | {100*d['all']['potential_explained_energy_ratio_vs_zero']:.3f}% | {100*s['all']['potential_explained_energy_ratio_vs_zero']:.3f}% |",
     f"| 短距离切片的方向差平方能量解释率 | {100*d['short_train_q25']['potential_explained_energy_ratio_vs_zero']:.3f}% | {100*s['short_train_q25']['potential_explained_energy_ratio_vs_zero']:.3f}% |",
     f"| 真实 S 的中点参考 MRE | {d['all']['oracle_symmetric_midpoint_mre_percent']:.6f}% | {s['all']['oracle_symmetric_midpoint_mre_percent']:.6f}% |",
     f"| 真实 S + 训练拟合势差的 MRE | {d['all']['oracle_S_plus_train_fitted_potential_mre_percent']:.6f}% | {s['all']['oracle_S_plus_train_fitted_potential_mre_percent']:.6f}% |",'',
     '这些解释率不是查询命中率，也不是完整学习模型误差下降比例。真实 S 的重构值不是可部署索引精度；中点也不是 MRE 最优的对称输出。跨训练查询连通分量的势差不可辨识，因此诊断排除相应极少数测试对。','',
     '## 30 组固定预算模型实验','',
     '同一 SAGE 编码器（128 隐藏维、64 输出维），同一批 20,000/5,000/5,000 训练/验证/测试 OD 组，正反向均训练；三种子 42/99/1234，30 epochs，Adam 0.001，batch 2048，SmoothL1。',
     '每种解码器只用训练数据固定校准初始预测均值；所有种子的校准常数在 pilot.json 中。相同种子五种解码器的初始可训练权重、参数量一致。该校准消除一个初始尺度混淆，但不是对所有优化/收敛因素的完整控制。',
     '验证 MRE 选择检查点，之后才预测测试集。测试标签倍增的合成检查在单线程确定性 CPU 下得到逐项完全相同的训练历史及所选权重。原并行 CPU 精确比较失败的微小数值变化已单独保留。','',
     '| 解码器 | 有向济南测试 MRE (%) | 同图无向化测试 MRE (%) |','|---|---:|---:|']
    names={'l1':'L1','tilde_63_1':'tilde-L1（63+1）','tilde_62_2':'tilde-L1（62+2）','linf_symmetric':'对称 L-infinity','linf_asymmetric':'非对称 tilde-L-infinity'}
    for mode,label in names.items():
        x=pilot['summary']['Jinan_native_directed'][mode];y=pilot['summary']['Jinan_native_undirected'][mode]
        lines.append(f"| {label} | {x['mean_test_mre_percent']:.6f} ± {x['std_test_mre_pp']:.6f} | {y['mean_test_mre_percent']:.6f} ± {y['std_test_mre_pp']:.6f} |")
    near=sum(r['best_epoch_in_last_five'] for r in pilot['runs'])
    lines+=['','± 是三个种子之间的样本标准差，不是置信区间。模型包含原始负预测率和高非对称切片结果，详见 pilot_summary.json。',
      f'30 次运行中，有 {near} 次最佳验证轮次落在最后 5 轮；这个数字只是训练预算边界的提醒，不单凭它宣称收敛或未收敛。',
      '有向与无向列的监督标签和图邻接都不同。判断非对称解码器作用应看每列内部同编码器的对比，不把两列整体差值归因于单一因素。','',
      '## 复现实验入口','',
      '源文件清单、Git blob / SHA-256、数据划分、坐标与标签校验：data_audit.json；30 次训练历史和检查点哈希：pilot.json；简表：pilot_summary.json。',
      '源文件恢复/校验：`python scripts/acquire_r2_pinned.py` / `--verify-only`。数据生成：`python scripts/prepare_survey_r2.py`。固定试跑：`python scripts/train_survey_r2.py --device cuda --epochs 30`。',
      '为防止覆盖历史，数据准备和试跑入口拒绝覆盖现有协议/结果文件。重新实验应使用新的独立输出目录或干净工作树，不删除历史证据。',
      '服务器原始文件保存在 data/survey_dcaa89d 与 data/figshare_native_20260913，派生数据在 data/protocol_r2，检查点与逐查询预测在 results/audit-r2-20260913/checkpoints。',
      '依赖新增仅 pyproj 3.7.2，安装在项目 .venv-audit。此报告未宣称完成历史 Harbin/Beijing 复现、原生精确算法效率比较或跨城市模型泛化。']
    (R/'R2_STATUS_ZH.md').write_text('\n'.join(lines)+'\n')
    print('R2_REPORT_WRITTEN',str(R/'R2_STATUS_ZH.md'),flush=True)

if __name__=='__main__':main()
