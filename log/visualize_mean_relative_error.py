import os
import re
import matplotlib.pyplot as plt

log_dir = 'log'
result_files = [
    f for f in os.listdir(log_dir)
    if f.endswith('_results_sample.txt')
]

model_names = []
mean_relative_errors = []

for filename in result_files:
    filepath = os.path.join(log_dir, filename)
    with open(filepath, 'r') as f:
        content = f.read()
        match = re.search(r'mean relative error:\s*([0-9\.eE+-]+)', content)
        if match:
            mean_relative_error = float(match.group(1))
            mean_relative_errors.append(mean_relative_error)
            # 模型名提取
            model_name = filename.replace('_results_sample.txt', '')
            model_names.append(model_name)
        else:
            print(f"Warning: mean relative error not found in {filename}")

plt.figure(figsize=(12, 6))
bars = plt.bar(model_names, mean_relative_errors, color=plt.cm.viridis([i/len(model_names) for i in range(len(model_names))]), edgecolor='black')

plt.ylabel('Mean Relative Error', fontsize=14)
plt.title('Comparison of Mean Relative Error Across Models', fontsize=16, pad=20)  # pad参数上移标题
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 去除顶部和右侧边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加数值标签，偏移量加大
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join('figure', 'mean_relative_error_comparison.png'))
