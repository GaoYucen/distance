import os
import re
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

log_dir = 'log'
result_files = [
    f for f in os.listdir(log_dir)
    if f.endswith('_results_sample.txt')
]

model_names = []
mean_relative_errors = []
mean_absolute_errors = []

for filename in result_files:
    filepath = os.path.join(log_dir, filename)
    with open(filepath, 'r') as f:
        content = f.read()
        
        # Extract MRE
        match_mre = re.search(r'mean relative error:\s*([0-9\.eE+-]+)', content)
        # Extract MAE
        match_mae = re.search(r'mean absolute error:\s*([0-9\.eE+-]+)', content)
        
        if match_mre and match_mae:
            mean_relative_error = float(match_mre.group(1))
            mean_absolute_error = float(match_mae.group(1))
            
            mean_relative_errors.append(mean_relative_error)
            mean_absolute_errors.append(mean_absolute_error)
            
            # Extract model name
            model_name = filename.replace('_results_sample.txt', '')
            model_names.append(model_name)
        else:
            print(f"Warning: Metrics not found in {filename}")

# Sort by MRE for better visualization
sorted_indices = np.argsort(mean_relative_errors)
model_names = [model_names[i] for i in sorted_indices]
mean_relative_errors = [mean_relative_errors[i] for i in sorted_indices]
mean_absolute_errors = [mean_absolute_errors[i] for i in sorted_indices]

# Create two subplots sharing the x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- Top Subplot: MAE (Line Chart) ---
x = np.arange(len(model_names))
ax1.plot(x, mean_absolute_errors, color='tab:red', marker='o', linewidth=2, markersize=8, label='MAE')
ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, color='tab:red', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax1.set_title('Comparison of MAE and MRE Across Models', fontsize=16, pad=15)

# Add values for MAE
for i, v in enumerate(mean_absolute_errors):
    # Add a small offset to not cover the point
    ax1.text(i, v * 1.05, f'{v:.1f}', ha='center', va='bottom', fontsize=10, color='tab:red', fontweight='bold')

# Expand Y limit slightly for MAE labels
ax1.set_ylim(0, max(mean_absolute_errors) * 1.2)

# --- Bottom Subplot: MRE (Bar Chart) ---
bars = ax2.bar(x, mean_relative_errors, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names))), alpha=0.8, label='MRE')
ax2.set_ylabel('Mean Relative Error (MRE)', fontsize=12, color='tab:blue', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.grid(axis='y', linestyle='--', alpha=0.3)

# X-axis labels (only needed on bottom plot)
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)

# Add values for MRE
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height * 1.02, f'{height:.4f}', ha='center', va='bottom', fontsize=10, color='black')

# Expand Y limit slightly for MRE labels
ax2.set_ylim(0, max(mean_relative_errors) * 1.15)

plt.tight_layout()
plt.savefig(os.path.join('figure', 'mre_mae_comparison.png'))
print("Figure saved to figure/mre_mae_comparison.png")
