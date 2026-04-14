#!/usr/bin/env python3
"""
导出LineVD项目中baseline实验的训练数据
专门用于导出不同baseline模型的对比数据
包括验证损失、AUROC、F1值等指标，并生成对比图
用户可以直接在代码中填写数据
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

print("Starting to generate baseline model comparison charts...")

# Set output directory to storage/outputs/baseline_results
output_dir = "storage/outputs/baseline_results"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# ==============================================
# Fill in the performance data for different baseline models here
# Format: {"baseline_name": "baseline model name", "val_loss": validation loss, "f1": F1 score, "auroc": AUROC score}
# Example:
# results = [
#     {"baseline_name": "baseline1", "val_loss": 0.15, "f1": 0.85, "auroc": 0.92},
#     {"baseline_name": "baseline2", "val_loss": 0.12, "f1": 0.88, "auroc": 0.94},
# ]
# ╭─────────────────────────────────────────────────────────────╮
# │ Trial train_linevd_7ea25_00000 result                       │
# ├─────────────────────────────────────────────────────────────┤
# │ checkpoint_dir_name                       checkpoint_000099 │
# │ time_this_iter_s                                    0.53414 │
# │ time_total_s                                    50567.90744 │
# │ training_iteration                                      200 │
# │ train_loss                                          0.73039 │
# │ val_auroc                                           0.54643 │
# │ val_loss                                             0.7853 │
# ╰─────────────────────────────────────────────────────────────╯
results = [
    # Add data here
    {"baseline_name": "linevd", "val_loss": 0.7853, "f1": 0.0647, "auroc": 0.54643},
    {"baseline_name": "mine",  "val_loss": 0.22905, "f1": 0.1752, "auroc": 0.70938},
]
# ==============================================

# Check if there is data
if not results:
    print("No data entered, exiting program")
    exit(1)

# Export results
results_df = pd.DataFrame(results)
output_file = os.path.join(output_dir, "baseline_results.csv")
results_df.to_csv(output_file, index=False)
print(f"\n✓ Experiment results exported to {output_file}")

# Print results
print("\nbaseline experiment results:")
print(results_df)

# Generate comparison chart
print("\nGenerating comparison chart...")

# Indicator comparison chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# F1 score comparison
if 'f1' in results_df.columns and results_df['f1'].notna().any():
    axes[0].bar(results_df['baseline_name'], results_df['f1'])
    axes[0].set_title('F1 Score Comparison')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_ylim(0, 1)
else:
    axes[0].set_title('F1 Score Comparison')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_ylim(0, 1)
    axes[0].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[0].transAxes)

# AUROC comparison
if 'auroc' in results_df.columns and results_df['auroc'].notna().any():
    axes[1].bar(results_df['baseline_name'], results_df['auroc'])
    axes[1].set_title('AUROC Comparison')
    axes[1].set_ylabel('AUROC Score')
    axes[1].set_ylim(0, 1)
else:
    axes[1].set_title('AUROC Comparison')
    axes[1].set_ylabel('AUROC Score')
    axes[1].set_ylim(0, 1)
    axes[1].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[1].transAxes)

# Validation loss comparison
if 'val_loss' in results_df.columns and results_df['val_loss'].notna().any():
    axes[2].bar(results_df['baseline_name'], results_df['val_loss'])
    axes[2].set_title('Validation Loss Comparison')
    axes[2].set_ylabel('Validation Loss')
else:
    axes[2].set_title('Validation Loss Comparison')
    axes[2].set_ylabel('Validation Loss')
    axes[2].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[2].transAxes)

plt.tight_layout()

# Save chart
plot_file = os.path.join(output_dir, "baseline_comparison.png")
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✓ Comparison chart saved to {plot_file}")

# Display chart
plt.show()

print(f"\nExport completed! All files saved to: {output_dir}")
