#!/usr/bin/env python3
"""
导出LineVD项目中myrq3实验的训练数据
专门用于导出GNN类型对比实验的对比数据
包括验证损失、AUROC、F1值等指标，并生成对比图
用户可以直接在代码中填写数据
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

print("开始生成GNN类型对比图...")

# 设置输出目录为storage/outputs/myrq3_results
output_dir = "storage/outputs/myrq3_results"
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录: {output_dir}")

# ==============================================
# 在这里填写不同GNN类型的性能数据
# 格式：{"gnn_type": "GNN类型", "val_loss": 验证损失, "f1": F1值, "auroc": AUROC值}
# 示例：
# results = [
#     {"gnn_type": "gcn", "val_loss": 0.18, "f1": 0.82, "auroc": 0.90},
#     {"gnn_type": "gat", "val_loss": 0.15, "f1": 0.85, "auroc": 0.92},
#     {"gnn_type": "gatv2", "val_loss": 0.13, "f1": 0.87, "auroc": 0.93},
# ]

# ╭─────────────────────────────────────────────────────────────╮
# │ Trial train_linevd_35c87_00000 result                       │
# ├─────────────────────────────────────────────────────────────┤
# │ checkpoint_dir_name                       checkpoint_000094 │
# │ time_this_iter_s                                    0.76143 │
# │ time_total_s                                    70337.41173 │
# │ training_iteration                                      190 │
# │ train_loss                                          0.15638 │
# │ val_auroc                                           0.70503 │
# │ val_loss                                            0.22458 │
# ╰─────────────────────────────────────────────────────────────╯
# ╭─────────────────────────────────────────────────────────────╮
# │ Trial train_linevd_35c87_00001 result                       │
# ├─────────────────────────────────────────────────────────────┤
# │ checkpoint_dir_name                       checkpoint_000045 │
# │ time_this_iter_s                                    1.13522 │
# │ time_total_s                                    69508.15659 │
# │ training_iteration                                       92 │
# │ train_loss                                          0.18234 │
# │ val_auroc                                           0.71267 │
# │ val_loss                                            0.20863 │
# ╰─────────────────────────────────────────────────────────────╯
# ╭─────────────────────────────────────────────────────────────╮
# │ Trial train_linevd_35c87_00002 result                       │
# ├─────────────────────────────────────────────────────────────┤
# │ checkpoint_dir_name                       checkpoint_000099 │
# │ time_this_iter_s                                    0.12354 │
# │ time_total_s                                    21383.95245 │
# │ training_iteration                                      200 │
# │ train_loss                                          0.15727 │
# │ val_auroc                                           0.65384 │
# │ val_loss                                            0.24141 │
# ╰─────────────────────────────────────────────────────────────╯
results = [
    {"gnn_type": "gcn", "val_loss": 0.24141, "f1": 0.1508, "auroc": 0.65384},
    {"gnn_type": "gat", "val_loss": 0.22458, "f1": 0.1321, "auroc": 0.70503},
    {"gnn_type": "gatv2", "val_loss": 0.20863, "f1": 0.1623, "auroc": 0.71267},
]
# ==============================================

# 检查是否有数据
if not results:
    print("未填写任何数据，退出程序")
    exit(1)

# 导出结果
if results:
    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "myrq3_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ 实验结果已导出到 {output_file}")
    
    # 打印结果
    print("\nmyrq3实验结果:")
    print(results_df)
    
    # 生成对比图
    print("\n生成对比图...")
    
    # 指标对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # F1值对比
    axes[0].bar(results_df['gnn_type'], results_df['f1'])
    axes[0].set_title('F1 Score Comparison')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_ylim(0, 1)
    
    # AUROC对比
    axes[1].bar(results_df['gnn_type'], results_df['auroc'])
    axes[1].set_title('AUROC Comparison')
    axes[1].set_ylabel('AUROC Score')
    axes[1].set_ylim(0, 1)
    
    # 验证损失对比
    axes[2].bar(results_df['gnn_type'], results_df['val_loss'])
    axes[2].set_title('Validation Loss Comparison')
    axes[2].set_ylabel('Validation Loss')
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = os.path.join(output_dir, "myrq3_gnn_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存到 {plot_file}")
    
    # 显示图表
    plt.show()
else:
    print("\n未找到任何实验结果")

print(f"\n导出完成！所有文件已保存到: {output_dir}")
