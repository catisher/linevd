#!/usr/bin/env python3
"""
导出LineVD项目中myrq4实验的训练数据
专门用于导出损失函数对比实验的对比数据
包括验证损失、AUROC、F1值等指标，并生成对比图
用户可以直接在代码中填写数据
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

print("开始生成损失函数对比图...")

# 设置输出目录为storage/outputs/myrq4_results
output_dir = "storage/outputs/myrq4_results"
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录: {output_dir}")

# ==============================================
# 在这里填写不同损失函数的性能数据
# 格式：{"loss_type": "损失函数类型", "val_loss": 验证损失, "f1": F1值, "auroc": AUROC值}
# 示例：
# results = [
#     {"loss_type": "ce", "val_loss": 0.16, "f1": 0.83, "auroc": 0.91},
#     {"loss_type": "sce", "val_loss": 0.14, "f1": 0.86, "auroc": 0.93},
#     {"loss_type": "focal", "val_loss": 0.12, "f1": 0.88, "auroc": 0.94},
# ]
# ╭─────────────────────────────────────────────────────────────╮
# │ Trial train_linevd_6e233_00000 result                       │
# ├─────────────────────────────────────────────────────────────┤
# │ checkpoint_dir_name                       checkpoint_000049 │
# │ time_this_iter_s                                    1.31206 │
# │ time_total_s                                    75816.19383 │
# │ training_iteration                                      100 │
# │ train_loss                                          0.56152 │
# │ val_auroc                                           0.71849 │
# │ val_loss                                             0.8064 │
# ╰─────────────────────────────────────────────────────────────╯
# ╭─────────────────────────────────────────────────────────────╮
# │ Trial train_linevd_6e233_00002 result                       │
# ├─────────────────────────────────────────────────────────────┤
# │ checkpoint_dir_name                       checkpoint_000049 │
# │ time_this_iter_s                                    1.06649 │
# │ time_total_s                                    75586.02146 │
# │ training_iteration                                      100 │
# │ train_loss                                          0.18678 │
# │ val_auroc                                           0.70962 │
# │ val_loss                                            0.21572 │
# ╰─────────────────────────────────────────────────────────────╯
results = [
    # 在此处添加数据
    {"loss_type": "ce", "val_loss": 0.8064, "f1": 0.1375, "auroc": 0.70962},
    {"loss_type": "focal", "val_loss": 0.21572, "f1": 0.1402, "auroc": 0.71849},
]
# ==============================================

# 检查是否有数据
if not results:
    print("未填写任何数据，退出程序")
    exit(1)

# 导出结果
if results:
    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "myrq4_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ 实验结果已导出到 {output_file}")
    
    # 打印结果
    print("\nmyrq4实验结果:")
    print(results_df)
    
    # 生成对比图
    print("\n生成对比图...")
    
    # 指标对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # F1值对比
    axes[0].bar(results_df['loss_type'], results_df['f1'])
    axes[0].set_title('F1 Score Comparison')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_ylim(0, 1)
    
    # AUROC对比
    axes[1].bar(results_df['loss_type'], results_df['auroc'])
    axes[1].set_title('AUROC Comparison')
    axes[1].set_ylabel('AUROC Score')
    axes[1].set_ylim(0, 1)
    
    # 验证损失对比
    axes[2].bar(results_df['loss_type'], results_df['val_loss'])
    axes[2].set_title('Validation Loss Comparison')
    axes[2].set_ylabel('Validation Loss')
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = os.path.join(output_dir, "myrq4_loss_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存到 {plot_file}")
    
    # 显示图表
    plt.show()
else:
    print("\n未找到任何实验结果")

print(f"\n导出完成！所有文件已保存到: {output_dir}")
