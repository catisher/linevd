#!/usr/bin/env python3
"""
导出LineVD项目中myrq1实验的训练数据
专门用于导出不同嵌入类型的对比数据
包括验证损失、AUROC、F1值等指标，并生成对比图
用户可以直接在代码中填写数据
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

print("开始生成不同嵌入类型的对比图...")

# 设置输出目录为storage/outputs/myrq1_results
output_dir = "storage/outputs/myrq1_results"
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录: {output_dir}")

# ==============================================
# 在这里填写不同嵌入类型的性能数据
# 格式：{"embedding_type": "嵌入类型名称", "val_loss": 验证损失, "f1": F1值, "auroc": AUROC值}
# 示例：
# results = [
#     {"embedding_type": "codebert", "val_loss": 0.15, "f1": 0.85, "auroc": 0.92},
#     {"embedding_type": "graphcodebert", "val_loss": 0.12, "f1": 0.88, "auroc": 0.94},
# ]
results = [
    # 在此处添加数据
    # 示例：{"embedding_type": "codebert", "val_loss": 0.15, "f1": 0.85, "auroc": 0.92},
]
# ==============================================

# 检查是否有数据
if not results:
    print("未填写任何数据，退出程序")
    exit(1)

# 导出结果
results_df = pd.DataFrame(results)
output_file = os.path.join(output_dir, "myrq1_results.csv")
results_df.to_csv(output_file, index=False)
print(f"\n✓ 实验结果已导出到 {output_file}")

# 打印结果
print("\nmyrq1实验结果:")
print(results_df)

# 生成对比图
print("\n生成对比图...")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 指标对比图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# F1值对比
if 'f1' in results_df.columns and results_df['f1'].notna().any():
    axes[0].bar(results_df['embedding_type'], results_df['f1'])
    axes[0].set_title('F1值对比')
    axes[0].set_ylabel('F1值')
    axes[0].set_ylim(0, 1)
else:
    axes[0].set_title('F1值对比')
    axes[0].set_ylabel('F1值')
    axes[0].set_ylim(0, 1)
    axes[0].text(0.5, 0.5, '无数据', ha='center', va='center', transform=axes[0].transAxes)

# AUROC对比
if 'auroc' in results_df.columns and results_df['auroc'].notna().any():
    axes[1].bar(results_df['embedding_type'], results_df['auroc'])
    axes[1].set_title('AUROC对比')
    axes[1].set_ylabel('AUROC值')
    axes[1].set_ylim(0, 1)
else:
    axes[1].set_title('AUROC对比')
    axes[1].set_ylabel('AUROC值')
    axes[1].set_ylim(0, 1)
    axes[1].text(0.5, 0.5, '无数据', ha='center', va='center', transform=axes[1].transAxes)

# 验证损失对比
if 'val_loss' in results_df.columns and results_df['val_loss'].notna().any():
    axes[2].bar(results_df['embedding_type'], results_df['val_loss'])
    axes[2].set_title('验证损失对比')
    axes[2].set_ylabel('验证损失')
else:
    axes[2].set_title('验证损失对比')
    axes[2].set_ylabel('验证损失')
    axes[2].text(0.5, 0.5, '无数据', ha='center', va='center', transform=axes[2].transAxes)

plt.tight_layout()

# 保存图表
plot_file = os.path.join(output_dir, "myrq1_embedding_comparison.png")
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✓ 对比图已保存到 {plot_file}")

# 显示图表
plt.show()

print(f"\n导出完成！所有文件已保存到: {output_dir}")
