#!/usr/bin/env python3
"""
导出LineVD项目中myrq2实验的训练数据
专门用于导出有无残差连接的对比数据
包括验证损失、AUROC、F1值等指标，并生成对比图
"""

import os
import glob
import json
import pandas as pd
import torch
import sastvd as svd
import sastvd.linevd as lvd
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt

print("开始导出myrq2实验数据...")

# 设置输出目录为storage/outputs/myrq2_results
output_dir = str(svd.outputs_dir() / "myrq2_results")
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录: {output_dir}")

# 查找myrq2实验目录
print(f"在 {str(svd.processed_dir())} 中查找myrq2实验目录...")
myrq2_dirs = glob.glob(str(svd.processed_dir() / "raytune_myrq2_*"))
print(f"找到 {len(myrq2_dirs)} 个myrq2实验目录:")
for i, d in enumerate(myrq2_dirs):
    print(f"  {i+1}. {d}")

if not myrq2_dirs:
    print("未找到myrq2实验目录，退出程序")
    exit(1)

myrq2_dir = myrq2_dirs[0]
print(f"使用myrq2实验目录: {myrq2_dir}")

# 存储实验结果
results = []

# 查找所有试验目录（递归查找包含modeltype的目录）
print(f"\n查找 {myrq2_dir} 下的所有试验目录...")
trial_dirs = []
for root, dirs, files in os.walk(myrq2_dir):
    for d in dirs:
        if 'modeltype=' in d:
            trial_dirs.append(os.path.join(root, d))
print(f"找到 {len(trial_dirs)} 个试验目录:")
for i, d in enumerate(trial_dirs):
    print(f"  {i+1}. {d}")

for trial_dir in trial_dirs:
    print(f"\n处理试验目录: {os.path.basename(trial_dir)}")
    if not os.path.isdir(trial_dir):
        print(f"  不是目录，跳过")
        continue
    
    # 读取配置文件
    config_file = os.path.join(trial_dir, "params.json")
    if not os.path.exists(config_file):
        print(f"  未找到params.json文件，跳过")
        continue
    
    print(f"  找到params.json文件")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # 提取模型类型
    modeltype = config.get("modeltype", "")
    print(f"  模型类型: '{modeltype}'")
    
    # 确定是否有残差连接
    if "residual" in modeltype.lower():
        residual = "有残差连接"
    else:
        residual = "无残差连接"
    print(f"  残差连接: {residual}")
    
    # 打印调试信息
    print(f"  试验目录: {trial_dir}")
    parent_dir = os.path.dirname(trial_dir)
    print(f"  父目录: {parent_dir}")
    grandparent_dir = os.path.dirname(parent_dir)
    print(f"  祖父目录: {grandparent_dir}")
    
    # 检查祖父目录的内容（这是csv_logs所在的目录）
    print(f"\n  祖父目录内容:")
    if os.path.exists(grandparent_dir):
        for item in os.listdir(grandparent_dir):
            item_path = os.path.join(grandparent_dir, item)
            if os.path.isdir(item_path):
                print(f"    目录: {item}")
            else:
                print(f"    文件: {item}")
    
    # 检查csv_logs目录的内容
    csv_logs_dir = os.path.join(grandparent_dir, "csv_logs")
    print(f"\n  csv_logs目录内容:")
    if os.path.exists(csv_logs_dir):
        for item in os.listdir(csv_logs_dir):
            item_path = os.path.join(csv_logs_dir, item)
            if os.path.isdir(item_path):
                print(f"    目录: {item}")
                # 检查子目录的内容
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    print(f"      - {subitem}")
            else:
                print(f"    文件: {item}")
    else:
        print(f"    csv_logs目录不存在")
    
    # 查找训练日志（在多个目录中查找）
    csv_files = []
    # 直接检查csv_logs目录及其子目录（如version_0、version_1）
    if os.path.exists(csv_logs_dir):
        # 查找csv_logs目录下的所有CSV文件
        csv_files.extend(glob.glob(f"{csv_logs_dir}/**/*.csv", recursive=True))
    
    print(f"\n  找到 {len(csv_files)} 个CSV文件")
    for i, f in enumerate(csv_files):
        print(f"    {i+1}. {f}")
    
    if not csv_files:
        print(f"  未找到训练日志，跳过")
        continue
    
    # 选择第一个CSV文件
    selected_csv = csv_files[0]
    print(f"  找到训练日志: {selected_csv}")
    
    # 读取训练日志
    df = pd.read_csv(selected_csv)
    
    # 打印CSV文件的列名，以便了解文件结构
    print(f"  CSV文件列名: {list(df.columns)}")
    
    # 打印CSV文件的前几行，以便了解数据结构
    print(f"  CSV文件前5行:")
    print(df.head())
    
    # 提取最佳验证损失
    best_val_loss = None
    if 'val_loss_epoch' in df.columns:
        # 过滤掉NaN值，然后取最小值
        valid_losses = df['val_loss_epoch'].dropna()
        if not valid_losses.empty:
            best_val_loss = valid_losses.min()
    elif 'val_loss' in df.columns:
        valid_losses = df['val_loss'].dropna()
        if not valid_losses.empty:
            best_val_loss = valid_losses.min()
    elif 'valid_loss' in df.columns:
        valid_losses = df['valid_loss'].dropna()
        if not valid_losses.empty:
            best_val_loss = valid_losses.min()
    elif 'validation_loss' in df.columns:
        valid_losses = df['validation_loss'].dropna()
        if not valid_losses.empty:
            best_val_loss = valid_losses.min()
    
    # 尝试从CSV文件中提取F1值和AUROC（如果有）
    f1 = None
    auroc = None
    
    # 查找AUROC值
    if 'val_auroc' in df.columns:
        valid_aurocs = df['val_auroc'].dropna()
        if not valid_aurocs.empty:
            auroc = valid_aurocs.max()
    elif 'auroc' in df.columns:
        valid_aurocs = df['auroc'].dropna()
        if not valid_aurocs.empty:
            auroc = valid_aurocs.max()
    elif 'test_auroc' in df.columns:
        valid_aurocs = df['test_auroc'].dropna()
        if not valid_aurocs.empty:
            auroc = valid_aurocs.max()
    elif 'roc_auc' in df.columns:
        valid_aurocs = df['roc_auc'].dropna()
        if not valid_aurocs.empty:
            auroc = valid_aurocs.max()
    
    # 查找准确率作为参考（CSV文件中没有直接的F1值）
    accuracy = None
    if 'val_acc' in df.columns:
        valid_accs = df['val_acc'].dropna()
        if not valid_accs.empty:
            accuracy = valid_accs.max()
    
    # 打印结果
    print(f"  验证损失: {best_val_loss:.4f}" if best_val_loss is not None else "  验证损失: 未找到")
    print(f"  AUROC: {auroc:.4f}" if auroc is not None else "  AUROC: 未找到")
    print(f"  准确率: {accuracy:.4f}" if accuracy is not None else "  准确率: 未找到")
    
    # 存储结果
    result = {
        "residual": residual,
        "modeltype": modeltype,
        "val_loss": best_val_loss,
        "f1": f1,
        "auroc": auroc
    }
    
    results.append(result)
    print(f"✓ 成功导出 {residual} 的结果")

# 导出结果
if results:
    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "myrq2_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ 实验结果已导出到 {output_file}")
    
    # 打印结果
    print("\nmyrq2实验结果:")
    print(results_df)
    
    # 生成对比图
    print("\n生成对比图...")
    
    # 检查是否有有效的指标值
    has_valid_data = False
    for col in ['f1', 'auroc', 'val_loss']:
        if results_df[col].notna().any():
            has_valid_data = True
            break
    
    if has_valid_data:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 指标对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # F1值对比
        if 'f1' in results_df.columns and results_df['f1'].notna().any():
            axes[0].bar(results_df['residual'], results_df['f1'])
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
            axes[1].bar(results_df['residual'], results_df['auroc'])
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
            axes[2].bar(results_df['residual'], results_df['val_loss'])
            axes[2].set_title('验证损失对比')
            axes[2].set_ylabel('验证损失')
        else:
            axes[2].set_title('验证损失对比')
            axes[2].set_ylabel('验证损失')
            axes[2].text(0.5, 0.5, '无数据', ha='center', va='center', transform=axes[2].transAxes)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = os.path.join(output_dir, "myrq2_residual_comparison.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ 对比图已保存到 {plot_file}")
        
        # 显示图表
        plt.show()
    else:
        print("未找到有效的指标数据，跳过生成对比图")
else:
    print("\n未找到任何实验结果")

print(f"\n导出完成！所有文件已保存到: {output_dir}")