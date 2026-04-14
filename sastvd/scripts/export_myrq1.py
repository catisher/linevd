#!/usr/bin/env python3
"""
导出LineVD项目中myrq1实验的训练数据
专门用于导出CodeBERT和GraphCodeBERT的对比数据
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

print("开始导出myrq1实验数据...")

# 设置输出目录为storage/outputs/myrq1_results
output_dir = str(svd.outputs_dir() / "myrq1_results")
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录: {output_dir}")

# 查找myrq1实验目录
print(f"在 {str(svd.processed_dir())} 中查找myrq1实验目录...")
myrq1_dirs = glob.glob(str(svd.processed_dir() / "raytune_myrq1_*"))
print(f"找到 {len(myrq1_dirs)} 个myrq1实验目录:")
for i, d in enumerate(myrq1_dirs):
    print(f"  {i+1}. {d}")

if not myrq1_dirs:
    print("未找到myrq1实验目录，退出程序")
    exit(1)

myrq1_dir = myrq1_dirs[0]
print(f"使用myrq1实验目录: {myrq1_dir}")

# 存储实验结果
results = []

# 查找所有试验目录（递归查找包含embtype的目录）
print(f"\n查找 {myrq1_dir} 下的所有试验目录...")
trial_dirs = []
for root, dirs, files in os.walk(myrq1_dir):
    for d in dirs:
        if 'embtype=' in d:
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
    
    # 提取嵌入类型
    embtype = config.get("embtype", "")
    print(f"  嵌入类型: '{embtype}'")
    if embtype not in ["codebert", "graphcodebert"]:
        print(f"  嵌入类型不在预期列表中，跳过")
        continue
    
    print(f"处理 {embtype} 试验...")
    
    # 查找检查点文件
    checkpoint_files = glob.glob(os.path.join(trial_dir, "checkpoints", "*.ckpt"))
    if not checkpoint_files:
        print(f"  未找到检查点文件，跳过")
        continue
    
    print(f"  找到检查点: {checkpoint_files[0]}")
    checkpoint_path = checkpoint_files[0]
    
    # 查找CSV日志文件
    parent_dir = os.path.dirname(trial_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    csv_logs_dir = os.path.join(grandparent_dir, "csv_logs")
    
    csv_files = []
    if os.path.exists(csv_logs_dir):
        csv_files.extend(glob.glob(f"{csv_logs_dir}/**/*.csv", recursive=True))
    
    # 提取验证损失
    best_val_loss = None
    csv_f1 = None
    csv_auroc = None
    
    if csv_files:
        selected_csv = None
        for csv_file in csv_files:
            if embtype == "codebert" and "version_0" in csv_file:
                selected_csv = csv_file
                break
            elif embtype == "graphcodebert" and "version_1" in csv_file:
                selected_csv = csv_file
                break
        
        if not selected_csv and csv_files:
            selected_csv = csv_files[0]
        
        if selected_csv:
            df = pd.read_csv(selected_csv)
            print(f"  CSV文件列名: {list(df.columns)}")
            
            # 提取验证损失
            if 'val_loss_epoch' in df.columns:
                valid_losses = df['val_loss_epoch'].dropna()
                if not valid_losses.empty:
                    best_val_loss = valid_losses.min()
            elif 'val_loss' in df.columns:
                valid_losses = df['val_loss'].dropna()
                if not valid_losses.empty:
                    best_val_loss = valid_losses.min()
            
            # 尝试从CSV中提取F1和AUROC
            if 'val_f1' in df.columns:
                valid_f1s = df['val_f1'].dropna()
                if not valid_f1s.empty:
                    csv_f1 = valid_f1s.max()
            
            if 'val_auroc' in df.columns:
                valid_aurocs = df['val_auroc'].dropna()
                if not valid_aurocs.empty:
                    csv_auroc = valid_aurocs.max()
            elif 'auroc' in df.columns:
                valid_aurocs = df['auroc'].dropna()
                if not valid_aurocs.empty:
                    csv_auroc = valid_aurocs.max()
    
    # 如果CSV中没有F1和AUROC，则加载模型计算
    if csv_f1 is None or csv_auroc is None:
        print(f"  CSV中缺少F1或AUROC数据，加载模型计算...")
        try:
            # 加载模型
            model = lvd.LitGNN.load_from_checkpoint(checkpoint_path)
            model.eval()
            
            # 加载测试数据
            data_module = lvd.BigVulDatasetLineVDDataModule(
                batch_size=256,
                sample=-1,
                methodlevel=False,
                nsampling=False,
                gtype=config.get("gtype", "pdg+raw"),
                splits=config.get("splits", "default"),
                feat=config.get("embtype", "graphcodebert")
            )
            data_module.setup()
            test_loader = data_module.test_dataloader()
            
            # 收集预测结果
            predictions = []
            labels = []
            probabilities = []
            
            with torch.no_grad():
                for batch in test_loader:
                    logits = model(batch)
                    prob = torch.softmax(logits, dim=1)
                    pred = torch.argmax(prob, dim=1)
                    lbl = batch.ndata["_VULN"]
                    
                    predictions.extend(pred.cpu().numpy())
                    labels.extend(lbl.cpu().numpy())
                    probabilities.extend(prob[:, 1].cpu().numpy())
            
            # 计算F1值和AUROC
            if csv_f1 is None:
                csv_f1 = f1_score(labels, predictions)
            if csv_auroc is None:
                csv_auroc = roc_auc_score(labels, probabilities)
            
            print(f"  计算得到 F1值: {csv_f1:.4f}")
            print(f"  计算得到 AUROC: {csv_auroc:.4f}")
            
        except Exception as e:
            print(f"  加载模型计算失败: {e}")
    
    print(f"  验证损失: {best_val_loss:.4f}" if best_val_loss is not None else "  验证损失: 未找到")
    print(f"  F1值: {csv_f1:.4f}" if csv_f1 is not None else "  F1值: 未找到")
    print(f"  AUROC: {csv_auroc:.4f}" if csv_auroc is not None else "  AUROC: 未找到")
    
    # 存储结果
    result = {
        "embedding_type": embtype,
        "val_loss": best_val_loss,
        "f1": csv_f1,
        "auroc": csv_auroc
    }
    
    results.append(result)
    print(f"✓ 成功导出 {embtype} 的结果")

# 导出结果
if results:
    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "myrq1_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ 实验结果已导出到 {output_file}")
    
    # 打印结果
    print("\nmyrq1实验结果:")
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
    else:
        print("未找到有效的指标数据，跳过生成对比图")
else:
    print("\n未找到任何实验结果")

print(f"\n导出完成！所有文件已保存到: {output_dir}")