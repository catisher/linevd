#!/usr/bin/env python3
"""
导出LineVD项目中myrq4实验的训练数据
专门用于导出损失函数对比实验的对比数据
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

print("开始导出myrq4实验数据...")

# 设置输出目录为storage/outputs/myrq4_results
output_dir = str(svd.outputs_dir() / "myrq4_results")
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录: {output_dir}")

# 查找myrq4实验目录
print(f"在 {str(svd.processed_dir())} 中查找myrq4实验目录...")
myrq4_dirs = glob.glob(str(svd.processed_dir() / "raytune_myrq4_*"))
print(f"找到 {len(myrq4_dirs)} 个myrq4实验目录:")
for i, d in enumerate(myrq4_dirs):
    print(f"  {i+1}. {d}")

if not myrq4_dirs:
    print("未找到myrq4实验目录，退出程序")
    exit(1)

myrq4_dir = myrq4_dirs[0]
print(f"使用myrq4实验目录: {myrq4_dir}")

# 存储实验结果
results = []

# 查找所有试验目录（递归查找包含loss的目录）
print(f"\n查找 {myrq4_dir} 下的所有试验目录...")
trial_dirs = []
for root, dirs, files in os.walk(myrq4_dir):
    for d in dirs:
        if 'loss=' in d:
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
    
    # 提取损失函数类型
    loss_type = config.get("loss", "")
    print(f"  损失函数类型: '{loss_type}'")
    if loss_type not in ["ce", "sce", "focal"]:
        print(f"  损失函数类型不在预期列表中，跳过")
        continue
    
    print(f"处理 {loss_type} 试验...")
    
    # 查找训练日志
    csv_files = glob.glob(f"{trial_dir}/csv_logs/*.csv")
    if not csv_files:
        print(f"未找到{trial_dir}的训练日志")
        continue
    
    # 读取训练日志
    df = pd.read_csv(csv_files[0])
    
    # 提取最佳验证损失
    best_val_loss = df['val_loss'].min() if 'val_loss' in df.columns else None
    
    # 查找检查点
    checkpoint_files = glob.glob(f"{trial_dir}/checkpoint_*/checkpoint")
    if not checkpoint_files:
        print(f"未找到{trial_dir}的检查点")
        continue
    
    checkpoint_path = checkpoint_files[0]
    
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
                probabilities.extend(prob[:, 1].cpu().numpy())  # 漏洞概率
        
        # 计算F1值和AUROC
        f1 = f1_score(labels, predictions)
        auroc = roc_auc_score(labels, probabilities)
        
        # 存储结果
        result = {
            "loss_type": loss_type,
            "val_loss": best_val_loss,
            "f1": f1,
            "auroc": auroc
        }
        
        results.append(result)
        print(f"✓ 成功导出 {loss_type} 的结果")
        
    except Exception as e:
        print(f"✗ 处理 {trial_dir} 失败: {e}")

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
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 指标对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # F1值对比
    axes[0].bar(results_df['loss_type'], results_df['f1'])
    axes[0].set_title('F1值对比')
    axes[0].set_ylabel('F1值')
    axes[0].set_ylim(0, 1)
    
    # AUROC对比
    axes[1].bar(results_df['loss_type'], results_df['auroc'])
    axes[1].set_title('AUROC对比')
    axes[1].set_ylabel('AUROC值')
    axes[1].set_ylim(0, 1)
    
    # 验证损失对比
    axes[2].bar(results_df['loss_type'], results_df['val_loss'])
    axes[2].set_title('验证损失对比')
    axes[2].set_ylabel('验证损失')
    
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
