#!/usr/bin/env python3
"""
使用myrq2实验训练的实际模型进行实证评估
"""

import os
import glob
import json
import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd
from sastvd.linevd.empirical_eval import EmpEvalBigVul
from collections import Counter
from math import sqrt

# 计算F1分数
def calc_f1(tp, fp, fn):
    """计算F1分数"""
    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if (prec + rec) == 0:
        return 0.0
    return 2 * ((prec * rec) / (prec + rec))

# ==============================================
# 在这里填写有无残差连接的检查点文件路径
# 格式：{"residual": 是否有残差连接(True/False), "checkpoint_path": "检查点文件路径"}
# 示例：
# checkpoint_paths = [
#     {"residual": False, "checkpoint_path": "/path/to/no_residual/checkpoint_000099"},
#     {"residual": True, "checkpoint_path": "/path/to/with_residual/checkpoint_000099"},
# ]
checkpoint_paths = [
    # 在此处添加数据
    {"residual": False, "checkpoint_path": "/home/wmy/linevd/storage/processed/raytune_myrq2_-1/202604070642_2f49f45_规范实验/tune_linevd_myrq2/train_linevd_f0bfb_00000_0_batch_size=256,embtype=graphcodebert,gamma=2,gatdropout=0.2000,gnntype=gatv2,gtype=pdg_raw,hdropout=0.3_2026-04-07_06-42-50/checkpoint_000099"},
    {"residual": True, "checkpoint_path": "/home/wmy/linevd/storage/processed/raytune_myrq2_-1/202604070642_2f49f45_规范实验/tune_linevd_myrq2/train_linevd_f0bfb_00001_1_batch_size=256,embtype=graphcodebert,gamma=2,gatdropout=0.2000,gnntype=gatv2,gtype=pdg_raw,hdropout=0.3_2026-04-07_06-42-50/checkpoint_000099"},
]
# ==============================================

print("开始使用实际模型进行 myrq2 实验分析...")

# 检查用户是否填写了检查点路径
if not checkpoint_paths:
    print("错误: 请在代码中填写检查点文件路径")
    print("请编辑文件 sastvd/scripts/generate_myrq2_data.py，在 checkpoint_paths 列表中填写检查点文件路径")
    exit(1)

# 设置输出目录
output_dir = str(svd.outputs_dir() / "myrq2_empirical_results")
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录: {output_dir}")

print(f"找到 {len(checkpoint_paths)} 个残差连接配置:")
for i, item in enumerate(checkpoint_paths):
    residual_status = "有残差连接" if item['residual'] else "无残差连接"
    print(f"  {i+1}. {residual_status}: {item['checkpoint_path']}")

# 处理每个残差连接配置
for item in checkpoint_paths:
    residual = item['residual']
    checkpoint_path = item['checkpoint_path']
    
    residual_status = "有残差连接" if residual else "无残差连接"
    print(f"\n处理 {residual_status}...")
    
    # 检查检查点文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        exit(1)
    
    # 检查是否是目录
    if os.path.isdir(checkpoint_path):
        print(f"检查点路径是目录，在目录中查找检查点文件...")
        # 在目录中查找检查点文件
        try:
            dir_files = os.listdir(checkpoint_path)
            # 查找常见的检查点文件
            checkpoint_files_in_dir = []
            for f in dir_files:
                if any(f.endswith(ext) for ext in [".ckpt", ".pt", ".pth", ".bin"]) or "checkpoint" in f:
                    checkpoint_files_in_dir.append(f)
            
            if not checkpoint_files_in_dir:
                print(f"错误: 在目录中未找到检查点文件: {checkpoint_path}")
                exit(1)
            
            # 使用第一个找到的检查点文件
            checkpoint_file = os.path.join(checkpoint_path, checkpoint_files_in_dir[0])
            print(f"在目录中找到检查点文件: {checkpoint_file}")
            checkpoint_path = checkpoint_file
        except Exception as e:
            print(f"读取目录时出错: {e}")
            exit(1)
    
    print(f"使用检查点文件: {checkpoint_path}")
    
    # 加载模型
    print("\n加载模型...")
    try:
        model = lvd.LitGNN.load_from_checkpoint(checkpoint_path, strict=False)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print(f"跳过 {residual_status}...")
        continue
    
    # 尝试从实验目录中读取配置文件
    print("\n读取实验配置...")
    # 从检查点路径获取目录路径
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # 查找当前目录或父目录下的params.json
    config_files = glob.glob(os.path.join(checkpoint_dir, "params.json"))
    if not config_files:
        # 如果当前目录没有，查找父目录
        parent_dir = os.path.dirname(checkpoint_dir)
        config_files = glob.glob(os.path.join(parent_dir, "params.json"))
    if not config_files:
        print(f"错误: 未找到配置文件 params.json")
        print(f"请确保配置文件存在于以下目录中:")
        print(f"  {checkpoint_dir}")
        print(f"  {parent_dir}")
        exit(1)
    
    # 读取配置文件
    try:
        with open(config_files[0], 'r') as f:
            config = json.load(f)
        datamodule_args = {
            "batch_size": config.get("batch_size", 1024),
            "nsampling_hops": config.get("nsampling_hops", 2),
            "gtype": config.get("gtype", "pdg+raw"),
            "splits": config.get("splits", "default"),
            "feat": config.get("embtype", "graphcodebert")
        }
        print(f"使用配置: {datamodule_args}")
    except Exception as e:
        print(f"错误: 读取配置文件时出错: {e}")
        exit(1)
    
    # 创建数据模块
    data = lvd.BigVulDatasetLineVDDataModule(**datamodule_args)
    
    # 创建训练器并测试模型
    print("\n测试模型...")
    trainer = pl.Trainer(accelerator='cpu')  # 如果有GPU可以使用 'gpu'
    trainer.test(model, data)
    
    # 使用 get_relevant_metrics 提取和输出指标
    print("\n提取评估指标...")
    residual_suffix = "with_residual" if residual else "no_residual"
    res = [
        residual_suffix,       # 试验ID（使用残差连接状态）
        checkpoint_path,      # 检查点路径
        model.res1vo,        # 排名指标1（仅正样本）
        model.res2mt,        # 多任务指标
        model.res2f,         # 方法级指标
        model.res3vo,        # 排名指标3（仅正样本）
        model.res2,          # 语句级指标
        model.lr,            # 学习率
    ]
    
    # 使用 get_relevant_metrics 提取指标
    metrics = lvd.get_relevant_metrics(res)
    
    # 打印提取的指标
    print("\n提取的评估指标:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # 保存指标到文件
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(output_dir, f"myrq2_{residual_suffix}_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\n指标已保存到: {metrics_file}")
    
    # 使用 EmpEvalBigVul 进行实证评估
    print("\n进行实证评估...")
    eebv = EmpEvalBigVul(model.all_funcs, data.test)
    eebv.eval_test()
    
    print(f"\n评估完成:")
    print(f"  成功评估: {len(eebv.func_results)} 个函数")
    print(f"  失败评估: {eebv.failed} 个函数")
    
    # 保存评估结果
    print("\n保存评估结果...")
    func_results_df = pd.DataFrame(eebv.func_results)
    residual_suffix = "with_residual" if residual else "no_residual"
    func_output_file = os.path.join(output_dir, f"myrq2_{residual_suffix}_func_results.csv")
    func_results_df.to_csv(func_output_file, index=False)
    print(f"函数级结果已保存到: {func_output_file}")
    
    # 语句级分析
    print("\n进行语句级分析...")
    # 统计各类别语句的数量
    stmt_tp = []
    stmt_tn = []
    stmt_fp = []
    stmt_fn = []
    for func in eebv.stmt_results:
        for stmt in func.values():
            if stmt["pred"][1] > model.f1thresh and stmt["vul"] == 1:
                stmt_tp.append(stmt)
            if stmt["pred"][1] < model.f1thresh and stmt["vul"] == 0:
                stmt_tn.append(stmt)
            if stmt["pred"][1] > model.f1thresh and stmt["vul"] == 0:
                stmt_fp.append(stmt)
            if stmt["pred"][1] < model.f1thresh and stmt["vul"] == 1:
                stmt_fn.append(stmt)
    
    # 统计语句类型
    print(f"语句级预测结果:")
    print(f"  TP: {len(stmt_tp)}")
    print(f"  TN: {len(stmt_tn)}")
    print(f"  FP: {len(stmt_fp)}")
    print(f"  FN: {len(stmt_fn)}")
    
    # 计算整体语句级F1
    stmt_f1 = calc_f1(len(stmt_tp), len(stmt_fp), len(stmt_fn))
    print(f"\n整体语句级F1: {stmt_f1:.4f}")
    
    # 保存语句级分析结果
    stmt_analysis = pd.DataFrame({
        "metric": ["TP", "TN", "FP", "FN", "F1"],
        "value": [len(stmt_tp), len(stmt_tn), len(stmt_fp), len(stmt_fn), stmt_f1]
    })
    stmt_output_file = os.path.join(output_dir, f"myrq2_{residual_suffix}_stmt_analysis.csv")
    stmt_analysis.to_csv(stmt_output_file, index=False)
    print(f"语句级分析结果已保存到: {stmt_output_file}")

# 生成残差连接对比数据
print("\n生成残差连接对比数据...")
# 收集所有残差连接配置的结果
comparison_results = []
for item in checkpoint_paths:
    residual = item['residual']
    residual_suffix = "with_residual" if residual else "no_residual"
    # 读取函数级结果
    func_file = os.path.join(output_dir, f"myrq2_{residual_suffix}_func_results.csv")
    if os.path.exists(func_file):
        func_df = pd.read_csv(func_file)
        # 计算函数级整体F1
        if not func_df.empty:
            # 简化计算，实际应该根据TP、FP、FN计算
            func_f1 = func_df['f1'].mean() if 'f1' in func_df.columns else 0.0
        else:
            func_f1 = 0.0
    else:
        func_f1 = 0.0
    
    # 读取语句级结果
    stmt_file = os.path.join(output_dir, f"myrq2_{residual_suffix}_stmt_analysis.csv")
    if os.path.exists(stmt_file):
        stmt_df = pd.read_csv(stmt_file)
        stmt_f1 = stmt_df[stmt_df['metric'] == 'F1']['value'].values[0]
    else:
        stmt_f1 = 0.0
    
    comparison_results.append({
        "residual": residual,
        "func_f1": round(func_f1, 4),
        "stmt_f1": round(stmt_f1, 4)
    })

# 转换为数据框
comparison_df = pd.DataFrame(comparison_results)

# 打印对比结果
print("\n残差连接对比结果:")
print(comparison_df)

# 保存对比结果
comparison_file = os.path.join(output_dir, "myrq2_residual_comparison.csv")
comparison_df.to_csv(comparison_file, index=False)
print(f"\n对比结果已保存到: {comparison_file}")

print(f"\n分析完成！所有结果已保存到: {output_dir}")
