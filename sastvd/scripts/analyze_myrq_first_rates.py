#!/usr/bin/env python3
"""分析 myrq1-4 实验模型的首排名性能指标。

该脚本用于分析 myrq1-4 实验训练的模型在漏洞检测任务中的"首排名"性能指标，
即模型将最易受攻击的代码行预测为最高排名的频率。
用户需要指定模型检查点路径。
"""
import pickle as pkl
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd
import seaborn as sns


def get_fr(v):
    """获取首个漏洞预测的排名。
    
    参数:
        v: 包含函数预测结果的数据结构
        
    返回:
        rank + 1: 首个真实漏洞在预测中的排名（从1开始）
    """
    # 提取预测概率和真实标签
    zipped = list(zip([i[1] for i in v[0]], v[1]))
    # 按预测概率降序排序
    zipped.sort(reverse=True, key=lambda x: x[0])
    # 找到第一个真实漏洞的排名
    for rank, i in enumerate(zipped):
        if i[1] == 1:
            return rank + 1


def calculate_top10_accuracy(vulns):
    """计算 Top-10 准确率。
    
    衡量存在漏洞函数中，至少有一条真实漏洞代码出现在前十名推荐列表中的比例。
    
    参数:
        vulns: 包含真实漏洞的函数列表
        
    返回:
        float: Top-10 准确率
    """
    count = 0
    for v in vulns:
        # 提取预测概率和真实标签
        zipped = list(zip([i[1] for i in v[0]], v[1]))
        # 按预测概率降序排序
        zipped.sort(reverse=True, key=lambda x: x[0])
        # 检查前10个预测中是否有真实漏洞
        for i, item in enumerate(zipped[:10]):
            if item[1] == 1:
                count += 1
                break
    return count / len(vulns) if vulns else 0


def calculate_ifa(vulns):
    """计算初始误报率（IFA）。
    
    衡量安全分析师在定位特定函数首个真实漏洞代码前，需要排查的错误预测行数。
    
    参数:
        vulns: 包含真实漏洞的函数列表
        
    返回:
        list: 每个函数的 IFA 值
    """
    ifa_values = []
    for v in vulns:
        # 提取预测概率和真实标签
        zipped = list(zip([i[1] for i in v[0]], v[1]))
        # 按预测概率降序排序
        zipped.sort(reverse=True, key=lambda x: x[0])
        # 计算找到第一个真实漏洞前的误报数
        ifa = 0
        found = False
        for item in zipped:
            if item[1] == 1:
                found = True
                break
            else:
                ifa += 1
        if found:
            ifa_values.append(ifa)
    return ifa_values


def analyze_model(checkpoint_path, output_dir, experiment_name):
    """分析指定模型的首排名性能。
    
    参数:
        checkpoint_path: 模型检查点路径
        output_dir: 输出目录
        experiment_name: 实验名称
    """
    print(f"分析模型: {checkpoint_path}")
    
    # 加载模型
    model = lvd.LitGNN.load_from_checkpoint(checkpoint_path, strict=False)
    
    # 加载数据模块
    datamodule_args = {
        "batch_size": 1024,
        "nsampling_hops": 2,
        "gtype": "gat",  # 默认使用 gat，可根据实际情况修改
        "splits": "default",
    }
    data = lvd.BigVulDatasetLineVDDataModule(**datamodule_args)
    
    # 创建训练器
    trainer = pl.Trainer(accelerator="cpu", default_root_dir="/tmp/")
    
    # 测试模型
    print("运行模型测试...")
    trainer.test(model, data)
    
    # 检查语句级指标
    print("\n模型性能指标:")
    print("RESRANK1:")
    print(model.res1vo)
    print("RES2MT:")
    print(model.res2mt)
    print("RESF:")
    print(model.res2f)
    print("RESRANK:")
    print(model.res3vo)
    print("RESLINE:")
    print(model.res2)
    
    # 获取首排名的代码行
    # 筛选出包含漏洞的函数（最大预测概率为1）
    vulns = [i for i in model.all_funcs if max(i[1]) == 1]
    # 进一步筛选出存在真实漏洞的函数
    vulns = [i for i in vulns if i[2].max() == 1]
    
    print(f"\n找到 {len(vulns)} 个包含真实漏洞的函数")
    
    # 计算所有漏洞函数的首排名
    histogram_data = [get_fr(v) for v in vulns]
    
    # 确保输出目录存在
    output_dir = svd.get_dir(output_dir)
    
    # 绘制首排名直方图
    print("生成首排名直方图...")
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    sns.histplot(histogram_data, bins=30)
    plt.title(f"{experiment_name} - 首排名分布")
    plt.xlabel("首排名")
    plt.ylabel("函数数量")
    
    # 保存图表
    plot_file = output_dir / f"{experiment_name}_first_ranking.pdf"
    plt.savefig(plot_file, bbox_inches="tight")
    print(f"首排名直方图已保存到: {plot_file}")
    
    # 保存首排名数据
    data_file = output_dir / f"{experiment_name}_first_ranking_data.pkl"
    with open(data_file, "wb") as f:
        pkl.dump(histogram_data, f)
    print(f"首排名数据已保存到: {data_file}")
    
    # 更详细的绘图设置
    font = {"family": "normal", "weight": "normal", "size": 15}
    matplotlib.rc("font", **font)
    
    # 创建两个子图，分别显示排名<=5和>5的情况
    fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True, figsize=(8, 4))
    sns.histplot([i for i in histogram_data if i <= 5], ax=axs[0], bins=5)
    sns.histplot([i for i in histogram_data if i > 5], ax=axs[1], bins=10)
    axs[0].set_title("排名 ≤ 5")
    axs[1].set_title("排名 > 5")
    axs[0].set_xlabel("首排名")
    axs[1].set_xlabel("首排名")
    axs[0].set_ylabel("函数数量")
    axs[1].set_ylabel("")  # 移除右侧子图的y轴标签
    fig.suptitle(f"{experiment_name} - 首排名分布")
    
    # 保存详细图表
    detailed_plot_file = output_dir / f"{experiment_name}_first_ranking_detailed.pdf"
    plt.savefig(detailed_plot_file, bbox_inches="tight")
    print(f"详细首排名直方图已保存到: {detailed_plot_file}")
    
    # 计算排名<=5和>5的数量
    rank_le_5 = len([i for i in histogram_data if i <= 5])
    rank_gt_5 = len([i for i in histogram_data if i > 5])
    print(f"\n首排名统计:")
    print(f"排名 ≤ 5: {rank_le_5} ({rank_le_5 / len(histogram_data) * 100:.2f}%)")
    print(f"排名 > 5: {rank_gt_5} ({rank_gt_5 / len(histogram_data) * 100:.2f}%)")
    
    # 计算 Top-10 准确率
    top10_accuracy = calculate_top10_accuracy(vulns)
    print(f"\nTop-10 准确率: {top10_accuracy:.4f} ({top10_accuracy * 100:.2f}%)")
    
    # 计算 IFA
    ifa_values = calculate_ifa(vulns)
    if ifa_values:
        avg_ifa = sum(ifa_values) / len(ifa_values)
        print(f"初始误报率 (IFA):")
        print(f"平均 IFA: {avg_ifa:.2f}")
        print(f"最小 IFA: {min(ifa_values)}")
        print(f"最大 IFA: {max(ifa_values)}")
        
        # 保存 IFA 数据
        ifa_file = output_dir / f"{experiment_name}_ifa_data.pkl"
        with open(ifa_file, "wb") as f:
            pkl.dump(ifa_values, f)
        print(f"IFA 数据已保存到: {ifa_file}")
        
        # 绘制 IFA 分布
        print("生成 IFA 分布直方图...")
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        sns.histplot(ifa_values, bins=20)
        plt.title(f"{experiment_name} - IFA 分布")
        plt.xlabel("IFA 值")
        plt.ylabel("函数数量")
        ifa_plot_file = output_dir / f"{experiment_name}_ifa_distribution.pdf"
        plt.savefig(ifa_plot_file, bbox_inches="tight")
        print(f"IFA 分布直方图已保存到: {ifa_plot_file}")
    else:
        print("无法计算 IFA: 没有找到包含真实漏洞的函数")


if __name__ == "__main__":
    # 直接在代码中指定检查点路径和实验名称
    # 请根据实际情况修改以下路径
    checkpoint_paths = [
        {
            "path": "/path/to/myrq1_checkpoint",  # myrq1 模型检查点路径
            "experiment": "myrq1"
        },
        {
            "path": "/path/to/myrq2_checkpoint",  # myrq2 模型检查点路径
            "experiment": "myrq2"
        },
        {
            "path": "/path/to/myrq3_checkpoint",  # myrq3 模型检查点路径
            "experiment": "myrq3"
        },
        {
            "path": "/path/to/myrq4_checkpoint",  # myrq4 模型检查点路径
            "experiment": "myrq4"
        }
    ]
    
    output_dir = "storage/outputs/myrq_first_rates"
    
    # 分析每个模型
    for item in checkpoint_paths:
        checkpoint_path = item["path"]
        experiment_name = item["experiment"]
        
        # 验证检查点路径
        if not os.path.exists(checkpoint_path):
            print(f"错误: 检查点路径不存在: {checkpoint_path}")
            continue
        
        print(f"\n=== 分析 {experiment_name} 实验 ===")
        # 分析模型
        analyze_model(checkpoint_path, output_dir, experiment_name)
    
    print("\n分析完成！")




