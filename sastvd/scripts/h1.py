#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""代码漏洞预测结果可视化生成脚本。

该脚本用于生成LineVD模型预测结果的HTML可视化文件，帮助直观地展示模型对代码漏洞的检测结果。
主要功能包括：
1. 获取模型对特定代码片段的预测结果
2. 将预测结果转换为可视化HTML格式
3. 查找性能良好的预测示例并保存
"""

import os
import glob
import json
from glob import glob

import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.dclass as svddc
import sastvd.helpers.hljs as hljs
import sastvd.helpers.rank_eval as svdhr
import sastvd.linevd as lvd
import torch as th
from tqdm import tqdm


def preds(model, datapartition, vid):
    """获取模型对特定代码片段的预测结果。
    
    参数:
        model: 训练好的LineVD模型
        datapartition: 数据分区(训练集、验证集或测试集)
        vid: 代码片段的唯一标识符
        
    返回:
        list: 按预测置信度降序排列的结果列表，每个元素包含[预测置信度, 行号, 是否为漏洞]
    """
    # 创建ID到索引的映射
    id2idx = {v: k for k, v in datapartition.idx2id.items()}
    idx = id2idx[vid]
    # 获取图数据
    g = datapartition[idx]
    # 模型预测
    ret_logits = model(g, test=True)
    # 计算行级预测概率
    line_ranks = th.nn.functional.softmax(ret_logits[0], dim=1)[:, 1]
    # 对预测结果进行三次方处理，增强高置信度预测的区分度
    line_ranks = [i ** 3 for i in line_ranks]
    # 组合预测结果、行号和真实标签
    ret = list(zip(line_ranks, g.ndata["_LINE"], g.ndata["_VULN"]))
    # 转换为列表格式
    ret = [[i[0].item(), i[1].item(), i[2].item()] for i in ret]
    # 按预测置信度降序排序
    ret = sorted(ret, key=lambda x: x[0], reverse=True)
    return ret


def save_html_preds(vid, model, data):
    """将模型预测结果保存为HTML可视化文件。
    
    参数: 
        vid: 代码片段的唯一标识符
        model: 训练好的LineVD模型
        data: 数据分区
    """
    # 获取行级预测结果
    line_preds = preds(model, data, vid)
    # 获取真实漏洞行号
    vulns = [i[1] for i in line_preds if i[2] == 1]

    norm_vulns = []
    # 对前5个预测结果进行归一化处理
    for idx, i in enumerate(line_preds[:5]):
        norm_vulns.append([0.7 - (0.15 * (idx)), i[1], i[2]])

    # 构建行号到预测置信度的映射
    line_preds = {i[1] - 1: i[0] for i in norm_vulns}
    # 生成HTML可视化文件
    hljs.linevd_to_html(svddc.BigVulDataset.itempath(vid), line_preds, vulns)


if __name__ == "__main__":
    # 使用与generate_baseline_data.py相同的检查点路径
    checkpoint_path = "/home/wmy/linevd/storage/processed/raytune_baseline_-1/202604070632_2f49f45_规范实验/tune_linevd_baseline/train_linevd_7ea25_00000_0_batch_size=256,embtype=codebert,gamma=2,gatdropout=0.2000,gnntype=gat,gtype=pdg_raw,hdropout=0.3000,hfe_2026-04-07_06-32-29/checkpoint_000099"
    
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
        exit(1)
    
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
            "batch_size": config.get("batch_size", 256),
            "nsampling_hops": config.get("nsampling_hops", 2),
            "gtype": config.get("gtype", "pdg+raw"),
            "splits": config.get("splits", "default"),
            "feat": config.get("embtype", "codebert")
        }
        print(f"使用配置: {datamodule_args}")
    except Exception as e:
        print(f"错误: 读取配置文件时出错: {e}")
        exit(1)
    
    # 创建数据模块
    data = lvd.BigVulDatasetLineVDDataModule(**datamodule_args)
    
    # 创建训练器并测试模型
    print("\n测试模型...")
    trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir="/tmp/")
    trainer.test(model, data)

    # 打印语句级指标
    print(model.res2mt)

    # 查找合适的可视化示例
    datapartition = data.train
    # 加载CVE数据字典
    cve_dict = svdd.bigvul_cve()
    stats = []
    # 遍历所有数据分区
    for datapartition in [data.train, data.val, data.test]:
        # 筛选漏洞样本
        temp_df = datapartition.df[datapartition.df.vul == 1]
        # 筛选代码长度适中的样本(300-1000个字符)
        temp_df = temp_df[
            (temp_df.before.str.len() > 300) & (temp_df.before.str.len() < 1000)
        ]
        # 设置模型为评估模式
        model.eval()
        # 遍历筛选后的样本
        for i in tqdm(range(len(temp_df))):
            sample = temp_df.iloc[i]
            # 获取预测结果
            p = preds(model, datapartition, sample.id)
            # 提取真实标签
            sorted_pred = [i[2] for i in p]
            try:
                # 计算前5个预测结果的精确率
                prec5 = svdhr.precision_at_k(sorted_pred, 5)
                # 如果精确率大于0.5，保存可视化结果
                if prec5 > 0.5:
                    save_html_preds(sample.id, model, datapartition)
                    # 记录统计信息
                    stats.append(
                        {
                            "vid": sample.id,
                            "cve": cve_dict[sample.id],  # CVE编号
                            "p@5": prec5,  # 前5个预测的精确率
                            "gt_vul": sum(sorted_pred),  # 真实漏洞数量
                            "len": len(sorted_pred),  # 代码总行数
                            "vul_ratio": sum(sorted_pred) / len(sorted_pred),  # 漏洞比例
                        }
                    )
            except Exception as E:
                print(E)
                continue
    # 保存统计信息到CSV文件
    pd.DataFrame.from_records(stats).to_csv(
        svd.outputs_dir() / "visualise.csv", index=0
    )