#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取指定C文件的每行置信度信息
"""

import os
import sys
import glob
import json

import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.dclass as svddc
import sastvd.linevd as lvd
import torch as th


def preds(model, datapartition, vid):
    """获取模型对特定代码片段的预测结果。"""
    # 创建ID到索引的映射
    id2idx = {v: k for k, v in datapartition.idx2id.items()}
    idx = id2idx[vid]
    # 获取图数据
    g = datapartition[idx]
    # 模型预测
    ret_logits = model(g, test=True)
    # 计算行级预测概率
    line_probs = th.nn.functional.softmax(ret_logits[0], dim=1)
    # 取漏洞的概率（索引为1）
    line_ranks = line_probs[:, 1]
    # 组合预测结果、行号和真实标签
    ret = list(zip(line_ranks, g.ndata["_LINE"], g.ndata["_VULN"]))
    # 转换为列表格式
    ret = [[i[0].item(), i[1].item(), i[2].item()] for i in ret]
    # 按预测置信度降序排序
    ret = sorted(ret, key=lambda x: x[0], reverse=True)
    return ret


def get_file_id(filename):
    """从文件名中提取ID"""
    return int(os.path.basename(filename).split('.')[0])


def main():
    # 检查点路径
    checkpoint_path = "/home/wmy/linevd/storage/processed/raytune_baseline_-1/202604070632_2f49f45_规范实验/tune_linevd_baseline/train_linevd_7ea25_00000_0_batch_size=256,embtype=codebert,gamma=2,gatdropout=0.2000,gnntype=gat,gtype=pdg_raw,hdropout=0.3000,hfe_2026-04-07_06-32-29/checkpoint_000099"
    
    # 检查检查点文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        return
    
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
                return
            
            # 使用第一个找到的检查点文件
            checkpoint_file = os.path.join(checkpoint_path, checkpoint_files_in_dir[0])
            print(f"在目录中找到检查点文件: {checkpoint_file}")
            checkpoint_path = checkpoint_file
        except Exception as e:
            print(f"读取目录时出错: {e}")
            return
    
    print(f"使用检查点文件: {checkpoint_path}")
    
    # 加载模型
    print("\n加载模型...")
    try:
        model = lvd.LitGNN.load_from_checkpoint(checkpoint_path, strict=False)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
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
        return
    
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
        return
    
    # 创建数据模块
    data = lvd.BigVulDatasetLineVDDataModule(**datamodule_args)
    
    # 准备数据
    data.setup(stage="fit")
    
    # 目标文件ID
    target_id = 188591
    
    # 检查目标ID是否在数据集中
    found = False
    for datapartition in [data.train, data.val, data.test]:
        if target_id in datapartition.idx2id.values():
            print(f"\n找到目标ID {target_id} 在 {datapartition.partition} 分区中")
            # 获取预测结果
            predictions = preds(model, datapartition, target_id)
            
            # 打印预测结果
            print("\n每行的置信度信息:")
            print("行号 | 置信度 | 是否为漏洞")
            print("-" * 30)
            
            # 按行号排序
            predictions_sorted = sorted(predictions, key=lambda x: x[1])
            
            for pred in predictions_sorted:
                confidence = pred[0]
                line = pred[1]
                is_vuln = "是" if pred[2] == 1 else "否"
                print(f"{line:4d} | {confidence:.4f} | {is_vuln}")
            
            found = True
            break
    
    if not found:
        print(f"\n错误: 未找到目标ID {target_id} 在数据集中")


if __name__ == "__main__":
    main()
