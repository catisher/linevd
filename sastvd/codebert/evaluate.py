#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CodeBERT 模型评估脚本
使用被注释的代码逻辑进行评估
"""

import torch
import torch.nn.functional as F
import sastvd as svd
import sastvd.helpers.ml as ml
from tqdm import tqdm
from main import LitCodebert, BigVulDatasetNLPDataModule, BigVulDatasetNLP


def evaluate_model(checkpoint_path):
    """评估训练好的 CodeBERT 模型
    
    Args:
        checkpoint_path: 模型检查点路径
    """
    # 加载模型
    print(f"加载模型检查点: {checkpoint_path}")
    model = LitCodebert.load_from_checkpoint(checkpoint_path)
    
    # 准备测试数据
    print("准备测试数据...")
    data = BigVulDatasetNLPDataModule(BigVulDatasetNLP, batch_size=64)
    test_loader = data.test_dataloader()

    # 收集预测结果
    print("开始评估模型...")
    # 检查是否有 GPU
    if False:
        model.cuda()
        all_pred = torch.empty((0, 2)).long().cuda()
        all_true = torch.empty((0)).long().cuda()
        print("使用 GPU 进行评估")
    else:
        all_pred = torch.empty((0, 2)).long()
        all_true = torch.empty((0)).long()
        print("使用 CPU 进行评估")

    for batch in tqdm(test_loader):
        ids, att_mask, labels = batch
        if False:
            ids = ids.cuda()
            att_mask = att_mask.cuda()
            labels = labels.cuda()
        
        with torch.no_grad():
            logits = F.softmax(model(ids, att_mask), dim=1)
        
        all_pred = torch.cat([all_pred, logits])
        all_true = torch.cat([all_true, labels])

    # 计算详细指标
    print("\n计算评估指标...")
    ml.get_metrics_logits(all_true, all_pred)


if __name__ == "__main__":
    # 直接在代码中指定检查点路径
    # 请根据实际情况修改以下路径
    checkpoint_path = "/home/wmy/linevd/storage/processed/codebert/202604150104_663e703_test/lightning_logs/version_0/checkpoints/epoch=1-step=468.ckpt"
    
    # 也可以通过命令行参数指定（如果提供了命令行参数，则使用命令行参数）
    import argparse
    parser = argparse.ArgumentParser(description="评估 CodeBERT 模型")
    parser.add_argument("--checkpoint", help="模型检查点路径（可选，优先于硬编码路径）")
    args = parser.parse_args()
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    
    print(f"使用检查点路径: {checkpoint_path}")
    evaluate_model(checkpoint_path)
