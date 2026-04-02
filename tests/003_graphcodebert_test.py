#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphCodeBERT 功能测试脚本

该脚本测试 GraphCodeBERT 模型的基本功能，包括：
1. 模型初始化
2. 代码编码
3. 结构信息生成
4. 与 LineVD 模型的集成
"""

import os
import sys

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.abspath('.'))

import sastvd as svd
import sastvd.graphcodebert as gcb
import sastvd.linevd as lvd
import torch


def test_graphcodebert_init():
    """测试 GraphCodeBERT 模型初始化"""
    print("测试 GraphCodeBERT 模型初始化...")
    try:
        model = gcb.GraphCodeBert()
        print("✓ GraphCodeBERT 模型初始化成功")
        return model
    except Exception as e:
        print(f"✗ GraphCodeBERT 模型初始化失败: {e}")
        return None


def test_graphcodebert_encode(model):
    """测试 GraphCodeBERT 编码功能"""
    print("测试 GraphCodeBERT 编码功能...")
    try:
        # 测试代码片段
        code_snippets = [
            "int add(int a, int b) { return a + b; }",
            "void func() { int x = 0; x = 1; }"
        ]
        
        # 生成结构信息
        structure_snippets = []
        for code in code_snippets:
            structure = gcb.generate_structure_info(code)
            structure_snippets.append(structure)
        print(f"✓ 结构信息生成成功: {structure_snippets}")
        
        # 编码代码-结构对
        embeddings = model.encode(code_snippets, structure_snippets)
        print(f"✓ 编码成功，嵌入形状: {embeddings.shape}")
        
        # 测试仅代码编码
        embeddings_only_code = model.encode(code_snippets)
        print(f"✓ 仅代码编码成功，嵌入形状: {embeddings_only_code.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 编码功能测试失败: {e}")
        return False


def test_linevd_graphcodebert_integration():
    """测试 LineVD 与 GraphCodeBERT 的集成"""
    print("测试 LineVD 与 GraphCodeBERT 的集成...")
    try:
        # 测试配置
        config = {
            "hfeat": 512,
            "embtype": "graphcodebert",  # 使用 GraphCodeBERT 嵌入
            "modeltype": "gat2layer",
            "loss": "ce",
            "hdropout": 0.2,
            "gatdropout": 0.2,
            "multitask": "linemethod",
            "stmtweight": 5,
            "gnntype": "gat",
            "scea": 0.7,
            "lr": 1e-3,
            "batch_size": 32,
            "gtype": "cfgcdg",
            "splits": "default"
        }
        
        # 初始化模型
        model = lvd.LitGNN(
            hfeat=config["hfeat"],
            embtype=config["embtype"],
            model=config["modeltype"],
            loss=config["loss"],
            hdropout=config["hdropout"],
            gatdropout=config["gatdropout"],
            multitask=config["multitask"],
            stmtweight=config["stmtweight"],
            gnntype=config["gnntype"],
            scea=config["scea"],
            lr=config["lr"],
        )
        print("✓ LineVD 模型初始化成功（使用 GraphCodeBERT 嵌入）")
        print(f"  嵌入类型: {model.EMBED}")
        print(f"  嵌入维度: {model.hparams.embfeat}")
        
        return True
    except Exception as e:
        print(f"✗ LineVD 集成测试失败: {e}")
        return False


if __name__ == "__main__":
    print("开始测试 GraphCodeBERT 功能...\n")
    
    # 测试 1: 模型初始化
    model = test_graphcodebert_init()
    
    if model:
        # 测试 2: 编码功能
        test_graphcodebert_encode(model)
        
        # 测试 3: LineVD 集成
        test_linevd_graphcodebert_integration()
    
    print("\n测试完成！")
