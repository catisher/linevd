#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphCodeBERT 模型在漏洞检测中的应用

该模块实现了基于 GraphCodeBERT 预训练模型的漏洞检测功能，包括：
1. 函数级和代码行级数据处理
2. 模型定义和训练
3. 模型评估和测试

主要使用的库：
- pandas: 数据处理和分析
- pytorch_lightning: 深度学习训练框架
- transformers: 提供 GraphCodeBERT 预训练模型
- torch: 深度学习核心库
- torchmetrics: 模型评估指标
"""

import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.ivdetect.evaluate as ivde
import torch
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


class BigVulDatasetNLP:
    """函数级代码数据集，用于 GraphCodeBERT 模型输入
    
    该类处理函数级别的代码数据，将代码文本转换为 GraphCodeBERT 可接受的输入格式
    支持训练、验证和测试集的处理，并对训练和验证集进行类别平衡
    """

    def __init__(self, partition="train", random_labels=False):
        """初始化数据集
        
        Args:
            partition: 数据集划分，可选值："train", "val", "test"
            random_labels: 是否使用随机标签（用于基准测试）
        """
        # 加载 BigVul 数据集
        self.df = svdd.bigvul()
        # 筛选指定划分的数据
        self.df = self.df[self.df.label == partition]
        
        # 对于训练和验证集，进行类别平衡处理
        if partition == "train" or partition == "val":
            # 获取所有漏洞样本
            vul = self.df[self.df.vul == 1]
            # 从非漏洞样本中采样与漏洞样本数量相同的样本
            nonvul = self.df[self.df.vul == 0].sample(len(vul), random_state=0)
            # 合并漏洞和非漏洞样本，实现类别平衡
            self.df = pd.concat([vul, nonvul])
        
        # 检查本地是否存在 GraphCodeBERT 模型
        graphcodebert_base_path = svd.external_dir() / "graphcodebert-base"
        # 加载 GraphCodeBERT 分词器
        tokenizer = AutoTokenizer.from_pretrained(graphcodebert_base_path)
        # 分词参数：填充到最大长度、截断过长序列、返回 PyTorch 张量
        tk_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
        
        # 处理文本，添加分隔符作为前缀
        code_texts = [tokenizer.sep_token + " " + ct for ct in self.df.before.tolist()]
        
        # 生成结构信息
        structure_texts = []
        for code in self.df.before.tolist():
            lines = code.split('\n')
            structure = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('if') or line.startswith('for') or line.startswith('while') or line.startswith('do'):
                    structure.append('<control>')
                elif line.startswith('return'):
                    structure.append('<return>')
                elif '=' in line and not line.startswith('//'):
                    structure.append('<assignment>')
                elif line.startswith('{') or line.startswith('}'):
                    structure.append('<bracket>')
                else:
                    structure.append('<statement>')
            structure_texts.append(' '.join(structure))
        
        # 分词处理（代码-结构对）
        tokenized = tokenizer(code_texts, structure_texts, **tk_args)
        # 获取标签列表
        self.labels = self.df.vul.tolist()
        # 如果需要使用随机标签（用于基准测试）
        if random_labels:
            self.labels = torch.randint(0, 2, (len(self.df),)).tolist()
        # 保存分词结果：输入 ID 和注意力掩码
        self.ids = tokenized["input_ids"]
        self.att_mask = tokenized["attention_mask"]

    def __len__(self):
        """获取数据集长度"""
        return len(self.df)

    def __getitem__(self, idx):
        """获取数据集中的单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple: (input_ids, attention_mask, label)
                - input_ids: 分词后的输入 ID 张量
                - attention_mask: 注意力掩码张量
                - label: 样本标签（0=非漏洞，1=漏洞）
        """
        return self.ids[idx], self.att_mask[idx], self.labels[idx]


class BigVulDatasetNLPLine:
    """代码行级数据集，用于 GraphCodeBERT 模型输入
    
    该类处理代码行级别的数据，将每一行代码作为一个样本
    特别关注漏洞相关的代码行，用于更细粒度的漏洞检测
    """

    def __init__(self, partition="train"):
        """初始化数据集
        
        Args:
            partition: 数据集划分，可选值："train", "val", "test"
        """
        # 获取漏洞相关的代码行信息
        linedict = ivde.get_dep_add_lines_bigvul()
        # 加载 BigVul 数据集
        df = svdd.bigvul()
        # 筛选指定划分的数据
        df = df[df.label == partition]
        # 只保留漏洞样本
        df = df[df.vul == 1].copy()
        # 最多采样 1000 个样本，避免数据量过大
        df = df.sample(min(1000, len(df)))

        # 存储代码行和对应的标签
        code_texts = []
        structure_texts = []
        self.labels = []

        # 处理每个漏洞样本
        for row in df.itertuples():
            # 获取当前样本的漏洞行信息
            line_info = linedict[row.id]
            # 合并被移除和依赖添加的漏洞行
            vuln_lines = set(list(line_info["removed"]) + line_info["depadd"])
            # 处理每一行代码
            for idx, line in enumerate(row.before.splitlines(), start=1):
                # 去除空白字符
                line = line.strip()
                # 跳过太短的行（可能是空白行或只有括号的行）
                if len(line) < 5:
                    continue
                # 跳过多行注释
                if line[:2] == "//":
                    continue
                # 添加代码行到文本列表
                code_texts.append(line.strip())
                # 生成结构信息
                if line.startswith('if') or line.startswith('for') or line.startswith('while') or line.startswith('do'):
                    structure_texts.append('<control>')
                elif line.startswith('return'):
                    structure_texts.append('<return>')
                elif '=' in line and not line.startswith('//'):
                    structure_texts.append('<assignment>')
                elif line.startswith('{') or line.startswith('}'):
                    structure_texts.append('<bracket>')
                else:
                    structure_texts.append('<statement>')
                # 添加标签（1=漏洞行，0=非漏洞行）
                self.labels.append(1 if idx in vuln_lines else 0)

        # 检查本地是否存在 GraphCodeBERT 模型
        graphcodebert_base_path = svd.external_dir() / "graphcodebert-base"
        # 加载 GraphCodeBERT 分词器
        tokenizer = AutoTokenizer.from_pretrained(graphcodebert_base_path)
        # 分词参数
        tk_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
        # 处理文本，添加分隔符作为前缀
        code_texts = [tokenizer.sep_token + " " + ct for ct in code_texts]
        # 分词处理（代码-结构对）
        tokenized = tokenizer(code_texts, structure_texts, **tk_args)
        # 保存分词结果
        self.ids = tokenized["input_ids"]
        self.att_mask = tokenized["attention_mask"]

    def __len__(self):
        """获取数据集长度"""
        return len(self.labels)

    def __getitem__(self, idx):
        """获取数据集中的单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple: (input_ids, attention_mask, label)
                - input_ids: 分词后的输入 ID 张量
                - attention_mask: 注意力掩码张量
                - label: 样本标签（0=非漏洞行，1=漏洞行）
        """
        return self.ids[idx], self.att_mask[idx], self.labels[idx]


class BigVulDatasetNLPDataModule(pl.LightningDataModule):
    """PyTorch Lightning 数据模块，用于管理 BigVul 数据集
    
    该类封装了数据加载和批处理逻辑，为训练、验证和测试提供数据加载器
    """

    def __init__(self, DataClass, batch_size: int = 32, sample: int = -1):
        """初始化数据模块
        
        Args:
            DataClass: 数据集类（BigVulDatasetNLP 或 BigVulDatasetNLPLine）
            batch_size: 批处理大小，默认为 32
            sample: 采样数量（-1 表示使用全部数据）
        """
        super().__init__()
        # 初始化训练集
        self.train = DataClass(partition="train")
        # 初始化验证集
        self.val = DataClass(partition="val")
        # 初始化测试集
        self.test = DataClass(partition="test")
        # 设置批处理大小
        self.batch_size = batch_size

    def train_dataloader(self):
        """返回训练数据加载器
        
        Returns:
            DataLoader: 训练数据加载器，带有随机打乱
        """
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        """返回验证数据加载器
        
        Returns:
            DataLoader: 验证数据加载器
        """
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        """返回测试数据加载器
        
        Returns:
            DataLoader: 测试数据加载器
        """
        return DataLoader(self.test, batch_size=self.batch_size)


class LitGraphCodebert(pl.LightningModule):
    """GraphCodeBERT 模型的 PyTorch Lightning 实现
    
    该类封装了 GraphCodeBERT 模型的定义、训练和评估逻辑
    使用预训练的 GraphCodeBERT 模型进行漏洞检测
    """

    def __init__(self, lr: float = 1e-3):
        """初始化模型
        
        Args:
            lr: 学习率，默认为 0.001
        """
        super().__init__()
        # 设置学习率
        self.lr = lr
        # 保存超参数（用于模型检查点和日志）
        self.save_hyperparameters()
        # 检查本地是否存在 GraphCodeBERT 模型
        graphcodebert_base_path = svd.external_dir() / "graphcodebert-base"
        # 加载预训练的 GraphCodeBERT 模型
        self.bert = AutoModel.from_pretrained(graphcodebert_base_path)
        # 第一全连接层：将 GraphCodeBERT 输出的 768 维特征映射到 256 维
        self.fc1 = torch.nn.Linear(768, 256)
        # 第二全连接层：将 256 维特征映射到 2 维（二分类）
        self.fc2 = torch.nn.Linear(256, 2)
        # 准确率指标
        self.accuracy = torchmetrics.Accuracy(task="binary")
        # AUC-ROC 指标（在 epoch 结束时计算）
        self.auroc = torchmetrics.AUROC(task="binary")
        # Matthews 相关系数指标（适用于不平衡数据集）
        self.mcc = torchmetrics.MatthewsCorrCoef(task="binary")

    def forward(self, ids, mask):
        """前向传播
        
        Args:
            ids: 输入 ID 张量
            mask: 注意力掩码张量
            
        Returns:
            logits: 模型输出的 logits 张量（形状：[batch_size, 2]）
        """
        # 禁用 GraphCodeBERT 模型的梯度计算（冻结预训练模型）
        with torch.no_grad():
            bert_out = self.bert(ids, attention_mask=mask)
        # 提取 CLS 标记的输出（用于分类任务）
        fc1_out = self.fc1(bert_out["pooler_output"])
        # 通过第二个全连接层获取最终输出
        fc2_out = self.fc2(fc1_out)
        return fc2_out

    def training_step(self, batch, batch_idx):
        """训练步骤
        
        Args:
            batch: 批次数据，包含 (input_ids, attention_mask, labels)
            batch_idx: 批次索引
            
        Returns:
            loss: 训练损失
        """
        # 解包批次数据
        ids, att_mask, labels = batch
        # 前向传播获取 logits
        logits = self(ids, att_mask)
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)

        # 计算概率分布
        pred = F.softmax(logits, dim=1)
        # 计算准确率
        acc = self.accuracy(pred.argmax(1), labels)
        # 计算 Matthews 相关系数
        mcc = self.mcc(pred.argmax(1), labels)

        # 记录训练指标到日志和进度条
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        self.log("train_mcc", mcc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤
        
        Args:
            batch: 批次数据，包含 (input_ids, attention_mask, labels)
            batch_idx: 批次索引
            
        Returns:
            loss: 验证损失
        """
        # 解包批次数据
        ids, att_mask, labels = batch
        # 前向传播获取 logits
        logits = self(ids, att_mask)
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)

        # 计算概率分布
        pred = F.softmax(logits, dim=1)
        # 计算准确率
        acc = self.accuracy(pred.argmax(1), labels)
        # 计算 Matthews 相关系数
        mcc = self.mcc(pred.argmax(1), labels)

        # 记录验证指标到日志和进度条
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        # 更新 AUC-ROC 指标（使用正类的 logits）
        self.auroc.update(logits[:, 1], labels)
        # 记录 AUC-ROC 指标
        self.log("val_auroc", self.auroc, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        self.log("val_mcc", mcc, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """测试步骤
        
        Args:
            batch: 批次数据，包含 (input_ids, attention_mask, labels)
            batch_idx: 批次索引
            
        Returns:
            loss: 测试损失
        """
        # 解包批次数据
        ids, att_mask, labels = batch
        # 前向传播获取 logits
        logits = self(ids, att_mask)
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        # 更新 AUC-ROC 指标
        self.auroc.update(logits[:, 1], labels)
        # 记录测试 AUC-ROC 指标
        self.log("test_auroc", self.auroc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """配置优化器
        
        Returns:
            optimizer: AdamW 优化器
        """
        # 使用 AdamW 优化器，适用于 transformer 模型
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# 运行示例
if __name__ == "__main__":
    run_id = svd.get_run_id()
    savepath = svd.get_dir(svd.processed_dir() / "graphcodebert" / run_id)
    model = LitGraphCodebert()
    data = BigVulDatasetNLPDataModule(BigVulDatasetNLP, batch_size=64)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer(
        accelerator="cpu",
        auto_lr_find=True,
        default_root_dir=savepath,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
    )
    #tuned = trainer.tune(model, data)
    trainer.fit(model, data)
    trainer.test(model, data)
