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


class LitCodebert(pl.LightningModule):
    """基于 CodeBert 的漏洞检测模型
    
    该类实现了基于 PyTorch Lightning 的 CodeBert 漏洞检测模型
    使用 CodeBert 作为特征提取器，配合全连接层进行分类
    """

    def __init__(self, lr: float = 1e-3):
        """初始化模型
        
        Args:
            lr: 学习率
        """
        super().__init__()
        
        # 设置学习率
        self.lr = lr
        
        # 保存超参数
        self.save_hyperparameters()
        
        # 加载预训练的 CodeBert 模型
        self.bert = AutoModel.from_pretrained("microsoft/codebert-base")
        
        # 全连接层：将 768 维的 CodeBert 输出转换为 2 维的分类结果
        self.fc1 = torch.nn.Linear(768, 256)  # 隐藏层：768 -> 256
        self.fc2 = torch.nn.Linear(256, 2)     # 输出层：256 -> 2（漏洞/非漏洞）
        
        # 评估指标
        self.accuracy = torchmetrics.Accuracy()  # 准确率
        self.auroc = torchmetrics.AUROC(compute_on_step=False)  # ROC-AUC（在 epoch 结束时计算）
        self.mcc = torchmetrics.MatthewsCorrcoef(2)  # Matthew 相关系数（二分类）

    def forward(self, ids, mask):
        """前向传播
        
        Args:
            ids: 输入 token ID 张量
            mask: 注意力掩码张量
            
        Returns:
            torch.Tensor: 形状为 [batch_size, 2] 的分类 logits
        """
        # 禁用 CodeBert 部分的梯度计算，仅微调全连接层
        with torch.no_grad():
            bert_out = self.bert(ids, attention_mask=mask)
        
        # 获取池化后的输出（CLS token）
        pooler_output = bert_out["pooler_output"]
        
        # 全连接层前向传播
        fc1_out = self.fc1(pooler_output)
        fc2_out = self.fc2(fc1_out)
        
        return fc2_out

    def training_step(self, batch, batch_idx):
        """训练步骤
        
        Args:
            batch: 包含 (ids, att_mask, labels) 的批处理数据
            batch_idx: 批处理索引
            
        Returns:
            torch.Tensor: 损失值
        """
        ids, att_mask, labels = batch
        
        # 获取模型输出 logits
        logits = self(ids, att_mask)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)

        # 计算 softmax 概率
        pred = F.softmax(logits, dim=1)
        
        # 计算评估指标
        acc = self.accuracy(pred.argmax(1), labels)  # 准确率
        mcc = self.mcc(pred.argmax(1), labels)        # Matthew 相关系数

        # 记录训练指标
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        self.log("train_mcc", mcc, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤
        
        Args:
            batch: 包含 (ids, att_mask, labels) 的批处理数据
            batch_idx: 批处理索引
            
        Returns:
            torch.Tensor: 损失值
        """
        ids, att_mask, labels = batch
        
        # 获取模型输出 logits
        logits = self(ids, att_mask)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)

        # 计算 softmax 概率
        pred = F.softmax(logits, dim=1)
        
        # 计算评估指标
        acc = self.accuracy(pred.argmax(1), labels)  # 准确率
        mcc = self.mcc(pred.argmax(1), labels)        # Matthew 相关系数

        # 记录验证指标
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.auroc.update(logits[:, 1], labels)  # 更新 ROC-AUC 指标
        self.log("val_auroc", self.auroc, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        self.log("val_mcc", mcc, prog_bar=True, logger=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        """测试步骤
        
        Args:
            batch: 包含 (ids, att_mask, labels) 的批处理数据
            batch_idx: 批处理索引
            
        Returns:
            torch.Tensor: 损失值
        """
        ids, att_mask, labels = batch
        
        # 获取模型输出 logits
        logits = self(ids, att_mask)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        # 更新 ROC-AUC 指标
        self.auroc.update(logits[:, 1], labels)
        
        # 记录测试指标
        self.log("test_auroc", self.auroc, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        """配置优化器
        
        Returns:
            torch.optim.Optimizer: AdamW 优化器
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
