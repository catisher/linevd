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

class BigVulDatasetNLPDataModule(pl.LightningDataModule):
    """PyTorch Lightning 数据模块
    
    该类用于管理数据集的加载和批处理，简化训练流程
    """

    def __init__(self, DataClass, batch_size: int = 32, sample: int = -1):
        """初始化数据模块
        
        Args:
            DataClass: 数据集类（BigVulDatasetNLP 或 BigVulDatasetNLPLine）
            batch_size: 批处理大小
            sample: 采样大小（-1 表示使用全部数据）
        """
        super().__init__()
        
        # 创建训练、验证和测试数据集
        self.train = DataClass(partition="train")
        self.val = DataClass(partition="val")
        self.test = DataClass(partition="test")
        
        # 设置批处理大小
        self.batch_size = batch_size

    def train_dataloader(self):
        """创建训练数据加载器
        
        Returns:
            DataLoader: 训练数据加载器
        """
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        """创建验证数据加载器
        
        Returns:
            DataLoader: 验证数据加载器
        """
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        """创建测试数据加载器
        
        Returns:
            DataLoader: 测试数据加载器
        """
        return DataLoader(self.test, batch_size=self.batch_size)

