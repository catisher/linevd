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


class BigVulDatasetNLPLine:
    """基于行级别的 BigVul 漏洞检测数据集类
    
    该类用于加载和预处理 BigVul 数据集，适用于行级别的漏洞检测任务
    每个样本对应代码中的一行，标签指示该行是否为漏洞行
    """

    def __init__(self, partition="train"):
        """初始化行级别数据集
        
        Args:
            partition: 数据集划分 ("train", "val", "test")
        """
        # 获取漏洞行信息（依赖添加的行）
        linedict = ivde.get_dep_add_lines_bigvul()
        
        # 加载 BigVul 数据集
        df = svdd.bigvul()
        
        # 选择指定划分的漏洞样本
        df = df[df.label == partition]
        df = df[df.vul == 1].copy()
        
        # 最多使用 1000 个样本以提高效率
        df = df.sample(min(1000, len(df)))

        texts = []  # 存储代码行
        self.labels = []  # 存储行级标签

        # 处理每个漏洞样本的代码行
        for row in df.itertuples():
            line_info = linedict[row.id]
            # 获取该样本中的漏洞行号集合
            vuln_lines = set(list(line_info["removed"]) + line_info["depadd"])
            
            # 遍历代码中的每一行
            for idx, line in enumerate(row.before.splitlines(), start=1):
                line = line.strip()
                
                # 跳过空行和注释行
                if len(line) < 5:
                    continue
                if line[:2] == "//":
                    continue
                
                # 添加代码行和对应的标签（1表示漏洞行，0表示非漏洞行）
                texts.append(line.strip())
                self.labels.append(1 if idx in vuln_lines else 0)

        # 初始化 CodeBert tokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        
        # 配置 tokenizer 参数
        tk_args = {
            "padding": True,      # 自动填充到相同长度
            "truncation": True,   # 超过最大长度时截断
            "return_tensors": "pt" # 返回 PyTorch 张量
        }
        
        # 准备输入文本，添加分隔符
        text = [tokenizer.sep_token + " " + ct for ct in texts]
        
        # 分词处理
        tokenized = tokenizer(text, **tk_args)
        
        # 保存分词后的输入ID和注意力掩码
        self.ids = tokenized["input_ids"]
        self.att_mask = tokenized["attention_mask"]

    def __len__(self):
        """获取数据集大小
        
        Returns:
            int: 代码行样本数量
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """获取指定索引的代码行样本
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple: (input_ids, attention_mask, label) 三元组
        """
        return self.ids[idx], self.att_mask[idx], self.labels[idx]


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
