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
    """BigVul 漏洞检测数据集类
    
    该类用于加载和预处理 BigVul 数据集，适用于基于 CodeBert 的漏洞检测任务
    实现了 __getitem__ 方法，用于 CodeBert 模型的输入格式
    """

    def __init__(self, partition="train", random_labels=False):
        """初始化数据集
        
        Args:
            partition: 数据集划分 ("train", "val", "test")
            random_labels: 是否使用随机标签进行调试
        """
        # 加载 BigVul 数据集
        self.df = svdd.bigvul()
        
        # 选择指定划分的数据
        self.df = self.df[self.df.label == partition]
        
        # 训练和验证集进行类别平衡（漏洞和非漏洞样本数量相同）
        if partition == "train" or partition == "val":
            vul = self.df[self.df.vul == 1]  # 漏洞样本
            nonvul = self.df[self.df.vul == 0].sample(len(vul), random_state=0)  # 随机选择相同数量的非漏洞样本
            self.df = pd.concat([vul, nonvul])
        
        # 初始化 CodeBert tokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        
        # 配置 tokenizer 参数
        tk_args = {
            "padding": True,      # 自动填充到相同长度
            "truncation": True,   # 超过最大长度时截断
            "return_tensors": "pt" # 返回 PyTorch 张量
        }
        
        # 准备输入文本，在每个代码片段前添加分隔符
        text = [tokenizer.sep_token + " " + ct for ct in self.df.before.tolist()]
        
        # 分词处理
        tokenized = tokenizer(text, **tk_args)
        
        # 保存标签
        self.labels = self.df.vul.tolist()
        
        # 如果需要，使用随机标签进行调试
        if random_labels:
            self.labels = torch.randint(0, 2, (len(self.df),)).tolist()
        
        # 保存分词后的输入ID和注意力掩码
        self.ids = tokenized["input_ids"]
        self.att_mask = tokenized["attention_mask"]

    def __len__(self):
        """获取数据集大小
        
        Returns:
            int: 数据集样本数量
        """
        return len(self.df)

    def __getitem__(self, idx):
        """获取指定索引的样本
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple: (input_ids, attention_mask, label) 三元组
        """
        return self.ids[idx], self.att_mask[idx], self.labels[idx]
