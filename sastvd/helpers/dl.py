#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习辅助模块

该模块提供了LineVD项目中深度学习相关的辅助功能，包括：
1. 张量内存检查工具
2. 批处理字典类，方便数据在设备间转移
3. 自定义数据集模板
4. 动态RNN包装器，支持打包序列的RNN操作
5. 序列填充函数，用于DataLoader的collate_fn

主要使用的库：
- torch: 深度学习框架
- torch.nn: 神经网络模块
- torch.utils.data: 数据加载和处理
- fnmatch: 文件匹配工具
"""
from fnmatch import fnmatch

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset


def tensor_memory(debug="len", verbose=1):
    """检查内存中所有的张量对象
    
    Args:
        debug (str): 调试模式，可选值：
            - "len": 仅打印张量数量
            - "values": 打印所有张量的详细信息（类型、设备、尺寸）
            默认为"len"
        verbose (int): 详细程度，0表示不打印任何信息，1表示打印信息，默认为1
        
    Returns:
        list: 包含所有张量信息的列表
        
    实现细节：
    - 使用gc模块获取内存中的所有对象
    - 检查对象是否为张量或包含张量数据的对象
    - 收集张量的类型、设备和尺寸信息
    - 根据debug参数打印相应的信息
    """
    import gc

    import torch

    ret = []
    for obj in gc.get_objects():
        try:
            # 检查对象是否为张量或包含张量数据
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                # 收集张量的详细信息
                ret.append(f"{type(obj)} : {obj.device} : {obj.size()}")
        except:
            # 忽略处理过程中可能发生的异常
            pass
    
    # 根据verbose和debug参数打印信息
    if verbose > 0:
        if debug == "len":
            print(len(ret))  # 仅打印张量数量
        if debug == "values":
            print("\n".join(ret))  # 打印所有张量的详细信息
    
    return ret


class BatchDict:
    """字典包装类，提供将属性移动到指定设备的辅助功能
    
    该类将字典中的键值对转换为类的属性，并提供了将张量属性移动到GPU的方法。
    
    示例用法：
    bd = BatchDict({"feat": torch.Tensor([1, 2, 3]), "labels": [1, 2, 3]})
    bd.cuda()  # 将张量属性移动到GPU
    """

    def __init__(self, batch: dict):
        """初始化BatchDict类
        
        Args:
            batch (dict): 包含批量数据的字典
            
        实现细节：
        - 将字典中的每个键值对转换为类的属性
        - 自动检测并设置可用的设备（优先使用GPU）
        """
        # 将字典中的键值对转换为类的属性
        for k, v in batch.items():
            setattr(self, k, v)
        # 设置设备（优先使用GPU，如果不可用则使用CPU）
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def cuda(self, exclude: list = []):
        """将相关属性移动到指定设备
        
        Args:
            exclude (list): 需要排除的属性列表，支持通配符匹配（使用fnmatch）
            默认为空列表
            
        实现细节：
        - 遍历类的所有属性
        - 检查属性是否在排除列表中
        - 将张量类型的属性移动到指定设备
        """
        for i in self.__dict__:
            skip = False
            # 检查属性是否需要排除
            for j in exclude:
                if fnmatch(i, j):
                    skip = True
            if skip:
                continue
            # 将张量属性移动到设备
            if hasattr(self, i):
                if isinstance(getattr(self, i), torch.Tensor):
                    setattr(self, i, getattr(self, i).to(self._device))

    def __repr__(self):
        """获取类的字符串表示
        
        Returns:
            str: 类的字符串表示，包含所有属性
        """
        return str(self.__dict__)

    def __getitem__(self, key):
        """支持通过键访问属性
        
        Args:
            key (str): 属性名称
            
        Returns:
            Any: 属性值
        """
        return getattr(self, key)


class CustomDataset(Dataset):
    """自定义数据集模板类
    
    该类继承自torch.utils.data.Dataset，提供了一个简单的数据集模板，可以根据需要进行扩展。
    适用于处理序列数据，支持获取数据长度信息。
    """

    def __init__(self, X, y):
        """初始化CustomDataset类
        
        Args:
            X (torch.Tensor): 输入特征数据，形状应为[样本数量, 序列长度, 特征维度]
            y (torch.Tensor): 目标标签数据，形状应为[样本数量]
            
        实现细节：
        - 保存输入特征和目标标签
        - 计算每个样本的有效序列长度（非零元素的数量）
        """
        self.data = X  # 保存输入特征数据
        self.target = y  # 保存目标标签数据
        # 计算每个样本的有效序列长度（通过计算非零元素的数量）
        self.length = [x.count_nonzero(dim=1).count_nonzero().item() for x in X]

    def __getitem__(self, index):
        """通过索引获取样本
        
        Args:
            index (int): 样本索引
            
        Returns:
            tuple: (特征数据, 目标标签, 序列长度)
        """
        x = self.data[index]  # 获取输入特征
        y = self.target[index]  # 获取目标标签
        x_len = self.length[index]  # 获取序列长度
        return x, y, x_len

    def __len__(self):
        """获取数据集的样本数量
        
        Returns:
            int: 数据集的样本数量
        """
        return len(self.data)


class DynamicRNN(nn.Module):
    """RNN包装器，支持处理打包序列
    
    该类封装了标准的RNN模块，使其能够处理不同长度的序列，通过打包序列来提高计算效率。
    
    来源: https://gist.github.com/davidnvq/594c539b76fc52bef49ec2332e6bcd15
    """

    def __init__(self, rnn_module):
        """初始化DynamicRNN包装器
        
        Args:
            rnn_module (nn.Module): 要包装的RNN模块，可以是RNN、LSTM或GRU等
        """
        super().__init__()
        self.rnn_module = rnn_module  # 保存原始的RNN模块

    def forward(self, x, len_x, initial_state=None):
        """前向传播函数
        
        Args:
            x (torch.FloatTensor): 填充后的输入序列张量
                形状: [batch_size, max_seq_len, embed_size]
            len_x (torch.LongTensor): 每个序列的实际长度
                形状: [batch_size]
            initial_state (torch.FloatTensor, optional): RNN的初始（隐藏，细胞）状态
                默认值: None
                
        Returns:
            tuple: (填充后的输出, h_n) 或 (填充后的输出, (h_n, c_n))
                padded_output: torch.FloatTensor
                    所有元素的隐藏状态输出，填充元素的隐藏状态将被赋值为零向量
                    形状: [batch_size, max_seq_len, hidden_size]
                h_n: torch.FloatTensor
                    每个打包序列最后一步的隐藏状态（不包括填充元素）
                    形状: [batch_size, hidden_size]
                c_n: torch.FloatTensor
                    如果rnn_module是RNN，则c_n = None
                    每个打包序列最后一步的细胞状态（不包括填充元素）
                    形状: [batch_size, hidden_size]
        """
        # 1. 首先按照序列长度降序排列
        sorted_len, idx = len_x.sort(dim=0, descending=True)
        sorted_x = x[idx]

        # 2. 将输入转换为打包序列
        packed_x = pack_padded_sequence(sorted_x, lengths=sorted_len, batch_first=True)

        # 3. 处理初始状态（如果提供）
        if initial_state is not None:
            if isinstance(initial_state, tuple):  # LSTM的情况：(h_0, c_0)
                hx = [state[:, idx] for state in initial_state]
            else:
                hx = initial_state[:, idx]  # RNN的情况：h_0
        else:
            hx = None

        # 4. 执行前向传播
        self.rnn_module.flatten_parameters()  # 优化参数内存使用
        packed_output, last_s = self.rnn_module(packed_x, hx)

        # 5. 将打包输出转换为填充序列
        max_seq_len = x.size(1)
        padded_output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=max_seq_len
        )

        # 6. 将序列恢复到原始顺序
        _, reverse_idx = idx.sort(dim=0, descending=False)
        padded_output = padded_output[reverse_idx]

        return padded_output, last_s


def collate_fn_pad_seq(data):
    """用于DataLoader的序列填充函数
    
    该函数将一个批次的样本进行处理，对序列进行填充以保证同一批次的序列长度一致。
    处理后的数据以BatchDict的形式返回，方便后续模型使用。
    
    Args:
        data (list): 包含样本的列表，每个样本是一个元组(特征数据, 标签, 序列长度)
        
    Returns:
        BatchDict: 包含填充后序列的批量数据
        
    实现细节：
    - 从输入数据中解压特征、标签和长度
    - 对特征序列进行填充
    - 创建一个包含多种特征表示的字典
    - 将字典转换为BatchDict对象返回
    """
    # 从输入数据中解压特征、标签和长度
    feat, labels, lengths = zip(*data)
    
    # 对特征序列进行填充，使同一批次的序列长度一致
    feat_padded = pad_sequence(feat, batch_first=True)
    
    # 创建BatchDict对象，包含多种特征表示
    return BatchDict(
        {
            "subseq": feat_padded,  # 原始填充后的特征序列
            "nametype": feat_padded,  # 命名类型特征（与subseq相同，可能用于不同的处理路径）
            "data": torch.stack([feat_padded, feat_padded * 2, feat_padded * 4]),  # 多种特征变换
            "control": torch.stack([feat_padded, feat_padded * 1.5])[:, :, :-10, :],  # 控制特征（截取前n-10个时间步）
            "labels": torch.Tensor(labels).long(),  # 转换为LongTensor类型的标签
            "subseq_lens": torch.Tensor(lengths).long(),  # 转换为LongTensor类型的序列长度
        }
    )
