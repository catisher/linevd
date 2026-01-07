#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集类定义模块

该模块定义了BigVulDataset类，用于将BigVul数据集表示为图数据集，方便后续的图神经网络模型训练和评估。

主要功能：
1. 加载和过滤BigVul数据集
2. 平衡训练集和验证集（解决类别不平衡问题）
3. 验证样本有效性（确保包含必要的节点和边信息）
4. 提供数据集统计信息
5. 支持索引访问和长度查询
"""
import json
from glob import glob
from pathlib import Path

import pandas as pd
import sastvd as svd
import sastvd.helpers.datasets as svdds
import sastvd.helpers.glove as svdglove


class BigVulDataset:
    """将BigVul数据集表示为图数据集的类
    
    该类负责加载、过滤、平衡BigVul数据集，并提供必要的接口以便后续的图神经网络模型使用。
    """

    def __init__(self, partition="train", vulonly=False, sample=-1, splits="default"):
        """初始化BigVulDataset类
        
        Args:
            partition (str): 数据集分区，可选值："train"、"val"、"test"，默认为"train"
            vulonly (bool): 是否只包含有漏洞的样本，默认为False
            sample (int): 是否使用样例数据集（指定样本数量），-1表示使用全部样本，默认为-1
            splits (str): 数据集划分方式，与datasets.py中的bigvul函数参数一致，默认为"default"
            
        执行步骤：
        1. 获取已完成图构建的样本ID
        2. 加载BigVul数据集
        3. 过滤指定分区的样本
        4. 只保留已完成图构建的样本
        5. 平衡训练集和验证集（解决类别不平衡问题）
        6. 调整测试集的正负样本比例
        7. 如果sample>0，使用指定数量的样本（用于调试）
        8. 如果vulonly=True，只保留有漏洞的样本
        9. 验证样本有效性（确保包含必要的节点和边信息）
        10. 建立索引到样本ID的映射
        11. 加载GloVe词向量
        """
        # 1. 获取已完成图构建的样本ID（存在nodes.json文件的样本）
        self.finished = [
            int(Path(i).name.split(".")[0])
            for i in glob(str(svd.processed_dir() / "bigvul/before/*nodes*"))
        ]
        
        # 2. 加载BigVul数据集
        self.df = svdds.bigvul(splits=splits)
        
        # 3. 保存分区信息
        self.partition = partition
        
        # 4. 过滤指定分区的样本
        self.df = self.df[self.df.label == partition]
        
        # 5. 只保留已完成图构建的样本
        self.df = self.df[self.df.id.isin(self.finished)]

        # 6. 平衡训练集和验证集（解决类别不平衡问题）
        if partition == "train" or partition == "val":
            vul = self.df[self.df.vul == 1]  # 获取所有有漏洞的样本
            # 从无漏洞样本中随机采样与有漏洞样本数量相同的样本
            nonvul = self.df[self.df.vul == 0].sample(len(vul), random_state=0)
            # 合并有漏洞和无漏洞样本，实现1:1平衡
            self.df = pd.concat([vul, nonvul])

        # 7. 调整测试集的正负样本比例（无漏洞样本数量不超过有漏洞样本的20倍）
        if partition == "test":
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0]
            nonvul = nonvul.sample(min(len(nonvul), len(vul) * 20), random_state=0)
            self.df = pd.concat([vul, nonvul])

        # 8. 如果sample>0，使用指定数量的样本（用于调试）
        if sample > 0:
            self.df = self.df.sample(sample, random_state=0)

        # 9. 如果vulonly=True，只保留有漏洞的样本
        if vulonly:
            self.df = self.df[self.df.vul == 1]

        # 10. 验证样本有效性（过滤掉没有lineNumber信息的样本）
        self.df["valid"] = svd.dfmp(
            self.df, BigVulDataset.check_validity, "id", desc="Validate Samples: "
        )
        self.df = self.df[self.df.valid]

        # 11. 建立索引到样本ID的映射
        self.df = self.df.reset_index(drop=True).reset_index()  # 重置索引，添加idx列
        self.df = self.df.rename(columns={"index": "idx"})  # 重命名索引列为idx
        self.idx2id = pd.Series(self.df.id.values, index=self.df.idx).to_dict()  # 建立idx到id的映射

        # 12. 加载GloVe词向量
        glove_path = svd.processed_dir() / "bigvul/glove_False/vectors.txt"
        self.emb_dict, _ = svdglove.glove_dict(glove_path)

    def itempath(_id):
        """根据样本ID获取源代码文件路径
        
        Args:
            _id (int): 样本ID
            
        Returns:
            Path: 源代码文件的完整路径
        """
        return svd.processed_dir() / f"bigvul/before/{_id}.c"

    def check_validity(_id):
        """检查指定ID的样本是否包含有效的节点和边信息
        
        有效性检查条件：
        1. 节点文件(.nodes.json)中必须包含多个不同的lineNumber信息
        2. 边文件(.edges.json)中必须包含REACHING_DEF或CDG类型的边
        
        Args:
            _id (int): 样本ID
            
        Returns:
            bool: 如果样本有效则返回True，否则返回False
            
        实现细节：
        - 使用try-except捕获文件读取和解析错误
        - 检查节点文件中的lineNumber集合大小是否大于1
        - 检查边文件中的边类型是否包含REACHING_DEF或CDG
        - 如果任何检查失败或发生异常，返回False
        """
        valid = 0
        try:
            # 检查节点文件是否包含有效的lineNumber信息
            with open(str(BigVulDataset.itempath(_id)) + ".nodes.json", "r") as f:
                nodes = json.load(f)
                lineNums = set()
                for n in nodes:
                    if "lineNumber" in n.keys():
                        lineNums.add(n["lineNumber"])
                        if len(lineNums) > 1:  # 至少包含2个不同的行号
                            valid = 1
                            break
                if valid == 0:
                    return False
            
            # 检查边文件是否包含有效的边类型
            with open(str(BigVulDataset.itempath(_id)) + ".edges.json", "r") as f:
                edges = json.load(f)
                edge_set = set([i[2] for i in edges])  # 获取所有边类型
                # 必须包含REACHING_DEF（到达定义）或CDG（控制依赖图）类型的边
                if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                    return False
                return True
        except Exception as E:
            # 发生任何异常都返回False
            print(E, str(BigVulDataset.itempath(_id)))
            return False

    def get_vuln_indices(self, _id):
        """根据样本ID获取存在漏洞的行索引
        
        Args:
            _id (int): 样本ID
            
        Returns:
            dict: 键为行号，值为1的字典，表示存在漏洞的行
            
        实现细节：
        - 从数据框中查询指定ID的样本
        - 获取该样本的removed字段（包含漏洞行的行号列表）
        - 将行号列表转换为字典格式，方便后续处理
        """
        df = self.df[self.df.id == _id]  # 查询指定ID的样本
        removed = df.removed.item()  # 获取漏洞行的行号列表
        return dict([(i, 1) for i in removed])  # 转换为字典格式

    def stats(self):
        """打印数据集的统计信息
        
        统计信息包括：
        - 不同分区（label）的样本数量
        - 每个分区中不同类别（vul）的样本数量
        
        实现细节：
        - 使用pandas的groupby方法按label和vul进行分组
        - 计算每个分组的样本数量
        - 打印统计结果
        """
        print(self.df.groupby(["label", "vul"]).count()["id"])

    def __getitem__(self, idx):
        """通过索引获取数据集样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 样本的字典表示
            
        实现细节：
        - 使用pandas的iloc方法根据索引获取行
        - 将行转换为字典格式返回
        """
        return self.df.iloc[idx].to_dict()

    def __len__(self):
        """获取数据集的样本数量
        
        Returns:
            int: 数据集的样本数量
        """
        return len(self.df)

    def __repr__(self):
        """获取数据集的字符串表示
        
        Returns:
            str: 包含数据集分区、样本数量和漏洞比例的字符串
            
        实现细节：
        - 计算漏洞样本在总样本中的比例
        - 格式化返回字符串，包含关键信息
        """
        vulnperc = round(len(self.df[self.df.vul == 1]) / len(self), 3)  # 计算漏洞样本比例
        return f"BigVulDataset(partition={self.partition}, samples={len(self)}, vulnperc={vulnperc})"
