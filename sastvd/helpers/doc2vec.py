#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Doc2Vec模型工具模块

该模块提供了Doc2Vec模型的训练、加载和使用功能，用于将文本转换为向量表示。

主要功能：
1. 训练Doc2Vec模型
2. 加载已训练的Doc2Vec模型
3. 提供文本向量推断的接口

主要使用的库：
- gensim: 用于训练和加载Doc2Vec模型
- logging: 用于记录训练过程
- sastvd.helpers.tokenise: 用于文本分词
"""
import logging

import sastvd.helpers.tokenise as svdt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def train_d2v(
    train_corpus,
    vector_size=300,
    window=2,
    min_count=5,
    workers=4,
    epochs=100,
    dm_concat=1,
    dm=1,
):
    """训练Doc2Vec模型
    
    Args:
        train_corpus (list): 训练语料库，包含多个文档字符串
        vector_size (int): 生成的向量维度，默认为300
        window (int): 窗口大小，即当前词与预测词之间的最大距离，默认为2
        min_count (int): 忽略词频低于该值的词，默认为5
        workers (int): 训练时使用的线程数，默认为4
        epochs (int): 训练迭代次数，默认为100
        dm_concat (int): 是否在DM模型中连接上下文向量，1表示连接，0表示不连接，默认为1
        dm (int): 训练算法，1表示使用DM（Distributed Memory）模型，0表示使用DBOW（Distributed Bag of Words）模型，默认为1
        
    Returns:
        Doc2Vec: 训练好的Doc2Vec模型
        
    实现细节：
    - 配置日志记录，用于显示训练过程
    - 将训练语料库转换为TaggedDocument格式
    - 初始化Doc2Vec模型
    - 构建词汇表
    - 训练模型
    
    示例用法：
    model = train_d2v(train_corpus)
    model.save("d2v.model")  # 保存模型
    """
    # 配置日志记录，用于显示训练过程
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    # 将训练语料库转换为TaggedDocument格式
    train_corpus = [
        TaggedDocument(doc.split(), [i]) for i, doc in enumerate(train_corpus)
    ]
    
    # 初始化Doc2Vec模型
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        dm_concat=dm_concat,
        dm=dm,
    )
    
    # 构建词汇表
    model.build_vocab(train_corpus)
    
    # 训练模型
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    
    return model


def load_d2v(path: str):
    """加载已训练的Doc2Vec模型
    
    Args:
        path (str): 模型文件路径或包含模型文件的目录路径
        
    Returns:
        Doc2Vec: 加载的Doc2Vec模型
        
    实现细节：
    - 检查路径是否以"d2v.model"结尾
    - 如果不是，则自动添加"/d2v.model"到路径末尾
    - 使用Doc2Vec.load()方法加载模型
    
    示例用法：
    path = "bigvul/d2v_False"
    model = load_d2v(path)  # 等价于加载bigvul/d2v_False/d2v.model
    """
    path = str(path)
    
    # 检查路径是否以"d2v.model"结尾，如果不是则自动添加
    if path.split("/")[-1] != "d2v.model":
        path += "/d2v.model"
    
    return Doc2Vec.load(path)


class D2V:
    """Doc2Vec模型封装类
    
    该类封装了Doc2Vec模型，提供了更方便的接口来使用Doc2Vec模型进行文本向量推断。
    """

    def __init__(self, path: str):
        """初始化D2V类
        
        Args:
            path (str): 模型文件路径或包含模型文件的目录路径
            
        实现细节：
        - 调用load_d2v函数加载Doc2Vec模型
        - 将加载的模型保存为类的属性
        """
        self.model = load_d2v(path)  # 加载Doc2Vec模型

    def infer(self, text: str):
        """推断文本的向量表示
        
        Args:
            text (str): 要推断向量的文本
            
        Returns:
            numpy.ndarray: 文本的向量表示
            
        实现细节：
        - 使用svdt.tokenise()函数对文本进行分词
        - 使用Doc2Vec模型的infer_vector方法推断向量
        """
        text = svdt.tokenise(text)  # 对文本进行分词
        return self.model.infer_vector(text.split())  # 推断文本向量
