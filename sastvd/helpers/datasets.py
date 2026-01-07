#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集处理辅助模块

该模块提供了LineVD项目中数据集处理的核心功能，包括：
1. BigVul数据集的加载和预处理
2. 训练/验证/测试集的划分
3. 代码注释的移除
4. GloVe和Doc2Vec词嵌入模型的训练

主要使用的库：
- pandas: 数据处理和分析
- re: 正则表达式用于代码注释移除
- sklearn: 机器学习工具，用于数据集划分
- sastvd: 项目内部模块，提供核心功能支持
"""
import os
import re

import pandas as pd
import sastvd as svd
import sastvd.helpers.doc2vec as svdd2v
import sastvd.helpers.git as svdg
import sastvd.helpers.glove as svdglove
import sastvd.helpers.tokenise as svdt
from sklearn.model_selection import train_test_split


def train_val_test_split_df(df, idcol, labelcol):
    """将DataFrame划分为训练集、验证集和测试集，并添加label列标识样本所属集合
    
    Args:
        df (pd.DataFrame): 包含原始数据的DataFrame
        idcol (str): 用于标识唯一样本的列名
        labelcol (str): 用于分类的标签列名
    
    Returns:
        pd.DataFrame: 添加了"label"列的DataFrame，label值为"train"、"val"或"test"
    
    划分比例：
    - 训练集: 80%
    - 验证集: 10%
    - 测试集: 10%
    """
    X = df[idcol]
    y = df[labelcol]
    train_rat = 0.8
    val_rat = 0.1
    test_rat = 0.1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_rat, random_state=1
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=test_rat / (test_rat + val_rat), random_state=1
    )
    X_train = set(X_train)
    X_val = set(X_val)
    X_test = set(X_test)

    def path_to_label(path):
        if path in X_train:
            return "train"
        if path in X_val:
            return "val"
        if path in X_test:
            return "test"

    df["label"] = df[idcol].apply(path_to_label)
    return df


def remove_comments(text):
    """从代码中删除注释，同时保留字符串字面量中的内容
    
    Args:
        text (str): 包含注释的代码字符串
        
    Returns:
        str: 已删除注释的代码字符串
        
    实现原理：
    1. 使用正则表达式匹配以下内容：
       - // 单行注释
       - /* */ 多行注释
       - '...' 单引号字符串字面量
       - "..." 双引号字符串字面量
       
    2. 对于匹配到的内容：
       - 如果是注释（以/开头），则替换为空格（而非空字符串，避免破坏代码结构）
       - 如果是字符串字面量，则保持不变
       
    3. 正则表达式标志：
       - re.DOTALL: 使.匹配包括换行符在内的所有字符
       - re.MULTILINE: 使^和$匹配每一行的开头和结尾
    """

    def replacer(match):
        """替换函数，根据匹配内容决定替换策略"""
        s = match.group(0)
        # 如果是注释（以/开头），替换为空格
        if s.startswith("/"):
            return " "  # 使用空格而不是空字符串，避免破坏代码结构
        # 如果是字符串字面量，保持不变
        else:
            return s

    # 正则表达式模式，匹配注释和字符串字面量
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    # 执行替换操作
    return re.sub(pattern, replacer, text)


def generate_glove(dataset="bigvul", sample=False, cache=True):
    """为分词后的数据集生成GloVe词嵌入模型
    
    Args:
        dataset (str): 数据集名称，默认为"bigvul"
        sample (bool): 是否使用样例数据集（仅用于测试），默认为False
        cache (bool): 是否使用缓存结果，如果已存在预训练模型则直接返回，默认为True
        
    Returns:
        None
        
    实现流程：
    1. 检查是否已存在预训练的GloVe模型，如果存在且cache为True则直接返回
    2. 加载指定数据集（目前仅支持bigvul）
    3. 仅使用训练集样本进行词嵌入训练
    4. 对代码进行分词处理
    5. 将分词后的代码行保存为语料库文件
    6. 使用glove模块训练GloVe模型
    """
    # 创建GloVe模型保存目录
    savedir = svd.get_dir(svd.processed_dir() / dataset / f"glove_{sample}")
    
    # 检查是否已存在预训练模型
    if os.path.exists(savedir / "vectors.txt") and cache:
        svd.debug("Already trained GloVe.")
        return
    
    # 加载数据集（目前仅支持bigvul）
    if dataset == "bigvul":
        df = bigvul(sample=sample)
    
    # 设置训练迭代次数：样例模式下迭代2次，完整模式下迭代500次
    MAX_ITER = 2 if sample else 500

    # 仅使用训练集样本进行词嵌入训练
    samples = df[df.label == "train"].copy()

    # 预处理：对代码进行分词处理
    samples.before = svd.dfmp(
        samples, svdt.tokenise_lines, "before", cs=200, desc="Get lines: "
    )
    
    # 将分词后的代码行展平为一维列表
    lines = [i for j in samples.before.to_numpy() for i in j]

    # 保存语料库到文件
    savedir = svd.get_dir(svd.processed_dir() / dataset / f"glove_{sample}")
    with open(savedir / "corpus.txt", "w") as f:
        f.write("\n".join(lines))

    # 训练GloVe模型
    CORPUS = savedir / "corpus.txt"
    svdglove.glove(CORPUS, MAX_ITER=MAX_ITER)


def generate_d2v(dataset="bigvul", sample=False, cache=True, **kwargs):
    """为分词后的数据集训练Doc2Vec模型
    
    Args:
        dataset (str): 数据集名称，默认为"bigvul"
        sample (bool): 是否使用样例数据集（仅用于测试），默认为False
        cache (bool): 是否使用缓存结果，如果已存在预训练模型则直接返回，默认为True
        **kwargs: 传递给Doc2Vec训练函数的额外参数
        
    Returns:
        None
        
    实现流程：
    1. 检查是否已存在预训练的Doc2Vec模型，如果存在且cache为True则直接返回
    2. 加载指定数据集（目前仅支持bigvul）
    3. 仅使用训练集样本进行词嵌入训练
    4. 对代码进行分词处理
    5. 使用doc2vec模块训练Doc2Vec模型
    6. 测试模型，打印与"memcpy"最相似的词
    7. 保存训练好的模型
    """
    # 创建Doc2Vec模型保存目录
    savedir = svd.get_dir(svd.processed_dir() / dataset / f"d2v_{sample}")
    
    # 检查是否已存在预训练模型
    if os.path.exists(savedir / "d2v.model") and cache:
        svd.debug("Already trained Doc2Vec.")
        return
    
    # 加载数据集（目前仅支持bigvul）
    if dataset == "bigvul":
        df = bigvul(sample=sample)

    # 仅使用训练集样本进行词嵌入训练
    samples = df[df.label == "train"].copy()

    # 预处理：对代码进行分词处理
    samples.before = svd.dfmp(
        samples, svdt.tokenise_lines, "before", cs=200, desc="Get lines: "
    )
    
    # 将分词后的代码行展平为一维列表
    lines = [i for j in samples.before.to_numpy() for i in j]

    # 训练Doc2Vec模型
    model = svdd2v.train_d2v(lines, **kwargs)

    # 测试模型：打印与"memcpy"最相似的词
    most_sim = model.dv.most_similar([model.infer_vector("memcpy".split())])
    for i in most_sim:
        print(lines[i[0]])
    
    # 保存训练好的Doc2Vec模型
    model.save(str(savedir / "d2v.model"))


def bigvul(minimal=True, sample=False, return_raw=False, splits="default"):
    """加载和预处理BigVul数据集
    
    Args:
        minimal (bool): 是否使用最小化版本的数据集（经过预处理和过滤的版本），默认为True
        sample (bool): 是否使用样例数据集（仅用于测试，数据量较小），默认为False
        return_raw (bool): 是否返回原始数据（不进行后处理），默认为False
        splits (str): 数据集划分方式，可选值：
            - "default": 默认的随机划分
            - "crossproject-linux": 跨项目划分，Linux项目作为测试集
            - "crossproject-Chrome": 跨项目划分，Chrome项目作为测试集
            - "crossproject-Android": 跨项目划分，Android项目作为测试集
            - "crossproject-qemu": 跨项目划分，qemu项目作为测试集
    
    Returns:
        pd.DataFrame: 处理后的BigVul数据集
        
    特殊情况处理：
    - id = 177860的样本在before/after字段中不应该包含注释
    
    实现流程：
    1. 尝试加载最小化版本的数据集（如果minimal=True）
    2. 如果最小化版本不存在或加载失败，则加载原始数据集
    3. 移除代码中的注释
    4. 如果return_raw=True，返回原始数据
    5. 生成代码差异信息
    6. 提取函数信息
    7. 后处理：过滤无效样本
    8. 划分数据集（训练集、验证集、测试集）
    9. 保存最小化版本的数据集（如果minimal=True）
    10. 返回处理后的数据集
    """
    # 创建最小化数据集保存目录
    savedir = svd.get_dir(svd.cache_dir() / "minimal_datasets")
    
    # 如果minimal=True，尝试加载最小化版本的数据集
    if minimal:
        try:
            # 读取最小化版本的数据集并删除空值
            df = pd.read_parquet(
                savedir / f"minimal_bigvul_{sample}.pq", engine="fastparquet"
            ).dropna()

            # 加载元数据并按项目统计样本数量
            md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
            md.groupby("project").count().sort_values("id")

            # 加载默认的随机划分方案
            default_splits = svd.external_dir() / "bigvul_rand_splits.csv"
            if os.path.exists(default_splits):
                splits = pd.read_csv(default_splits)
                splits = splits.set_index("id").to_dict()["label"]
                df["label"] = df.id.map(splits)

            # 处理跨项目划分方案
            if "crossproject" in splits:
                # 提取目标项目名称
                project = splits.split("_")[-1]
                # 加载元数据
                md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
                # 获取非目标项目的样本ID
                nonproject = md[md.project != project].id.tolist()
                # 将非目标项目划分为训练集和验证集（9:1）
                trid, vaid = train_test_split(nonproject, test_size=0.1, random_state=1)
                # 获取目标项目的样本ID作为测试集
                teid = md[md.project == project].id.tolist()
                # 创建划分字典
                teid = {k: "test" for k in teid}
                trid = {k: "train" for k in trid}
                vaid = {k: "val" for k in vaid}
                # 合并所有划分
                cross_project_splits = {**trid, **vaid, **teid}
                # 添加label列
                df["label"] = df.id.map(cross_project_splits)

            # 返回加载的最小化数据集
            return df
        except Exception as E:
            # 如果加载失败，打印错误信息并继续加载原始数据集
            print(E)
            pass
    # 确定数据集文件名：样例模式使用小样本文件，否则使用完整数据集
    filename = "MSR_data_cleaned_SAMPLE.csv" if sample else "MSR_data_cleaned.csv"
    # 加载原始数据集
    df = pd.read_csv(svd.external_dir() / filename)
    # 重命名Unnamed: 0列为id
    df = df.rename(columns={"Unnamed: 0": "id"})
    # 添加dataset列，标识数据集名称
    df["dataset"] = "bigvul"

    # 移除代码中的注释
    df["func_before"] = svd.dfmp(df, remove_comments, "func_before", cs=500)
    df["func_after"] = svd.dfmp(df, remove_comments, "func_after", cs=500)

    # 如果return_raw=True，返回原始数据（不进行后处理）
    if return_raw:
        return df

    # 生成代码差异信息
    cols = ["func_before", "func_after", "id", "dataset"]
    svd.dfmp(df, svdg._c2dhelper, columns=cols, ordr=False, cs=300)

    # 提取函数信息并合并到DataFrame中
    df["info"] = svd.dfmp(df, svdg.allfunc, cs=500)
    df = pd.concat([df, pd.json_normalize(df["info"])], axis=1)

    # 后处理：过滤无效样本
    # 1. 只处理有漏洞的样本
    dfv = df[df.vul == 1]
    # 2. 移除没有添加或删除代码行但被标记为漏洞的样本
    dfv = dfv[~dfv.apply(lambda x: len(x.added) == 0 and len(x.removed) == 0, axis=1)]
    # 3. 移除函数结尾异常的样本（不是}或;结尾）
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_before.strip()[-1] != "}" and x.func_before.strip()[-1] != ";",
            axis=1,
        )
    ]
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_after.strip()[-1] != "}" and x.after.strip()[-1:] != ";",
            axis=1,
        )
    ]
    # 4. 移除以");"结尾的函数
    dfv = dfv[~dfv.before.apply(lambda x: x[-2:] == ");")]

    # 5. 移除修改比例过高的样本（修改比例>0.7）
    dfv["mod_prop"] = dfv.apply(
        lambda x: len(x.added + x.removed) / len(x["diff"].splitlines()), axis=1
    )
    dfv = dfv.sort_values("mod_prop", ascending=0)
    dfv = dfv[dfv.mod_prop < 0.7]
    # 6. 移除代码行数过少的函数（少于5行）
    dfv = dfv[dfv.apply(lambda x: len(x.before.splitlines()) > 5, axis=1)]
    # 7. 根据过滤后的漏洞样本ID，保留所有非漏洞样本和过滤后的漏洞样本
    keep_vuln = set(dfv.id.tolist())
    df = df[(df.vul == 0) | (df.id.isin(keep_vuln))].copy()

    # 划分数据集为训练集、验证集和测试集
    df = train_val_test_split_df(df, "id", "vul")

    # 定义需要保存的列
    keepcols = [
        "dataset",
        "id",
        "label",
        "removed",
        "added",
        "diff",
        "before",
        "after",
        "vul",
    ]
    # 保存最小化版本的数据集
    df_savedir = savedir / f"minimal_bigvul_{sample}.pq"
    df[keepcols].to_parquet(
        df_savedir,
        object_encoding="json",
        index=0,
        compression="gzip",
        engine="fastparquet",
    )
    # 保存元数据
    metadata_cols = df.columns[:17].tolist() + ["project"]
    df[metadata_cols].to_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv", index=0)
    # 返回处理后的数据集
    return df


def bigvul_cve():
    """获取BigVul数据集中样本ID到CVE漏洞编号的映射
    
    Returns:
        dict: 键为样本ID，值为对应的CVE漏洞编号
        
    实现流程：
    1. 加载BigVul元数据文件
    2. 提取id和CVE ID列
    3. 将数据转换为字典形式返回
    """
    md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
    ret = md[["id", "CVE ID"]]
    return ret.set_index("id").to_dict()["CVE ID"]
