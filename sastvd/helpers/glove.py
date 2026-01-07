"""
GloVe词向量工具模块

该模块提供了调用StanfordNLP GloVe的Python包装器，用于训练词向量和获取文本嵌入。

主要功能：
1. 在语料库上训练GloVe词向量模型
2. 加载已训练的GloVe嵌入和词汇表
3. 查找最接近的词嵌入
4. 从文本获取词嵌入（未知词使用零向量）
5. 从句子列表获取平均嵌入

主要使用的库：
- pickle: 用于缓存嵌入数据
- pathlib: 用于路径操作
- numpy: 用于向量计算
- scipy: 用于计算向量距离
- sastvd: 项目核心库
- sastvd.helpers.tokenise: 用于文本分词
"""

import pickle as pkl
from pathlib import Path

import numpy as np
import sastvd as svd
import sastvd.helpers.tokenise as svdt
from scipy import spatial


def glove(
    CORPUS,
    VOCAB_FILE="vocab.txt",
    COOCCURRENCE_FILE="cooccurrence.bin",
    COOCCURRENCE_SHUF_FILE="cooccurrence.shuf.bin",
    SAVE_FILE="vectors",
    VERBOSE=2,
    MEMORY=4.0,
    VOCAB_MIN_COUNT=5,
    VECTOR_SIZE=200,
    MAX_ITER=15,
    WINDOW_SIZE=15,
    BINARY=2,
    NUM_THREADS=8,
    X_MAX=10,
):
    """在语料库上运行StanfordNLP GloVe训练词向量
    
    该函数主要基于GloVe仓库中的demo.sh脚本实现，执行四个主要步骤：
    1. 构建词汇表
    2. 计算共现矩阵
    3. 打乱共现数据
    4. 训练GloVe词向量
    
    Args:
        CORPUS (str): 语料库文件路径
        VOCAB_FILE (str, optional): 词汇表输出文件，默认为"vocab.txt"
        COOCCURRENCE_FILE (str, optional): 共现矩阵输出文件，默认为"cooccurrence.bin"
        COOCCURRENCE_SHUF_FILE (str, optional): 打乱后的共现矩阵输出文件，默认为"cooccurrence.shuf.bin"
        SAVE_FILE (str, optional): 词向量输出文件前缀，默认为"vectors"
        VERBOSE (int, optional): 详细程度级别，默认为2
        MEMORY (float, optional): 分配的内存（GB），默认为4.0
        VOCAB_MIN_COUNT (int, optional): 词汇表中的最小词频，默认为5
        VECTOR_SIZE (int, optional): 词向量维度，默认为200
        MAX_ITER (int, optional): 训练迭代次数，默认为15
        WINDOW_SIZE (int, optional): 上下文窗口大小，默认为15
        BINARY (int, optional): 输出格式（0=文本，1=二进制，2=文本和二进制），默认为2
        NUM_THREADS (int, optional): 训练线程数，默认为8
        X_MAX (int, optional): 加权函数的截断参数，默认为10
    """
    savedir = Path(CORPUS).parent
    VOCAB_FILE = savedir / VOCAB_FILE
    COOCCURRENCE_FILE = savedir / COOCCURRENCE_FILE
    COOCCURRENCE_SHUF_FILE = savedir / COOCCURRENCE_SHUF_FILE
    SAVE_FILE = savedir / SAVE_FILE

    cmd1 = f"vocab_count \
        -min-count {VOCAB_MIN_COUNT} \
        -verbose {VERBOSE} \
        < {CORPUS} > {VOCAB_FILE}"

    cmd2 = f"cooccur \
        -memory {MEMORY} \
        -vocab-file {VOCAB_FILE} \
        -verbose {VERBOSE} \
        -window-size {WINDOW_SIZE} \
        < {CORPUS} > {COOCCURRENCE_FILE}"

    cmd3 = f"shuffle \
        -memory {MEMORY} \
        -verbose {VERBOSE} \
        < {COOCCURRENCE_FILE} > {COOCCURRENCE_SHUF_FILE}"

    cmd4 = f"glove \
        -save-file {SAVE_FILE} \
        -threads {NUM_THREADS} \
        -input-file {COOCCURRENCE_SHUF_FILE} \
        -x-max {X_MAX} -iter {MAX_ITER} \
        -vector-size {VECTOR_SIZE} \
        -binary {BINARY} \
        -vocab-file {VOCAB_FILE} \
        -verbose {VERBOSE}"

    svd.watch_subprocess_cmd(cmd1)
    svd.watch_subprocess_cmd(cmd2)
    svd.watch_subprocess_cmd(cmd3)
    svd.watch_subprocess_cmd(cmd4)


def glove_dict(vectors_path, cache=True):
    """加载GloVe嵌入和词汇表
    
    Args:
        vectors_path (str or Path): GloVe向量文件的路径
        cache (bool, optional): 是否缓存加载结果，默认为True
    
    Returns:
        tuple: 包含两个元素的元组：
            - embeddings_dict (dict): 词向量字典，键为词，值为向量
            - vocab (dict): 词汇表字典，键为词，值为索引
    
    示例用法：
    vectors_path = svd.processed_dir() / "bigvul/glove/vectors.txt"
    emb_dict, vocab = glove_dict(vectors_path)
    
    实现细节：
    1. 检查是否有缓存的结果，如果有且cache=True则直接返回
    2. 从文本文件加载词向量到字典
    3. 加载词汇表文件
    4. 将词汇表转换为字典格式（词到索引的映射）
    5. 如果cache=True，将结果缓存到pickle文件
    6. 返回词向量字典和词汇表
    """
    # Caching
    savepath = svd.get_dir(svd.cache_dir() / "glove")
    savepath /= str(svd.hashstr(str(vectors_path)))
    if cache:
        try:
            with open(savepath, "rb") as f:
                return pkl.load(f)
        except Exception as E:
            print(E)
            pass

    # Read into dict
    embeddings_dict = {}
    with open(vectors_path, "r") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    # Read vocab
    with open(vectors_path.parent / "vocab.txt", "r") as f:
        vocab = [i.split()[0] for i in f.readlines()]
        vocab = dict([(j, i) for i, j in enumerate(vocab)])

    # Cache
    with open(savepath, "wb") as f:
        pkl.dump([embeddings_dict, vocab], f)

    return embeddings_dict, vocab


def find_closest_embeddings(word, embeddings_dict, topn=10):
    """查找与给定词最接近的GloVe嵌入
    
    Args:
        word (str): 要查找相似词的目标词
        embeddings_dict (dict): 词向量字典，键为词，值为向量
        topn (int, optional): 返回的最接近词的数量，默认为10
    
    Returns:
        list: 包含最接近词的列表，按相似度降序排列
    
    实现细节：
    1. 获取目标词的词向量
    2. 计算目标词向量与所有其他词向量的欧几里得距离
    3. 按距离排序并返回前topn个最接近的词
    """
    embedding = embeddings_dict[word]
    return sorted(
        embeddings_dict.keys(),
        key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding),
    )[:topn]


def get_embeddings(text: str, emb_dict: dict, emb_size: int = 100) -> np.array:
    """从文本中获取词嵌入，对不在词汇表(OoV)的词使用零向量
    
    Args:
        text (str): 文本字符串，应已预处理并按空格分词
        emb_dict (dict): 词嵌入字典，键为词，值为嵌入向量
        emb_size (int, optional): 嵌入向量的维度，默认为100
    
    Returns:
        np.array: 嵌入向量数组，形状为(序列长度, 嵌入维度)
    """
    return np.array([
        emb_dict[i] if i in emb_dict else np.full(emb_size, 0.001) for i in text.split()
    ])


def get_embeddings_list(li: list, emb_dict: dict, emb_size: int = 100) -> list:
    """从句子列表中获取嵌入，然后对每个句子进行平均
    
    Args:
        li (list): 句子列表
        emb_dict (dict): 词嵌入字典，键为词，值为嵌入向量
        emb_size (int, optional): 嵌入向量的维度，默认为100
    
    Returns:
        list: 包含每个句子平均嵌入的列表
    
    示例用法：
    li = ['static long ec device ioctl xcmd struct cros ec dev ec void user arg',
        'struct cros ec dev ec',
        'void user arg',
        'static long',
        'struct cros ec dev ec',
        'void user arg',
        '']

    glove_path = svd.processed_dir() / "bigvul/glove/vectors.txt"
    emb_dict, _ = glove_dict(glove_path)
    emb_size = 200
    
    实现细节：
    1. 对列表中的每个句子进行分词
    2. 处理空句子，将其替换为"<EMPTY>"
    3. 对每个句子调用get_embeddings获取词嵌入
    4. 对每个句子的词嵌入进行平均，得到句子级别的嵌入
    """
    li = [svdt.tokenise(i) for i in li]
    li = [i if len(i) > 0 else "<EMPTY>" for i in li]
    return [np.mean(get_embeddings(i, emb_dict, emb_size), axis=0) for i in li]
