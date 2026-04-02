"""LineVD 项目核心工具模块

该模块提供了项目中常用的工具函数，包括：
1. 路径管理：获取项目和存储目录路径
2. 调试工具：打印调试信息
3. Git 工具：获取版本控制信息
4. 子进程管理：执行命令行操作
5. 并行处理：高效处理数据
6. 辅助功能：生成 ID、哈希计算等
"""
import hashlib
import inspect
import os
import random
import string
import subprocess
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def project_dir() -> Path:
    """获取项目根目录路径
    
    Returns:
        Path: 项目根目录的路径对象
    """
    return Path(__file__).parent.parent


def storage_dir() -> Path:
    """获取存储目录路径
    Returns:
        Path: 存储目录的路径对象
    """
    ## 可以根据实际情况精简一下
    return Path(__file__).parent.parent / "storage"


def external_dir() -> Path:
    """获取外部数据存储目录路径
    
    用于存放原始数据集，如 MSR_data_cleaned.csv
    
    Returns:
        Path: 外部数据目录的路径对象
    """
    path = storage_dir() / "external"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def interim_dir() -> Path:
    """获取中间数据存储目录路径
    
    用于存放处理过程中的中间数据
    
    Returns:
        Path: 中间数据目录的路径对象
    """
    path = storage_dir() / "interim"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def processed_dir() -> Path:
    """获取处理后数据存储目录路径
    
    用于存放处理完成的数据，如图结构、词嵌入模型等
    
    Returns:
        Path: 处理后数据目录的路径对象
    """
    path = storage_dir() / "processed"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def outputs_dir() -> Path:
    """获取输出文件存储目录路径
    
    用于存放模型训练结果、评估指标等
    
    Returns:
        Path: 输出文件目录的路径对象
    """
    path = storage_dir() / "outputs"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def cache_dir() -> Path:
    """获取缓存目录路径
    
    用于存放缓存文件，如最小化数据集、元数据等
    
    Returns:
        Path: 缓存目录的路径对象
    """
    path = storage_dir() / "cache"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def get_dir(path) -> Path:
    """获取或创建目录路径
    
    如果目录不存在，自动创建
    
    Args:
        path: 目录路径
    
    Returns:
        Path: 目录的路径对象
    """
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def debug(msg, noheader=False, sep="\t"):
    """打印调试信息到控制台
    
    包含时间戳、文件名和行号等上下文信息
    
    Args:
        msg: 要打印的消息
        noheader: 是否省略头部信息
        sep: 分隔符
    """
    caller = inspect.stack()[1]
    file_name = caller.filename
    ln = caller.lineno
    now = datetime.now()
    time = now.strftime("%m/%d/%Y - %H:%M:%S")
    if noheader:
        print("\t\x1b[94m{}\x1b[0m".format(msg), end="")
        return
    print(
        '\x1b[40m[{}] File "{}", line {}\x1b[0m\n\t\x1b[94m{}\x1b[0m'.format(
            time, file_name, ln, msg
        )
    )


def gitsha():
    """获取当前 Git 提交的 SHA 哈希值
    
    用于确保实验的可重现性
    
    Returns:
        str: 短格式的 Git 提交 SHA
    """
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode()
    )


def gitmessage():
    """获取当前 Git 提交的消息
    
    用于生成运行 ID，确保实验的可追溯性
    
    Returns:
        str: 转换为小写并使用下划线连接的提交消息
    """
    m = subprocess.check_output(["git", "log", "-1", "--format=%s"]).strip().decode()
    return "_".join(m.lower().split())


def subprocess_cmd(command: str, verbose: int = 0, force_shell: bool = False):
    """执行命令行进程
    Args:
        command: 要执行的命令
        verbose: 详细程度，大于 1 时打印输出
        force_shell: 是否强制使用本地 shell 执行
    
    Returns:
        tuple: (stdout, stderr) 输出
    
    Example:
        subprocess_cmd('echo a; echo b', verbose=1)
        >>> a
        >>> b
    """
    ## 怀疑这里作者埋了坑
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output = process.communicate()
    if verbose > 1:
        debug(output[0].decode())
        debug(output[1].decode())
    #返回值是一个元组 (stdout, stderr)
    return output


def watch_subprocess_cmd(command: str, force_shell: bool = False):
    """执行子进程并实时监控输出
    
    用于调试目的，实时显示命令执行结果
    
    Args:
        command: 要执行的命令
        force_shell: 是否强制使用本地 shell 执行
    """
    ## 埋坑
    #singularity = os.getenv("SINGULARITY")
    #if singularity != "true" and not force_shell:
    #    command = f"singularity exec {project_dir() / 'main.sif'} " + command
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    # 轮询进程输出直到完成
    noheader = False
    while True:
        nextline = process.stdout.readline()
        if nextline == b"" and process.poll() is not None:
            break
        debug(nextline.decode(), noheader=noheader)
        noheader = True


def genid():
    """生成随机字符串
    
    用于生成唯一标识符
    
    Returns:
        str: 10位随机字符串（大写字母和数字）
    """
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))


def get_run_id(args=None):
    """生成运行 ID   
    包含时间戳、Git 提交信息和可选的参数信息
    Returns:
        str: 唯一的运行 ID
    """
    if not args:
        ID = datetime.now().strftime("%Y%m%d%H%M_{}".format(gitsha()))
        return ID + "_" + gitmessage()
    ID = datetime.now().strftime(
        "%Y%m%d%H%M_{}_{}".format(
            gitsha(), "_".join([f"{v}" for _, v in vars(args).items()])
        )
    )
    return ID


def hashstr(s):
    """对字符串进行哈希处理
    将字符串转换为 8 位数字哈希值
    Args:
        s: 要哈希的字符串
    
    Returns:
        int: 8 位数字哈希值
    """
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


def dfmp(df, function, columns=None, ordr=True, workers=6, cs=10, desc="Run: "):
    """并行应用函数到数据框
    
    支持 DataFrame 或列表作为输入，使用多进程提高处理速度
    
    Args:
        df: 输入数据，可以是 pd.DataFrame 或 list
        function: 要应用的函数
        columns: 要处理的列，None 表示处理整个行
        ordr: 是否保持原始顺序
        workers: 并行工作进程数
        cs: 分块大小
        desc: 进度条描述
    
    Returns:
        list: 处理结果列表
    
    """
    # 根据输入类型和columns参数，提取要处理的数据项
    if isinstance(columns, str):
        # 如果columns是字符串，提取该列的值并转换为列表
        items = df[columns].tolist()
    elif isinstance(columns, list):
        # 如果columns是列表，提取多列并转换为字典记录列表
        items = df[columns].to_dict("records")
    elif isinstance(df, pd.DataFrame):
        # 如果df是DataFrame且columns为None，将整个DataFrame转换为字典记录列表
        items = df.to_dict("records")
    elif isinstance(df, list):
        # 如果df本身就是列表，直接使用
        items = df
    else:
        # 如果输入类型不支持，抛出异常
        raise ValueError("First argument of dfmp should be pd.DataFrame or list.")

    # 初始化一个空列表，用于存储处理后的结果
    processed = []
    # 更新进度条描述，添加工作进程数信息
    desc = f"({workers} Workers) {desc}"
    # 使用多进程池进行并行处理
    with Pool(processes=workers) as p:
        # 根据ordr参数选择imap（保持顺序）或imap_unordered（不保持顺序）
        map_func = getattr(p, "imap" if ordr else "imap_unordered")
        # 使用tqdm显示进度条，遍历处理结果
        for ret in tqdm(map_func(function, items, cs), total=len(items), desc=desc):
            # 将每个处理结果添加到processed列表中
            processed.append(ret)
    # 返回处理后的结果列表
    return processed


def chunks(lst, n):
    """将列表分割成指定大小的块
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
