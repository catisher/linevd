"""生成代码图数据脚本。

该脚本用于预处理代码文件并使用Joern和SAST工具提取图信息，主要用于漏洞检测和分析。
"""

import os
import pickle as pkl
import sys

import numpy as np
import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.joern as svdj
import sastvd.helpers.sast as sast

# 设置配置
NUM_JOBS = 100  # 并行任务数量
# 如果在Jupyter中运行，JOB_ARRAY_NUMBER为0，否则从命令行参数获取
JOB_ARRAY_NUMBER = 0 if "ipykernel" in sys.argv[0] else int(sys.argv[1]) - 1

# 读取数据
df = svdd.bigvul()  # 加载BigVul数据集
df = df.iloc[::-1]  # 反转数据顺序
splits = np.array_split(df, NUM_JOBS)  # 将数据集分割为多个任务


def preprocess(row):
    """并行处理每行代码数据。

    示例：
    df = svdd.bigvul()  # 加载数据集
    row = df.iloc[180189]  # 论文示例
    row = df.iloc[177860]  # 边缘情况1
    preprocess(row)  # 调用预处理函数
    """
    # 创建保存目录
    savedir_before = svd.get_dir(svd.processed_dir() / row["dataset"] / "before")
    savedir_after = svd.get_dir(svd.processed_dir() / row["dataset"] / "after")

    # 写入C文件
    fpath1 = savedir_before / f"{row['id']}.c"
    with open(fpath1, "w") as f:
        f.write(row["before"])
    
    fpath2 = savedir_after / f"{row['id']}.c"
    # 如果存在差异，写入修改后的代码
    if len(row["diff"]) > 0:
        with open(fpath2, "w") as f:
            f.write(row["after"])

    # 对原始代码（before）运行Joern
    if not os.path.exists(f"{fpath1}.edges.json"):
        svdj.full_run_joern(fpath1, verbose=3)

    # 对修改后的代码（after）运行Joern
    if not os.path.exists(f"{fpath2}.edges.json") and len(row["diff"]) > 0:
        svdj.full_run_joern(fpath2, verbose=3)

    # 运行SAST提取
    fpath3 = savedir_before / f"{row['id']}.c.sast.pkl"
    if not os.path.exists(fpath3):
        sast_before = sast.run_sast(row["before"])
        with open(fpath3, "wb") as f:
            pkl.dump(sast_before, f)


if __name__ == "__main__":
    # 并行处理当前任务的数据
    svd.dfmp(splits[JOB_ARRAY_NUMBER], preprocess, ordr=False, workers=8)
