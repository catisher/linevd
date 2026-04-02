"""RQ1研究问题实验脚本。

该脚本用于运行RQ1（Research Question 1）的实验，使用Ray Tune进行LineVD模型的超参数调优。
究问题 ：不同代码嵌入类型对漏洞检测性能的影响

对比内容 ：

- 嵌入类型 ： codebert vs glove vs doc2vec
- 模型类型 ：仅 mlponly （多层感知机，不使用GNN）
- 图类型 ： pdg+raw （程序依赖图+原始特征）
- 多任务类型 ：仅 line （行级预测）
消融目的 ：

- 比较不同代码嵌入表示的效果
- 验证 CodeBERT 相比传统词向量的优势
- 在不使用图神经网络的情况下，纯粹比较嵌入质量
"""

import os

import sastvd as svd
import sastvd.linevd.run as lvdrun
from ray import tune

# 设置环境变量
os.environ["SLURM_JOB_NAME"] = "bash"

# 超参数搜索空间配置
config = {
    "hfeat": tune.choice([512]),  # 隐藏特征维度
    "embtype": tune.choice(["codebert", "glove", "doc2vec"]),  # 嵌入类型
    "stmtweight": tune.choice([1, 5, 10]),  # 语句权重
    "hdropout": tune.choice([0.1, 0.15, 0.2, 0.25]),  # 隐藏层dropout率
    "gatdropout": tune.choice([0.15, 0.2]),  # GAT层dropout率
    "modeltype": tune.choice(["mlponly"]),  # 模型类型
    "gnntype": tune.choice(["gat"]),  # GNN类型
    "loss": tune.choice(["ce"]),  # 损失函数类型
    "scea": tune.choice([0.4, 0.5, 0.6]),  # SCEA参数
    "gtype": tune.choice(["pdg+raw"]),  # 图类型
    "batch_size": tune.choice([1024]),  # 批次大小
    "multitask": tune.choice(["line"]),  # 多任务类型
    "splits": tune.choice(["default"]),  # 数据集分割方式
    "lr": tune.choice([1e-3, 1e-4, 3e-4, 5e-4]),  # 学习率
}

# 样本大小设置（-1表示使用所有样本）
samplesz = -1
run_id = svd.get_run_id()  # 获取运行ID
savepath = svd.get_dir(svd.processed_dir() / f"raytune_features_{samplesz}" / run_id)  # 保存路径
# 创建可训练函数
trainable = tune.with_parameters(lvdrun.train_linevd, samplesz=samplesz, savepath=savepath)

# 运行超参数调优
analysis = tune.run(
    trainable,  # 可训练函数
    resources_per_trial={"cpu": 1, "gpu": 1},  # 每个试验的资源需求
    metric="val_loss",  # 优化指标
    mode="min",  # 优化模式（最小化验证损失）
    config=config,  # 超参数配置
    num_samples=1000,  # 试验次数
    name="tune_linevd",  # 实验名称
    local_dir=savepath,  # 本地保存目录
    keep_checkpoints_num=2,  # 保留的检查点数量
    checkpoint_score_attr="min-val_loss",  # 检查点评分属性
)
