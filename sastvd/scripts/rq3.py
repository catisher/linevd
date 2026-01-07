"""RQ3研究问题实验脚本。

该脚本用于运行RQ3（Research Question 3）的实验，使用Ray Tune进行LineVD模型的超参数调优，
重点研究不同多任务学习设置（行级和方法级）的效果。
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
    "embtype": tune.choice(["codebert"]),  # 嵌入类型（仅使用CodeBERT）
    "stmtweight": tune.choice([1, 5, 10]),  # 语句权重
    "hdropout": tune.choice([0.25, 0.3]),  # 隐藏层dropout率
    "gatdropout": tune.choice([0.15, 0.2]),  # GAT层dropout率
    "modeltype": tune.choice(["gat2layer"]),  # 模型类型（2层GAT）
    "gnntype": tune.choice(["gat"]),  # GNN类型（GAT）
    "loss": tune.choice(["ce"]),  # 损失函数类型
    "scea": tune.choice([0.4, 0.5, 0.6]),  # SCEA参数
    "gtype": tune.choice(["pdg-raw"]),  # 图类型（程序依赖图+原始特征）
    "batch_size": tune.choice([1024]),  # 批次大小
    "multitask": tune.choice(["line", "linemethod"]),  # 多任务类型（行级或行+方法级）
    "splits": tune.choice(["default"]),  # 数据集分割方式
    "lr": tune.choice([1e-3, 1e-4, 3e-4, 5e-4]),  # 学习率
}

# 样本大小设置（-1表示使用所有样本）
samplesz = -1
run_id = svd.get_run_id()  # 获取运行ID
# 设置保存路径（针对多任务学习实验）
sp = svd.get_dir(svd.processed_dir() / f"raytune_method_{samplesz}" / run_id)
# 创建可训练函数
trainable = tune.with_parameters(lvdrun.train_linevd, samplesz=samplesz, savepath=sp)

# 运行超参数调优
analysis = tune.run(
    trainable,  # 可训练函数
    resources_per_trial={"cpu": 1, "gpu": 1},  # 每个试验的资源需求
    metric="val_loss",  # 优化指标
    mode="min",  # 优化模式（最小化验证损失）
    config=config,  # 超参数配置
    num_samples=1000,  # 试验次数
    name="tune_linevd",  # 实验名称
    local_dir=sp,  # 本地保存目录
    keep_checkpoints_num=2,  # 保留的检查点数量
    checkpoint_score_attr="min-val_loss",  # 检查点评分属性
)
