"""RQ1: CodeBERT vs GraphCodeBERT 对比实验。

该脚本用于对比 CodeBERT 和 GraphCodeBERT 两种预训练模型的嵌入效果。
"""

import os

import sastvd as svd
import sastvd.linevd.run as lvdrun
from ray import tune

# 设置环境变量
os.environ["SLURM_JOB_NAME"] = "bash"

# RQ1: CodeBERT vs GraphCodeBERT 配置
config = {
    "hfeat": tune.choice([512]),  # 隐藏特征维度
    "embtype": tune.choice(["codebert", "graphcodebert"]),  # 对比两种嵌入类型
    "stmtweight": tune.choice([1]),  # 语句权重
    "hdropout": tune.choice([0.3]),  # 隐藏层dropout率
    "gatdropout": tune.choice([0.2]),  # GAT层dropout率
    "modeltype": tune.choice(["gat2layer"]),  # 模型类型
    "gnntype": tune.choice(["gat"]),  # GNN类型
    "loss": tune.choice(["ce"]),  # 损失函数类型
    "scea": tune.choice([0.5]),  # SCEA参数
    "gtype": tune.choice(["pdg+raw"]),  # 图类型
    "batch_size": tune.choice([256]),  # 批次大小
    "multitask": tune.choice(["linemethod"]),  # 多任务类型
    "splits": tune.choice(["default"]),  # 数据集分割方式
    "lr": tune.choice([1e-4]),  # 学习率
}

# 样本大小设置（-1表示使用所有样本）
samplesz = -1
run_id = svd.get_run_id()  # 获取运行ID
# 设置保存路径
sp = svd.get_dir(svd.processed_dir() / f"raytune_rq1_{samplesz}" / run_id)

# 创建可训练函数
trainable = tune.with_parameters(
    lvdrun.train_linevd, 
    max_epochs=100,  # 最大训练轮数
    samplesz=samplesz, 
    savepath=sp
)

# 运行超参数调优
analysis = tune.run(
    trainable,  # 可训练函数
    resources_per_trial={"cpu": 2, "gpu": 0.5},  # 每个试验的资源需求
    metric="val_loss",  # 优化指标
    mode="min",  # 优化模式（最小化验证损失）
    config=config,  # 超参数配置
    num_samples=5,  # 试验次数
    name="tune_linevd_rq1",  # 实验名称
    storage_path=sp,  # 本地保存目录
    keep_checkpoints_num=1,  # 保留的检查点数量（仅保留最佳模型）
    checkpoint_score_attr="min-val_loss",  # 检查点评分属性
)