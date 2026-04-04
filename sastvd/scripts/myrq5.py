"""RQ5: 对比不同损失函数。

该脚本用于对比 CE（标准交叉熵）、SCE（对称交叉熵）和 Focal Loss 三种损失函数的效果。
"""

import os

import sastvd as svd
import sastvd.linevd.run as lvdrun
from ray import tune

# 设置环境变量
os.environ["SLURM_JOB_NAME"] = "bash"

# RQ5: 对比不同损失函数配置
config = {
    "hfeat": tune.grid_search([512]),  # 隐藏特征维度
    "embtype": tune.grid_search(["graphcodebert"]),  # 嵌入类型
    "stmtweight": tune.grid_search([1]),  # 语句权重
    "hdropout": tune.grid_search([0.3]),  # 隐藏层dropout率
    "gatdropout": tune.grid_search([0.2]),  # GAT层dropout率
    "modeltype": tune.grid_search(["gat2layer+residual"]),  # 模型架构
    "gnntype": tune.grid_search(["gatv2"]),  # GNN类型
    "loss": tune.grid_search(["ce", "sce", "focal"]),  # 对比不同损失函数
    "gamma": tune.grid_search([2]),  # Focal Loss参数gamma
    "scea": tune.grid_search([0.5]),  # SCEA参数（仅对loss="sce"有效）
    "gtype": tune.grid_search(["pdg+raw"]),  # 图类型
    "batch_size": tune.grid_search([256]),  # 批次大小
    "multitask": tune.grid_search(["linemethod"]),  # 多任务类型
    "splits": tune.grid_search(["default"]),  # 数据集分割方式
    "lr": tune.grid_search([1e-4]),  # 学习率
    "nsampling": tune.grid_search([False]),  # 是否使用邻居采样
    "mlp_layers": tune.grid_search([1]),
    "use_bn": tune.grid_search([True]),
}

# 样本大小设置（-1表示使用所有样本）
samplesz = -1
run_id = svd.get_run_id()  # 获取运行ID
# 设置保存路径
sp = svd.get_dir(svd.processed_dir() / f"raytune_myrq5_{samplesz}" / run_id)

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
    num_samples=1,  # 试验次数（grid_search时每个组合运行1次）
    name="tune_linevd_myrq5",  # 实验名称
    storage_path=sp,  # 本地保存目录
    keep_checkpoints_num=1,  # 保留的检查点数量（仅保留最佳模型）
    checkpoint_score_attr="min-val_loss",  # 检查点评分属性
)
