"""RQ3: 不同图结构 对比实验。

该脚本用于对比 CFG+DFG、DFG only 和 CFG only 三种图结构的效果。
"""

import os

import sastvd as svd
import sastvd.linevd.run as lvdrun
from ray import tune

# 设置环境变量
os.environ["SLURM_JOB_NAME"] = "bash"

# RQ3: 不同图结构 配置
config = {
    "hfeat": tune.grid_search([512]),  # 隐藏特征维度
    "embtype": tune.grid_search(["graphcodebert"]),  # 嵌入类型
    "stmtweight": tune.grid_search([1]),  # 语句权重
    "hdropout": tune.grid_search([0.3]),  # 隐藏层dropout率
    "gatdropout": tune.grid_search([0.2]),  # GAT层dropout率
    "modeltype": tune.grid_search(["gat2layer+residual"]),  # 模型类型
    "gnntype": tune.grid_search(["gat"]),  # GNN类型
    "loss": tune.grid_search(["ce"]),  # 损失函数类型
    "gamma": tune.grid_search([2]),  # Focal Loss参数gamma
    "scea": tune.grid_search([0.5]),  # SCEA参数
    "gtype": tune.grid_search(["pdg+raw", "cfg+raw", "dfg+raw"]),  # 对比不同图结构
    "batch_size": tune.grid_search([256]),  # 批次大小
    "multitask": tune.grid_search(["linemethod"]),  # 多任务类型
    "splits": tune.grid_search(["default"]),  # 数据集分割方式
    "lr": tune.grid_search([1e-4]),  # 学习率
    "nsampling": tune.grid_search([False]),  # 是否使用邻居采样
    "mlp_layers": tune.grid_search([1]),  # MLP层数
    "use_bn": tune.grid_search([True]),
    "use_multichannel": tune.grid_search([False]),  # 不使用多通道 GNN
    "num_edge_types": tune.grid_search([7]),  # 边类型数量
}

# 样本大小设置（-1表示使用所有样本）
samplesz = -1
run_id = svd.get_run_id()  # 获取运行ID
# 设置保存路径
sp = svd.get_dir(svd.processed_dir() / f"raytune_myrq3_{samplesz}" / run_id)

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
    name="tune_linevd_myrq3",  # 实验名称
    storage_path=sp,  # 本地保存目录
    keep_checkpoints_num=1,  # 保留的检查点数量（仅保留最佳模型）
    checkpoint_score_attr="min-val_loss",  # 检查点评分属性
)
