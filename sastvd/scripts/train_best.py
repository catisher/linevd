"""最佳超参数配置训练脚本。

该脚本用于使用最佳超参数配置训练LineVD模型，这些参数是通过之前的超参数调优实验确定的。
使用Ray Tune进行训练，确保模型达到最佳性能。
"""

import os

import sastvd as svd
import sastvd.linevd.run as lvdrun
from ray import tune

# 设置环境变量
os.environ["SLURM_JOB_NAME"] = "bash"

# 最佳超参数配置
# 这些参数是通过之前的超参数调优实验确定的最优配置
config = {
    "hfeat": tune.choice([512]),  # 隐藏特征维度（最优值：512）
    "embtype": tune.choice(["codebert"]),  # 嵌入类型（最优值：CodeBERT）
    "stmtweight": tune.choice([1]),  # 语句权重（最优值：1）
    "hdropout": tune.choice([0.3]),  # 隐藏层dropout率（最优值：0.3）
    "gatdropout": tune.choice([0.2]),  # GAT层dropout率（最优值：0.2）
    "modeltype": tune.choice(["gat2layer"]),  # 模型类型（最优值：2层GAT）
    "gnntype": tune.choice(["gat"]),  # GNN类型（最优值：GAT）
    "loss": tune.choice(["ce"]),  # 损失函数类型（最优值：交叉熵）
    "scea": tune.choice([0.5]),  # SCEA参数（最优值：0.5）
    "gtype": tune.choice(["pdg+raw"]),  # 图类型（最优值：程序依赖图+原始特征）
    "batch_size": tune.choice([1024]),  # 批次大小（最优值：1024）
    "multitask": tune.choice(["linemethod"]),  # 多任务类型（最优值：行+方法级）
    "splits": tune.choice(["default"]),  # 数据集分割方式（最优值：默认分割）
    "lr": tune.choice([1e-4]),  # 学习率（最优值：1e-4）
}

# 样本大小设置（-1表示使用所有样本）
samplesz = -1
run_id = svd.get_run_id()  # 获取运行ID
# 设置保存路径（针对最佳配置实验）
sp = svd.get_dir(svd.processed_dir() / f"raytune_best_{samplesz}" / run_id)
# 创建可训练函数
# 传递最大训练轮数、样本大小和保存路径参数

trainable = tune.with_parameters(
    lvdrun.train_linevd, max_epochs=130, samplesz=samplesz, savepath=sp
)

# 运行最佳配置的超参数调优
analysis = tune.run(
    trainable,  # 可训练函数
    resources_per_trial={"cpu": 2, "gpu": 0.5},  # 每个试验的资源需求
    metric="val_loss",  # 优化指标
    mode="min",  # 优化模式（最小化验证损失）
    config=config,  # 最佳超参数配置
    num_samples=1000,  # 试验次数
    name="tune_linevd",  # 实验名称
    local_dir=sp,  # 本地保存目录
    keep_checkpoints_num=1,  # 保留的检查点数量（仅保留最佳模型）
    checkpoint_score_attr="min-val_loss",  # 检查点评分属性
)
