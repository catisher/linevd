"""方法级漏洞检测实验脚本。

该脚本用于运行方法级别的漏洞检测实验，使用Ray Tune进行超参数调优，
重点研究不同图神经网络架构在方法级别的检测效果。
"""

import os

import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd
from ray import tune
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)

# 设置环境变量
os.environ["SLURM_JOB_NAME"] = "bash"


def train_ml(
    config, savepath, samplesz=-1, max_epochs=130, num_gpus=1, checkpoint_dir=None
):
    """训练方法级漏洞检测模型并传递给Ray Tune。

    参数:
        config: 包含超参数的字典
        savepath: 模型和结果保存路径
        samplesz: 样本大小，-1表示使用所有样本
        max_epochs: 最大训练轮数
        num_gpus: 使用的GPU数量
        checkpoint_dir: 检查点目录，用于恢复训练
    """
    # 创建模型实例
    model = lvd.LitGNN(
        methodlevel=True,  # 使用方法级别的表示
        nsampling=False,  # 不使用邻居采样
        model=config["modeltype"],  # 模型类型（来自超参数配置）
        embtype="glove",  # 嵌入类型
        loss="ce",  # 损失函数类型
        hdropout=config["hdropout"],  # 隐藏层dropout率（来自超参数配置）
        gatdropout=config["gatdropout"],  # GAT层dropout率（来自超参数配置）
        num_heads=4,  # 注意力头数量
        multitask="line",  # 多任务类型
        stmtweight=1,  # 语句权重
        gnntype=config["gnntype"],  # GNN类型（来自超参数配置）
        lr=1e-4,  # 学习率
    )

    # 加载数据
    data = lvd.BigVulDatasetLineVDDataModule(
        batch_size=32,  # 批次大小
        sample=samplesz,  # 样本大小
        methodlevel=True,  # 使用方法级别的数据
        nsampling=False,  # 不使用邻居采样
        nsampling_hops=2,  # 采样跳数（仅当nsampling=True时有效）
        gtype="pdg+raw",  # 图类型（程序依赖图+原始特征）
        splits="default",  # 数据集分割方式
        feat="glove",  # 特征嵌入类型
    )

    # 创建回调函数
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")  # 检查点回调
    metrics = ["train_loss", "val_loss", "val_auroc"]  # 监控指标
    raytune_callback = TuneReportCallback(metrics, on="validation_end")  # Ray Tune报告回调
    rtckpt_callback = TuneReportCheckpointCallback(metrics, on="validation_end")  # Ray Tune检查点回调
    
    # 创建训练器
    trainer = pl.Trainer(
        gpus=1,  # 使用的GPU数量
        auto_lr_find=False,  # 不自动查找学习率
        default_root_dir=savepath,  # 模型保存根目录
        num_sanity_val_steps=3,  # 验证前的健全性检查步骤数
        callbacks=[checkpoint_callback, raytune_callback, rtckpt_callback],  # 回调函数列表
        max_epochs=max_epochs,  # 最大训练轮数
    )

    # 训练模型
    trainer.fit(model, data)

    # 保存测试结果
    main_savedir = svd.get_dir(svd.outputs_dir() / "rq_results_methodonly")  # 结果保存目录
    trainer.test(model, data, ckpt_path="best")  # 在测试集上运行最佳模型
    
    # 收集测试结果
    res = [
        "methodonly",  # 模型类型
        "methodonly",  # 配置类型
        model.res1vo,  # 结果1：漏洞位置预测
        model.res2mt,  # 结果2：多任务预测
        model.res2f,  # 结果2：最终预测
        model.res3vo,  # 结果3：漏洞位置预测
        model.res2,  # 结果2：综合预测
        model.lr,  # 学习率
    ]
    mets = lvd.get_relevant_metrics(res)  # 获取相关指标
    res_df = pd.DataFrame.from_records([mets])  # 转换为DataFrame
    res_df.to_csv(str(main_savedir / svd.get_run_id()) + ".csv", index=0)  # 保存为CSV文件

    # 保存最佳模型结果（当前已注释）
    # trainer.test(model, data, ckpt_path="best")
    # res = [
    #     "methodonly",
    #     "methodonly",
    #     model.res1vo,
    #     model.res2mt,
    #     model.res2f,
    #     model.res3vo,
    #     model.res2,
    #     model.lr,
    # ]
    # mets = lvd.get_relevant_metrics(res)
    # res_df = pd.DataFrame.from_records([mets])
    # res_df.to_csv(str(main_savedir / svd.get_run_id()) + ".best.csv", index=0)


# 超参数搜索空间配置
config = {
    "gnntype": tune.choice(["gat", "gcn"]),  # GNN类型选择（GAT或GCN）
    "hdropout": tune.choice([0.1, 0.15, 0.2, 0.25]),  # 隐藏层dropout率选择
    "gatdropout": tune.choice([0.15, 0.2]),  # GAT层dropout率选择
    "modeltype": tune.choice(["gat1layer", "gat2layer"]),  # 模型类型选择（1层或2层GAT）
}

# 样本大小设置（-1表示使用所有样本）
samplesz = -1
# 获取运行ID
run_id = svd.get_run_id()
# 设置保存路径（针对方法级实验）
sp = svd.get_dir(svd.processed_dir() / f"raytune_methodlevel_{samplesz}" / run_id)
# 创建可训练函数

trainable = tune.with_parameters(train_ml, samplesz=samplesz, savepath=sp)

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
