"""LineVD模型训练脚本。

该脚本提供了一个封装函数，用于通过RayTune进行LineVD模型的训练和超参数调优。
主要功能包括：
1. 初始化LineVD模型（LitGNN）
2. 加载数据集模块
3. 配置训练器和回调函数
4. 启动模型训练流程

该脚本是LineVD模型训练的主要入口点，支持与RayTune集成进行超参数优化。
"""


# sastvd/script/run_method.py的修改版本，适用于LineVD模型训练
import pytorch_lightning as pl
import sastvd.linevd as lvd
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)


def train_linevd(
    config, savepath, samplesz=-1, max_epochs=130, num_gpus=1, checkpoint_dir=None
):
    """封装PyTorch Lightning训练流程，用于RayTune超参数调优。
    
    参数:
        config: 配置字典，包含模型和训练的超参数
            - hfeat: 隐藏层特征维度
            - embtype: 嵌入类型
            - modeltype: 模型类型
            - loss: 损失函数类型
            - hdropout: 隐藏层dropout率
            - gatdropout: GAT模型dropout率
            - multitask: 多任务学习类型（"linemethod", "line", "method"）
            - stmtweight: 语句级任务的权重
            - gnntype: GNN类型（"gat", "gcn"等）
            - scea: 是否使用SCEA优化
            - lr: 学习率
            - batch_size: 批次大小
            - gtype: 图类型
            - splits: 数据集分割方式
        savepath: 模型保存路径
        samplesz: 样本大小，-1表示使用完整数据集
        max_epochs: 最大训练轮数
        num_gpus: 使用的GPU数量
        checkpoint_dir: 检查点目录，用于从检查点恢复训练
    """
    # 初始化LineVD模型
    model = lvd.LitGNN(
        hfeat=config["hfeat"],
        embtype=config["embtype"],
        methodlevel=False,  # 不使用方法级预测
        nsampling=True,  # 使用邻居采样
        model=config["modeltype"],
        loss=config["loss"],
        hdropout=config["hdropout"],
        gatdropout=config["gatdropout"],
        num_heads=4,  # GAT模型的注意力头数量
        multitask=config["multitask"],
        stmtweight=config["stmtweight"],
        gnntype=config["gnntype"],
        scea=config["scea"],
        lr=config["lr"],
    )

    # 加载数据集
    data = lvd.BigVulDatasetLineVDDataModule(
        batch_size=config["batch_size"],
        sample=samplesz,
        methodlevel=False,
        nsampling=True,
        nsampling_hops=2,  # 邻居采样的跳数
        gtype=config["gtype"],
        splits=config["splits"],
    )

    # 配置训练器和回调函数
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")  # 基于验证损失保存最佳模型
    metrics = ["train_loss", "val_loss", "val_auroc"]  # 要报告的指标
    raytune_callback = TuneReportCallback(metrics, on="validation_end")  # RayTune报告回调
    rtckpt_callback = TuneReportCheckpointCallback(metrics, on="validation_end")  # RayTune检查点回调
    trainer = pl.Trainer(
        gpus=1,
        auto_lr_find=False,  # 不自动寻找学习率
        default_root_dir=savepath,
        num_sanity_val_steps=0,  # 不进行验证前的 sanity 检查
        callbacks=[checkpoint_callback, raytune_callback, rtckpt_callback],
        max_epochs=max_epochs,
    )
    trainer.fit(model, data)  # 启动训练流程
