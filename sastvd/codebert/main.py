import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.ivdetect.evaluate as ivde
import torch
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from BigVulDatasetNLPDataModule import BigVulDatasetNLPDataModule
from BigVulDatasetNLPLine import BigVulDatasetNLPLine
from LitCodebert import LitCodebert
from BigVulDatasetNLPLine import BigVulDatasetNLPLine
# 导入必要的库和模块
# pandas: 数据处理
# pytorch_lightning: 高级训练框架
# sastvd: 项目核心模块
# torch: 深度学习框架
# transformers: Hugging Face 预训练模型库


# 主程序：训练和测试 CodeBert 漏洞检测模型

# 获取运行 ID（用于保存模型和日志）
run_id = svd.get_run_id()

# 设置模型保存路径
savepath = svd.get_dir(svd.processed_dir() / "codebert" / run_id)

# 初始化模型
model = LitCodebert()

# 初始化数据模块（使用 BigVulDatasetNLP，批处理大小为 64）
data = BigVulDatasetNLPDataModule(BigVulDatasetNLP, batch_size=64)

# 检查点回调：保存验证损失最小的模型
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")

# 初始化 Trainer
# gpus=1: 使用 1 个 GPU（如果可用）
# auto_lr_find=True: 自动查找最佳学习率
# default_root_dir: 模型和日志的保存路径
# num_sanity_val_steps=0: 跳过验证集的 sanity 检查
# callbacks: 使用的回调函数列表
trainer = pl.Trainer(
    gpus=1,
    auto_lr_find=True,
    default_root_dir=savepath,
    num_sanity_val_steps=0,
    callbacks=[checkpoint_callback],
)

# 自动调整学习率
tuned = trainer.tune(model, data)

# 训练模型
trainer.fit(model, data)

# 测试模型
trainer.test(model, data)

# import sastvd.helpers.ml as ml
# from tqdm import tqdm

# run_id = "202108191652_2a65b8c_update_default_getitem_bigvul"
# chkpoint = (
#     svd.processed_dir()
#     / f"codebert/{run_id}/lightning_logs/version_0/checkpoints/epoch=188-step=18900.ckpt"
# )
# model = LitCodebert.load_from_checkpoint(chkpoint)
# model.cuda()
# all_pred = torch.empty((0, 2)).long().cuda()
# all_true = torch.empty((0)).long().cuda()
# for batch in tqdm(data.test_dataloader()):
#     ids, att_mask, labels = batch
#     ids = ids.cuda()
#     att_mask = att_mask.cuda()
#     labels = labels.cuda()
#     with torch.no_grad():
#         logits = F.softmax(model(ids, att_mask), dim=1)
#     all_pred = torch.cat([all_pred, logits])
#     all_true = torch.cat([all_true, labels])
# ml.get_metrics_logits(all_true, all_pred)
