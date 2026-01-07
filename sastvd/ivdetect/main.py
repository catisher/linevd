"""IVDetect漏洞检测方法的实现。

该文件包含IVDetect模型的完整训练、验证和测试流程，包括：
1. 数据加载和预处理
2. 模型创建和初始化
3. 训练循环实现（含早停机制）
4. 模型评估和性能指标计算
5. 使用GNNExplainer进行语句级漏洞检测分析
"""


import pickle as pkl
from importlib import reload

import dgl
import sastvd as svd
import sastvd.helpers.ml as ml
import sastvd.helpers.rank_eval as svdr
import sastvd.ivdetect.evaluate as ivde
import sastvd.ivdetect.gnnexplainer as ge
import sastvd.ivdetect.helpers as ivd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader

# 加载数据集
# 重新加载ivd模块以确保使用最新的实现
reload(ivd)
# 创建训练、验证和测试数据集
# 使用BigVulDatasetIVDetect类处理数据并生成图结构
# partition参数指定数据集划分
# "train"：训练集，用于模型训练
# "val"：验证集，用于训练过程中的模型评估和早停
# "test"：测试集，用于最终评估模型性能
train_ds = ivd.BigVulDatasetIVDetect(partition="train")
val_ds = ivd.BigVulDatasetIVDetect(partition="val")
test_ds = ivd.BigVulDatasetIVDetect(partition="test")

# 数据加载器配置参数
dl_args = {
    "drop_last": False,  # 是否丢弃最后一个不完整的批次
    "shuffle": True,     # 是否在每个epoch前打乱数据顺序
    "num_workers": 6     # 数据加载的并行工作进程数
}

# 创建图数据加载器，用于批量加载图数据
# 训练集和验证集使用较小的批次大小(16)，测试集使用较大的批次大小(64)
train_dl = GraphDataLoader(train_ds, batch_size=16, **dl_args)
val_dl = GraphDataLoader(val_ds, batch_size=16, **dl_args)
test_dl = GraphDataLoader(test_ds, batch_size=64, **dl_args)

# 创建模型
# 选择计算设备，优先使用CUDA GPU，如果不可用则使用CPU
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 打印调试信息，显示当前使用的设备
svd.debug(dev)
# 创建IVDetect模型实例
# 参数1(200)：输入特征维度，对应GloVe词嵌入的维度
# 参数2(64)：隐藏层维度，控制模型复杂度
model = ivd.IVDetect(200, 64)
# 将模型移动到指定设备
model.to(dev)

# 调试单个样本
# 获取第一个训练批次用于模型调试
batch = next(iter(train_dl))
# 将批次数据移动到指定设备
batch = batch.to(dev)
# 模型前向传播，获取预测结果
logits = model(batch, train_ds)

# 优化器和损失函数配置
# 使用交叉熵损失函数，适用于多分类问题
criterion = nn.CrossEntropyLoss()
# 使用Adam优化器，学习率设置为0.0001
# model.parameters()：模型的所有可训练参数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 训练循环设置
# 获取唯一的运行ID，用于保存模型和日志
ID = svd.get_run_id({})
# 可选：使用已有的运行ID进行模型加载和继续训练
# ID = "202108121558_79d3273"

# 创建日志记录器
# 参数说明：
# model：要保存的模型
# svd.processed_dir() / "ivdetect" / ID：日志和模型的保存路径
# max_patience=10000：早停机制的最大耐心值
# val_every=30：每30个训练步骤进行一次验证评估
logger = ml.LogWriter(
    model, svd.processed_dir() / "ivdetect" / ID, max_patience=10000, val_every=30
)

# 可选：加载已有的日志记录器，继续之前的训练
# logger.load_logger()
while True:
    for batch in train_dl:

        # Training
        model.train()
        batch = batch.to(dev)
        logits = model(batch, train_ds)
        labels = dgl.max_nodes(batch, "_VULN").long()
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation
        train_mets = ml.get_metrics_logits(labels, logits)
        val_mets = train_mets
        if logger.log_val():
            model.eval()
            with torch.no_grad():
                all_pred = torch.empty((0, 2)).long().to(dev)
                all_true = torch.empty((0)).long().to(dev)
                for val_batch in val_dl:
                    val_batch = val_batch.to(dev)
                    val_labels = dgl.max_nodes(val_batch, "_VULN").long()
                    val_logits = model(val_batch, val_ds)
                    all_pred = torch.cat([all_pred, val_logits])
                    all_true = torch.cat([all_true, val_labels])
                val_mets = ml.get_metrics_logits(all_true, all_pred)
        logger.log(train_mets, val_mets)
        logger.save_logger()

    # Early Stopping
    if logger.stop():
        break
    logger.epoch()

# Print test results
logger.load_best_model()
model.eval()
all_pred = torch.empty((0, 2)).long().to(dev)
all_true = torch.empty((0)).long().to(dev)
with torch.no_grad():
    for test_batch in test_dl:
        test_batch = test_batch.to(dev)
        test_labels = dgl.max_nodes(test_batch, "_VULN").long()
        test_logits = model(test_batch, test_ds)
        all_pred = torch.cat([all_pred, test_logits])
        all_true = torch.cat([all_true, test_labels])
        test_mets = ml.get_metrics_logits(all_true, all_pred)
        logger.test(test_mets)
logger.test(test_mets)
rank_metr_test = ml.met_dict_to_str(svdr.rank_metr(all_pred, all_true))

# 使用GNNExplainer进行语句级漏洞检测分析
# 获取BigVul数据集中的正确依赖添加行
correct_lines = ivde.get_dep_add_lines_bigvul()

# 存储预测的漏洞行
pred_lines = dict()

# 遍历测试集批次
for batch in test_dl:
    # 将批次拆分为单个图
    for g in dgl.unbatch(batch):
        # 获取样本ID
        sampleid = g.ndata["_SAMPLE"].max().int().item()
        
        # 跳过不在正确行数据中的样本
        if sampleid not in correct_lines:
            continue
        # 跳过已处理的样本
        if sampleid in pred_lines:
            continue
        
        try:
            # 使用GNNExplainer获取按重要性排序的代码行
            lines = ge.gnnexplainer(model, g.to(dev), test_ds)
        except Exception as E:
            print(E)
        
        # 存储预测结果
        pred_lines[sampleid] = lines

# 将预测结果保存到文件
with open(svd.cache_dir() / "pred_lines.pkl", "wb") as f:
    pkl.dump(pred_lines, f)

# 计算平均首次命中排名（MFR）
# MFR衡量模型在检测到真正的漏洞行之前需要检查的行数
MFR = []
for sampleid, pred in pred_lines.items():
    # 获取该样本的真实漏洞行
    true = correct_lines[sampleid]
    true = list(true["removed"]) + list(true["depadd"])
    
    # 查找真实漏洞行在预测列表中的位置
    for i, p in enumerate(pred):
        if p in true:
            MFR += [i]
            break

# 打印平均MFR值，值越小表示模型性能越好
print(sum(MFR) / len(MFR))
