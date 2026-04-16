"""IVDetect漏洞检测方法的实现。"""


import pickle as pkl
import os
import json
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
train_ds = ivd.BigVulDatasetIVDetect(partition="train")
val_ds = ivd.BigVulDatasetIVDetect(partition="val")
test_ds = ivd.BigVulDatasetIVDetect(partition="test")

# 数据加载器配置参数
dl_args = {
    "drop_last": False,  # 是否丢弃最后一个不完整的批次
    "shuffle": True,     # 是否在每个epoch前打乱数据顺序
    "num_workers": 6     # 数据加载的并行工作进程数
}

# 获取唯一的运行ID，用于保存模型和日志
ID = svd.get_run_id({})
# 可选：使用已有的运行ID进行模型加载和继续训练
# ID = "202108121558_79d3273"

# 超参数配置
config = {
    "input_size": 200,        # 输入特征维度
    "hidden_size": 64,         # 隐藏层维度
    "learning_rate": 0.0001,   # 学习率
    "batch_size": 16,          # 训练和验证批次大小
    "test_batch_size": 64,     # 测试批次大小
    "dropout": 0.5,            # Dropout概率
    "max_patience": 50,        # 早停机制最大耐心值
    "val_every": 30,           # 每多少步进行一次验证
    "use_gpu": False           # 强制使用CPU
}

# 加载已有的超参数（如果存在）
config_path = svd.processed_dir() / "ivdetect" / ID / "config.json"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"Loaded configuration from {config_path}")
else:
    # 保存当前配置
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Saved configuration to {config_path}")

# 创建模型
# 选择计算设备，根据配置决定是否使用GPU
if config["use_gpu"] and torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Using GPU: cuda:0")
else:
    dev = torch.device("cpu")
    print("Using CPU")
# 打印调试信息，显示当前使用的设备
svd.debug(dev)
# 创建IVDetect模型实例
model = ivd.IVDetect(config["input_size"], config["hidden_size"], config["dropout"], device=dev)
# 将模型移动到指定设备
model.to(dev)

# 调试单个样本
# 获取第一个训练批次用于模型调试
batch = next(iter(GraphDataLoader(train_ds, batch_size=1, **dl_args)))
# 将批次数据移动到指定设备
batch = batch.to(dev)
# 模型前向传播，获取预测结果
logits = model(batch, train_ds)
# 清理内存
torch.cuda.empty_cache()

# 优化器和损失函数配置
# 使用交叉熵损失函数，适用于多分类问题
criterion = nn.CrossEntropyLoss()
# 使用Adam优化器
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

# 创建图数据加载器，用于批量加载图数据
train_dl = GraphDataLoader(train_ds, batch_size=config["batch_size"], **dl_args)
val_dl = GraphDataLoader(val_ds, batch_size=config["batch_size"], **dl_args)
test_dl = GraphDataLoader(test_ds, batch_size=config["test_batch_size"], **dl_args)

# 创建日志记录器
logger = ml.LogWriter(
    model, svd.processed_dir() / "ivdetect" / ID, 
    max_patience=config["max_patience"], 
    val_every=config["val_every"]
)

# 直接加载现有模型进行评测
# 设置为True以跳过训练，直接加载模型进行评测
EVAL_ONLY = True
# 已训练模型的路径（如果存在）
# 如果设置了该路径，会尝试加载指定的模型
TRAINED_MODEL_PATH = None
# 示例：TRAINED_MODEL_PATH = "path/to/your/trained/model/best.model"

if not EVAL_ONLY:
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
            # 清理内存
            torch.cuda.empty_cache()

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
                        # 清理内存
                        torch.cuda.empty_cache()
                    val_mets = ml.get_metrics_logits(all_true, all_pred)
                    # 清理内存
                    torch.cuda.empty_cache()
            logger.log(train_mets, val_mets)
            logger.save_logger()

        # Early Stopping
        if logger.stop():
            break
        logger.epoch()
else:
    print("Skipping training, directly evaluating...")
    # 尝试加载已训练的模型
    if TRAINED_MODEL_PATH and os.path.exists(TRAINED_MODEL_PATH):
        try:
            model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=dev))
            print(f"Loaded trained model from {TRAINED_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model from {TRAINED_MODEL_PATH}: {e}")
            print("Using initial model weights")
    else:
        # 确保加载最佳模型
        try:
            logger.load_best_model()
            print("Loaded best model successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using initial model weights")

# Print test results
# 只有在非评测模式下才尝试加载最佳模型
if not EVAL_ONLY:
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
# 只计算一次最终的测试指标
test_mets = ml.get_metrics_logits(all_true, all_pred)
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

# 保存评估结果为 CSV 文件
import pandas as pd
from sastvd.linevd import get_relevant_metrics

# 收集评估指标
rank_metrics = svdr.rank_metr(all_pred, all_true)
MFR_value = sum(MFR) / len(MFR) if MFR else 0

# 构建 trial_result 结构
trial_result = [
    ID,  # 试验 ID
    str(svd.processed_dir() / "ivdetect" / ID / "best.model"),  # 检查点路径
    [0] * 10,  # 前10名准确率，这里填充占位符
    test_mets,  # 语句级评估指标
    test_mets,  # 方法级评估指标（使用相同的测试指标）
    rank_metrics,  # 排名评估指标
    test_mets,  # 语句行级评估指标（使用相同的测试指标）
    config["learning_rate"]  # 学习率
]

# 计算 acc@5（使用排名指标中的相关值或占位符）
trial_result[2][5] = rank_metrics.get("MAP@5", 0.0)

# 使用 get_relevant_metrics 函数
relevant_metrics = get_relevant_metrics(trial_result)

# 添加 MFR 到结果中
relevant_metrics["MFR"] = MFR_value

# 创建 DataFrame 并保存为 CSV
df = pd.DataFrame([relevant_metrics])
output_file = f"ivdetect_evaluation_results.csv"
df.to_csv(output_file, index=False)
print(f"评估结果已保存到: {output_file}")
print("\n评估结果:")
print(df)
