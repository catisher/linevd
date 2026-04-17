"""评估训练好的IVDetect模型。"""

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
from dgl.dataloading import GraphDataLoader

# 配置
MODEL_PATH = "/home/wmy/linevd/storage/processed/ivdetect/202604170054_0077d02_test/best.model"  # 替换为实际模型路径
dev = torch.device("cpu")  # 与训练时保持一致

# 加载测试数据
print("加载测试数据...")
reload(ivd)
test_ds = ivd.BigVulDatasetIVDetect(partition="test")  # 使用完整测试集
dl_args = {"drop_last": False, "shuffle": False, "num_workers": 4}
test_dl = GraphDataLoader(test_ds, batch_size=64, **dl_args)

# 创建模型并加载权重
print("加载模型...")
model = ivd.IVDetect(200, 64)
model.to(dev)
model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
model.eval()

# 测试评估
print("开始测试评估...")
all_pred = []
all_true = []

with torch.no_grad():
    batch_count = 0
    for test_batch in test_dl:
        test_batch = test_batch.to(dev)
        test_labels = dgl.max_nodes(test_batch, "_VULN").long()
        test_logits = model(test_batch, test_ds)
        all_pred.append(test_logits.cpu())
        all_true.append(test_labels.cpu())
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"已处理 {batch_count} 个批次...")

# 合并所有预测和标签
all_pred = torch.cat(all_pred, dim=0)
all_true = torch.cat(all_true, dim=0)

print(f"总样本数: {len(all_true)}")

# 计算测试指标
print("计算最终指标...")
test_mets = ml.get_metrics_logits(all_true, all_pred)
rank_metr_test = ml.met_dict_to_str(svdr.rank_metr(all_pred, all_true))

print("\n=== 测试结果 ===")
print(f"准确率: {test_mets['acc']:.4f}")
print(f"精确率: {test_mets['prec']:.4f}")
print(f"召回率: {test_mets['rec']:.4f}")
print(f"F1值: {test_mets['f1']:.4f}")
print(f"AUC: {test_mets['roc_auc']:.4f}")
print(f"排名指标: {rank_metr_test}")

# 语句级分析（与main.py一致）
print("\n开始GNNExplainer分析...")
correct_lines = ivde.get_dep_add_lines_bigvul()
pred_lines = dict()

sample_count = 0
for batch in test_dl:
    for g in dgl.unbatch(batch):
        sampleid = g.ndata["_SAMPLE"].max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in pred_lines:
            continue
        try:
            lines = ge.gnnexplainer(model, g.to(dev), test_ds)
        except Exception as E:
            print(E)
        pred_lines[sampleid] = lines
        sample_count += 1
        if sample_count % 10 == 0:
            print(f"已分析 {sample_count} 个样本...")

# 保存预测结果
with open(svd.cache_dir() / "pred_lines.pkl", "wb") as f:
    pkl.dump(pred_lines, f)

# 计算MFR
MFR = []
for sampleid, pred in pred_lines.items():
    true = correct_lines[sampleid]
    true = list(true["removed"]) + list(true["depadd"])
    for i, p in enumerate(pred):
        if p in true:
            MFR += [i]
            break

if MFR:
    mean_mfr = sum(MFR) / len(MFR)
    print(f"\n=== GNNExplainer 结果 ===")
    print(f"Mean First Rank (MFR): {mean_mfr:.2f}")
    print(f"MFR总和: {sum(MFR)}")
    print(f"MFR数量: {len(MFR)}")
else:
    print("\n没有找到匹配的漏洞行")

print("\n评估完成！")
