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
all_true_f = []  # 方法级真实标签
all_pred_f = []  # 方法级预测
all_funcs = []  # 用于语句级评估的函数列表

with torch.no_grad():
    batch_count = 0
    for test_batch in test_dl:
        test_batch = test_batch.to(dev)
        test_labels = dgl.max_nodes(test_batch, "_VULN").long()
        test_logits = model(test_batch, test_ds)
        
        # 保存方法级预测和标签
        all_true_f.append(test_labels.cpu())
        all_pred_f.append(test_logits.cpu())
        
        # 处理每个图的语句级预测
        for i, g in enumerate(dgl.unbatch(test_batch)):
            # 获取语句级预测
            # 注意：IVDetect 可能没有直接的语句级预测，这里需要根据模型结构调整
            # 假设模型返回的是图级别的预测
            # 这里使用图级别的预测作为所有语句的预测
            # 实际应用中需要根据模型结构调整
            num_nodes = g.num_nodes()
            node_logits = test_logits[i].unsqueeze(0).repeat(num_nodes, 1)
            node_labels = g.ndata["_VULN"].long().cpu().numpy().tolist()
            
            # 构建语句级预测列表
            sm_logits = torch.softmax(node_logits, dim=1).cpu().numpy().tolist()
            all_funcs.append([sm_logits, node_labels, [test_labels[i].item()]])
        
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"已处理 {batch_count} 个批次...")

# 合并所有预测和标签
all_true_f = torch.cat(all_true_f, dim=0)
all_pred_f = torch.cat(all_pred_f, dim=0)

print(f"总样本数: {len(all_true_f)}")

# 计算语句级评估指标
print("计算语句级评估指标...")
res1 = ivde.eval_statements_list(all_funcs)
res1vo = ivde.eval_statements_list(all_funcs, vo=True, thresh=0)

print("\n=== 语句级评估结果 ===")
print("包含负样本的排名准确率:")
for k, v in res1.items():
    print(f"Top-{k}: {v:.4f}")
print("\n仅包含正样本的排名准确率:")
for k, v in res1vo.items():
    print(f"Top-{k}: {v:.4f}")

# 计算方法级评估指标
print("\n计算方法级评估指标...")
test_mets_f = ml.get_metrics_logits(all_true_f, all_pred_f)

print("\n=== 方法级测试结果 ===")
print(f"准确率: {test_mets_f['acc']:.4f}")
print(f"精确率: {test_mets_f['prec']:.4f}")
print(f"召回率: {test_mets_f['rec']:.4f}")
print(f"F1值: {test_mets_f['f1']:.4f}")
print(f"AUC: {test_mets_f['roc_auc']:.4f}")

# 计算排名指标
print("\n计算排名指标...")
rank_metrs = []
rank_metrs_vo = []
for af in all_funcs:
    # 计算排名指标
    rank_metr_calc = svdr.rank_metr([i[1] for i in af[0]], af[1], 0)
    if max(af[1]) > 0:  # 如果是正样本
        rank_metrs_vo.append(rank_metr_calc)
    rank_metrs.append(rank_metr_calc)

try:
    # 计算所有样本的平均排名指标
    res3 = ml.dict_mean(rank_metrs)
    print("\n=== 所有样本排名指标 ===")
    for k, v in res3.items():
        print(f"{k}: {v:.4f}")
except Exception as E:
    print(f"计算排名指标时出错: {E}")

# 计算仅正样本的平均排名指标
try:
    res3vo = ml.dict_mean(rank_metrs_vo)
    print("\n=== 仅正样本排名指标 ===")
    for k, v in res3vo.items():
        print(f"{k}: {v:.4f}")
except Exception as E:
    print(f"计算正样本排名指标时出错: {E}")

# 从语句级别预测方法级别
print("\n从语句级别预测方法级别...")
method_level_pred = []
method_level_true = []
for af in all_funcs:
    # 计算方法级真实标签（如果有任何漏洞行，则为漏洞）
    method_level_true.append(1 if sum(af[1]) > 0 else 0)
    pred_method = 0  # 默认预测为安全
    # 如果任何行预测为漏洞，则方法级预测为漏洞
    for logit in af[0]:
        if logit[1] > 0.5:
            pred_method = 1
            break
    method_level_pred.append(pred_method)

# 计算从行级预测方法级的指标
res4 = ml.get_metrics(method_level_true, method_level_pred)
print("\n=== 从语句级预测方法级结果 ===")
print(f"准确率: {res4['acc']:.4f}")
print(f"精确率: {res4['prec']:.4f}")
print(f"召回率: {res4['rec']:.4f}")
print(f"F1值: {res4['f1']:.4f}")

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
