"""Phoenix数据集实验结果分析脚本。

该脚本用于分析LineVD模型在Phoenix数据集上的实验结果，
主要包含五个研究问题(RQ1-RQ5)的数据分析和结果生成，
最终输出LaTeX格式的结果表格用于学术论文。
"""
from glob import glob

import pandas as pd
import sastvd as svd

# 设置pandas显示所有列
pd.set_option("display.max_columns", None)

# %% Phoenix数据集分析
# 加载Phoenix数据集的实验结果
results = glob(str(svd.outputs_dir() / "phoenix/rq_results/*.csv"))
results2 = glob(str(svd.outputs_dir() / "phoenix_new/rq_results_new/*.csv"))
# 合并所有结果文件路径
results += results2
# 读取并合并所有结果数据框
res_df = pd.concat([pd.read_csv(i) for i in results])
# 去除重复的试验结果
res_df = res_df.drop_duplicates(["trial_id", "checkpoint"])

# 定义不同类型的指标列
metrics = [i for i in res_df.columns if "stmt" in i]  # 语句级指标
metricsf = [i for i in res_df.columns if "func" in i]  # 函数级指标
rankedcols = ["acc@5", "MAP@5", "nDCG@5", "MFR"]  # 排名相关指标
metricsline = [i for i in res_df.columns if "stmtline" in i]  # 语句行级指标
configcols = [i for i in res_df.columns if "config" in i]  # 配置参数列

# 处理图类型配置，移除"+raw"后缀
res_df["config/gtype"] = res_df["config/gtype"].apply(lambda x: x.replace("+raw", ""))

# RQ2分析的前置设置
# 为mlponly模型类型设置特殊的gnntype和gtype值
res_df["config/gnntype"] = res_df.apply(
    lambda x: "nognn" if x["config/modeltype"] == "mlponly" else x["config/gnntype"],
    axis=1,
)
res_df["config/gtype"] = res_df.apply(
    lambda x: "nognn" if x["config/modeltype"] == "mlponly" else x["config/gtype"],
    axis=1,
)

# RQ1: 不同嵌入类型(embtype)的性能分析
rq1_cg = "config/embtype"  # 分组列
# 按语句行级F1分数降序排序，每个试验取最佳结果
rq1 = res_df.sort_values("stmtline_f1", ascending=0).groupby("trial_id").head(1)
# 筛选使用默认数据分割的结果
rq1 = rq1[rq1["config/splits"] == "default"]
# 按嵌入类型分组，每个组取前5个最佳结果，然后计算平均值
rq1 = rq1.groupby(rq1_cg).head(5).groupby(rq1_cg).mean()[metricsline]

# RQ2: 不同GNN类型和图类型的性能分析
rq2_cg = ["config/gnntype", "config/gtype"]  # 分组列
# 筛选使用默认数据分割的结果
rq2 = res_df[res_df["config/splits"] == "default"]
# 按语句行级F1分数排序的结果
rq2a = rq2.sort_values("stmtline_f1", ascending=0).groupby("trial_id").head(1)
# 按语句级F1分数排序的结果
rq2b = rq2.sort_values("stmt_f1", ascending=0).groupby("trial_id").head(1)
# 筛选多任务类型为"linemethod"的结果
rq2b = rq2b[rq2b["config/multitask"] == "linemethod"]
# 计算不同GNN类型和图类型组合的平均语句行级指标
rq2a = rq2a.groupby(rq2_cg).head(5).groupby(rq2_cg).mean()[metricsline]
# 重命名列名，移除"line"前缀
rq2a.columns = [i.replace("line", "") for i in rq2a.columns]
# 计算不同GNN类型和图类型组合的平均语句级指标
rq2b = rq2b.groupby(rq2_cg).head(5).groupby(rq2_cg).mean()[metrics]
# 添加多任务类型标识
rq2a["multitask"] = "line"
rq2b["multitask"] = "line+method"
# 合并并分组计算最终结果
rq2final = pd.concat([rq2a, rq2b]).reset_index().groupby(rq2_cg + ["multitask"]).sum()

# RQ3: 不同多任务类型的性能分析
rq3_cg = "config/multitask"  # 分组列
# 按语句行级F1分数排序的结果
rq3a = res_df.sort_values("stmtline_f1", ascending=0).groupby("trial_id").head(1)
# 按语句级F1分数排序的结果
rq3b = res_df.sort_values("stmt_f1", ascending=0).groupby("trial_id").head(1)
# 筛选使用默认数据分割的结果
rq3a = rq3a[rq3a["config/splits"] == "default"]
rq3b = rq3b[rq3b["config/splits"] == "default"]
# 计算前5个最佳结果的平均语句行级指标
rq3a = rq3a.head(5).groupby(lambda x: True).mean()[metricsline]
# 重命名列名，移除"line"前缀
rq3a.columns = [i.replace("line", "") for i in rq3a.columns]
# 计算前5个最佳结果的所有指标
rq3b = rq3b.head(5).groupby(lambda x: True).mean()
# 合并语句行级和语句级指标结果
rq3 = pd.concat([rq3a, rq3b[metrics]])

# RQ5: 不同数据分割方式的性能分析
rq5_cg = "config/splits"  # 分组列
# 按语句级F1分数降序排序，每个试验取最佳结果
rq5 = res_df.sort_values("stmt_f1", ascending=0).groupby("trial_id").head(1)
# 按数据分割方式分组，每个组取前5个最佳结果，然后计算平均值
rq5 = rq5.groupby(rq5_cg).head(5).groupby(rq5_cg).mean()[metrics + ["MFR"]]

# 生成LaTeX格式的结果表格
rq1.round(3)[metricsline]  # RQ1结果，保留3位小数
rq2final.round(3)[metrics]  # RQ2最终结果，保留3位小数
rq3.round(3)[metrics]  # RQ3结果，保留3位小数
rq3b.round(3)[rankedcols]  # RQ3排名相关指标，保留3位小数
rq3b.round(3)[metricsf]  # RQ3函数级指标，保留3位小数
# 打印RQ5的LaTeX格式表格
print(
    rq5.round(3)[metrics][
        ["stmt_f1", "stmt_rec", "stmt_prec", "stmt_rocauc", "stmt_prauc"]
    ].to_latex()
)
