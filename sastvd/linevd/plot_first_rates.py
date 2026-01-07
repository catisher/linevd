"""LineVD模型首排名性能分析脚本。

该脚本用于分析LineVD模型在漏洞检测任务中的"首排名"性能指标，
即模型将最易受攻击的代码行预测为最高排名的频率。
主要功能包括：
1. 加载并合并Ray Tune的超参数调优结果
2. 选择性能最佳的模型
3. 加载模型并进行测试
4. 计算并可视化首排名指标
"""
import pickle as pkl
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd
import seaborn as sns
from ray.tune import Analysis

if __name__ == "__main__":

    # 获取存储在storage/processed目录下的Ray Tune分析目录
    raytune_dirs = glob(str(svd.processed_dir() / "raytune_*_-1"))
    tune_dirs = [i for j in [glob(f"{rd}/*") for rd in raytune_dirs] for i in j]

    # 加载完整的数据框
    df_list = []
    for d in tune_dirs:
        df_list.append(Analysis(d).dataframe())
    df = pd.concat(df_list)
    # 筛选使用默认数据分割的结果
    df = df[df["config/splits"] == "default"]

    # 加载结果数据框
    results = glob(str(svd.outputs_dir() / "rq_results_new/*.csv"))
    res_df = pd.concat([pd.read_csv(i) for i in results])

    # 合并数据框并选择性能最佳的模型
    mdf = df.merge(res_df[["trial_id", "checkpoint", "stmt_f1"]], on="trial_id")
    best = mdf.sort_values("stmt_f1", ascending=0).iloc[0]
    best_path = f"{best['logdir']}/{best['checkpoint']}/checkpoint"

    # 加载模型和数据模块
    datamodule_args = {
        "batch_size": 1024,
        "nsampling_hops": 2,
        "gtype": best["config/gtype"],  # 使用最佳模型的图类型配置
        "splits": best["config/splits"],  # 使用最佳模型的数据分割配置
    }
    data = lvd.BigVulDatasetLineVDDataModule(**datamodule_args)
    trainer = pl.Trainer(gpus=1, default_root_dir="/tmp/")
    model = lvd.LitGNN.load_from_checkpoint(best_path, strict=False)
    # 对模型进行测试
    trainer.test(model, data)

    # 检查语句级指标
    print("RESRANK1")  # 排名1的指标
    print(model.res1vo)
    print("RES2MT")  # 多任务指标
    print(model.res2mt)
    print("RESF")  # F1分数相关指标
    print(model.res2f)
    print("RESRANK")  # 排名相关指标
    print(model.res3vo)
    print("RESLINE")  # 行级指标
    print(model.res2)

    # 获取首排名的代码行
    # 筛选出包含漏洞的函数（最大预测概率为1）
    vulns = [i for i in model.all_funcs if max(i[1]) == 1]
    # 进一步筛选出存在真实漏洞的函数
    vulns = [i for i in vulns if i[2].max() == 1]

    def get_fr(v):
        """获取首个漏洞预测的排名。
        
        参数:
            v: 包含函数预测结果的数据结构
            
        返回:
            rank + 1: 首个真实漏洞在预测中的排名（从1开始）
        """
        # 提取预测概率和真实标签
        zipped = list(zip([i[1] for i in v[0]], v[1]))
        # 按预测概率降序排序
        zipped.sort(reverse=True, key=lambda x: x[0])
        # 找到第一个真实漏洞的排名
        for rank, i in enumerate(zipped):
            if i[1] == 1:
                return rank + 1

    # 计算所有漏洞函数的首排名
    histogram_data = [get_fr(v) for v in vulns]
    
    # 绘制首排名直方图
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    sns.histplot(histogram_data, bins=30)
    savedir = svd.get_dir(svd.outputs_dir() / "mfrplots")
    plt.savefig(savedir / "mfrhist.pdf", bbox_inches="tight")
    # 保存首排名数据
    with open(savedir / "histdata.pkl", "wb") as f:
        pkl.dump(histogram_data, f)

    # 更详细的绘图设置
    font = {"family": "normal", "weight": "normal", "size": 15}
    matplotlib.rc("font", **font)
    hist_data = pkl.load(open(savedir / "histdata.pkl", "rb"))
    
    # 创建两个子图，分别显示排名<=5和>5的情况
    fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True, figsize=(8, 4))
    sns.histplot([i for i in hist_data if i <= 5], ax=axs[0], bins=5)
    sns.histplot([i for i in hist_data if i > 5], ax=axs[1], bins=10)
    axs[1].set_ylabel("")  # 移除右侧子图的y轴标签
    fig.text(0.54, -0.02, "First Ranking", ha="center")  # 添加共享的x轴标签
    plt.savefig(savedir / "mfrhist.pdf", bbox_inches="tight")

    # 计算排名<=5和>5的数量
    len([i for i in hist_data if i <= 5])
    len([i for i in hist_data if i > 5])
