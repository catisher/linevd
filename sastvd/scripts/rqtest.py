"""RQ实验测试结果生成脚本。

该脚本用于从Ray Tune的超参数调优结果中加载模型检查点，并在测试集上运行测试，
生成并保存测试结果。它会定期检查新的实验结果并自动处理未测试的模型。
"""

import os
import time
from glob import glob
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd
from ray.tune import Analysis


def main(config, df):
    """运行模型测试并生成测试结果。

    参数:
        config: 包含模型配置的字典，包括gtype（图类型）、splits（数据集分割）和embtype（嵌入类型）
        df: 包含所有实验结果的DataFrame
    """
    # 设置测试结果保存目录
    main_savedir = svd.get_dir(svd.outputs_dir() / "rq_results_new")
    # 根据配置筛选相关实验结果
    df_gtype = df[
        (df["config/gtype"] == config["config/gtype"])
        & (df["config/splits"] == config["config/splits"])
        & (df["config/embtype"] == config["config/embtype"])
    ]

    # 检查是否所有检查点都已测试
    skipall = True
    for row in df_gtype.itertuples():
        chkpt_list = glob(row.logdir + "/checkpoint_*")  # 获取所有检查点目录
        chkpt_list = [i + "/checkpoint" for i in chkpt_list]  # 构造完整检查点路径
        for chkpt in chkpt_list:
            chkpt_info = Path(chkpt).parent.name
            chkpt_res_path = main_savedir / f"{row.trial_id}_{chkpt_info}.csv"
            if not os.path.exists(chkpt_res_path):
                skipall = False  # 存在未测试的检查点
                break
    if skipall:
        return  # 所有检查点都已测试，返回

    # 获取超参数列
    hparam_cols = df_gtype.columns[df_gtype.columns.str.contains("config")].tolist()
    hparam_cols += ["experiment_id", "logdir"]
    # 创建数据集模块
    data = lvd.BigVulDatasetLineVDDataModule(
        batch_size=1024,  # 批次大小
        nsampling_hops=2,  # 采样跳数
        gtype=config["config/gtype"],  # 图类型
        splits=config["config/splits"],  # 数据集分割方式
        feat=config["config/embtype"],  # 特征嵌入类型
    )
    # 遍历每个实验结果
    for row in df_gtype.itertuples():
        chkpt_list = glob(row.logdir + "/checkpoint_*")
        chkpt_list = [i + "/checkpoint" for i in chkpt_list]
        try:
            # 遍历每个检查点
            for chkpt in chkpt_list:
                chkpt_info = Path(chkpt).parent.name
                chkpt_res_path = main_savedir / f"{row.trial_id}_{chkpt_info}.csv"
                if os.path.exists(chkpt_res_path):
                    continue  # 跳过已测试的检查点
                # 加载模型并测试
                model = lvd.LitGNN()
                model = lvd.LitGNN.load_from_checkpoint(chkpt, strict=False)
                trainer = pl.Trainer(gpus=1, default_root_dir="/tmp/")  # 创建训练器
                trainer.test(model, data)  # 运行测试
                # 收集测试结果
                res = [
                    row.trial_id,
                    chkpt_info,
                    model.res1vo,
                    model.res2mt,
                    model.res2f,
                    model.res3vo,
                    model.res2,
                    model.lr,
                ]
                # 保存结果到CSV
                mets = lvd.get_relevant_metrics(res)  # 获取相关指标
                hparams = df[df.trial_id == res[0]][hparam_cols].to_dict("records")[0]  # 获取超参数
                res_df = pd.DataFrame.from_records([{**mets, **hparams}])  # 合并指标和超参数
                res_df.to_csv(chkpt_res_path, index=0)  # 保存为CSV
        except Exception as E:
            print(E)  # 捕获并打印异常


if __name__ == "__main__":
    # 无限循环，定期检查新的实验结果
    while True:
        try:
            # 获取所有Ray Tune实验目录
            raytune_dirs = glob(str(svd.processed_dir() / "raytune_*_-1"))
            tune_dirs = [i for j in [glob(f"{rd}/*") for rd in raytune_dirs] for i in j]

            # 加载所有实验结果数据
            df_list = []
            for d in tune_dirs:
                df_list.append(Analysis(d).dataframe())  # 加载每个实验的结果
            df = pd.concat(df_list)  # 合并所有实验结果

            # 确保配置列存在
            if "config/splits" not in df.columns:
                df["config/splits"] = "default"
            if "config/embtype" not in df.columns:
                df["config/embtype"] = "codebert"
            # 获取唯一的配置组合
            configs = df[["config/gtype", "config/splits", "config/embtype"]]
            configs = configs.drop_duplicates().to_dict("records")  # 去除重复配置

            # 对每个配置运行测试
            for config in configs:
                main(config, df)
        except Exception as E:
            print(E)  # 捕获并打印异常
            pass

        # 休眠60秒后再次检查
        print("Sleeping...")
        time.sleep(60)
