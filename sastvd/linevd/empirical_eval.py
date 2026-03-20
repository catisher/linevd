import os
from collections import Counter, defaultdict
from glob import glob
from math import sqrt

import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.helpers.dclass as svddc
import sastvd.linevd as lvd
import sastvd.linevd.c_builtins as cbuiltin
from ray.tune import ExperimentAnalysis
from tqdm import tqdm


class EmpEvalBigVul:
    """执行实证评估的类。
    
    该类用于对模型预测结果进行实证分析，包括函数级和语句级的评估。
    """

    def __init__(self, all_funcs: list, test_data: lvd.BigVulDatasetLineVD):
        """初始化实证评估类。

        Args:
            all_funcs (list): 从LitGNN返回的所有函数预测结果
            test_data (lvd.BigVulDatasetLineVD): 测试数据集

        Example:
            model = lvd.LitGNN()
            model = lvd.LitGNN.load_from_checkpoint($BESTMODEL$, strict=False)
            trainer.test(model, data)
            all_funcs = model.all_funcs

            datamodule_args = {"batch_size": 1024, "nsampling_hops": 2, "gtype": "pdg+raw"}

            eebv = EmpEvalBigVul(model.all_funcs, data.test)
            eebv.eval_test()
        """
        # 加载函数元数据
        self.func_df = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
        self.func_df = self.func_df.set_index("id")
        # 存储所有函数预测结果
        self.all_funcs = all_funcs
        # 存储测试数据
        self.test_data = test_data

    def func_metadata(self, _id):
        """获取函数元数据。
        
        Args:
            _id: 函数ID
            
        Returns:
            dict: 函数的元数据信息
        """
        return self.func_df.loc[_id].to_dict()

    def stmt_metadata(self, _id):
        """获取语句元数据。
        
        Args:
            _id: 样本ID
            
        Returns:
            dict: 语句的元数据信息，以行号为键
        """
        # 提取代码属性图中的节点信息
        n = lvd.feature_extraction(svddc.BigVulDataset.itempath(_id), return_nodes=True)
        # 保留需要的列
        keepcols = ["_label", "name", "controlStructureType", "local_type"]
        n = n.set_index("lineNumber")[keepcols]
        return n.to_dict("index")

    def test_item(self, idx):
        """获取测试项信息。
        
        Args:
            idx: 测试项索引
            
        Returns:
            tuple: (函数数据, 语句数据)的元组
        """
        # 获取样本ID
        _id = self.test_data.idx2id[idx]
        # 获取预测结果
        preds = self.all_funcs[idx]
        # 获取函数元数据
        f_data = self.func_metadata(_id)
        # 获取语句元数据
        s_data = self.stmt_metadata(_id)

        # 格式化函数数据
        f_data["pred"] = preds[2].max().item()
        f_data["vul"] = max(preds[1])

        # 格式化语句数据
        s_pred_data = defaultdict(dict)

        for i in range(len(preds[0])):
            s_pred_data[preds[3][i]]["vul"] = preds[1][i]
            if f_data["pred"] == 1:
                s_pred_data[preds[3][i]]["pred"] = list(preds[0][i])
            else:
                s_pred_data[preds[3][i]]["pred"] = [1, 0]
            s_pred_data[preds[3][i]].update(s_data[preds[3][i]])

        return f_data, dict(s_pred_data)

    def eval_test(self):
        """评估所有测试数据。
        
        遍历所有测试数据，收集函数和语句级别的预测结果。
        """
        # 初始化结果列表
        self.func_results = []
        self.stmt_results = []
        # 记录失败次数和错误信息
        self.failed = 0
        self.err = []
        # 遍历所有测试数据
        for i in tqdm(range(len(self.test_data))):
            try:
                # 获取测试项信息
                f_ret, s_ret = self.test_item(i)
            except Exception as E:
                # 记录失败情况
                self.failed += 1
                self.err.append(E)
                continue
            # 添加到结果列表
            self.func_results.append(f_ret)
            self.stmt_results.append(s_ret)
        return


if __name__ == "__main__":

    # 获取存储/处理目录中的分析目录
    raytune_dirs = glob(str(svd.processed_dir() / "raytune_*_-1"))
    
    # 加载完整的数据框架 - 搜索 tune_linevd 子目录
    df_list = []
    for rd in raytune_dirs:
        # 查找 tune_linevd 子目录
        tune_subdirs = glob(f"{rd}/*/tune_linevd")
        for d in tune_subdirs:
            try:
                # 加载实验分析
                analysis = ExperimentAnalysis(d)
                df_list.append(analysis.dataframe())
                print(f"Successfully loaded: {d}")
            except Exception as e:
                print(f"Warning: Skipping {d} - {e}")
                continue
    
    # 检查是否找到实验数据
    if not df_list:
        print("Error: No experiment data found")
        exit(1)
    
    # 合并数据框
    df = pd.concat(df_list)
    # 筛选默认分割的数据
    df = df[df["config/splits"] == "default"]
    
    # 搜索实际的检查点文件
    checkpoint_files = []
    
    for base_dir in raytune_dirs:
        # 递归查找所有 checkpoint 文件
        trial_dirs = glob(f"{base_dir}/**/train_linevd_*", recursive=True)
        for trial_dir in trial_dirs:
            # 查找 checkpoint 子目录
            checkpoint_dirs = glob(f"{trial_dir}/checkpoint_*")
            for checkpoint_dir in checkpoint_dirs:
                checkpoint_file = os.path.join(checkpoint_dir, "checkpoint")
                if os.path.exists(checkpoint_file):
                    checkpoint_files.append(checkpoint_file)
    
    # 检查是否找到检查点文件
    if not checkpoint_files:
        print("Error: No checkpoint files found")
        exit(1)
    
    # 选择第一个找到的检查点
    best_path = checkpoint_files[0]
    print(f"Found {len(checkpoint_files)} checkpoint files")
    print(f"Using checkpoint: {best_path}")
    
    # 从检查点路径中提取配置信息（简化处理）
    best = df.iloc[0]  # 使用第一个实验的配置

    # 加载模块
    model = lvd.LitGNN()
    datamodule_args = {
        "batch_size": 1024,
        "nsampling_hops": 2,
        "gtype": best["config/gtype"],
        "splits": best["config/splits"],
    }
    # 创建数据模块
    data = lvd.BigVulDatasetLineVDDataModule(**datamodule_args)
    # 创建训练器
    trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir="/tmp/")
    # 从检查点加载模型
    model = lvd.LitGNN.load_from_checkpoint(best_path, strict=False)
    # 测试模型
    trainer.test(model, data)

    # 检查语句级指标
    print(model.res2mt)

    # 进行实证评估
    eebv = EmpEvalBigVul(model.all_funcs, data.test)
    eebv.eval_test()

    # 函数级矩阵评估
    func_tp = [i for i in eebv.func_results if i["pred"] == 1 and i["vul"] == 1]
    func_tn = [i for i in eebv.func_results if i["pred"] == 0 and i["vul"] == 0]
    func_fp = [i for i in eebv.func_results if i["pred"] == 1 and i["vul"] == 0]
    func_fn = [i for i in eebv.func_results if i["pred"] == 0 and i["vul"] == 1]

    # 语句级矩阵评估
    stmt_tp = []
    stmt_tn = []
    stmt_fp = []
    stmt_fn = []
    for func in eebv.stmt_results:
        for stmt in func.values():
            if stmt["pred"][1] > model.f1thresh and stmt["vul"] == 1:
                stmt_tp.append(stmt)
            if stmt["pred"][1] < model.f1thresh and stmt["vul"] == 0:
                stmt_tn.append(stmt)
            if stmt["pred"][1] > model.f1thresh and stmt["vul"] == 0:
                stmt_fp.append(stmt)
            if stmt["pred"][1] < model.f1thresh and stmt["vul"] == 1:
                stmt_fn.append(stmt)

    # 主要分析 - 语句级
    def count_labels(stmts):
        """获取语句信息。
        
        Args:
            stmts: 语句列表
            
        Returns:
            Counter: 语句类型计数
        """
        label_info = []
        for info in stmts:
            if info["_label"] == "CALL":
                if "<operator>" in info["name"]:
                    if "assignment" in info["name"]:
                        label_info.append("Assignment Operator")
                        continue
                    if (
                        "addition" in info["name"]
                        or "subtraction" in info["name"]
                        or "division" in info["name"]
                        or "Plus" in info["name"]
                        or "Minus" in info["name"]
                        or "minus" in info["name"]
                        or "plus" in info["name"]
                        or "modulo" in info["name"]
                        or "multiplication" in info["name"]
                    ):
                        label_info.append("Arithmetic Operator")
                        continue
                    if (
                        "lessThan" in info["name"]
                        or "greaterThan" in info["name"]
                        or "EqualsThan" in info["name"]
                        or "equals" in info["name"]
                    ):
                        label_info.append("Comparison Operator")
                        continue
                    if (
                        "FieldAccess" in info["name"]
                        or "IndexAccess" in info["name"]
                        or "fieldAccess" in info["name"]
                        or "indexAccess" in info["name"]
                    ):
                        label_info.append("Access Operator")
                        continue
                    if (
                        "logical" in info["name"]
                        or "<operator>.not" in info["name"]
                        or "<operator>.or" in info["name"]
                        or "<operator>.and" in info["name"]
                        or "conditional" in info["name"]
                    ):
                        label_info.append("Logical Operator")
                        continue
                    if "<operator>.cast" in info["name"]:
                        label_info.append("Cast Operator")
                        continue
                    if "<operator>" in info["name"]:
                        label_info.append("Other Operator")
                        continue
                elif info["name"] in cbuiltin.l_funcs:
                    label_info.append("Builtin Function Call")
                    continue
                else:
                    label_info.append("External Function Call")
                    continue
            if info["_label"] == "CONTROL_STRUCTURE":
                label_info.append(info["controlStructureType"])
                continue
            label_info.append(info["_label"])
        return Counter(label_info)

    def calc_mcc(tp, fp, tn, fn):
        """计算马修斯相关系数(MCC)。
        
        Args:
            tp: 真阳性
            fp: 假阳性
            tn: 真阴性
            fn: 假阴性
            
        Returns:
            float: MCC值
        """
        x = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        return ((tp * tn) - (fp * fn)) / sqrt(x)

    def calc_f1(tp, fp, tn, fn):
        """计算F1分数。
        
        Args:
            tp: 真阳性
            fp: 假阳性
            tn: 真阴性
            fn: 假阴性
            
        Returns:
            float: F1分数
        """
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        return 2 * ((prec * rec) / (prec + rec))

    # 统计各类别语句的数量
    counts = [
        count_labels(stmt_fp),
        count_labels(stmt_fn),
        count_labels(stmt_tp),
        count_labels(stmt_tn),
    ]
    # 获取所有节点类型
    all_node_types = set([j for k in [list(i.keys()) for i in counts] for j in k])
    ntmat = []
    # 遍历所有节点类型
    for node_type in all_node_types:
        # 获取各类别的计数
        fp = counts[0][node_type] if node_type in counts[0] else 0
        fn = counts[1][node_type] if node_type in counts[1] else 0
        tp = counts[2][node_type] if node_type in counts[2] else 0
        tn = counts[3][node_type] if node_type in counts[3] else 0
        try:
            # 计算MCC
            mcc = calc_mcc(tp, fp, tn, fn)
        except:
            mcc = None
        try:
            # 计算F1
            f1 = calc_f1(tp, fp, tn, fn)
        except:
            f1 = None
        # 美化节点类型名称
        if node_type == "METHOD_PARAMETER_IN":
            node_type = "Parameter In"
        if node_type == "METHOD_PARAMETER_OUT":
            node_type = "Parameter Out"
        if node_type == "ThrowStatement":
            node_type = "Throw Statement"
        if node_type == "JUMP_TARGET":
            node_type = "Jump Target"
        if node_type == "FIELD_IDENTIFIER":
            node_type = "Field Identifier"
        if node_type == "ThrowStatement":
            continue
        # 添加到结果列表
        ntmat.append(
            {
                "Statement Type": node_type.title(),
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                # "MCC": abs(round(mcc, 2)) if mcc else None,
                "F1": abs(round(f1, 2)) if f1 else None,
            }
        )

    # 获取最终数据框
    stmt_analysis = pd.DataFrame.from_records(ntmat).sort_values("F1", ascending=0)
    # 输出LaTeX格式的分析结果
    print(stmt_analysis.to_latex(index=0))
    # Func analysis: length of function, number of interprocedural calls, other software metrics.
