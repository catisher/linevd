"""BigVul数据集漏洞统计脚本。

该脚本用于统计BigVul数据集在方法级别和语句级别上的漏洞分布情况，
包括训练集、验证集和测试集中的漏洞函数数量、非漏洞函数数量、
漏洞语句数量和非漏洞语句数量，并计算漏洞比例，最终将结果保存为CSV文件。
"""

import pandas as pd
import sastvd as svd
import sastvd.linevd as lvd
from tqdm import tqdm

if __name__ == "__main__":
    # 初始化BigVul数据集数据模块
    # 配置：批量大小=1024，全量数据，语句级别，不使用邻居采样，图类型为PDG，默认数据划分
    data = lvd.BigVulDatasetLineVDDataModule(
        batch_size=1024,
        sample=-1,
        methodlevel=False,
        nsampling=False,
        gtype="pdg",
        splits="default",
    )

    # 初始化统计数据存储列表
    # train_func: 训练集方法级漏洞标签
    # train_stmt: 训练集语句级漏洞标签
    # val_func: 验证集方法级漏洞标签
    # val_stmt: 验证集语句级漏洞标签
    # test_func: 测试集方法级漏洞标签
    # test_stmt: 测试集语句级漏洞标签
    train_func = []
    train_stmt = []
    val_func = []
    val_stmt = []
    test_func = []
    test_stmt = []

    # 统计训练集漏洞分布
    # 遍历每个训练图，收集方法级和语句级漏洞标签
    for i in tqdm(range(len(data.train))):
        # 方法级漏洞标签：取图中所有节点的_FVULN标签的最大值
        train_func.append(data.train[i].ndata["_FVULN"].max().item())
        # 语句级漏洞标签：收集图中所有节点的_VULN标签
        train_stmt += data.train[i].ndata["_VULN"].tolist()

    # 统计验证集漏洞分布
    for i in tqdm(range(len(data.val))):
        val_func.append(data.val[i].ndata["_FVULN"].max().item())
        val_stmt += data.val[i].ndata["_VULN"].tolist()

    # 统计测试集漏洞分布
    for i in tqdm(range(len(data.test))):
        test_func.append(data.test[i].ndata["_FVULN"].max().item())
        test_stmt += data.test[i].ndata["_VULN"].tolist()

    def funcstmt_helper(funcs, stmts):
        """统计漏洞和非漏洞的数量。
        
        参数:
            funcs: 方法级漏洞标签列表 (1表示漏洞，0表示非漏洞)
            stmts: 语句级漏洞标签列表 (1表示漏洞，0表示非漏洞)
            
        返回:
            包含统计结果的字典，包括：
            - vul_funcs: 漏洞方法数量
            - nonvul_funcs: 非漏洞方法数量
            - vul_stmts: 漏洞语句数量
            - nonvul_stmts: 非漏洞语句数量
        """
        ret = {}
        ret["vul_funcs"] = funcs.count(1)  # 统计漏洞方法数量
        ret["nonvul_funcs"] = funcs.count(0)  # 统计非漏洞方法数量
        ret["vul_stmts"] = stmts.count(1)  # 统计漏洞语句数量
        ret["nonvul_stmts"] = stmts.count(0)  # 统计非漏洞语句数量
        return ret

    # 收集所有分区的统计结果
    stats = []
    stats.append({"partition": "train", **funcstmt_helper(train_func, train_stmt)})  # 添加训练集统计
    stats.append({"partition": "val", **funcstmt_helper(val_func, val_stmt)})  # 添加验证集统计
    stats.append({"partition": "test", **funcstmt_helper(test_func, test_stmt)})  # 添加测试集统计

    # 将统计结果转换为DataFrame
    df = pd.DataFrame.from_records(stats)
    
    # 计算漏洞比例
    df["func_ratio"] = df.vul_funcs / (df.vul_funcs + df.nonvul_funcs)  # 方法级漏洞比例
    df["stmt_ratio"] = df.vul_stmts / (df.vul_stmts + df.nonvul_stmts)  # 语句级漏洞比例
    
    # 将结果保存为CSV文件
    df.to_csv(svd.outputs_dir() / "bigvul_stats.csv", index=0)
