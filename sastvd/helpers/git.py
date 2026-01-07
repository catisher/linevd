"""
Git差异处理工具模块

该模块提供了与Git差异相关的功能，用于比较、解析和处理代码变更。

主要功能：
1. 比较两个字符串的Git差异
2. 解析Git patch获取修改和删除的行
3. 并行处理代码差异计算
4. 缓存和加载代码差异
5. 生成组合函数（提交前后的合并）

主要使用的库：
- os: 用于文件操作
- uuid: 用于生成唯一文件名
- multiprocessing.Pool: 用于并行处理
- sastvd: 项目核心库
- tqdm: 用于显示进度条
- unidiff: 用于解析Git patch
"""
import os
import pickle as pkl
import uuid
from multiprocessing import Pool

import sastvd as svd
from tqdm import tqdm
from unidiff import PatchSet


def gitdiff(old: str, new: str):
    """计算两个字符串之间的Git差异
    
    Args:
        old (str): 旧版本的字符串
        new (str): 新版本的字符串
    
    Returns:
        str: Git差异的字符串表示
    
    实现细节：
    1. 在缓存目录中创建两个临时文件，分别写入旧字符串和新字符串
    2. 执行git diff命令比较这两个文件
    3. 使用--no-index参数比较不在Git仓库中的文件
    4. 使用--no-prefix参数避免显示文件路径前缀
    5. 设置足够大的上下文行数(-U参数)以包含完整的文件内容
    6. 执行命令并获取输出
    7. 删除临时文件
    8. 将输出从字节转换为字符串并返回
    """
    cachedir = svd.cache_dir()
    oldfile = cachedir / uuid.uuid4().hex
    newfile = cachedir / uuid.uuid4().hex
    with open(oldfile, "w") as f:
        f.write(old)
    with open(newfile, "w") as f:
        f.write(new)
    cmd = " ".join(
        [
            "git",
            "diff",
            "--no-index",
            "--no-prefix",
            f"-U{len(old.splitlines()) + len(new.splitlines())}",
            str(oldfile),
            str(newfile),
        ]
    )
    process = svd.subprocess_cmd(cmd)
    os.remove(oldfile)
    os.remove(newfile)
    return process[0].decode()


def md_lines(patch: str):
    r"""从Git patch中获取修改和删除的行

    Args:
        patch (str): Git diff生成的patch字符串
    
    Returns:
        dict: 包含修改和删除行信息的字典，格式为{
            'added': [int],  # 添加的行号列表
            'removed': [int],  # 删除的行号列表
            'diff': str  # 差异内容
        }
    
    示例用法：
    old = "bool asn1_write_GeneralString(struct asn1_data *data, const char *s)\n\
    {\n\
       asn1_push_tag(data, ASN1_GENERAL_STRING);\n\
       asn1_write_LDAPString(data, s);\n\
       asn1_pop_tag(data);\n\
       return !data->has_error;\n\
    }\n\
    \
\
    "

    new = "bool asn1_write_GeneralString(struct asn1_data *data, const char *s)\n\
    {\n\
        if (!asn1_push_tag(data, ASN1_GENERAL_STRING)) return false;\n\
        if (!asn1_write_LDAPString(data, s)) return false;\n\
        return asn1_pop_tag(data);\n\
    }\n\
    \
\
    int test() {\n\
        return 1;\n\
    }\n\
    "

    patch = gitdiff(old, new)
    """
    parsed_patch = PatchSet(patch)
    ret = {"added": [], "removed": [], "diff": ""}
    if len(parsed_patch) == 0:
        return ret
    parsed_file = parsed_patch[0]
    hunks = list(parsed_file)
    assert len(hunks) == 1
    hunk = hunks[0]
    ret["diff"] = str(hunk).split("\n", 1)[1]
    for idx, ad in enumerate([i for i in ret["diff"].splitlines()], start=1):
        if len(ad) > 0:
            ad = ad[0]
            if ad == "+" or ad == "-":
                ret["added" if ad == "+" else "removed"].append(idx)
    return ret


def code2diff(old: str, new: str):
    """获取旧字符串和新字符串之间的添加和删除行
    
    Args:
        old (str): 旧版本的字符串
        new (str): 新版本的字符串
    
    Returns:
        dict: 包含修改和删除行信息的字典，格式与md_lines函数返回值相同
    
    实现细节：
    1. 调用gitdiff函数获取两个字符串的Git差异
    2. 调用md_lines函数解析差异获取修改和删除的行
    """
    patch = gitdiff(old, new)
    return md_lines(patch)


def _c2dhelper(item):
    """并行处理的辅助函数，计算并保存代码差异
    
    Args:
        item (dict): 包含以下键的字典：
            - func_before: 修改前的函数代码
            - func_after: 修改后的函数代码
            - id: 唯一标识符
            - dataset: 数据集名称
    
    实现细节：
    1. 确定缓存目录路径
    2. 构建保存路径，如果文件已存在则直接返回
    3. 如果修改前后的函数代码相同，则直接返回
    4. 调用code2diff函数计算代码差异
    5. 将结果保存到pickle文件中
    """
    savedir = svd.get_dir(svd.cache_dir() / item["dataset"] / "gitdiff")
    savepath = savedir / f"{item['id']}.git.pkl"
    if os.path.exists(savepath):
        return
    if item["func_before"] == item["func_after"]:
        return
    ret = code2diff(item["func_before"], item["func_after"])
    with open(savepath, "wb") as f:
        pkl.dump(ret, f)


def mp_code2diff(df):
    """并行计算代码差异
    
    Args:
        df (pandas.DataFrame): 包含以下列的数据框：
            - func_before: 修改前的函数代码
            - func_after: 修改后的函数代码
            - id: 唯一标识符
            - dataset: 数据集名称
    
    实现细节：
    1. 将数据框转换为字典列表，每个字典包含一行数据
    2. 创建包含6个进程的进程池
    3. 使用imap_unordered并行处理每个项目
    4. 使用tqdm显示处理进度
    """
    items = df[["func_before", "func_after", "id", "dataset"]].to_dict("records")
    with Pool(processes=6) as pool:
        for _ in tqdm(pool.imap_unordered(_c2dhelper, items), total=len(items)):
            pass


def get_codediff(dataset, iid):
    """从文件中获取代码差异
    
    Args:
        dataset (str): 数据集名称
        iid (str): 唯一标识符
    
    Returns:
        dict or list: 如果找到差异文件则返回差异字典，否则返回空列表
    
    实现细节：
    1. 确定缓存目录路径
    2. 构建保存路径
    3. 尝试从pickle文件中加载差异数据
    4. 如果加载失败（文件不存在或格式错误），返回空列表
    """
    savedir = svd.get_dir(svd.cache_dir() / dataset / "gitdiff")
    savepath = savedir / f"{iid}.git.pkl"
    try:
        with open(savepath, "rb") as f:
            return pkl.load(f)
    except:
        return []


def allfunc(row):
    """根据差异生成组合函数（提交前后的合并版本）
    
    Args:
        row (dict): 包含以下键的字典：
            - dataset: 数据集名称
            - id: 唯一标识符
            - func_before: 修改前的函数代码
    
    Returns:
        dict: 包含以下键的字典：
            - diff: 组合函数的原始差异
            - added: 添加的行号列表（相对于组合函数，从1开始）
            - removed: 删除的行号列表（相对于组合函数，从1开始）
            - before: 组合函数，其中添加的行被注释掉
            - after: 组合函数，其中删除的行被注释掉
    
    实现细节：
    1. 调用get_codediff函数获取差异数据
    2. 初始化返回字典，设置默认值
    3. 如果差异数据存在，处理差异行：
       - 对于删除行(-)，在before版本中保留，在after版本中注释掉
       - 对于添加行(+)，在before版本中注释掉，在after版本中保留
    4. 将处理后的行组合成完整的函数代码
    """
    readfile = get_codediff(row["dataset"], row["id"])

    ret = dict()
    ret["diff"] = "" if len(readfile) == 0 else readfile["diff"]
    ret["added"] = [] if len(readfile) == 0 else readfile["added"]
    ret["removed"] = [] if len(readfile) == 0 else readfile["removed"]
    ret["before"] = row["func_before"]
    ret["after"] = row["func_before"]

    if len(readfile) > 0:
        lines_before = []
        lines_after = []
        for li in ret["diff"].splitlines():
            if len(li) == 0:
                continue
            li_before = li
            li_after = li
            if li[0] == "-":
                li_before = li[1:]
                li_after = "// " + li[1:]
            if li[0] == "+":
                li_before = "// " + li[1:]
                li_after = li[1:]
            lines_before.append(li_before)
            lines_after.append(li_after)
        ret["before"] = "\n".join(lines_before)
        ret["after"] = "\n".join(lines_after)

    return ret
