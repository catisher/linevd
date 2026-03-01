import pandas as pd
import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.git as svdg
from tqdm import tqdm

# 启用pandas的tqdm进度条功能
tqdm.pandas()


def insert_bigvul_comments(diff: str):
    """在git diff补丁中插入注释行，替换+和-符号
    
    Args:
        diff: git diff格式的字符串
        
    Returns:
        插入注释后的代码字符串
    """
    lines = []
    # 遍历diff中的每一行
    for li in diff.splitlines():
        # 跳过空行
        if len(li) == 0:
            continue
        # 处理删除的行（以-开头）
        if li[0] == "-":
            # 添加注释标记删除的行
            lines.append("//flaw_line_below:")
            # 移除开头的-符号
            li = li[1:]
        # 处理添加的行（以+开头）
        if li[0] == "+":
            # 添加注释标记修复的行
            lines.append("//fix_flaw_line_below:")
            # 在行前添加//注释
            li = "//" + li[1:]
        # 添加处理后的行
        lines.append(li)
    # 将所有行重新组合成字符串并返回
    return "\n".join(lines)


def apply_bigvul_comments(row):
    """使用pandas应用get_codediff函数
    
    Args:
        row: DataFrame中的一行数据
        
    Returns:
        获取到的diff字符串，如果没有则返回空字符串
    """
    # 调用git助手函数获取代码差异
    ret = svdg.get_codediff(row.dataset, row.id)
    # 如果返回结果为空，返回空字符串，否则返回diff部分
    return "" if len(ret) == 0 else ret["diff"]


def fine_grain_diff(row, diff=False):
    """获取代码的细粒度差异
    
    Args:
        row: DataFrame中的一行数据
        diff: 是否将差异写入文件
        
    Returns:
        添加和删除的行总数
    """
    # 如果两行代码相等，返回0
    if row.equality:
        return 0
    
    # 获取两个版本的函数代码
    f1 = row.vfwf
    f2 = row.vul_func_with_fix
    
    # 去除每行的首尾空格
    f1 = "\n".join([i.strip() for i in f1.splitlines()])
    f2 = "\n".join([i.strip() for i in f2.splitlines()])
    
    # 计算代码差异
    cd = svdg.code2diff(f1, f2)
    added = cd["added"]  # 添加的行号
    removed = cd["removed"]  # 删除的行号
    
    # 如果需要，将差异写入文件
    if diff:
        with open(svd.cache_dir() / "difftest.c", "w") as f:
            f.write(cd["diff"])
        with open(svd.cache_dir() / "difftest1.c", "w") as f:
            f.write(f1)
        with open(svd.cache_dir() / "difftest2.c", "w") as f:
            f.write(f2)
    
    # 返回添加和删除的行总数
    return len(added) + len(removed)


def test_bigvul_diff_similarity():
    """测试BigVul数据集的差异相似性
    
    测试目的：验证添加注释后的代码与修复后的代码是否具有足够的相似性
    """
    # 加载BigVul数据集，非最小化版本，使用样本数据
    df = svdd.bigvul(minimal=False, sample=True)
    # 筛选出漏洞样本
    df_vul = df[df.vul == 1].copy()
    # 多进程计算代码差异
    svdg.mp_code2diff(df_vul)
    # 应用注释到原始差异
    df_vul["vfwf_orig"] = df_vul.progress_apply(apply_bigvul_comments, axis=1)
    # 插入注释到差异中
    df_vul["vfwf"] = df_vul.vfwf_orig.progress_apply(insert_bigvul_comments)
    # 计算注释后代码与修复后代码的相等性（去除空格后比较）
    df_vul["equality"] = df_vul.progress_apply(
        lambda x: " ".join(x.vfwf.split()) == " ".join(x.vul_func_with_fix.split()),
        axis=1,
    )
    # 断言至少30%的样本具有相似性
    assert len(df_vul[df_vul.equality]) / len(df_vul) >= 0.3


def test_bigvul_diff_similarity_2():
    """测试BigVul最小化数据集的差异相似性
    
    测试目的：验证最小化版本的BigVul数据集中，修复前后的代码行数是否相同
    """
    # 加载BigVul最小化数据集，使用样本数据
    df = svdd.bigvul(minimal=True, sample=True)
    # 计算修复前代码的行数
    df["len_1"] = df.before.apply(lambda x: len(x.splitlines()))
    # 计算修复后代码的行数
    df["len_2"] = df.after.apply(lambda x: len(x.splitlines()))
    # 断言所有样本的修复前后代码行数相同
    assert len(df[df.len_1 != df.len_2]) == 0


def test_code2diff_cases():
    """测试code2diff函数的具体案例
    
    测试目的：验证code2diff函数在特定案例中的表现是否正确
    """
    # 加载BigVul数据集，非最小化版本，返回原始数据
    df = svdd.bigvul(minimal=False, return_raw=True)
    # 筛选出漏洞样本
    df = df[df.vul == 1]
    # 将数据转换为字典，方便按id访问
    dfd = df.set_index("id")[["func_before", "func_after"]].to_dict()

    # 测试案例1：验证特定id的代码差异计算
    codediff = svdg.code2diff(dfd["func_before"][177775], dfd["func_after"][177775])
    assert codediff["removed"] == [16]  # 验证删除的行号
    assert codediff["added"] == [17]    # 验证添加的行号

    # 测试案例2：验证另一个特定id的代码差异计算
    codediff = svdg.code2diff(dfd["func_before"][180189], dfd["func_after"][180189])
    assert codediff["removed"] == [36]                    # 验证删除的行号
    assert codediff["added"] == [24, 25, 26, 27, 28, 29, 37]  # 验证添加的行号
