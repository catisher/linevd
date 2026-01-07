import os
import pickle as pkl
import uuid
from xml.etree import cElementTree

import sastvd as svd

"""
SAST工具集成模块

该模块提供了与多种静态代码分析工具（SAST）的集成接口，用于检测代码中的潜在漏洞。

支持的工具：
1. FlawFinder - 检测C/C++代码中的安全漏洞
2. RATS - Rough Auditing Tool for Security，安全审计工具
3. CppCheck - C/C++静态代码分析工具

主要功能：
- 运行单个SAST工具
- 同时运行多个SAST工具
- 解析并统一处理各工具的输出格式
- 提取有问题的代码行号
"""


def file_helper(content: str) -> str:
    """将内容保存到临时文件并返回文件路径。
    
    参数:
        content (str): 要保存的内容
    
    返回:
        str: 保存内容的临时文件路径
    """
    uid = uuid.uuid4().hex
    savefile = str(svd.cache_dir() / uid) + ".c"
    with open(savefile, "w") as f:
        f.write(content)
    return savefile


def flawfinder(code: str):
    """在代码字符串上运行flawfinder工具检测安全漏洞。
    
    参数:
        code (str): 要分析的C/C++代码字符串
    
    返回:
        list: 包含检测到的漏洞记录的列表，每个记录包含sast工具名称、行号和漏洞信息
    """
    savefile = file_helper(code)
    opts = "--dataonly --quiet --singleline"
    cmd = f"flawfinder {opts} {savefile}"
    ret = svd.subprocess_cmd(cmd)[0].decode().splitlines()
    os.remove(savefile)
    records = []
    for i in ret:
        item = {"sast": "flawfinder"}
        splits = i.split(":", 2)
        item["line"] = int(splits[1])
        item["message"] = splits[2]
        records.append(item)
    return records


def rats(code: str):
    """在代码字符串上运行RATS（Rough Auditing Tool for Security）工具检测安全漏洞。
    
    参数:
        code (str): 要分析的代码字符串
    
    返回:
        list: 包含检测到的漏洞记录的列表，每个记录包含sast工具名称、行号、严重性和漏洞信息
    """
    savefile = file_helper(code)
    cmd = f"rats --resultsonly --xml {savefile}"
    ret = svd.subprocess_cmd(cmd)[0].decode()
    os.remove(savefile)
    records = []
    tree = cElementTree.ElementTree(cElementTree.fromstring(ret))
    for i in tree.findall("./vulnerability"):
        item = {"sast": "rats"}
        for v in i.iter():
            if v.tag == "line":
                item["line"] = int(v.text.strip())
            if v.tag == "severity":
                item["severity"] = v.text.strip()
            if v.tag == "message":
                item["message"] = v.text.strip()
        records.append(item)
    return records


def cppcheck(code: str):
    """在代码字符串上运行CppCheck工具检测C/C++代码中的错误和漏洞。
    
    参数:
        code (str): 要分析的C/C++代码字符串
    
    返回:
        list: 包含检测到的错误和漏洞记录的列表，每个记录包含sast工具名称、行号、严重性、ID和详细信息
    """
    savefile = file_helper(code)
    cmd = (
        f"cppcheck --enable=all --inconclusive --library=posix --force --xml {savefile}"
    )
    ret = svd.subprocess_cmd(cmd)[1].decode()
    os.remove(savefile)
    records = []
    tree = cElementTree.ElementTree(cElementTree.fromstring(ret))
    for i in tree.iter("error"):
        item = {"sast": "cppcheck"}
        vul_attribs = i.attrib
        loc_attribs = list(i.iter("location"))
        if len(loc_attribs) == 0:
            continue
        loc_attribs = loc_attribs[0].attrib
        item["line"] = loc_attribs["line"]
        item["message"] = vul_attribs["msg"]
        item["severity"] = vul_attribs["severity"]
        item["id"] = vul_attribs["id"]
        records.append(item)
    return records


def run_sast(code: str, verbose: int = 0):
    """在代码字符串上运行所有SAST工具并返回合并的检测结果。
    
    参数:
        code (str): 要分析的代码字符串
        verbose (int): 输出详细程度，默认值为0
    
    返回:
        list: 所有SAST工具检测到的漏洞记录的合并列表
    """
    rflaw = flawfinder(code)
    rrats = rats(code)
    rcpp = cppcheck(code)
    if verbose > 0:
        svd.debug(
            f"FlawFinder: {len(rflaw)} | RATS: {len(rrats)} | CppCheck: {len(rcpp)}"
        )
    return rflaw + rrats + rcpp


def get_sast_lines(sast_pkl_path):
    """从SAST工具的输出文件中提取有问题的代码行号。
    
    参数:
        sast_pkl_path: SAST工具输出的pickle文件路径
    
    返回:
        dict: 包含不同SAST工具检测到的问题行号集合的字典
              格式: {"cppcheck": set(), "rats": set(), "flawfinder": set()}
    """
    ret = dict()
    ret["cppcheck"] = set()
    ret["rats"] = set()
    ret["flawfinder"] = set()

    try:
        with open(sast_pkl_path, "rb") as f:
            sast_data = pkl.load(f)
        for i in sast_data:
            if i["sast"] == "cppcheck":
                if i["severity"] == "error" and i["id"] != "syntaxError":
                    ret["cppcheck"].add(i["line"])
            elif i["sast"] == "flawfinder":
                if "CWE" in i["message"]:
                    ret["flawfinder"].add(i["line"])
            elif i["sast"] == "rats":
                ret["rats"].add(i["line"])
    except Exception as E:
        print(E)
        pass
    return ret
