import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import sastvd as svd
import scipy.sparse as sparse
from graphviz import Digraph


def nodelabel2line(label: str):
    """从节点标签中提取行号。

    示例：
    s = "METHOD_1.0: static long main()..."
    nodelabel2line(s)
    >>> '1.0'
    """
    try:
        return str(int(label))
    except:
        return label.split(":")[0].split("_")[-1]


def randcolor():
    """生成随机颜色。"""

    def r():
        return random.randint(0, 255)

    return "#%02X%02X%02X" % (r(), r(), r())


def get_digraph(nodes, edges, edge_label=True):
    """根据节点和边列表绘制有向图。"""
    dot = Digraph(comment="Combined PDG")

    nodes = [n + [nodelabel2line(n[1])] for n in nodes]
    colormap = {"": "white"}
    for n in nodes:
        if n[2] not in colormap:
            colormap[n[2]] = randcolor()

    for n in nodes:
        style = {"style": "filled", "fillcolor": colormap[n[2]]}
        dot.node(str(n[0]), str(n[1]), **style)
    for e in edges:
        style = {"color": "black"}
        if e[2] == "CALL":
            style["style"] = "solid"
            style["color"] = "purple"
        elif e[2] == "AST":
            style["style"] = "solid"
            style["color"] = "black"
        elif e[2] == "CFG":
            style["style"] = "solid"
            style["color"] = "red"
        elif e[2] == "CDG":
            style["style"] = "solid"
            style["color"] = "blue"
        elif e[2] == "REACHING_DEF":
            style["style"] = "solid"
            style["color"] = "orange"
        elif "DDG" in e[2]:
            style["style"] = "dashed"
            style["color"] = "darkgreen"
        else:
            style["style"] = "solid"
            style["color"] = "black"
        style["penwidth"] = "1"
        if edge_label:
            dot.edge(str(e[0]), str(e[1]), e[2], **style)
        else:
            dot.edge(str(e[0]), str(e[1]), **style)
    return dot


def run_joern(filepath: str, verbose: int):
    """使用最新版本的Joern提取代码属性图(CPG)。

    该函数调用Joern工具分析指定的代码文件，生成代码属性图(CPG)的JSON输出。
    Joern是一个静态分析框架，能够提取代码的AST、CFG、PDG等多种图结构。

    参数
    ----------
    filepath: str
        要分析的代码文件路径（如："storage/processed/bigvul/before/53.c"）
    verbose: int
        详细程度级别，控制调试信息的输出
        - 0: 静默模式
        - 1-2: 基本信息
        - 3+: 输出执行的命令
        - 5+: 输出清理工作空间的错误信息

    执行流程
    --------
    1. 构建Joern Scala脚本路径
    2. 构建目标代码文件的完整路径
    3. 构造Joern命令行命令
    4. 执行Joern分析（调用subprocess_cmd）
    5. 清理Joern工作空间中的临时文件

    输出文件
    --------
    在代码文件所在目录生成：
    - {filename}.edges.json: 边信息（数据流、控制流等）
    - {filename}.nodes.json: 节点信息（代码元素、行号等）

    示例
    ------
    >>> run_joern("/path/to/code.c", verbose=3)
    # 生成：/path/to/code.c.edges.json
    # 生成：/path/to/code.c.nodes.json
    """
    # 构建Joern Scala脚本的路径
    # 该脚本定义了如何遍历CPG并提取节点和边信息
    script_file = svd.external_dir() / "get_func_graph.scala"

    # 构建目标代码文件的完整路径
    # 将相对路径转换为绝对路径，基于external_dir
    filename = svd.external_dir() / filepath

    # 构造传递给Scala脚本的参数（文件名）
    params = f"filename={filename}"

    # 构造Joern命令行命令
    # --script: 指定要执行的Scala脚本
    # --params: 传递给脚本的参数
    command = f"joern --script {script_file} --params='{params}'"

    # 构建完整的命令路径（joern-cli目录下的joern可执行文件）
    # 神经，作者藏的坑还挺多

    #command = str(svd.external_dir() / "joern-cli" / command)
    command = str(Path.home() / "joern" / "joern-cli" / command)
   
    # 如果verbose大于2，打印执行的命令（用于调试）
    if verbose > 2:
        svd.debug(command)

    # 执行Joern命令（调用子进程）
    # 这会生成.edges.json和.nodes.json文件
    svd.subprocess_cmd(command, verbose=verbose)

    # 清理Joern工作空间中的临时文件
    # Joern会在workspace目录下创建项目文件，分析完成后可以删除
    try:
        shutil.rmtree(svd.external_dir() / "joern-cli" / "workspace" / filename.name)
    except Exception as E:
        # 如果verbose大于4，打印清理过程中的错误
        # 清理失败不影响主要功能，所以使用pass忽略异常
        if verbose > 4:
            print(E)
        pass


def get_node_edges(filepath: str, verbose=0):
    """根据文件路径获取节点和边信息（必须在 run_joern 之后运行）。

    filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/53.c"
    
    参数
    ----------
    filepath: str
        代码文件的路径
    verbose: int, 可选
        详细程度级别（默认为 0）
    
    返回
    -------
    tuple 或 None
        - nodes: pandas.DataFrame，包含节点信息
        - edges: pandas.DataFrame，包含边信息
        如果处理失败，返回 None
    """
    # 获取文件所在目录
    outdir = Path(filepath).parent
    # 获取文件名
    outfile = outdir / Path(filepath).name

    # 读取边信息 JSON 文件
    with open(str(outfile) + ".edges.json", "r") as f:
        # 加载 JSON 数据
        edges = json.load(f)
        # 转换为 DataFrame，指定列名
        edges = pd.DataFrame(edges, columns=["innode", "outnode", "etype", "dataflow"])
        # 用空字符串填充 NaN 值
        edges = edges.fillna("")

    # 读取节点信息 JSON 文件
    with open(str(outfile) + ".nodes.json", "r") as f:
        # 加载 JSON 数据
        nodes = json.load(f)
        # 从记录创建 DataFrame
        nodes = pd.DataFrame.from_records(nodes)
        # 如果没有 controlStructureType 列，添加空列
        if "controlStructureType" not in nodes.columns:
            nodes["controlStructureType"] = ""
        # 用空字符串填充 NaN 值
        nodes = nodes.fillna("")
        try:
            # 选择需要的列
            nodes = nodes[
                ["id", "_label", "name", "code", "lineNumber", "controlStructureType"]
            ]
        except Exception as E:
            # 如果选择列失败，根据 verbose 级别输出调试信息
            if verbose > 1:
                svd.debug(f"Failed {filepath}: {E}")
            # 返回 None 表示处理失败
            return None

    # 为局部变量分配行号
    # 读取源代码文件
    with open(filepath, "r") as f:
        # 读取所有行
        code = f.readlines()
    # 调用辅助函数为节点分配行号
    lmap = assign_line_num_to_local(nodes, edges, code)
    # 根据映射更新行号，如果节点 ID 在映射中则使用映射值，否则保持原值
    nodes.lineNumber = nodes.apply(
        lambda x: lmap[x.id] if x.id in lmap else x.lineNumber, axis=1
    )
    # 用空字符串填充 NaN 值
    nodes = nodes.fillna("")

    # 如果代码为空则分配节点名称
    # 如果 code 是"<empty>"则替换为空字符串
    nodes.code = nodes.apply(lambda x: "" if x.code == "<empty>" else x.code, axis=1)
    # 如果 code 为空则使用 name 填充
    nodes.code = nodes.apply(lambda x: x.code if x.code != "" else x["name"], axis=1)

    # 为图显示分配节点标签
    # 构建节点标签：标签_行号：代码
    nodes["node_label"] = (
        nodes._label + "_" + nodes.lineNumber.astype(str) + ": " + nodes.code
    )

    # 按节点类型过滤
    # 移除 COMMENT 类型的节点
    nodes = nodes[nodes._label != "COMMENT"]
    # 移除 FILE 类型的节点
    nodes = nodes[nodes._label != "FILE"]

    # 按边类型过滤
    # 移除 CONTAINS 类型的边
    edges = edges[edges.etype != "CONTAINS"]
    # 移除 SOURCE_FILE 类型的边
    edges = edges[edges.etype != "SOURCE_FILE"]
    # 移除 DOMINATE 类型的边
    edges = edges[edges.etype != "DOMINATE"]
    # 移除 POST_DOMINATE 类型的边
    edges = edges[edges.etype != "POST_DOMINATE"]

    # 移除未连接到行号节点的节点（可能效率不高）
    # 合并 outnode 的行号信息
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_out"}),
        left_on="outnode",
        right_on="id",
    )
    # 合并 innode 的行号信息
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_in"}),
        left_on="innode",
        right_on="id",
    )
    # 保留至少一端有行号的边
    edges = edges[(edges.line_out != "") | (edges.line_in != "")]

    # 唯一化类型
    # 如果 outnode 没有行号，则创建唯一标识符
    edges.outnode = edges.apply(
        lambda x: f"{x.outnode}_{x.innode}" if x.line_out == "" else x.outnode, axis=1
    )
    # 创建节点 ID 到名称的映射
    typemap = nodes[["id", "name"]].set_index("id").to_dict()["name"]

    # 创建节点 ID 到行号的映射
    linemap = nodes.set_index("id").to_dict()["lineNumber"]
    # 遍历每条边
    for e in edges.itertuples():
        # 如果 outnode 是字符串（表示是创建的唯一标识符）
        if type(e.outnode) == str:
            # 获取 innode 的行号
            lineNum = linemap[e.innode]
            # 构建节点标签
            node_label = f"TYPE_{lineNum}: {typemap[int(e.outnode.split('_')[0])]}"
            # 添加新节点到 nodes DataFrame
            ## 这里显示append 是 pandas已弃用的方法
            ## to be modified
            # nodes = nodes.append(
            #     {"id": e.outnode, "node_label": node_label, "lineNumber": lineNum},
            #     ignore_index=True,
            # )
              # ========== 核心修改：替换 append 为 pd.concat ==========
            # 1. 构建新行的 DataFrame（必须用列表包裹字典，否则会按列解析）
            new_node = pd.DataFrame([{
                "id": e.outnode, 
                "node_label": node_label, 
                "lineNumber": lineNum
            }])
            # 2. 合并原 nodes 和新行，重置索引（对应 ignore_index=True）
            nodes = pd.concat([nodes, new_node], ignore_index=True)
            # ======================================================

    # 返回节点和边
    return nodes, edges


def plot_node_edges(filepath: str, lineNumber: int = -1, filter_edges=[]):
    """根据文件路径绘制节点和边（必须在get_node_edges之后运行）。

    TO BE DEPRECATED.（即将被废弃）
    """
    nodes, edges = get_node_edges(filepath)

    if len(filter_edges) > 0:
        edges = edges[edges.etype.isin(filter_edges)]

    # Draw graph
    if lineNumber > 0:
        nodesforline = set(nodes[nodes.lineNumber == lineNumber].id.tolist())
    else:
        nodesforline = set(nodes.id.tolist())

    edges_new = edges[
        (edges.outnode.isin(nodesforline)) | (edges.innode.isin(nodesforline))
    ]
    nodes_new = nodes[
        nodes.id.isin(set(edges_new.outnode.tolist() + edges_new.innode.tolist()))
    ]
    dot = get_digraph(
        nodes_new[["id", "node_label"]].to_numpy().tolist(),
        edges_new[["outnode", "innode", "etype"]].to_numpy().tolist(),
    )
    dot.render("/tmp/tmp.gv", view=True)


def full_run_joern(filepath: str, verbose=0):
    """运行完整的Joern提取流程并保存输出。"""
    try:
        run_joern(filepath, verbose)
        nodes, edges = get_node_edges(filepath)
        return {"nodes": nodes, "edges": edges}
    except Exception as E:
        if verbose > 0:
            svd.debug(f"Failed {filepath}: {E}")
        return None


def full_run_joern_from_string(code: str, dataset: str, iid: str, verbose=0):
    """从字符串而不是文件运行完整的Joern提取流程。"""
    savedir = svd.get_dir(svd.interim_dir() / dataset)
    savepath = savedir / f"{iid}.c"
    with open(savepath, "w") as f:
        f.write(code)
    return full_run_joern(savepath, verbose)


def neighbour_nodes(nodes, edges, nodeids: list, hop: int = 1, intermediate=True):
    """给定节点、边和节点ID，返回指定跳数的邻居节点。

    nodes = pd.DataFrame()

    """
    nodes_new = (
        nodes.reset_index(drop=True).reset_index().rename(columns={"index": "adj"})
    )
    id2adj = pd.Series(nodes_new.adj.values, index=nodes_new.id).to_dict()
    adj2id = {v: k for k, v in id2adj.items()}

    arr = []
    for e in zip(edges.innode.map(id2adj), edges.outnode.map(id2adj)):
        arr.append([e[0], e[1]])
        arr.append([e[1], e[0]])

    arr = np.array(arr)
    shape = tuple(arr.max(axis=0)[:2] + 1)
    coo = sparse.coo_matrix((np.ones(len(arr)), (arr[:, 0], arr[:, 1])), shape=shape)

    def nodeid_neighbours_from_csr(nodeid):
        return [
            adj2id[i]
            for i in csr[
                id2adj[nodeid],
            ]
            .toarray()[0]
            .nonzero()[0]
        ]

    neighbours = defaultdict(list)
    if intermediate:
        for h in range(1, hop + 1):
            csr = coo.tocsr()
            csr **= h
            for nodeid in nodeids:
                neighbours[nodeid] += nodeid_neighbours_from_csr(nodeid)
        return neighbours
    else:
        csr = coo.tocsr()
        csr **= hop
        for nodeid in nodeids:
            neighbours[nodeid] += nodeid_neighbours_from_csr(nodeid)
        return neighbours


def rdg(edges, gtype):
    """根据图类型过滤和简化图。"""
    if gtype == "reftype":
        return edges[(edges.etype == "EVAL_TYPE") | (edges.etype == "REF")]
    if gtype == "ast":
        return edges[(edges.etype == "AST")]
    if gtype == "pdg":
        return edges[(edges.etype == "REACHING_DEF") | (edges.etype == "CDG")]
    if gtype == "cfgcdg":
        return edges[(edges.etype == "CFG") | (edges.etype == "CDG")]
    if gtype == "all":
        return edges[
            (edges.etype == "REACHING_DEF")
            | (edges.etype == "CDG")
            | (edges.etype == "AST")
            | (edges.etype == "EVAL_TYPE")
            | (edges.etype == "REF")
        ]


def assign_line_num_to_local(nodes, edges, code):
    """为代码属性图(CPG)中的局部变量分配行号。"""
    label_nodes = nodes[nodes._label == "LOCAL"].id.tolist()
    onehop_labels = neighbour_nodes(nodes, rdg(edges, "ast"), label_nodes, 1, False)
    twohop_labels = neighbour_nodes(nodes, rdg(edges, "reftype"), label_nodes, 2, False)
    node_types = nodes[nodes._label == "TYPE"]
    id2name = pd.Series(node_types.name.values, index=node_types.id).to_dict()
    node_blocks = nodes[
        (nodes._label == "BLOCK") | (nodes._label == "CONTROL_STRUCTURE")
    ]
    blocknode2line = pd.Series(
        node_blocks.lineNumber.values, index=node_blocks.id
    ).to_dict()
    local_vars = dict()
    local_vars_block = dict()
    for k, v in twohop_labels.items():
        types = [i for i in v if i in id2name and i < 1000]
        if len(types) == 0:
            continue
        assert len(types) == 1, "Incorrect Type Assumption."
        block = onehop_labels[k]
        assert len(block) == 1, "Incorrect block Assumption."
        block = block[0]
        local_vars[k] = id2name[types[0]]
        local_vars_block[k] = blocknode2line[block]
    nodes["local_type"] = nodes.id.map(local_vars)
    nodes["local_block"] = nodes.id.map(local_vars_block)
    local_line_map = dict()
    for row in nodes.dropna().itertuples():
        localstr = "".join((row.local_type + row.name).split()) + ";"
        try:
            ln = ["".join(i.split()) for i in code][int(row.local_block) :].index(
                localstr
            )
            rel_ln = row.local_block + ln + 1
            local_line_map[row.id] = rel_ln
        except:
            continue
    return local_line_map


def drop_lone_nodes(nodes, edges):
    """移除没有边连接的孤立节点。

    参数:
        nodes (pd.DataFrame): 包含id和node_label列的节点数据框
        edges (pd.DataFrame): 包含outnode, innode和etype列的边数据框
    """
    nodes = nodes[(nodes.id.isin(edges.innode)) | (nodes.id.isin(edges.outnode))]
    return nodes


def plot_graph_node_edge_df(
    nodes, edges, nodeids=[], hop=1, drop_lone_nodes=True, edge_label=True
):
    """从节点和边的数据框绘制图形。

    参数:
        nodes (pd.DataFrame): 包含id和node_label列的节点数据框
        edges (pd.DataFrame): 包含outnode, innode和etype列的边数据框
        drop_lone_nodes (bool): 是否隐藏没有入边/出边的节点
        lineNumber (int): 围绕此节点绘制子图
    """
    # Drop lone nodes
    if drop_lone_nodes:
        nodes = nodes[(nodes.id.isin(edges.innode)) | (nodes.id.isin(edges.outnode))]

    # Get subgraph
    if len(nodeids) > 0:
        nodeids = nodes[nodes.lineNumber.isin(nodeids)].id
        keep_nodes = neighbour_nodes(nodes, edges, nodeids, hop)
        keep_nodes = set(list(nodeids) + [i for j in keep_nodes.values() for i in j])
        nodes = nodes[nodes.id.isin(keep_nodes)]
        edges = edges[
            (edges.innode.isin(keep_nodes)) & (edges.outnode.isin(keep_nodes))
        ]

    dot = get_digraph(
        nodes[["id", "node_label"]].to_numpy().tolist(),
        edges[["outnode", "innode", "etype"]].to_numpy().tolist(),
        edge_label=edge_label,
    )
    dot.render("/tmp/tmp.gv", view=True)
