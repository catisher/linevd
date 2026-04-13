from importlib import reload

import sastvd.helpers.dclass as svddc
import sastvd.helpers.joern as svdj
import sastvd.linevd as lvd
from graphviz import Digraph

# 重新加载joern模块，确保使用最新版本
reload(svdj)


def get_digraph(nodes, edges, edge_label=True):
    """根据节点和边列表绘制有向图。
    """
    # 创建有向图，使用neato引擎进行布局
    dot = Digraph(comment="Combined PDG", engine="neato")

    # 为每个节点添加行号信息
    nodes = [n + [svdj.nodelabel2line(n[1])] for n in nodes]
    # 初始化颜色映射表
    colormap = {"": "white"}
    for n in nodes:
        # 为每种节点类型分配随机颜色
        if n[2] not in colormap:
            colormap[n[2]] = svdj.randcolor()

    # 添加节点到图中
    for n in nodes:
        # 设置节点样式：圆形、固定大小
        style = {"shape": "circle", "fixedsize": "true", "width": "0.5"}
        dot.node(str(n[0]), str(n[1]), **style)
    
    # 添加边到图中
    for e in edges:
        # 默认边样式
        style = {"color": "black"}
        # 根据边类型设置不同样式
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
            style["color"] = "black"
        elif e[2] == "REACHING_DEF":
            style["style"] = "dashed"
            style["color"] = "black"
        elif "DDG" in e[2]:
            style["style"] = "dashed"
            style["color"] = "red"
            # style["dir"] = "back"
        else:
            style["style"] = "solid"
            style["color"] = "black"
        style["penwidth"] = "1"
        # 根据edge_label参数决定是否显示边标签
        if edge_label:
            dot.edge(str(e[0]), str(e[1]), e[2], **style)
        else:
            dot.edge(str(e[0]), str(e[1]), **style)
    return dot


# 获取样本182352的文件路径
_id = svddc.BigVulDataset.itempath(182352)

# 行号映射表：将原始行号映射到新的行号
lineMap = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    12: 11,
    13: 12,
    14: 13,
    16: 14,
    17: 15,
    21: 18,
    22: 19,
    25: 21,
    26: 22,
    29: 25,
    31: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
}


# 获取代码属性图（CPG）的节点和边
n, e = svdj.get_node_edges(_id)
# 应用行号映射
n.lineNumber = n.lineNumber.map(lineMap).fillna("")
e.line_in = e.line_in.map(lineMap).fillna("")
e.line_out = e.line_out.map(lineMap).fillna("")

# 交换边的输入输出节点（用于调整边的方向）
e["tmp1"] = e.line_in
e["tmp2"] = e.line_out
e.line_out = e.tmp1
e.line_in = e.tmp2

# 对节点进行分组（合并相同行号的节点）
n, e = lvd.ne_groupnodes(n, e)

# 反转方法声明相关的DDG边
alt_e = e[(e.line_out == 1) & (e.dataflow != "")].copy()
alt_e.outnode = alt_e.innode
alt_e.innode = 1
e = e[e.line_out != 1]
# 使用pd.concat替代已弃用的append方法
import pandas as pd
e = pd.concat([e, alt_e], ignore_index=True)

# 绘制图
# 设置节点标签为行号
n["node_label"] = n["lineNumber"].astype(str)
# 移除自环边
e = e[e.innode != e.outnode]

# 为DDG边添加数据流信息
e.etype = e.apply(
    lambda x: f"DDG: {x.dataflow}" if len(x.dataflow) > 0 else x.etype, axis=1
)
# 查看特定数据流的边
e[e.dataflow == "!sig_none"]

# 过滤边：只保留特定类型的边
en = e[e.etype != "CFG"]
en = en[en.etype != "AST"]
en = en[en.etype != "REACHING_DEF"]
en = en[en.etype != "DDG: <RET>"]
en = en[en.etype != "DDG: !sig_none"]
en = en[en.etype != "DDG: now = timespec64_to_ktime(ts64)"]
# 合并节点信息到边
en = en.merge(n[["id", "name", "code"]], left_on="line_in", right_on="id")
# en = en[~((en.etype.str.contains("=")) & (~en.name.str.contains("assignment")))]

# 只保留赋值操作的DDG边
en.name = en.name.fillna("<operator>.assignment")
en = en[en.name == "<operator>.assignment"]
en.dataflow = en.dataflow.fillna("")
# 提取赋值操作的左侧变量
en["left_assign"] = en.code.apply(lambda x: x.split("=")[0].strip())
# 过滤：只保留数据流中包含左侧变量的边
en = en[en.apply(lambda x: x.left_assign in x.dataflow, axis=1)]

# 添加CDG边回图中
en = en[(en.etype.str.contains("DDG"))]
en = en.append(e[e.etype == "CDG"])

# Add other edges
en = en.append({"innode": 3, "outnode": 18, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 3, "outnode": 22, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 3, "outnode": 25, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 4, "outnode": 9, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 4, "outnode": 19, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 4, "outnode": 25, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 5, "outnode": 18, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 5, "outnode": 19, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 6, "outnode": 8, "etype": "DDG"}, ignore_index=1)

# Reverse edges nack
en["tmp"] = en.innode
en["innode"] = en.outnode
en["outnode"] = en.tmp

# 移除孤立节点
n = svdj.drop_lone_nodes(n, en)
# 手动添加节点4
n = pd.concat([n, pd.DataFrame([{"id": 4, "node_label": "4"}])], ignore_index=True)

# 生成最终的有向图
dot = get_digraph(
    n[["id", "node_label"]].to_numpy().tolist(),
    en[["outnode", "innode", "etype"]].to_numpy().tolist(),
    edge_label=False,
)
dot.render("/tmp/tmp.gv", view=True)