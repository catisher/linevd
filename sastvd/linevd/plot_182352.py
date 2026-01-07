"""代码属性图(CPG)可视化脚本。

该脚本用于处理和可视化特定漏洞ID(182352)的代码属性图，
通过Graphviz生成有向图，帮助理解代码的控制流、数据流和语法结构。
"""
from importlib import reload

import sastvd.helpers.dclass as svddc
import sastvd.helpers.joern as svdj
import sastvd.linevd as lvd
from graphviz import Digraph

reload(svdj)


def get_digraph(nodes, edges, edge_label=True):
    """根据节点和边列表绘制有向图。
    
    参数:
        nodes: 节点列表，每个节点包含ID和标签
        edges: 边列表，每个边包含源节点、目标节点和边类型
        edge_label: 是否显示边标签，默认为True
        
    返回:
        dot: Graphviz的Digraph对象，包含绘制好的有向图
    """
    dot = Digraph(comment="Combined PDG", engine="neato")

    # 为每个节点添加行号信息
    nodes = [n + [svdj.nodelabel2line(n[1])] for n in nodes]
    # 初始化颜色映射，用于节点着色
    colormap = {"": "white"}
    for n in nodes:
        if n[2] not in colormap:
            colormap[n[2]] = svdj.randcolor()

    # 添加所有节点到图中
    for n in nodes:
        style = {"shape": "circle", "fixedsize": "true", "width": "0.5"}
        dot.node(str(n[0]), str(n[1]), **style)
    
    # 添加所有边到图中，并根据边类型设置不同的样式
    for e in edges:
        style = {"color": "black"}
        # 根据边类型设置不同的颜色和样式
        if e[2] == "CALL":  # 调用边
            style["style"] = "solid"
            style["color"] = "purple"
        elif e[2] == "AST":  # 抽象语法树边
            style["style"] = "solid"
            style["color"] = "black"
        elif e[2] == "CFG":  # 控制流图边
            style["style"] = "solid"
            style["color"] = "red"
        elif e[2] == "CDG":  # 控制依赖图边
            style["style"] = "solid"
            style["color"] = "black"
        elif e[2] == "REACHING_DEF":  # 可达定义边
            style["style"] = "dashed"
            style["color"] = "black"
        elif "DDG" in e[2]:  # 数据流图边
            style["style"] = "dashed"
            style["color"] = "red"
            # style["dir"] = "back"
        else:  # 其他类型边
            style["style"] = "solid"
            style["color"] = "black"
        
        style["penwidth"] = "1"
        if edge_label:
            dot.edge(str(e[0]), str(e[1]), e[2], **style)
        else:
            dot.edge(str(e[0]), str(e[1]), **style)
    
    return dot


# 获取特定漏洞ID(182352)的文件路径
_id = svddc.BigVulDataset.itempath(182352)


# 行号映射表，用于将原始行号映射到新的行号
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


# 获取代码属性图(CPG)的节点和边
n, e = svdj.get_node_edges(_id)
# 应用行号映射
n.lineNumber = n.lineNumber.map(lineMap).fillna("")
e.line_in = e.line_in.map(lineMap).fillna("")
e.line_out = e.line_out.map(lineMap).fillna("")

# 交换行号（将入边和出边的行号互换）
e["tmp1"] = e.line_in
e["tmp2"] = e.line_out
e.line_out = e.tmp1
e.line_in = e.tmp2

# 对节点进行分组
n, e = lvd.ne_groupnodes(n, e)

# 为方法声明反转数据流图(DDG)边
alt_e = e[(e.line_out == 1) & (e.dataflow != "")].copy()
alt_e.outnode = alt_e.innode
alt_e.innode = 1
e = e[e.line_out != 1]
e = e.append(alt_e)

# 绘制图的准备工作
n["node_label"] = n["lineNumber"].astype(str)
# 移除自环边（源节点和目标节点相同的边）
e = e[e.innode != e.outnode]

# 为数据流边添加具体的数据流动信息

e.etype = e.apply(
    lambda x: f"DDG: {x.dataflow}" if len(x.dataflow) > 0 else x.etype, axis=1
)
# 筛选出需要的边类型
en = e[e.etype != "CFG"]
en = en[en.etype != "AST"]
en = en[en.etype != "REACHING_DEF"]
en = en[en.etype != "DDG: <RET>"]
en = en[en.etype != "DDG: !sig_none"]
en = en[en.etype != "DDG: now = timespec64_to_ktime(ts64)"]
# 合并节点信息到边数据中
en = en.merge(n[["id", "name", "code"]], left_on="line_in", right_on="id")
# en = en[~((en.etype.str.contains("=")) & (~en.name.str.contains("assignment")))]

# 只保留赋值语句的数据流边
en.name = en.name.fillna("<operator>.assignment")
en = en[en.name == "<operator>.assignment"]
en.dataflow = en.dataflow.fillna("")
# 提取赋值语句左侧的变量名
en["left_assign"] = en.code.apply(lambda x: x.split("=")[0].strip())
# 只保留数据流中包含赋值左侧变量的边
en = en[en.apply(lambda x: x.left_assign in x.dataflow, axis=1)]

# 添加控制依赖图(CDG)边回来
en = en[(en.etype.str.contains("DDG"))]
en = en.append(e[e.etype == "CDG"])

# 添加其他必要的边
en = en.append({"innode": 3, "outnode": 18, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 3, "outnode": 22, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 3, "outnode": 25, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 4, "outnode": 9, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 4, "outnode": 19, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 4, "outnode": 25, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 5, "outnode": 18, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 5, "outnode": 19, "etype": "DDG"}, ignore_index=1)
en = en.append({"innode": 6, "outnode": 8, "etype": "DDG"}, ignore_index=1)

# 反转边的方向

en["tmp"] = en.innode
en["innode"] = en.outnode
en["outnode"] = en.tmp

# 删除孤立节点
n = svdj.drop_lone_nodes(n, en)
# 添加缺失的节点
n = n.append({"id": 4, "node_label": "4"}, ignore_index=1)

# 生成有向图

dot = get_digraph(
    n[["id", "node_label"]].to_numpy().tolist(),
    en[["outnode", "innode", "etype"]].to_numpy().tolist(),
    edge_label=False,
)

# 渲染并显示图形
dot.render("/tmp/tmp.gv", view=True)
