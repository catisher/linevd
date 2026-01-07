import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

mpl.rcParams["figure.dpi"] = 300

"""
图形可视化工具模块

该模块提供了用于绘制和可视化图结构的功能，支持NetworkX和DGL图格式。

主要功能：
1. 绘制简单的NetworkX有向图
2. 绘制简单的DGL图

主要使用的库：
- matplotlib: 用于图形绘制
- networkx: 用于NetworkX图操作
- nx_pydot: 用于使用Graphviz布局绘制图
"""


def simple_nx_plot(outnodes, innodes, node_labels):
    """绘制简单的NetworkX有向图，用于调试目的
    
    Args:
        outnodes (list): 出边节点列表，与innodes一一对应
        innodes (list): 入边节点列表，与outnodes一一对应
        node_labels (list): 节点标签列表，按节点索引顺序排列
    
    实现细节：
    1. 创建节点标签字典，映射节点索引到标签
    2. 根据出边和入边节点列表创建有向图
    3. 添加所有节点（包括孤立节点）
    4. 设置绘图参数，包括字体大小、节点大小、颜色等
    5. 使用Graphviz的dot布局算法定位节点
    6. 绘制图形并显示
    """
    labels = dict([(i, j) for i, j in enumerate(node_labels)])
    G = nx.DiGraph(
        list(zip(outnodes, innodes)),
    )
    G.add_nodes_from(range(len(node_labels)))

    options = {
        "font_size": 6,
        "node_size": 300,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 0.5,
        "width": 0.5,
        "labels": labels,
        "with_labels": True,
    }
    pos = graphviz_layout(G, prog="dot")
    nx.draw_networkx(G, pos, **options)
    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.10)
    plt.axis("off")
    plt.show()


def simple_dgl_plot(dglgraph):
    """绘制简单的DGL图
    
    Args:
        dglgraph (dgl.DGLGraph): 要绘制的DGL图对象
    
    实现细节：
    1. 将DGL图转换为NetworkX图
    2. 设置绘图参数，包括字体大小、节点大小、颜色等
    3. 使用Graphviz的dot布局算法定位节点
    4. 绘制图形并显示
    """
    G = dglgraph.to_networkx()
    options = {
        "font_size": 6,
        "node_size": 300,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 0.5,
        "width": 0.5,
    }
    pos = graphviz_layout(G, prog="dot")
    nx.draw_networkx(G, pos, **options)
    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.10)
    plt.axis("off")
    plt.show()
