"""树状LSTM（TreeLSTM）实现。

该文件是从DGL的Tree-LSTM实现修改而来，实现了用于处理AST等树形结构的LSTM模型。
包含Child-Sum TreeLSTM单元和自定义的N元TreeLSTM模型。
"""

import warnings

import dgl
import torch as th
import torch.nn as nn

# This warning also appears in official DGL Tree-LSTM docs, so ignore it.
warnings.filterwarnings("ignore", message="The input graph for the user-defined edge")


class ChildSumTreeLSTMCell(nn.Module):
    """Child-Sum树状LSTM单元。
    
    实现了Child-Sum TreeLSTM的核心计算单元，用于处理树形结构数据。
    该单元通过聚合子节点的信息来更新当前节点的隐藏状态。
    """

    def __init__(self, x_size, h_size):
        """初始化Child-Sum树状LSTM单元。
        
        参数
        ----------
        x_size : int
            输入特征的维度大小
        h_size : int
            隐藏状态的维度大小
        """
        super(ChildSumTreeLSTMCell, self).__init__()
        # 输入门、输出门和更新门的权重矩阵
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        # 隐藏状态的权重矩阵
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        # 偏置项
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        # 遗忘门的权重矩阵
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        """消息传递函数（Message UDF）。
        
        从子节点向父节点传递隐藏状态和细胞状态。
        
        参数
        ----------
        edges : dgl.EdgeBatch
            边批次数据
        
        返回
        -------
        dict
            包含子节点的隐藏状态和细胞状态
        """
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes):
        """聚合函数（Reduce UDF）。
        
        聚合所有子节点的信息，计算当前节点的输入。
        
        参数
        ----------
        nodes : dgl.NodeBatch
            节点批次数据
        
        返回
        -------
        dict
            包含聚合后的输入和细胞状态
        """
        # 聚合所有子节点的隐藏状态
        h_tild = th.sum(nodes.mailbox["h"], 1)
        # 计算遗忘门
        f = th.sigmoid(self.U_f(nodes.mailbox["h"]))
        # 更新细胞状态
        c = th.sum(f * nodes.mailbox["c"], 1)
        return {"iou": self.U_iou(h_tild), "c": c}

    def apply_node_func(self, nodes):
        """节点应用函数（Apply UDF）。
        
        更新节点的隐藏状态和细胞状态。
        
        参数
        ----------
        nodes : dgl.NodeBatch
            节点批次数据
        
        返回
        -------
        dict
            包含更新后的隐藏状态和细胞状态
        """
        # 计算输入门、输出门和更新门
        iou = nodes.data["iou"] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        # 更新细胞状态
        c = i * u + nodes.data["c"]
        # 计算隐藏状态
        h = o * th.tanh(c)
        return {"h": h, "c": c}


class TreeLSTM(nn.Module):
    """自定义N元树状LSTM模型。

    该模型用于处理树形结构数据，如抽象语法树（AST），能够捕获树的层次结构信息。

    示例：
    a = BigVulGraphDataset(sample=10)  # 创建数据集
    asts = a.item(180189)["asts"]  # 获取代码的AST列表
    batched_g = dgl.batch([i for i in asts if i])  # 批次处理AST图
    model = TreeLSTM(200, 200)  # 创建TreeLSTM模型
    model(batched_g)  # 前向传播
    """

    def __init__(
        self,
        x_size,
        h_size,
        dropout=0,
    ):
        """初始化TreeLSTM模型。

        参数
        ----------
        x_size : int
            输入特征的维度大小
        h_size : int
            隐藏状态的维度大小
        dropout : float, 可选
            Dropout概率，应用于最终层，默认为0
        """
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.dropout = nn.Dropout(dropout)
        self.cell = ChildSumTreeLSTMCell(x_size, h_size)
        self.h_size = h_size
        self.dev = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    def forward(self, g):
        """给定批次树结构，计算Tree-LSTM的预测结果。

        参数
        ----------
        g : dgl.DGLGraph
            用于计算的树结构
        
        返回
        -------
        dict
            包含每个树的根节点隐藏状态的字典，键为(树ID, 行号)，值为隐藏状态向量
        """
        # 输入嵌入
        embeds = g.ndata["_FEAT"].to(self.dev)
        n = g.number_of_nodes()
        # 计算输入门、输出门和更新门
        g.ndata["iou"] = self.cell.W_iou(self.dropout(embeds))
        # 初始化隐藏状态和细胞状态
        g.ndata["h"] = th.zeros((n, self.h_size)).to(self.dev)
        g.ndata["c"] = th.zeros((n, self.h_size)).to(self.dev)

        # 按拓扑顺序传播
        dgl.prop_nodes_topo(
            g,
            self.cell.message_func,
            self.cell.reduce_func,
            apply_node_func=self.cell.apply_node_func,
        )
        # 获取隐藏状态
        h = self.dropout(g.ndata.pop("h"))

        # 解批次并获取根节点（假设根节点索引为0）
        g.ndata["h"] = h
        unbatched = dgl.unbatch(g)
        return dict(
            [
                [
                    (i.ndata["_ID"][0].int().item(), i.ndata["_LINE"][0].int().item()),
                    i.ndata["h"][0],
                ]
                for i in unbatched
            ]
        )
