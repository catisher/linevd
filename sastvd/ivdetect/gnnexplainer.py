import math

import dgl
import dgl.function as fn
import matplotlib.pylab as plt
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class NodeExplainerModule(nn.Module):
    """
    一个PyTorch模块，用于基于计算图和节点特征解释节点的预测结果。
    使用两种掩码：一种用于边，另一种用于节点特征。
    由于DGL在边掩码操作上的限制，此解释器需要被解释的模型
    接受一个额外的输入参数（边掩码），并在其内部消息解析操作中应用此掩码。
    这是当前使用边掩码的解决方法。
    """

    # 类内部变量：损失组件的系数权重
    # g_size: 边掩码大小损失的系数
    # feat_size: 节点特征掩码大小损失的系数
    # g_ent: 边掩码熵损失的系数
    # feat_ent: 节点特征掩码熵损失的系数
    loss_coef = {"g_size": 0.05, "feat_size": 1.0, "g_ent": 0.1, "feat_ent": 0.1}

    def __init__(
        self,
        model,
        num_edges,
        node_feat_dim,
        activation="sigmoid",
        agg_fn="sum",
        mask_bias=False,
    ):
        """初始化节点解释器模块。

        参数:
            model: 要解释的GNN模型
            num_edges: 计算图中的边数
            node_feat_dim: 节点特征的维度
            activation: 掩码使用的激活函数（默认为"sigmoid"）
            agg_fn: 聚合函数（默认为"sum"）
            mask_bias: 是否为边掩码添加偏置（默认为False）
        """
        super(NodeExplainerModule, self).__init__()
        self.model = model
        self.model.eval()
        self.num_edges = num_edges
        self.node_feat_dim = node_feat_dim
        self.activation = activation
        self.agg_fn = agg_fn
        self.mask_bias = mask_bias

        # 初始化掩码参数
        self.edge_mask, self.edge_mask_bias = self.create_edge_mask(self.num_edges)
        self.node_feat_mask = self.create_node_feat_mask(self.node_feat_dim)

    def create_edge_mask(self, num_edges, init_strategy="normal", const=1.0):
        """
        根据计算图中的边数，创建可学习的边掩码。
        为了适配DGL，将此掩码从N*N邻接矩阵转换为边的数量

        参数
        ----------
        num_edges: 整数N，指定边的数量。
        init_strategy: 字符串，指定参数初始化方法
        const: 浮点数，常量初始化的值

        返回
        -------
        mask和mask_bias: 张量，形状均为N*1
        """
        mask = nn.Parameter(th.Tensor(num_edges, 1))

        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(1.0 / num_edges)
            with th.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "const":
            nn.init.constant_(mask, const)

        if self.mask_bias:
            mask_bias = nn.Parameter(th.Tensor(num_edges, 1))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def create_node_feat_mask(self, node_feat_dim, init_strategy="normal"):
        """
        根据计算图中节点特征的维度，创建可学习的特征掩码。

        参数
        ----------
        node_feat_dim: 整数N，节点特征的维度
        init_strategy: 字符串，指定参数初始化方法

        返回
        -------
        mask: 张量，形状为N
        """
        mask = nn.Parameter(th.Tensor(node_feat_dim))

        if init_strategy == "normal":
            std = 0.1
            with th.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with th.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

    def forward(self, graph, n_feats, dataset):
        """
        计算对给定模型的输入应用掩码后的预测结果。

        参数
        ----------
        graph: DGLGraph，应该是要解释的目标节点的子图。
        n_feats: 节点特征张量
        dataset: 数据集对象

        返回
        -------
        new_logits: 张量，形状为N * Num_Classes
        """

        # 步骤1: 使用内部特征掩码对节点特征进行掩码
        new_n_feats = n_feats * self.node_feat_mask.sigmoid()
        edge_mask = self.edge_mask.sigmoid()

        # 步骤2: 在对节点特征和边进行掩码后计算logits
        graph.ndata["_FEAT"] = new_n_feats
        new_logits = self.model(graph, dataset, edge_mask)

        return new_logits

    def _loss(self, pred_logits, pred_label):
        """
        计算此解释器的损失，在作者的代码中包括6个部分：
        1. 节点和边掩码前后预测logits之间的预测损失；
        2. 边掩码本身的损失，尝试将掩码值设为0或1；
        3. 节点特征掩码本身的损失，尝试将掩码值设为0或1；
        4. 边掩码权重的L2损失，但使用总和而不是平均值；
        5. 节点特征掩码权重的L2损失，在作者的代码中未使用；
        6. 邻接矩阵的拉普拉斯损失。
        在PyG实现中，有5种类型的损失：
        1. 节点和边掩码前后logits之间的预测损失；
        2. 边掩码权重的总和损失；
        3. 边掩码熵损失，尝试将掩码值设为0或1；
        4. 节点特征掩码权重的总和损失；
        5. 节点特征掩码熵损失，尝试将掩码值设为0或1；

        参数
        ----------
        pred_logits：张量，模型输出的N维logits
        pred_label: 张量，N维的one-hot标签

        返回
        -------
        loss: 标量，此解释器的总体损失。
        """
        # 1. 预测损失
        log_logit = -F.log_softmax(pred_logits, dim=-1)
        pred_loss = th.sum(log_logit * pred_label)

        # 2. 边掩码损失
        if self.activation == "sigmoid":
            edge_mask = th.sigmoid(self.edge_mask)
        elif self.activation == "relu":
            edge_mask = F.relu(self.edge_mask)
        else:
            raise ValueError()
        edge_mask_loss = self.loss_coef["g_size"] * th.sum(edge_mask)

        # 3. 边掩码熵损失
        edge_ent = -edge_mask * th.log(edge_mask + 1e-8) - (1 - edge_mask) * th.log(
            1 - edge_mask + 1e-8
        )
        edge_ent_loss = self.loss_coef["g_ent"] * th.mean(edge_ent)

        # 4. 节点特征掩码损失
        if self.activation == "sigmoid":
            node_feat_mask = th.sigmoid(self.node_feat_mask)
        elif self.activation == "relu":
            node_feat_mask = F.relu(self.node_feat_mask)
        else:
            raise ValueError()
        node_feat_mask_loss = self.loss_coef["feat_size"] * th.sum(node_feat_mask)

        # 5. 节点特征掩码熵损失
        node_feat_ent = -node_feat_mask * th.log(node_feat_mask + 1e-8) - (
            1 - node_feat_mask
        ) * th.log(1 - node_feat_mask + 1e-8)
        node_feat_ent_loss = self.loss_coef["feat_ent"] * th.mean(node_feat_ent)

        total_loss = (
            pred_loss
            + edge_mask_loss
            + edge_ent_loss
            + node_feat_mask_loss
            + node_feat_ent_loss
        )

        return total_loss


def visualize_sub_graph(
    sub_graph, edge_weights=None, origin_nodes=None, center_node=None
):
    """
    使用networkx可视化子图，如果提供了边权重，将使用不同的蓝色淡化效果设置边。

    参数
    ----------
    sub_graph: DGLGraph，要可视化的子图。
    edge_weights: 张量，与边数相同。值范围为(0,1)，默认为None
    origin_nodes: 列表，将用于在可视化中替换子图中节点ID的节点ID列表
    center_node: 张量，要在原始节点列表中用不同颜色突出显示的节点ID

    返回
    -------
    显示子图
    """
    # 提取原始索引并映射到新的networkx图
    # 转换为networkx图
    g = dgl.to_networkx(sub_graph)
    nx_edges = g.edges(data=True)

    if not (origin_nodes is None):
        n_mapping = {
            new_id: old_id for new_id, old_id in enumerate(origin_nodes.tolist())
        }
        g = nx.relabel_nodes(g, mapping=n_mapping)

    pos = nx.spring_layout(g)

    if edge_weights is None:
        options = {
            "node_size": 1000,
            "alpha": 0.9,
            "font_size": 24,
            "width": 4,
        }
    else:

        ec = [edge_weights[e[2]["id"]][0] for e in nx_edges]
        options = {
            "node_size": 1000,
            "alpha": 0.3,
            "font_size": 12,
            "edge_color": ec,
            "width": 4,
            "edge_cmap": plt.cm.Reds,
            "edge_vmin": 0,
            "edge_vmax": 1,
            "connectionstyle": "arc3,rad=0.1",
        }

    nx.draw(g, pos, with_labels=True, node_color="b", **options)
    if not (center_node is None):
        nx.draw(
            g, 
            pos,
            nodelist=center_node.tolist(),
            with_labels=True,
            node_color="r",
            **options
        )

    plt.show()


def gnnexplainer(model, graph, dataset):
    """运行GNNExplainer来获取按重要性排序的代码行。

    GNNExplainer是一种用于解释图神经网络预测的方法，
    通过训练可学习的边掩码和节点特征掩码来识别对预测最重要的子图结构和特征。

    参数
    ----------
    model: 要解释的GNN模型
    graph: DGLGraph，包含要解释的目标节点的子图
    dataset: 数据集对象，提供模型所需的额外信息

    返回
    -------
    list: 按重要性降序排列的代码行列表
    """
    dev = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    # 创建解释器
    explainer = NodeExplainerModule(
        model=model, num_edges=graph.number_of_edges(), node_feat_dim=200
    )
    explainer.to(dev)

    # 定义优化器
    optim = th.optim.Adam(explainer.parameters(), lr=0.01, weight_decay=0)

    # 为给定节点训练解释器
    model.eval()
    model_logits = model(graph, dataset)
    model_predict = F.one_hot(th.argmax(model_logits, dim=-1), 2)
    sub_feats = graph.ndata["_FEAT"]

    for epoch in tqdm(range(50)):
        explainer.train()
        exp_logits = explainer(graph, sub_feats, dataset)
        loss = explainer._loss(exp_logits, model_predict[0])

        optim.zero_grad()
        loss.backward()
        optim.step()

    # 可视化边的重要性
    edge_weights = explainer.edge_mask.sigmoid().detach()

    # 将权重重要性聚合到节点
    graph.ndata["line_importance"] = th.ones(graph.number_of_nodes(), device=dev) * 2
    graph.edata["edge_mask"] = edge_weights
    graph.update_all(
        fn.u_mul_e("line_importance", "edge_mask", "m"), fn.mean("m", "line_importance")
    )

    # 返回按重要性排序的行列表
    ret = sorted(
        list(
            zip(
                graph.ndata["line_importance"].squeeze().detach().cpu().numpy(),
                graph.ndata["_LINE"].detach().cpu().numpy(),
            )
        ),
        reverse=True,
    )

    return [i[1] for i in ret]
