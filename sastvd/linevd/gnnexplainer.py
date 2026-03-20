"""图神经网络解释器模块。

该模块实现了GNNExplainer算法，用于解释图神经网络模型对节点预测的结果。
主要功能包括：
1. 使用边掩码和节点特征掩码来识别影响模型预测的关键边和特征
2. 基于PyTorch Lightning实现训练流程
3. 提供节点重要性分数计算功能

该实现针对DGL框架进行了优化，能够有效解释基于图的漏洞检测模型。
"""

import math

import dgl
import pytorch_lightning as pl
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class NodeExplainerModule(nn.Module):
    """
    基于计算图和节点特征解释节点预测的PyTorch模块。
    使用两种掩码：一种用于边，另一种用于节点特征。
    由于DGL对边掩码操作的限制，被解释的模型需要接受额外的边掩码参数
    并在其内部消息传递操作中应用此掩码。
    """

    # 类内部变量：损失函数系数
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
            model: 要解释的图神经网络模型
            num_edges: 图中边的数量
            node_feat_dim: 节点特征的维度
            activation: 激活函数类型，可选"sigmoid"或"relu"
            agg_fn: 聚合函数类型
            mask_bias: 是否使用边掩码偏置
        """
        super(NodeExplainerModule, self).__init__()
        # 保存要解释的模型
        self.model = model
        # 设置模型为评估模式，冻结模型参数
        self.model.eval()
        # 保存边的数量
        self.num_edges = num_edges
        # 保存节点特征维度
        self.node_feat_dim = node_feat_dim
        # 保存激活函数类型
        self.activation = activation
        # 保存聚合函数类型
        self.agg_fn = agg_fn
        # 保存是否使用掩码偏置
        self.mask_bias = mask_bias

        # 初始化边掩码参数
        self.edge_mask, self.edge_mask_bias = self.create_edge_mask(self.num_edges)
        # 初始化节点特征掩码参数
        self.node_feat_mask = self.create_node_feat_mask(self.node_feat_dim)

    def create_edge_mask(self, num_edges, init_strategy="normal", const=1.0):
        """创建边掩码参数。

        参数:
            num_edges: 边的数量
            init_strategy: 初始化策略，可选"normal"或"const"
            const: 如果使用"const"策略，设置的常数值

        返回:
            mask: 边掩码参数
            mask_bias: 边掩码偏置参数（如果启用）
        """
        # 创建可学习的边掩码参数
        mask = nn.Parameter(th.Tensor(num_edges, 1))

        # 根据初始化策略初始化掩码
        if init_strategy == "normal":
            # 使用正态分布初始化，均值为1.0
            std = nn.init.calculate_gain("relu") * math.sqrt(1.0 / num_edges)
            with th.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "const":
            # 使用常数值初始化
            nn.init.constant_(mask, const)

        # 如果启用了掩码偏置，创建偏置参数
        if self.mask_bias:
            mask_bias = nn.Parameter(th.Tensor(num_edges, 1))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def create_node_feat_mask(self, node_feat_dim, init_strategy="normal"):
        """创建节点特征掩码参数。

        参数:
            node_feat_dim: 节点特征的维度
            init_strategy: 初始化策略，可选"normal"或"constant"

        返回:
            mask: 节点特征掩码参数
        """
        # 创建可学习的节点特征掩码参数
        mask = nn.Parameter(th.Tensor(node_feat_dim))

        # 根据初始化策略初始化掩码
        if init_strategy == "normal":
            # 使用正态分布初始化，均值为1.0，标准差为0.1
            std = 0.1
            with th.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            # 使用常数值初始化
            with th.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

    def forward(self, graph, n_feats):
        """
        在对输入进行掩码处理后计算模型的预测结果。
        
        参数:
            graph: DGL图，应为要解释的目标节点的子图
            n_feats: 节点特征
            
        返回:
            new_logits: 形状为N * Num_Classes的张量
        """

        # 步骤1: 使用内部特征掩码掩码节点特征
        # 通过sigmoid激活函数将掩码值限制在[0,1]范围内
        new_n_feats = n_feats * self.node_feat_mask.sigmoid()
        # 计算边掩码的激活值
        edge_mask = self.edge_mask.sigmoid()

        # 步骤2: 计算掩码节点特征和边后的logits
        # 将掩码后的特征存储到图的节点数据中
        graph.ndata["_maskedfeat"] = new_n_feats
        # 调用模型进行预测，传入边权重和特征覆盖参数
        new_logits = self.model(
            graph, test=True, e_weights=edge_mask, feat_override="_maskedfeat"
        )[0]

        return new_logits

    def _loss(self, pred_logits, pred_label):
        """
        计算解释器的损失，包括以下几个部分：
        1. 掩码前后预测logits之间的预测损失；
        2. 边掩码自身的损失，试图将掩码值推向0或1；
        3. 节点特征掩码自身的损失，试图将掩码值推向0或1；
        4. 边掩码权重的L2损失（求和而不是平均）；
        5. 节点特征掩码的熵损失，试图将掩码值推向0或1。
        
        参数:
            pred_logits: 模型输出的N维logits张量
            pred_label: N维的one-hot标签张量
            
        返回:
            loss: 标量，解释器的总体损失
        """
        # 1. 预测损失：确保掩码后的预测结果与原始预测一致
        log_logit = -F.log_softmax(pred_logits, dim=-1)
        pred_loss = th.sum(log_logit * pred_label)

        # 2. 边掩码损失：鼓励边掩码值趋向于0（移除边）
        if self.activation == "sigmoid":
            edge_mask = th.sigmoid(self.edge_mask)
        elif self.activation == "relu":
            edge_mask = F.relu(self.edge_mask)
        else:
            raise ValueError()
        # 使用系数g_size控制边掩码损失的权重
        edge_mask_loss = self.loss_coef["g_size"] * th.sum(edge_mask)

        # 3. 边掩码熵损失：鼓励边掩码值趋向于0或1（二元化）
        edge_ent = -edge_mask * th.log(edge_mask + 1e-8) - (1 - edge_mask) * th.log(
            1 - edge_mask + 1e-8
        )
        # 使用系数g_ent控制边掩码熵损失的权重
        edge_ent_loss = self.loss_coef["g_ent"] * th.mean(edge_ent)

        # 4. 节点特征掩码损失：鼓励节点特征掩码值趋向于0（移除特征）
        if self.activation == "sigmoid":
            node_feat_mask = th.sigmoid(self.node_feat_mask)
        elif self.activation == "relu":
            node_feat_mask = F.relu(self.node_feat_mask)
        else:
            raise ValueError()
        # 使用系数feat_size控制节点特征掩码损失的权重
        node_feat_mask_loss = self.loss_coef["feat_size"] * th.sum(node_feat_mask)

        # 5. 节点特征掩码熵损失：鼓励节点特征掩码值趋向于0或1（二元化）
        node_feat_ent = -node_feat_mask * th.log(node_feat_mask + 1e-8) - (
            1 - node_feat_mask
        ) * th.log(1 - node_feat_mask + 1e-8)
        # 使用系数feat_ent控制节点特征掩码熵损失的权重
        node_feat_ent_loss = self.loss_coef["feat_ent"] * th.mean(node_feat_ent)

        # 总损失：所有损失项的加权和
        total_loss = (
            pred_loss
            + edge_mask_loss
            + edge_ent_loss
            + node_feat_mask_loss
            + node_feat_ent_loss
        )

        return total_loss


class GNNExplainerLit(pl.LightningModule):
    """GNNExplainer的PyTorch Lightning训练器类。"""

    def __init__(self, model, g):
        """初始化训练器。
        
        参数:
            model: 要解释的图神经网络模型
            g: 要解释的DGL图
        """
        super().__init__()
        # 冻结模型参数，只训练解释器
        for param in model.parameters():
            param.requires_grad = False
        # 初始化节点解释器
        self.explainer = NodeExplainerModule(
            model=model, num_edges=g.number_of_edges(), node_feat_dim=768
        )
        # 获取模型原始预测结果
        self.model_logits = model(g, test=True)[0]
        # 将logits转换为one-hot标签
        self.model_predict = F.one_hot(th.argmax(self.model_logits, dim=-1), 2)
        # 提取节点的CodeBERT特征
        self.sub_feats = g.ndata["_CODEBERT"]
        # 保存要解释的图
        self.g = g

    def forward(self, g):
        """前向传播。
        
        参数:
            g: DGL图
            
        返回:
            exp_logits: 经过解释器处理后的预测logits
        """
        exp_logits = self.explainer(g, self.sub_feats)
        return exp_logits

    def training_step(self, batch, batch_idx):
        """训练步骤。
        
        参数:
            batch: 批次数据
            batch_idx: 批次索引
            
        返回:
            loss: 训练损失
        """
        # 执行前向传播
        exp_logits = self(batch)[0]
        # 计算损失
        loss = self.explainer._loss(exp_logits, self.model_predict)
        # 记录训练损失
        self.log("train_loss", loss, on_epoch=True, prog_bar=False)
        return loss

    def train_dataloader(self):
        """训练数据加载器。
        
        返回:
            DGL图数据加载器，仅包含单个图
        """
        return dgl.dataloading.GraphDataLoader([self.g])

    def configure_optimizers(self):
        """配置优化器。
        
        返回:
            AdamW优化器
        """
        return th.optim.AdamW(self.parameters(), lr=0.01, weight_decay=0)


def get_node_importances(model, g):
    """基于GNNExplainer为DGL图分配节点重要性分数。
    
    参数:
        model: 图神经网络模型
        g: DGL图
        
    返回:
        节点重要性分数张量
    """
    # 初始化GNNExplainer训练器，将模型和图移动到GPU
    gnne = GNNExplainerLit(model.cuda(), g.to("cuda"))
    # 创建PyTorch Lightning训练器
    trainer = pl.Trainer(
        gpus=1, max_epochs=20, default_root_dir="/tmp/", log_every_n_steps=1
    )
    # 训练解释器，学习最优的边和特征掩码
    trainer.fit(gnne)
    # 获取训练后的边权重（经过sigmoid激活）
    edge_weights = gnne.explainer.edge_mask.sigmoid().detach()
    # 初始化节点重要性分数为2.0
    g.ndata["line_importance"] = th.ones(g.number_of_nodes(), device="cuda") * 2
    # 将边权重存储到图的边数据中
    g.edata["edge_mask"] = edge_weights.cuda()
    # 使用DGL的消息传递机制聚合边权重到节点
    # u_mul_e: 源节点数据乘以边数据
    # mean: 对消息取平均值
    g.update_all(
        dgl.function.u_mul_e("line_importance", "edge_mask", "m"),
        dgl.function.mean("m", "line_importance"),
    )
    # 返回节点重要性分数
    return g.ndata["line_importance"].squeeze()
