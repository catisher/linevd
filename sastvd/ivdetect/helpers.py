"""IVDetect漏洞检测方法的实现。

该模块包含IVDetect模型的核心组件，包括特征提取、模型架构和数据集处理。
IVDetect是一种基于图神经网络的漏洞检测方法，能够解释模型预测结果。
"""


import pickle as pkl
from collections import defaultdict
from pathlib import Path

import dgl
import networkx as nx
import pandas as pd
import sastvd as svd
import sastvd.helpers.dclass as svddc
import sastvd.helpers.dl as dl
import sastvd.helpers.glove as svdg
import sastvd.helpers.joern as svdj
import sastvd.helpers.tokenise as svdt
import sastvd.ivdetect.treelstm as ivdts
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from torch.nn.utils.rnn import pad_sequence

# def global_mean_pool(x, batch, size=None):
#     """Global mean pool (copied)."""
#     import numpy as np
#     from torch_scatter import scatter

#     size = int(batch.max().item() + 1) if size is None else size
#     return scatter(x, batch, dim=0, dim_size=size, reduce="mean")


def feature_extraction(filepath):
    """提取IVDetect代码表示的相关组件。

    该函数从给定的代码文件中提取IVDetect模型所需的各种特征表示，
    包括：
    1. 分词后的子标记序列（subseq）
    2. 行级AST（抽象语法树）表示
    3. 变量名和类型信息
    4. 数据依赖和控制依赖上下文

    调试信息：
    filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/180189.c"
    filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/182480.c"

    打印信息：
    svdj.plot_graph_node_edge_df(nodes, svdj.rdg(edges, "ast"), [24], 0)
    svdj.plot_graph_node_edge_df(nodes, svdj.rdg(edges, "reftype"))
    pd.options.display.max_colwidth = 500
    print(subseq.to_markdown(mode="github", index=0))
    print(nametypes.to_markdown(mode="github", index=0))
    print(uedge.to_markdown(mode="github", index=0))

    4/5 比较：
    Theirs: 31, 22, 13, 10, 6, 29, 25, 23
    Ours  : 40, 30, 19, 14, 7, 38, 33, 31
    Pred  : 40,   , 19, 14, 7, 38, 33, 31

    参数
    ----------
    filepath: str
        代码文件的路径

    返回
    -------
    tuple或None
        如果成功，返回包含两个元素的元组：
        - pdg_nodes: pandas.DataFrame，包含节点信息和特征
        - pdg_edges: tuple，包含边的源节点和目标节点列表
        如果处理失败，返回None
    """
    # 生成缓存文件名，基于文件路径的最后三部分
    cache_name = "_".join(str(filepath).split("/")[-3:])
    # 构建缓存文件路径
    cachefp = svd.get_dir(svd.cache_dir() / "ivdetect_feat_ext") / Path(cache_name).stem
    # 尝试从缓存加载结果
    try:
        with open(cachefp, "rb") as f:
            return pkl.load(f)
    except:
        pass

    # 尝试从代码文件中提取节点和边
    try:
        nodes, edges = svdj.get_node_edges(filepath)
    except:
        return None

    # 1. 生成分词后的子标记序列
    # 按代码长度降序排序，保留每行最长的代码
    subseq = (
        nodes.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )
    # 选择需要的列
    subseq = subseq[["lineNumber", "code", "local_type"]].copy()
    # 将local_type和code合并
    subseq.code = subseq.local_type + " " + subseq.code
    # 删除local_type列
    subseq = subseq.drop(columns="local_type")
    # 过滤空字符串行
    subseq = subseq[~subseq.eq("").any(1)]
    # 过滤只有空格的行
    subseq = subseq[subseq.code != " "]
    # 将lineNumber转换为整数
    subseq.lineNumber = subseq.lineNumber.astype(int)
    # 按行号排序
    subseq = subseq.sort_values("lineNumber")
    # 对代码进行分词
    subseq.code = subseq.code.apply(svdt.tokenise)
    # 转换为字典，键为行号，值为分词后的代码
    subseq = subseq.set_index("lineNumber").to_dict()["code"]

    # 2. 行级AST构建
    # 提取AST边
    ast_edges = svdj.rdg(edges, "ast")
    # 移除孤立节点
    ast_nodes = svdj.drop_lone_nodes(nodes, ast_edges)
    # 过滤空行号的节点
    ast_nodes = ast_nodes[ast_nodes.lineNumber != ""]
    # 将行号转换为整数
    ast_nodes.lineNumber = ast_nodes.lineNumber.astype(int)
    # 为每行内的节点添加索引
    ast_nodes["lineidx"] = ast_nodes.groupby("lineNumber").cumcount().values
    # 只保留行内的边
    ast_edges = ast_edges[ast_edges.line_out == ast_edges.line_in]
    # 创建节点ID到行内索引的映射
    ast_dict = pd.Series(ast_nodes.lineidx.values, index=ast_nodes.id).to_dict()
    # 将边的节点ID替换为行内索引
    ast_edges.innode = ast_edges.innode.map(ast_dict)
    ast_edges.outnode = ast_edges.outnode.map(ast_dict)
    # 按行号分组，聚合边信息
    ast_edges = ast_edges.groupby("line_in").agg({"innode": list, "outnode": list})
    # 对代码进行分词
    ast_nodes.code = ast_nodes.code.fillna("").apply(svdt.tokenise)
    # 获取每行的节点索引列表
    nodes_per_line = (
        ast_nodes.groupby("lineNumber").agg({"lineidx": list}).to_dict()["lineidx"]
    )
    # 按行号分组，聚合代码信息
    ast_nodes = ast_nodes.groupby("lineNumber").agg({"code": list})
    # 合并边和代码信息
    ast = ast_edges.join(ast_nodes, how="inner")
    # 构建AST表示
    ast["ast"] = ast.apply(lambda x: [x.outnode, x.innode, x.code], axis=1)
    # 转换为字典，键为行号，值为AST表示
    ast = ast.to_dict()["ast"]

    # If it is a lone node (nodeid doesn't appear in edges) or it is a node with no
    # incoming connections (parent node), then add an edge from that node to the node
    # with id = 0 (unless it is zero itself).
    # DEBUG:
    # import sastvd.helpers.graphs as svdgr
    # svdgr.simple_nx_plot(ast[20][0], ast[20][1], ast[20][2])
    for k, v in ast.items():
        allnodes = nodes_per_line[k]  # 该行的所有节点
        outnodes = v[0]  # 出边节点
        innodes = v[1]   # 入边节点
        # 找出孤立节点（不在出边或入边中的节点）
        lonenodes = [i for i in allnodes if i not in outnodes + innodes]
        # 找出父节点（在出边中但不在入边中的节点）
        parentnodes = [i for i in outnodes if i not in innodes]
        # 为孤立节点和父节点添加到0的边
        for n in set(lonenodes + parentnodes) - set([0]):
            outnodes.append(0)
            innodes.append(n)
        # 更新AST表示
        ast[k] = [outnodes, innodes, v[2]]

    # 3. 变量名和类型信息
    # 提取reftype边
    reftype_edges = svdj.rdg(edges, "reftype")
    # 移除孤立节点
    reftype_nodes = svdj.drop_lone_nodes(nodes, reftype_edges)
    # 创建图
    reftype_nx = nx.Graph()
    # 添加边
    reftype_nx.add_edges_from(reftype_edges[["innode", "outnode"]].to_numpy())
    # 获取连通组件
    reftype_cc = list(nx.connected_components(reftype_nx))
    # 存储变量名和类型
    varnametypes = list()
    # 处理每个连通组件
    for cc in reftype_cc:
        # 获取该组件的所有节点
        cc_nodes = reftype_nodes[reftype_nodes.id.isin(cc)]
        # 获取类型信息
        var_type = cc_nodes[cc_nodes["_label"] == "TYPE"].name.item()
        # 处理每个标识符节点
        for idrow in cc_nodes[cc_nodes["_label"] == "IDENTIFIER"].itertuples():
            # 添加行号、类型和名称
            varnametypes += [[idrow.lineNumber, var_type, idrow.name]]
    # 创建DataFrame
    nametypes = pd.DataFrame(varnametypes, columns=["lineNumber", "type", "name"])
    # 去重并按行号排序
    nametypes = nametypes.drop_duplicates().sort_values("lineNumber")
    # 对类型进行分词
    nametypes.type = nametypes.type.apply(svdt.tokenise)
    # 对名称进行分词
    nametypes.name = nametypes.name.apply(svdt.tokenise)
    # 合并类型和名称
    nametypes["nametype"] = nametypes.type + " " + nametypes.name
    # 按行号分组，合并信息
    nametypes = nametypes.groupby("lineNumber").agg({"nametype": lambda x: " ".join(x)})
    # 转换为字典
    nametypes = nametypes.to_dict()["nametype"]

    # 4/5. 数据依赖/控制依赖上下文
    # 按语句分组节点
    nodesline = nodes[nodes.lineNumber != ""].copy()
    # 将行号转换为整数
    nodesline.lineNumber = nodesline.lineNumber.astype(int)
    # 按代码长度降序排序，保留每行最长的代码
    nodesline = (
        nodesline.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )
    # 复制边
    edgesline = edges.copy()
    # 将边的节点ID替换为行号
    edgesline.innode = edgesline.line_in
    edgesline.outnode = edgesline.line_out
    # 将节点ID替换为行号
    nodesline.id = nodesline.lineNumber
    # 提取PDG边
    edgesline = svdj.rdg(edgesline, "pdg")
    # 移除孤立节点
    nodesline = svdj.drop_lone_nodes(nodesline, edgesline)
    # 去重边
    edgesline = edgesline.drop_duplicates(subset=["innode", "outnode", "etype"])
    # 将REACHING_DEF类型转换为DDG
    edgesline["etype"] = edgesline.apply(
        lambda x: "DDG" if x.etype == "REACHING_DEF" else x.etype, axis=1
    )
    # 过滤非数字节点
    edgesline = edgesline[edgesline.innode.apply(lambda x: isinstance(x, float))]
    edgesline = edgesline[edgesline.outnode.apply(lambda x: isinstance(x, float))]
    # 创建反向边
    edgesline_reverse = edgesline[["innode", "outnode", "etype"]].copy()
    edgesline_reverse.columns = ["outnode", "innode", "etype"]
    # 合并正向和反向边
    uedge = pd.concat([edgesline, edgesline_reverse])
    # 过滤自环边
    uedge = uedge[uedge.innode != uedge.outnode]
    # 按入节点和边类型分组，聚合出节点
    uedge = uedge.groupby(["innode", "etype"]).agg({"outnode": set})
    # 重置索引
    uedge = uedge.reset_index()
    # 如果有边
    if len(uedge) > 0:
        # 透视表，行是入节点，列是边类型
        uedge = uedge.pivot("innode", "etype", "outnode")
        # 添加缺失的DDG列
        if "DDG" not in uedge.columns:
            uedge["DDG"] = None
        # 添加缺失的CDG列
        if "CDG" not in uedge.columns:
            uedge["CDG"] = None
        uedge = uedge.reset_index()[["innode", "CDG", "DDG"]]
        uedge.columns = ["lineNumber", "control", "data"]
        # 将集合转换为列表
        uedge.control = uedge.control.apply(
            lambda x: list(x) if isinstance(x, set) else []
        )
        uedge.data = uedge.data.apply(lambda x: list(x) if isinstance(x, set) else [])
        # 转换为字典
        data = uedge.set_index("lineNumber").to_dict()["data"]
        control = uedge.set_index("lineNumber").to_dict()["control"]
    else:
        # 空字典
        data = {}
        control = {}

    # 生成PDG
    pdg_nodes = nodesline.copy()
    # 选择ID列并排序
    pdg_nodes = pdg_nodes[["id"]].sort_values("id")
    # 添加子标记序列
    pdg_nodes["subseq"] = pdg_nodes.id.map(subseq).fillna("")
    # 添加AST表示
    pdg_nodes["ast"] = pdg_nodes.id.map(ast).fillna("")
    # 添加变量名和类型
    pdg_nodes["nametypes"] = pdg_nodes.id.map(nametypes).fillna("")
    # 添加数据依赖
    pdg_nodes["data"] = pdg_nodes.id.map(data)
    # 添加控制依赖
    pdg_nodes["control"] = pdg_nodes.id.map(control)
    # 复制边
    pdg_edges = edgesline.copy()
    # 重置索引
    pdg_nodes = pdg_nodes.reset_index(drop=True).reset_index()
    # 创建节点ID到索引的映射
    pdg_dict = pd.Series(pdg_nodes.index.values, index=pdg_nodes.id).to_dict()
    # 将边的节点ID替换为索引
    pdg_edges.innode = pdg_edges.innode.map(pdg_dict)
    pdg_edges.outnode = pdg_edges.outnode.map(pdg_dict)
    # 过滤NaN值
    pdg_edges = pdg_edges.dropna()
    # 转换为元组，包含出节点和入节点列表
    pdg_edges = (pdg_edges.outnode.tolist(), pdg_edges.innode.tolist())

    # 缓存结果
    with open(cachefp, "wb") as f:
        pkl.dump([pdg_nodes, pdg_edges], f)
    # 返回结果
    return pdg_nodes, pdg_edges


class GruWrapper(nn.Module):
    """GRU包装器，用于获取GRU的最后一个状态。

    该类封装了PyTorch的GRU模型，提供了从动态序列中提取最后一个隐藏状态的功能。
    使用dl.DynamicRNN来处理可变长度的序列输入。
    """

    def __init__(
        self, input_size, hidden_size, num_layers=1, dropout=0, bidirectional=False
    ):
        """初始化GRU包装器。

        参数
        ----------
        input_size: int
            输入特征的维度
        hidden_size: int
            隐藏状态的维度
        num_layers: int, 可选
            GRU的层数（默认为1）
        dropout: float, 可选
            层间 dropout 概率（默认为0）
        bidirectional: bool, 可选
            是否使用双向GRU（默认为False）
        """
        super(GruWrapper, self).__init__()
        self.gru = dl.DynamicRNN(
            nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True,
            )
        )

    def forward(self, x, x_lens, return_sequence=False):
        """前向传播。

        参数
        ----------
        x: torch.Tensor
            输入序列张量，形状为 [batch_size, seq_len, input_size]
        x_lens: torch.Tensor
            每个序列的实际长度，形状为 [batch_size]
        return_sequence: bool, 可选
            是否返回完整序列的输出（默认为False）

        返回
        -------
        tuple
            - out: torch.Tensor
                如果 return_sequence 为 True，则形状为 [batch_size, seq_len, hidden_size]
                否则，形状为 [batch_size, hidden_size]
            - hidden: torch.Tensor
                最后一个时间步的隐藏状态
        """
        # 前向传播通过GRU
        out, hidden = self.gru(x, x_lens)
        if return_sequence:
            return out, hidden
        # 提取每个序列的最后一个有效时间步的输出
        out = out[range(out.shape[0]), x_lens - 1, :]
        return out, hidden


class IVDetect(nn.Module):
    """IVDetect漏洞检测模型。

    IVDetect是一种基于图神经网络的漏洞检测方法，能够结合多种代码表示
    （包括序列特征、AST结构和数据/控制依赖）来检测代码中的漏洞，
    并提供可解释的预测结果。

    模型架构：
    1. 多个GRU网络用于处理不同的序列特征
    2. TreeLSTM用于处理AST结构特征
    3. BiGRU用于融合不同特征表示
    4. GCN用于捕获代码行之间的依赖关系
    5. 全连接层用于最终的漏洞分类
    """

    def __init__(self, input_size, hidden_size, dropout=0.5):
        """初始化IVDetect模型。

        参数
        ----------
        input_size: int
            输入特征的维度
        hidden_size: int
            隐藏状态的维度
        dropout: float, 可选
            dropout 概率（默认为0.5）
        """
        super(IVDetect, self).__init__()
        # 初始化GRU包装器用于处理不同的序列特征
        self.gru = GruWrapper(input_size, hidden_size)
        self.gru2 = GruWrapper(input_size, hidden_size)
        self.gru3 = GruWrapper(input_size, hidden_size)
        self.gru4 = GruWrapper(input_size, hidden_size)
        # 双向GRU用于特征融合
        self.bigru = nn.GRU(
            hidden_size, hidden_size, bidirectional=True, batch_first=True
        )
        # 设备选择
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # TreeLSTM用于处理AST结构
        self.treelstm = ivdts.TreeLSTM(input_size, hidden_size, dropout=0)
        # GCN用于图表示学习
        self.gcn = GraphConv(hidden_size, 2)
        # 全连接层用于特征连接
        self.connect = nn.Linear(hidden_size * 3 * 2, hidden_size)
        # Dropout概率
        self.dropout = dropout
        # 隐藏层大小
        self.h_size = hidden_size

    def forward(self, g, dataset, e_weights=[]):
        """前向传播。

        执行IVDetect模型的前向传播，处理图输入并生成预测结果。

        调试示例：
        import sastvd.helpers.graphs as svdgr
        from importlib import reload
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset = BigVulDatasetIVDetect(partition="train", sample=10)
        g = dgl.batch([dataset[0], dataset[1]]).to(dev)

        input_size = 200
        hidden_size = 200
        num_layers = 2

        reload(ivdts)
        model = IVDetect(200, 64).to(dev)
        ret = model(g, dataset)

        参数
        ----------
        g: DGLGraph
            输入图，包含节点特征和边信息
        dataset: BigVulDatasetIVDetect
            数据集对象，用于获取额外的特征信息
        e_weights: list, 可选
            边权重列表，用于加权图连接（默认为空列表）

        返回
        -------
        torch.Tensor
            处理后的图节点特征，形状为 [num_nodes, 2]，表示每个节点的漏洞预测概率
        """
        # 从CPU上的磁盘加载数据
        nodes = list(
            zip(
                g.ndata["_SAMPLE"].detach().cpu().int().numpy(),
                g.ndata["_LINE"].detach().cpu().int().numpy(),
            )
        )
        data = dict()
        asts = []
        for sampleid in set([n[0] for n in nodes]):
            datasetitem = dataset.item(sampleid)
            for row in datasetitem["df"].to_dict(orient="records"):
                data[(sampleid, row["id"])] = row
            asts += datasetitem["asts"]
        asts = [i for i in asts if i is not None]
        asts = dgl.batch(asts).to(self.dev)

        feat = defaultdict(list)
        for n in nodes:
            f1 = torch.Tensor(data[n]["subseq"])
            f1 = f1 if f1.shape[0] > 0 else torch.zeros(1, 200)
            f1_lens = len(f1)
            feat["f1"].append(f1)
            feat["f1_lens"].append(f1_lens)

            f3 = torch.Tensor(data[n]["nametypes"])
            f3 = f3 if f3.shape[0] > 0 else torch.zeros(1, 200)
            f3_lens = len(f3)
            feat["f3"].append(f3)
            feat["f3_lens"].append(f3_lens)

        # 通过GRU / TreeLSTM处理
        F1, _ = self.gru(
            pad_sequence(feat["f1"], batch_first=True).to(self.dev),
            torch.Tensor(feat["f1_lens"]).long(),
        )
        F2 = self.treelstm(asts)
        F3, _ = self.gru2(
            pad_sequence(feat["f3"], batch_first=True).to(self.dev),
            torch.Tensor(feat["f3_lens"]).long(),
        )
        # F4, _ = self.gru3(
        #     pad_sequence(feat["f1"], batch_first=True).to(self.dev),
        #     torch.Tensor(feat["f1_lens"]).long(),
        # )
        # F5, _ = self.gru4(
        #     pad_sequence(feat["f1"], batch_first=True).to(self.dev),
        #     torch.Tensor(feat["f1_lens"]).long(),
        # )

        # 填充空值（例如，行没有AST表示或数据/控制依赖）
        F2 = torch.stack(
            [F2[i] if i in F2 else torch.zeros(self.h_size).to(self.dev) for i in nodes]
        )

        F1 = F1.unsqueeze(1)
        F2 = F2.unsqueeze(1)
        F3 = F3.unsqueeze(1)

        feat_vec, _ = self.bigru(torch.cat((F1, F2, F3), dim=1))
        feat_vec = F.dropout(feat_vec, self.dropout)
        feat_vec = torch.flatten(feat_vec, 1)
        feat_vec = self.connect(feat_vec)

        g.ndata["h"] = self.gcn(g, feat_vec)
        batch_pooled = torch.empty(size=(0, 2)).to(self.dev)
        # for g_i in dgl.unbatch(g):
        #     conv_output = g_i.ndata["h"]
        #     pooled = global_mean_pool(
        #         conv_output,
        #         torch.tensor(
        #             np.zeros(shape=(conv_output.shape[0]), dtype=int), device=self.dev
        #         ),
        #     )
        #     batch_pooled = torch.cat([batch_pooled, pooled])
        return batch_pooled


class BigVulDatasetIVDetect(svddc.BigVulDataset):
    """IVDetect版本的BigVul数据集。

    继承自BigVulDataset类，为IVDetect模型提供特定的数据加载和处理功能。
    该类负责将原始代码转换为IVDetect模型所需的特征表示，包括：
    1. 加载GloVe词嵌入用于特征向量化
    2. 提取代码的AST结构、数据依赖和控制依赖
    3. 构建用于模型训练和推理的图表示
    """

    def __init__(self, **kwargs):
        """初始化IVDetect版本的BigVul数据集。

        参数
        ----------
        **kwargs: dict
            传递给父类BigVulDataset的关键字参数
        """
        super(BigVulDatasetIVDetect, self).__init__(**kwargs)
        # 加载GloVe词嵌入用于代码特征向量化
        glove_path = svd.processed_dir() / "bigvul/glove_False/vectors.txt"
        self.emb_dict, _ = svdg.glove_dict(glove_path)

    def item(self, _id):
        """获取指定ID的数据项。

        根据给定的ID获取数据集项，包括代码的各种特征表示和AST图。

        参数
        ----------
        _id: int
            数据项的ID

        返回
        -------
        dict
            包含以下键的字典：
            - df: pandas.DataFrame
                包含代码行级别的特征数据
            - asts: list
                包含每个代码行的AST图表示的列表
        """
        n, _ = feature_extraction(svddc.BigVulDataset.itempath(_id))
        n.subseq = n.subseq.apply(lambda x: svdg.get_embeddings(x, self.emb_dict, 200))
        n.nametypes = n.nametypes.apply(
            lambda x: svdg.get_embeddings(x, self.emb_dict, 200)
        )

        asts = []

        def ast_dgl(row, lineid):
            if len(row) == 0:
                return None
            outnode, innode, ndata = row
            g = dgl.graph((outnode, innode))
            g.ndata["_FEAT"] = torch.Tensor(
                svdg.get_embeddings_list(ndata, self.emb_dict, 200)
            )
            g.ndata["_ID"] = torch.Tensor([_id] * g.number_of_nodes())
            g.ndata["_LINE"] = torch.Tensor([lineid] * g.number_of_nodes())
            return g

        for row in n.itertuples():
            asts.append(ast_dgl(row.ast, row.id))

        return {"df": n, "asts": asts}

    def _feat_ext_itempath(_id):
        """Run feature extraction with itempath."""
        feature_extraction(svddc.BigVulDataset.itempath(_id))

    def cache_features(self):
        """Save features to disk as cache."""
        svd.dfmp(
            self.df,
            svddc.BigVulDataset._feat_ext_itempath,
            "id",
            ordr=False,
            desc="Cache features: ",
        )

    def __getitem__(self, idx):
        """Override getitem."""
        _id = self.idx2id[idx]
        n, e = feature_extraction(svddc.BigVulDataset.itempath(_id))
        n["vuln"] = n.id.map(self.get_vuln_indices(_id)).fillna(0)
        g = dgl.graph(e)
        g.ndata["_LINE"] = torch.Tensor(n["id"].astype(int).to_numpy())
        g.ndata["_VULN"] = torch.Tensor(n["vuln"].astype(int).to_numpy())
        g.ndata["_SAMPLE"] = torch.Tensor([_id] * len(n))
        g = dgl.add_self_loop(g)
        return g