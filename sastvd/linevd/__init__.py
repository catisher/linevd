"""LineVD模型训练和实现的核心代码。

该文件包含了LineVD模型的主要实现，包括图特征提取、数据集处理、模型架构和训练逻辑。
LineVD是一种基于图神经网络的漏洞检测模型，能够在代码行级别进行漏洞检测。
"""
import os
from glob import glob

import dgl
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.codebert as cb
import sastvd.graphcodebert as gcb
import sastvd.helpers.dclass as svddc
import sastvd.helpers.doc2vec as svdd2v
import sastvd.helpers.glove as svdg
import sastvd.helpers.joern as svdj
import sastvd.helpers.losses as svdloss
import sastvd.helpers.ml as ml
import sastvd.helpers.rank_eval as svdr
import sastvd.helpers.sast as sast
import sastvd.ivdetect.evaluate as ivde
import sastvd.linevd.gnnexplainer as lvdgne
import torch as th
import torch.nn.functional as F
import torchmetrics
from dgl.data.utils import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GATConv, GraphConv
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from tqdm import tqdm


def ne_groupnodes(n, e):
    """将具有相同行号的节点分组。
    
    该函数将代码属性图(CPG)中属于同一源代码行的多个AST节点合并为一个节点。
    在Joern提取的CPG中，一行代码可能对应多个AST节点（如声明、表达式等），
    但在行级漏洞检测中，我们需要以代码行为单位进行分析，因此需要合并节点。
    返回:
        nl: 分组后的节点DataFrame，每个行号只保留一个代表性节点
        el: 调整后的边DataFrame，使用行号作为节点标识符
    """
    # 步骤1: 过滤掉没有行号的节点
    # 某些AST节点（如文件节点、方法节点）可能没有具体的行号
    nl = n[n.lineNumber != ""].copy()
    
    # 将行号从字符串转换为整数，便于后续排序和比较
    nl.lineNumber = nl.lineNumber.astype(int)
    
    # 步骤2: 按代码长度降序排序
    # 同一行可能有多个节点，保留代码最长的节点作为代表
    # 这样可以保留更多的语义信息
    nl = nl.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
    
    # 步骤3: 按行号分组，每组只保留第一个节点（代码最长的）
    # groupby后每组取head(1)实现去重
    nl = nl.groupby("lineNumber").head(1)
    
    # 步骤4: 复制边DataFrame，准备调整边的节点引用
    el = e.copy()
    
    # 将边的节点引用从节点ID改为行号
    # line_in 和 line_out 是预先计算好的行号映射
    el.innode = el.line_in
    el.outnode = el.line_out
    
    # 步骤5: 将节点的ID设置为行号
    # 这样节点ID和行号一致，便于后续处理
    nl.id = nl.lineNumber
    
    # 删除没有边连接的孤立节点
    nl = svdj.drop_lone_nodes(nl, el)
    
    # 步骤6: 清理边数据
    # 删除重复的边（相同的起点、终点和边类型）
    el = el.drop_duplicates(subset=["innode", "outnode", "etype"])
    
    # 过滤掉节点引用不是浮点数的边（可能是无效数据）
    el = el[el.innode.apply(lambda x: isinstance(x, float))]
    el = el[el.outnode.apply(lambda x: isinstance(x, float))]
    
    # 将边的节点引用转换为整数类型
    el.innode = el.innode.astype(int)
    el.outnode = el.outnode.astype(int)
    
    return nl, el


def feature_extraction(_id, graph_type="cfgcdg", return_nodes=False):
    """提取代码图的基本特征。
    
    该函数从代码文件中提取代码属性图(CPG)，并将其转换为适合图神经网络处理的格式。
    主要步骤包括：获取节点和边、按行号分组节点、过滤孤立节点、映射索引和边类型。
    返回:
        如果return_nodes=True: 返回包含节点信息的DataFrame
        否则: 返回元组(code, lineno, ei, eo, et)，分别表示
            code: 代码行列表（每行代码的文本内容）
            lineno: 行号列表（代码行对应的源代码行号）
            ei: 边的起始节点列表（边的源节点索引）
            eo: 边的结束节点列表（边的目标节点索引）
            et: 边类型列表（边的类型编码）

    """
    # 步骤1: 获取代码属性图(CPG)的节点和边
    # svdj.get_node_edges 从Joern生成的JSON文件中读取节点和边信息
    n, e = svdj.get_node_edges(_id)
    
    # 步骤2: 按行号分组节点
    # 同一行代码可能对应多个AST节点，这里将它们合并为一个节点
    # 保留代码最长的节点作为代表
    n, e = ne_groupnodes(n, e)

    # 如果只需要节点元数据（用于实证评估），直接返回节点DataFrame
    if return_nodes:
        return n

    # 步骤3: 根据图类型过滤边
    # graph_type可能包含"+"，如"pdg+raw"，取第一部分作为图类型
    # svdj.rdg 函数根据图类型筛选边（如只保留CFG边或PDG边）
    e = svdj.rdg(e, graph_type.split("+")[0])
    
    # 删除没有边连接的孤立节点
    n = svdj.drop_lone_nodes(n, e)

    # 可选：绘制图结构（已注释）
    # svdj.plot_graph_node_edge_df(n, e)

    # 步骤4: 将行号映射为连续索引
    # 重置索引，创建从0开始的连续索引
    n = n.reset_index(drop=True).reset_index()
    
    # 创建行号到索引的映射字典
    # 键为原始行号，值为新的连续索引
    iddict = pd.Series(n.index.values, index=n.id).to_dict()
    
    # 将边中的节点标识符从行号映射为索引
    e.innode = e.innode.map(iddict)
    e.outnode = e.outnode.map(iddict)

    # 步骤5: 将边类型映射为整数编码
    # 获取所有边类型列表
    etypes = e.etype.tolist()
    
    # 创建边类型到整数的映射（按字母顺序排序后编号）
    # 例如: {"AST": 0, "CFG": 1, "DDG": 2}
    d = dict([(y, x) for x, y in enumerate(sorted(set(etypes)))])
    
    # 将边类型转换为整数编码
    etypes = [d[i] for i in etypes]

    # 步骤6: 构建代码文本表示
    # 如果图类型不包含"+raw"，则在代码前添加函数名和节点名
    if "+raw" not in graph_type:
        try:
            # 获取第1行的节点名称作为函数名
            func_name = n[n.lineNumber == 1].name.item()
        except:
            # 如果获取失败（如没有第1行），使用空字符串
            print(_id)
            func_name = ""
        # 构建代码格式: 函数名 + 节点名 + </s> + 原始代码
        # </s> 是分隔符，用于CodeBERT等预训练模型
        n.code = func_name + " " + n.name + " " + "</s>" + " " + n.code
    else:
        # 对于"+raw"类型，只添加分隔符
        n.code = "</s>" + " " + n.code

    # 步骤7: 返回处理后的特征
    # 返回代码列表、行号列表、边的起始节点、结束节点和边类型
    return n.code.tolist(), n.id.tolist(), e.innode.tolist(), e.outnode.tolist(), etypes


# %%
class BigVulDatasetLineVD(svddc.BigVulDataset):
    """LineVD版本的BigVul数据集实现。    
    该类继承自BigVulDataset，用于为LineVD模型准备代码图数据。
    主要负责图数据的处理，包括提取代码属性图(CPG)、映射索引、构建图结构等。
    支持多种图类型（如pdg、cfgcdg）和特征类型（如codebert、glove、doc2vec）。
    """

    def __init__(self, gtype="pdg", feat="all", **kwargs):
        """初始化LineVD版本的BigVul数据集。
        
        参数:
            gtype: 图类型，如"pdg"（程序依赖图）或"cfgcdg"（控制流图+数据依赖图）
            feat: 使用的特征类型，如"all"（所有特征）、"codebert"（CodeBERT特征）、"graphcodebert"（Graphcodebert特征）、"glove"或"doc2vec"
            **kwargs: 传递给父类的其他参数
        """
        # 调用父类 BigVulDataset 的初始化方法，传递所有参数
        super(BigVulDatasetLineVD, self).__init__(**kwargs)
        # 从 IVDetect 获取依赖添加行的数据，包含漏洞行和依赖添加行信息
        lines = ivde.get_dep_add_lines_bigvul()
        # 处理漏洞行数据：将 removed（漏洞行）和 depadd（依赖添加行）合并为一个集合
        # 使用 set 去重，确保每个行号只出现一次
        lines = {k: set(list(v["removed"]) + v["depadd"]) for k, v in lines.items()}
        # 将处理后的漏洞行数据保存为实例属性，供后续使用
        self.lines = lines
        # 保存图类型（如 "pdg"、"cfgcdg" 等），用于后续图构建
        self.graph_type = gtype
        # 构建 GloVe 词向量文件的路径
        glove_path = svd.processed_dir() / "bigvul/glove_False/vectors.txt"
        # 加载 GloVe 词向量字典，返回词到向量的映射字典
        self.glove_dict, _ = svdg.glove_dict(glove_path)
        # 初始化 Doc2Vec 模型，用于生成文档级嵌入
        self.d2v = svdd2v.D2V(svd.processed_dir() / "bigvul/d2v_False")
        # 保存特征类型（如 "all"、"codebert"、"graphcodebert"、""glove"、"doc2vec"），控制使用哪种特征
        self.feat = feat

    def item(self, _id, codebert=None, graphcodebert=None):
        """获取并缓存指定ID的代码图数据项。
        """
        if codebert :
            g = self.item_codebert(_id)
            return g 
        elif graphcodebert :
            g= self.item_graphcodebert(_id)
            return g
        else:
            raise ValueError("codebert or graphcodebert must be provided")

    
    def item_codebert(self, _id, codebert):
        """缓存单个数据项的代码图。
        """
        """获取并缓存指定ID的代码图数据项。
        """
        # 构建缓存目录路径，包含图类型信息
        savedir = svd.get_dir(
            svd.cache_dir() / f"bigvul_linevd_codebert_{self.graph_type}"
        ) / str(_id)
        # 检查缓存目录是否存在，如果存在则直接加载缓存的图
        if os.path.exists(savedir):
            # 从缓存目录加载图数据
            g = load_graphs(str(savedir))[0][0]
            # 注释掉的代码：将函数级漏洞标签复制到所有节点
            # g.ndata["_FVULN"] = g.ndata["_VULN"].max().repeat((g.number_of_nodes()))
            # 注释掉的代码：移除 SAST 工具的标签
            # if "_SASTRATS" in g.ndata:
            #     g.ndata.pop("_SASTRATS")
            #     g.ndata.pop("_SASTCPP")
            #     g.ndata.pop("_SASTFF")
            #     g.ndata.pop("_GLOVE")
            #     g.ndata.pop("_DOC2VEC")
            # 检查图中是否包含 CodeBERT 嵌入
            if "_CODEBERT" in g.ndata:
                # 如果使用 CodeBERT 特征，移除其他特征以节省内存
                if self.feat == "codebert":
                    for i in ["_GLOVE", "_DOC2VEC", "_RANDFEAT","_GRAPHCODEBERT"]:
                        g.ndata.pop(i, None)
                # 如果使用 GloVe 特征，移除其他特征
                if self.feat == "glove":
                    for i in ["_CODEBERT", "_DOC2VEC", "_RANDFEAT","_GRAPHCODEBERT"]:
                        g.ndata.pop(i, None)
                # 如果使用 Doc2Vec 特征，移除其他特征
                if self.feat == "doc2vec":
                    for i in ["_CODEBERT", "_GLOVE", "_RANDFEAT","_GRAPHCODEBERT"]:
                        g.ndata.pop(i, None)
                # 返回处理后的图
                return g
        # 如果缓存不存在，则进行特征提取
        code, lineno, ei, eo, et = feature_extraction(
            svddc.BigVulDataset.itempath(_id), self.graph_type
        )
        # 根据漏洞行号列表生成漏洞标签
        if _id in self.lines:
            vuln = [1 if i in self.lines[_id] else 0 for i in lineno]
        else:
            # 如果没有漏洞行号，所有行都标记为安全
            vuln = [0 for _ in lineno]
        # 创建 DGL 图对象，使用边索引构建图
        g = dgl.graph((eo, ei))
        # 生成 GloVe 词向量嵌入
        #gembeds = th.Tensor(svdg.get_embeddings_list(code, self.glove_dict, 200))
        gembeds = th.Tensor(np.array(svdg.get_embeddings_list(code, self.glove_dict, 200)))
        # 将 GloVe 嵌入存储到图的节点数据中
        g.ndata["_GLOVE"] = gembeds
        
        # 生成 Doc2Vec 嵌入并存储到节点数据中
        # 下面这种更快
        #g.ndata["_DOC2VEC"] = th.Tensor([self.d2v.infer(i) for i in code])
        g.ndata["_DOC2VEC"] = th.Tensor(np.array([self.d2v.infer(i) for i in code]))
        # 如果提供了 CodeBERT 模型，则生成 CodeBERT 嵌入
        if codebert:
            # 清理代码文本中的制表符和换行符
            code = [c.replace("\\t", "").replace("\\n", "") for c in code]
            # 将代码分成批次，每批 128 行
            chunked_batches = svd.chunks(code, 128)
            # 对每个批次进行 CodeBERT 编码，并移到 CPU
            features = [codebert.encode(c).detach().cpu() for c in chunked_batches]
            # 将所有批次的特征连接起来，存储到节点数据中
            g.ndata["_CODEBERT"] = th.cat(features)

        # 生成随机特征，维度为 100
        g.ndata["_RANDFEAT"] = th.rand(size=(g.number_of_nodes(), 100))
        # 存储行号信息
        g.ndata["_LINE"] = th.Tensor(lineno).int()
        # 存储漏洞标签
        g.ndata["_VULN"] = th.Tensor(vuln).float()

        # 获取 SAST（静态应用安全测试）标签
        s = sast.get_sast_lines(svd.processed_dir() / f"bigvul/before/{_id}.c.sast.pkl")
        # 生成 RATS 工具的标签
        rats = [1 if i in s["rats"] else 0 for i in g.ndata["_LINE"]]
        # 生成 cppcheck 工具的标签
        cppcheck = [1 if i in s["cppcheck"] else 0 for i in g.ndata["_LINE"]]
        # 生成 flawfinder 工具的标签
        flawfinder = [1 if i in s["flawfinder"] else 0 for i in g.ndata["_LINE"]]
        # 存储 RATS 标签到节点数据中
        g.ndata["_SASTRATS"] = th.tensor(rats).long()
        # 存储 cppcheck 标签到节点数据中
        g.ndata["_SASTCPP"] = th.tensor(cppcheck).long()
        # 存储 flawfinder 标签到节点数据中
        g.ndata["_SASTFF"] = th.tensor(flawfinder).long()

        # 从行级标签倒推函数级标签，将函数级漏洞标签复制到所有节点
        g.ndata["_FVULN"] = g.ndata["_VULN"].max().repeat((g.number_of_nodes()))
        # 存储边类型信息
        g.edata["_ETYPE"] = th.Tensor(et).long()
        # 构建方法级 CodeBERT 嵌入的文件路径
        emb_path = svd.cache_dir() / f"codebert_method_level/{_id}.pt"
        # 加载方法级嵌入并复制到所有节点
        g.ndata["_FUNC_EMB"] = th.load(emb_path, weights_only=True).repeat((g.number_of_nodes(), 1))
        
    def item_graphcodebert(self, _id, graphcodebert):
        """缓存单个数据项的 GraphCodeBERT 代码图。
        """
        # 构建缓存目录路径，包含图类型信息
        savedir = svd.get_dir(
            svd.cache_dir() / f"bigvul_linevd_graphcodebert_{self.graph_type}"
        ) / str(_id)
        # 检查缓存目录是否存在，如果存在则直接加载缓存的图
        if os.path.exists(savedir):
            # 从缓存目录加载图数据
            g = load_graphs(str(savedir))[0][0]
            # 检查图中是否包含 GraphCodeBERT 嵌入
            if "_GRAPHCODEBERT" in g.ndata:
                # 如果使用 GraphCodeBERT 特征，移除其他特征以节省内存
                if self.feat == "graphcodebert":
                    for i in ["_GLOVE", "_DOC2VEC", "_RANDFEAT","_CODEBERT"]:
                        g.ndata.pop(i, None)
                # 如果使用 GloVe 特征，移除其他特征
                if self.feat == "glove":
                    for i in ["_GRAPHCODEBERT", "_DOC2VEC", "_RANDFEAT","_CODEBERT"]:
                        g.ndata.pop(i, None)
                # 如果使用 Doc2Vec 特征，移除其他特征
                if self.feat == "doc2vec":
                    for i in ["_GRAPHCODEBERT", "_GLOVE", "_RANDFEAT","_CODEBERT"]:
                        g.ndata.pop(i, None)
                # 返回处理后的图
                return g
        # 如果缓存不存在，则进行特征提取
        code, lineno, ei, eo, et = feature_extraction(
            svddc.BigVulDataset.itempath(_id), self.graph_type
        )
        # 根据漏洞行号列表生成漏洞标签
        if _id in self.lines:
            vuln = [1 if i in self.lines[_id] else 0 for i in lineno]
        else:
            # 如果没有漏洞行号，所有行都标记为安全
            vuln = [0 for _ in lineno]
        # 创建 DGL 图对象，使用边索引构建图
        g = dgl.graph((eo, ei))

        # # 生成 GloVe 词向量嵌入
        # gembeds = th.Tensor(svdg.get_embeddings_list(code, self.glove_dict, 200))
        # # 将 GloVe 嵌入存储到图的节点数据中
        # g.ndata["_GLOVE"] = gembeds
        # # 生成 Doc2Vec 嵌入并存储到节点数据中
        # g.ndata["_DOC2VEC"] = th.Tensor([self.d2v.infer(i) for i in code])
        
         # 生成 GloVe 词向量嵌入
        gembeds = th.Tensor(np.array(svdg.get_embeddings_list(code, self.glove_dict, 200)))
        # 将 GloVe 嵌入存储到图的节点数据中
        g.ndata["_GLOVE"] = gembeds
        # 生成 Doc2Vec 嵌入并存储到节点数据中
        g.ndata["_DOC2VEC"] = th.Tensor(np.array([self.d2v.infer(i) for i in code]))
        # 如果提供了 GraphCodeBERT 模型，则生成 GraphCodeBERT 嵌入

        # 如果提供了 GraphCodeBERT 模型，则生成 GraphCodeBERT 嵌入

        if graphcodebert:
            code = [c.replace("\t", " ").replace("\n", " ").strip() for c in code]
            structure = [graphcodebert.generate_structure_info(c) for c in code]

            BATCH_SIZE = 128
            # 安全分批：代码和结构使用完全相同的索引
            code_batches = [code[i:i+BATCH_SIZE] for i in range(0, len(code), BATCH_SIZE)]
            struct_batches = [structure[i:i+BATCH_SIZE] for i in range(0, len(structure), BATCH_SIZE)]

            graphcodebert_features = []
            for cb, sb in zip(code_batches, struct_batches):
                feat = graphcodebert.encode(cb, sb).detach().cpu()
                graphcodebert_features.append(feat)

            g.ndata["_GRAPHCODEBERT"] = th.cat(graphcodebert_features)

        # 生成随机特征，维度为 100
        g.ndata["_RANDFEAT"] = th.rand(size=(g.number_of_nodes(), 100))
        # 存储行号信息
        g.ndata["_LINE"] = th.Tensor(lineno).int()
        # 存储漏洞标签
        g.ndata["_VULN"] = th.Tensor(vuln).float()

        # 获取 SAST（静态应用安全测试）标签
        s = sast.get_sast_lines(svd.processed_dir() / f"bigvul/before/{_id}.c.sast.pkl")
        # 生成 RATS 工具的标签
        rats = [1 if i in s["rats"] else 0 for i in g.ndata["_LINE"]]
        # 生成 cppcheck 工具的标签
        cppcheck = [1 if i in s["cppcheck"] else 0 for i in g.ndata["_LINE"]]
        # 生成 flawfinder 工具的标签
        flawfinder = [1 if i in s["flawfinder"] else 0 for i in g.ndata["_LINE"]]
        # 存储 RATS 标签到节点数据中
        g.ndata["_SASTRATS"] = th.tensor(rats).long()
        # 存储 cppcheck 标签到节点数据中
        g.ndata["_SASTCPP"] = th.tensor(cppcheck).long()
        # 存储 flawfinder 标签到节点数据中
        g.ndata["_SASTFF"] = th.tensor(flawfinder).long()

        # 从行级标签倒推函数级标签，将函数级漏洞标签复制到所有节点
        g.ndata["_FVULN"] = g.ndata["_VULN"].max().repeat((g.number_of_nodes()))
        # 存储边类型信息
        g.edata["_ETYPE"] = th.Tensor(et).long()
        # 构建方法级 GraphCodeBERT 嵌入的文件路径
        graph_emb_path = svd.cache_dir() / f"graphcodebert_method_level/{_id}.pt"
        # 加载方法级嵌入并复制到所有节点
        g.ndata["_GRAPH_FUNC_EMB"] = th.load(graph_emb_path, weights_only=True).repeat((g.number_of_nodes(), 1))

        # 为图添加自环边
        g = dgl.add_self_loop(g)
        # 将处理后的图保存到缓存目录
        save_graphs(str(savedir), [g])
        # 返回图对象
        return g
    
      

    def cache_items(self, codebert , graphcodebert):
        """缓存所有数据项的代码图。
        """
        for i in tqdm(self.df.sample(len(self.df)).id.tolist()):
            try:
                self.item_codebert(i, codebert)
                self.item_graphcodebert(i, graphcodebert)
            except Exception as E:
                print(E)

    def cache_codebert_method_level(self, codebert):
        """缓存所有数据项的方法级CodeBERT嵌入。
        就是直接对_id文件所对应的代码进行编码，生成方法级的CodeBERT嵌入向量。
        """
        # 获取缓存目录，用于存储CodeBERT嵌入
        # 对应文件在item方法中生成
        savedir = svd.get_dir(svd.cache_dir() / "codebert_method_level")
        # 从缓存目录中读取已处理的文件，提取ID
        # glob(str(savedir / "*"))  作用 ：获取缓存目录中的所有文件路径
        # i.split("/")  作用 ：按斜杠 / 分割路径字符串，获取最后一个元素（文件名）
        # i.split(".")[0]  作用 ：将文件名中的扩展名（.pt）去掉，保留ID
        done = [int(i.split("/")[-1].split(".")[0]) for i in glob(str(savedir / "*"))]
        # 将已处理的ID转换为集合，以便快速查找
        done = set(done)
        # 将数据分为每批128个样本
        ## 修改大小，内存不够
        #batches = svd.chunks((range(len(self.df))), 128)
        batches = svd.chunks((range(len(self.df))), 64)
        
        # 遍历每个批次，使用tqdm显示处理进度
        for idx_batch in tqdm(batches):
            # 从DataFrame中获取当前批次的文本数据（before列）
            batch_texts = self.df.iloc[idx_batch[0] : idx_batch[-1] + 1].before.tolist()
            # 从DataFrame中获取当前批次的ID
            batch_ids = self.df.iloc[idx_batch[0] : idx_batch[-1] + 1].id.tolist()
            # 检查当前批次的所有ID是否都已处理过
            if set(batch_ids).issubset(done):
                # 如果已处理，跳过当前批次
                continue
            # 在每个文本前添加结束标记</s>，这是CodeBERT的输入格式要求
            texts = ["</s> " + ct for ct in batch_texts]
            # 使用CodeBERT编码文本，生成嵌入向量，然后将结果从GPU移至CPU
            embedded = codebert.encode(texts).detach().cpu()
            # 确保文本数量和ID数量一致
            assert len(batch_texts) == len(batch_ids)
            # 遍历每个样本
            for i in range(len(batch_texts)):
                # 将每个样本的嵌入保存到文件，文件名使用ID
                th.save(embedded[i], savedir / f"{batch_ids[i]}.pt")

    def cache_graphcodebert_method_level(self, codebert):
        # 获取缓存目录，用于存储GraphCodeBERT嵌入
        savedir = svd.get_dir(svd.cache_dir() / "graphcodebert_method_level")
        # 从缓存目录中读取已处理的文件，提取ID
        done = [int(i.split("/")[-1].split(".")[0]) for i in glob(str(savedir / "*"))]
        # 将已处理的ID转换为集合，以便快速查找
        done = set(done)
        # 将数据分为每批128个样本
        batches = svd.chunks((range(len(self.df))), 64)
        
        # 遍历每个批次，使用tqdm显示处理进度
        for idx_batch in tqdm(batches):
            # 从DataFrame中获取当前批次的文本数据（before列）
            batch_texts = self.df.iloc[idx_batch[0] : idx_batch[-1] + 1].before.tolist()
            # 从DataFrame中获取当前批次的ID
            batch_ids = self.df.iloc[idx_batch[0] : idx_batch[-1] + 1].id.tolist()
            # 检查当前批次的所有ID是否都已处理过
            if set(batch_ids).issubset(done):
                # 如果已处理，跳过当前批次
                continue
            # 在每个文本前添加结束标记</s>，这是GraphCodeBERT的输入格式要求
            texts = ["</s> " + ct for ct in batch_texts]
            # 为每个文本生成结构信息
            structures = []
            for text in batch_texts:
                lines = text.split('\n')
                structure = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('if') or line.startswith('for') or line.startswith('while') or line.startswith('do'):
                        structure.append('<control>')
                    elif line.startswith('return'):
                        structure.append('<return>')
                    elif '=' in line and not line.startswith('//'):
                        structure.append('<assignment>')
                    elif line.startswith('{') or line.startswith('}'):
                        structure.append('<bracket>')
                    else:
                        structure.append('<statement>')
                structures.append(' '.join(structure))
            # 使用GraphCodeBERT编码文本，生成嵌入向量，然后将结果从GPU移至CPU
            embedded = codebert.encode(texts, structures).detach().cpu()
            # 确保文本数量和ID数量一致
            assert len(batch_texts) == len(batch_ids)
            # 遍历每个样本
            for i in range(len(batch_texts)):
                # 将每个样本的嵌入保存到文件，文件名使用ID
                th.save(embedded[i], savedir / f"{batch_ids[i]}.pt")

    def __getitem__(self, idx):
        """Override getitem."""
        _id = self.idx2id[idx]
    
        if self.feat == "codebert":
            return self.item_codebert(_id)
        elif self.feat == "graphcodebert":
            return self.item_graphcodebert(_id)
        else:
            # 默认使用 codebert 路径
            return self.item_codebert(_id)
        #return self.item(self.idx2id[idx])


class BigVulDatasetLineVDDataModule(pl.LightningDataModule):
    '''
    缓存数据，包括CodeBERT和GraphCodeBERT的嵌入向量
    配置各种加载器，包括训练集、验证集和测试集的加载器
    '''
    def __init__(
        self,
        batch_size: int = 32,
        sample: int = -1,
        methodlevel: bool = False,
        nsampling: bool = False,
        nsampling_hops: int = 1,
        gtype: str = "cfgcdg",
        splits: str = "default",
        feat: str = "all",
    ):
        """初始化BigVul数据集的数据模块。
        
        参数:
            batch_size: 批次大小
            sample: 样本数量，-1表示使用所有样本
            methodlevel: 是否使用方法级别的表示
            nsampling: 是否使用邻居采样
            nsampling_hops: 邻居采样的跳数
            gtype: 图类型，如"cfgcdg"或"pdg"
            splits: 数据集分割方式，如"default"或"crossproject"
            feat: 使用的特征类型，如"all"、"codebert"、"graphcodebert"、"glove"或"doc2vec"
        """
        super().__init__()
        dataargs = {"sample": sample, "gtype": gtype, "splits": splits, "feat": feat}
        self.train = BigVulDatasetLineVD(partition="train", **dataargs)
        self.val = BigVulDatasetLineVD(partition="val", **dataargs)
        self.test = BigVulDatasetLineVD(partition="test", **dataargs)
        # 初始化 CodeBERT 模型并缓存方法级嵌入
        codebert = cb.CodeBert()
        self.train.cache_codebert_method_level(codebert)
        self.val.cache_codebert_method_level(codebert)
        self.test.cache_codebert_method_level(codebert)
        
        # 初始化 GraphCodeBERT 模型并缓存方法级嵌入
        graphcodebert = gcb.GraphCodeBert()
        self.train.cache_graphcodebert_method_level(graphcodebert)
        self.val.cache_graphcodebert_method_level(graphcodebert)
        self.test.cache_graphcodebert_method_level(graphcodebert)
        
        # 缓存所有数据项
        self.train.cache_items(codebert, graphcodebert)
        self.val.cache_items(codebert, graphcodebert)
        self.test.cache_items(codebert, graphcodebert)

        self.batch_size = batch_size
        self.nsampling = nsampling
        self.nsampling_hops = nsampling_hops
        self.feat = feat

    def node_dl(self, g, shuffle=False):
        # 处理邻居采样时做的封装
        # MultiLayerFullNeighborSampler 是 DGL 提供的采样器，用于图神经网络训练
        # self.nsampling_hops: 采样跳数，控制采样多少层的邻居
        # 例如：如果设置为 2，则采样 2 层邻居（直接邻居 + 邻居的邻居）
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.nsampling_hops)
        return dgl.dataloading.NodeDataLoader(
            g,
            g.nodes(),
            sampler,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=1,
        )

    def train_dataloader(self):
        """返回训练集数据加载器。
        
        返回:
            图数据加载器，支持批量加载训练数据
        """
        if self.nsampling:
            #g = next(iter(GraphDataLoader(self.train, batch_size=len(self.train))))
            #return self.node_dl(g, shuffle=True)
            return self.node_dl(self.train, shuffle=True, batch_size=self.batch_size)
        return GraphDataLoader(self.train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        """返回验证集数据加载器。
        
        返回:
            图数据加载器，支持批量加载验证数据
        """
        if self.nsampling:
            #g = next(iter(GraphDataLoader(self.val, batch_size=len(self.val))))
            #return self.node_dl(g)
            # 使用分批加载，避免一次性加载全部数据
            return self.node_dl(self.val, shuffle=True, batch_size=self.batch_size)
        return GraphDataLoader(self.val, batch_size=self.batch_size)

    def val_graph_dataloader(self):
        """返回验证集的图数据加载器。
        
        返回:
            图数据加载器，用于验证集的图级处理
        """
        return GraphDataLoader(self.val, batch_size=32)

    def test_dataloader(self):
        """返回测试集数据加载器。
        
        返回:
            图数据加载器，支持批量加载测试数据
        """
        return GraphDataLoader(self.test, batch_size=32)


# %%
class LitGNN(pl.LightningModule):
    """LineVD模型的PyTorch Lightning实现。
    """

    def __init__(
        self,
        hfeat: int = 512,
        embtype: str = "codebert",
        embfeat: int = -1,  # Keep for legacy purposes
        num_heads: int = 4,
        lr: float = 1e-3,
        hdropout: float = 0.2,
        mlpdropout: float = 0.2,
        gatdropout: float = 0.2,
        methodlevel: bool = False,
        nsampling: bool = False,
        model: str = "gat2layer",
        loss: str = "ce",
        multitask: str = "linemethod",
        stmtweight: int = 5,
        gnntype: str = "gat",
        random: bool = False,
        scea: float = 0.7,
    ):
        """初始化LineVD模型。
        
        参数:
            hfeat: 隐藏特征维度
            embtype: 嵌入类型，如"codebert"、"glove"或"doc2vec"
            embfeat: 嵌入特征维度（用于遗留目的）
            num_heads: GAT注意力头的数量
            methodlevel: 是否使用方法级别的表示
            nsampling: 是否使用邻居采样
            model: 模型类型，如"gat2layer"（2层GAT）或"gat1layer"（1层GAT）
            loss: 损失函数类型，如"ce"（交叉熵）或"sce"（对称交叉熵）
            multitask: 多任务类型，如"linemethod"（行级+方法级）或"line"（仅行级）
            stmtweight: 语句级别的权重（用于不平衡数据）
            gnntype: GNN类型，如"gat"（图注意力网络）或"gcn"（图卷积网络）
            random: 是否使用随机权重（用于基线比较）
            scea: SCE损失函数的α参数
        """
        super().__init__()
        self.lr = lr  # 保存学习率
        self.random = random  # 保存是否使用随机权重的标志
        self.save_hyperparameters()  # 保存超参数到检查点

        # 根据嵌入类型设置参数
        if self.hparams.embtype == "codebert":
            self.hparams.embfeat = 768
            self.EMBED = "_CODEBERT"
            self.FUNC_EMB = "_FUNC_EMB"
        if self.hparams.embtype == "graphcodebert":
            self.hparams.embfeat = 768
            self.EMBED = "_GRAPHCODEBERT"
            self.FUNC_EMB = "_GRAPH_FUNC_EMB"
        if self.hparams.embtype == "glove":
            self.hparams.embfeat = 200
            self.EMBED = "_GLOVE"
        if self.hparams.embtype == "doc2vec":
            self.hparams.embfeat = 300
            self.EMBED = "_DOC2VEC"

        # Loss
        if self.hparams.loss == "sce":
            # 使用对称交叉熵损失
            self.loss = svdloss.SCELoss(self.hparams.scea, 1 - self.hparams.scea)
            self.loss_f = th.nn.CrossEntropyLoss()
        else:
            # 使用标准交叉熵损失，设置类别权重
            self.loss = th.nn.CrossEntropyLoss(
                weight=th.Tensor([1, self.hparams.stmtweight])
                #weight=th.Tensor([1, self.hparams.stmtweight]).cuda()
            )
            self.loss_f = th.nn.CrossEntropyLoss()

        # 评估指标配置
        self.accuracy = torchmetrics.Accuracy(task="binary") # 准确率指标
        self.auroc = torchmetrics.AUROC(task="binary") # AUC-ROC 指标
        self.mcc = torchmetrics.MatthewsCorrCoef(task="binary", num_classes=2) # Matthews 相关系数

        # 图卷积层配置

        hfeat = self.hparams.hfeat
        # 隐藏特征维度

        gatdrop = self.hparams.gatdropout
        # GAT 层的 dropout 率

        numheads = self.hparams.num_heads
        # GAT 注意力头数量

        embfeat = self.hparams.embfeat
        # 嵌入特征维度

        gnn_args = {"out_feats": hfeat}
        # GNN 输出特征维度
        if self.hparams.gnntype == "gat":
            gnn = GATConv
            gat_args = {"num_heads": numheads, "feat_drop": gatdrop}
            gnn1_args = {**gnn_args, **gat_args, "in_feats": embfeat}
            gnn2_args = {**gnn_args, **gat_args, "in_feats": hfeat * numheads}
        elif self.hparams.gnntype == "gcn":
            gnn = GraphConv
            gnn1_args = {"in_feats": embfeat, **gnn_args}
            gnn2_args = {"in_feats": hfeat, **gnn_args}

        # model: gat2layer
        # 配置 2 层 GAT 模型
        if "gat" in self.hparams.model:
            # 第一层
            self.gat = gnn(**gnn1_args)
            # 第二层
            self.gat2 = gnn(**gnn2_args)

            fcin = hfeat * numheads if self.hparams.gnntype == "gat" else hfeat
            self.fc = th.nn.Linear(fcin, self.hparams.hfeat)
            self.fconly = th.nn.Linear(embfeat, self.hparams.hfeat)
            self.mlpdropout = th.nn.Dropout(self.hparams.mlpdropout)

        # model: mlp-only
        # 配置仅 MLP 模型
        if "mlponly" in self.hparams.model:
            self.fconly = th.nn.Linear(embfeat, self.hparams.hfeat)
            self.mlpdropout = th.nn.Dropout(self.hparams.mlpdropout)

        # model: contains femb
        # 配置包含函数嵌入的模型
        ## 神经，根本没用到
        if "+femb" in self.hparams.model:
            self.fc_femb = th.nn.Linear(embfeat * 2, self.hparams.hfeat)
        
        # 在 __init__ 方法中添加
        if "residual" in self.hparams.model:
            self.use_residual = True
        else:
            self.use_residual = False

        # self.resrgat = ResRGAT(hdim=768, rdim=1, numlayers=1, dropout=0)
        # self.gcn = GraphConv(embfeat, hfeat)
        # self.gcn2 = GraphConv(hfeat, hfeat)

        # Transform codebert embedding
        # 转换 CodeBERT 嵌入维度
        self.codebertfc = th.nn.Linear(768, self.hparams.hfeat)

        # Hidden Layers   
        # 隐藏层配置
        self.fch = []
        for _ in range(8):
            # 添加 8 个隐藏层，每层都是全连接层
            self.fch.append(th.nn.Linear(self.hparams.hfeat, self.hparams.hfeat))
        # 将隐藏层包装为 ModuleList
        self.hidden = th.nn.ModuleList(self.fch)
        # 隐藏层的 dropout
        self.hdropout = th.nn.Dropout(self.hparams.hdropout)
        # 输出层，2 个类别（安全/漏洞）
        self.fc2 = th.nn.Linear(self.hparams.hfeat, 2)

    def forward(self, g, test=False, e_weights=[], feat_override=""):
        # 根据具体模式选择输入数据
        if self.hparams.nsampling and not test:
        # 处理邻居采样的情况（训练时使用）
            # 目标节点的嵌入特征
            hdst = g[2][-1].dstdata[self.EMBED]
            h_func = g[2][-1].dstdata[self.FUNC_EMB] # 目标节点的函数嵌入
            g2 = g[2][1] # 采样后的子图（第1层）
            g = g[2][0] # 采样后的子图（第0层）
            if "gat2layer" in self.hparams.model:
                # 对于2层GAT，使用第0层子图的源节点特征
                h = g.srcdata[self.EMBED]
            elif "gat1layer" in self.hparams.model:
                # 对于1层GAT，使用第1层子图的源节点特征
                h = g2.srcdata[self.EMBED]
        else:
            # 非邻居采样的情况（测试或不使用邻居采样时）
            g2 = g  # 完整图
            h = g.ndata[self.EMBED]  # 所有节点的嵌入特征
            if len(feat_override) > 0:
                # 如果指定了特征覆盖，使用指定的特征
                h = g.ndata[feat_override]
            h_func = g.ndata[self.FUNC_EMB]  # 所有节点的函数嵌入
            hdst = h # 目标节点特征就是所有节点特征

        if self.random:
            # 如果设置为随机模式，返回随机预测结果
            return th.rand((h.shape[0], 2)).to(self.device), th.rand(
                h_func.shape[0], 2
            ).to(self.device)

        # model: contains femb
        ## 没用到，可忽略
        if "+femb" in self.hparams.model:
            # 如果模型包含函数嵌入，将代码嵌入和函数嵌入拼接
            # 形状: [N, hfeat + func_feat_dim]
            h = th.cat([h, h_func], dim=1)
            h = F.elu(self.fc_femb(h))

        # Transform h_func if wrong size
        ## 这个地方有点问题
        if self.hparams.embfeat != 768:
            h_func = self.codebertfc(h_func)

        # model: gat2layer
        if "gat" in self.hparams.model:
            if "gat2layer" in self.hparams.model:
                # 2层GAT处理
                h_residual = h  # 保存原始特征用于残差连接
                
                h = self.gat(g, h)
                # 第一层GAT
                if self.hparams.gnntype == "gat":
                    # 如果是GAT，需要将多头注意力的输出展平
                    h = h.view(-1, h.size(1) * h.size(2))
                
                # 添加残差连接（如果启用且维度匹配）
                if "residual" in self.hparams.model and h.shape == h_residual.shape:
                    h = h + h_residual
                
                h_residual2 = h  # 保存第一层输出用于第二层残差连接
                
                h = self.gat2(g2, h) # 第二层GAT
                if self.hparams.gnntype == "gat":
                    # 再次展平多头注意力的输出
                    h = h.view(-1, h.size(1) * h.size(2))
                
                # 添加残差连接（如果启用且维度匹配）
                if "residual" in self.hparams.model and h.shape == h_residual2.shape:
                    h = h + h_residual2
                    
            elif "gat1layer" in self.hparams.model:
                # 1层GAT处理
                h_residual = h  # 保存原始特征用于残差连接
                
                h = self.gat(g2, h) # 单层GAT
                if self.hparams.gnntype == "gat":
                    # 展平多头注意力的输出
                    h = h.view(-1, h.size(1) * h.size(2))
                
                # 添加残差连接（如果启用且维度匹配）
                if "residual" in self.hparams.model and h.shape == h_residual.shape:
                    h = h + h_residual
            # 通过全连接层和激活函数处理
            h = self.mlpdropout(F.elu(self.fc(h)))
            # 处理函数嵌入
            h_func = self.mlpdropout(F.elu(self.fconly(h_func)))

        # Edge masking (for GNNExplainer)
        if test and len(e_weights) > 0:
            # 测试时使用边权重进行GNNExplainer解释
            g.ndata["h"] = h  # 将特征存储到图中
            g.edata["ew"] = e_weights  # 设置边权重
            # 使用边权重更新节点特征
            g.update_all(
                dgl.function.u_mul_e("h", "ew", "m"), dgl.function.mean("m", "h")
            )
            h = g.ndata["h"]
            # 提取更新后的特征

        # model: mlp-only
        if "mlponly" in self.hparams.model:
            # 仅MLP模型处理
            h = self.mlpdropout(F.elu(self.fconly(hdst)))# 处理目标节点特征
            h_func = self.mlpdropout(F.elu(self.fconly(h_func)))# 处理函数嵌入

        # Hidden layers
        for idx, hlayer in enumerate(self.hidden):
            # 遍历8个隐藏层
            h = self.hdropout(F.elu(hlayer(h))) # 处理行级特征
            h_func = self.hdropout(F.elu(hlayer(h_func))) # 处理函数级特征
        h = self.fc2(h) # 行级预测输出
        h_func = self.fc2(
            h_func
        )  ## 方法级预测输出，与行级预测共享权重

        if self.hparams.methodlevel:
            # 如果是方法级预测模式
            g.ndata["h"] = h  # 将行级特征存储到图中
            return dgl.mean_nodes(g, "h"), None  # 返回图的平均特征作为方法级预测
        else:
            return h, h_func  ## 返回行级和方法级预测，用于多任务训练

    def shared_step(self, batch, test=False):
        """模型的共享步骤，用于训练、验证和测试阶段。
        """
        logits = self(batch, test) # 执行前向传播，获取预测结果
        if self.hparams.methodlevel:
            # 方法级预测模式
            if self.hparams.nsampling:
                # 检查是否使用了邻居采样，如果是则抛出错误
                # 因为方法级预测需要整个图的信息，不能使用邻居采样
                raise ValueError("Cannot train on method level with nsampling.")
            # 从图中提取最大的 _VULN 值作为方法级标签
            # 这里使用 max_nodes 是因为方法级标签是整个函数的标签，取最大值表示函数是否包含漏洞
            labels = dgl.max_nodes(batch, "_VULN").long()
            labels_func = None # 方法级模式下不需要单独的方法级标签
        else:
            # 行级预测模式
            if self.hparams.nsampling and not test:
                # 如果使用了邻居采样且不是测试模式
                # 从采样后的子图中提取目标节点的标签
                labels = batch[2][-1].dstdata["_VULN"].long()# 行级标签
                labels_func = batch[2][-1].dstdata["_FVULN"].long() # 方法级标签
            else:
                # 非邻居采样模式或测试模式
                # 从完整图中提取所有节点的标签
                labels = batch.ndata["_VULN"].long() # 行级标签
                labels_func = batch.ndata["_FVULN"].long() # 方法级标签
        return logits, labels, labels_func  # 返回预测结果、行级标签和方法级标签

    def training_step(self, batch, batch_idx):
        """模型的训练步骤。
        """
        logits, labels, labels_func = self.shared_step(
            batch
        )  
        # 调用 shared_step 方法获取预测结果和标签
        # print(logits.argmax(1), labels_func)  # 调试用，打印预测结果和方法级标签
        loss1 = self.loss(logits[0], labels)   # 计算行级预测的损失
        if not self.hparams.methodlevel:
            loss2 = self.loss_f(logits[1], labels_func)   # 计算方法级预测的损失
        # Need some way of combining the losses for multitask training
        # 多任务训练的损失组合方式
        loss = 0 # 初始化总损失
        if "line" in self.hparams.multitask:
            # 如果多任务包含行级预测
            loss1 = self.loss(logits[0], labels) # 重新计算行级损失
            loss += loss1  # 加到总损失中
        if "method" in self.hparams.multitask and not self.hparams.methodlevel:
            # 如果多任务包含方法级预测且不是纯方法级模式
            loss2 = self.loss(logits[1], labels_func)# 计算方法级损失
            loss += loss2   # 加到总损失中
        # 选择要用于评估的预测结果
        # 如果多任务模式是纯方法级，则使用方法级预测，否则使用行级预测
        logits = logits[1] if self.hparams.multitask == "method" else logits[0]
        pred = F.softmax(logits, dim=1)
        # 对预测结果应用 softmax，得到概率分布
        # 取最大概率的类别作为预测结果

        acc = self.accuracy(pred.argmax(1), labels)# 计算准确率
        if not self.hparams.methodlevel:
            acc_func = self.accuracy(logits.argmax(1), labels_func)
            # 计算方法级准确率
        mcc = self.mcc(pred.argmax(1), labels)
        # 计算 Matthews 相关系数
        # print(pred.argmax(1), labels)  # 调试用，打印预测结果和标签

        # 获取batch_size
        if self.hparams.nsampling:
            # 如果使用了邻居采样，batch_size 就是标签的数量
            batch_size = labels.shape[0]
        else:
            # 否则，batch_size 是图中节点的数量
            batch_size = batch.number_of_nodes()
        #   记录训练损失到日志
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        # 记录训练准确率到日志
        self.log("train_acc", acc, prog_bar=True, logger=True, batch_size=batch_size)
        if not self.hparams.methodlevel:
            # 记录方法级准确率到日志
            self.log("train_acc_func", acc_func, prog_bar=True, logger=True, batch_size=batch_size)
        # 记录 Matthews 相关系数到日志
        self.log("train_mcc", mcc, prog_bar=True, logger=True, batch_size=batch_size)
        return loss # 返回总损失，用于反向传播

    def validation_step(self, batch, batch_idx):
        """模型的验证步骤。
        """
        # 调用 shared_step 方法获取预测结果和标签
        logits, labels, labels_func = self.shared_step(batch)
        loss = 0 # 初始化总损失
        if "line" in self.hparams.multitask:
            # 如果多任务包含行级预测，计算行级损失
            loss1 = self.loss(logits[0], labels)
            loss += loss1  # 加到总损失中
        if "method" in self.hparams.multitask:
            # 如果多任务包含方法级预测，计算方法级损失
            loss2 = self.loss_f(logits[1], labels_func)
            loss += loss2  # 加到总损失中

        # 选择要用于评估的预测结果
        # 如果多任务模式是纯方法级，则使用方法级预测，否则使用行级预测
        logits = logits[1] if self.hparams.multitask == "method" else logits[0]
        # 对预测结果应用 softmax，得到概率分布
        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        mcc = self.mcc(pred.argmax(1), labels)

        # 获取batch_size
        if self.hparams.nsampling:
            # 如果使用了邻居采样，batch_size 就是标签的数量
            batch_size = labels.shape[0]
        else:
            # 否则，batch_size 是图中节点的数量
            batch_size = batch.number_of_nodes()

        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.auroc.update(logits[:, 1], labels)
        self.log("val_auroc", self.auroc, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val_acc", acc, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val_mcc", mcc, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        logits, labels, _ = self.shared_step(
            batch, True
        )  # TODO: 使其支持多任务

        if self.hparams.methodlevel:
            # 方法级标签就是原始标签
            # 返回方法级预测、标签和解批后的图
            labels_f = labels
            return logits[0], labels_f, dgl.unbatch(batch)
        # 非方法级预测模式（行级或多任务）
        # 将行级预测概率存储到图的节点数据中
        batch.ndata["pred"] = F.softmax(logits[0], dim=1)
        # 将方法级预测概率存储到图的节点数据中
        batch.ndata["pred_func"] = F.softmax(logits[1], dim=1)
        logits_f = [] # 存储方法级预测的列表
        labels_f = []
        preds = []  # 存储详细预测结果的列表
        # 遍历解批后的图（每个子图对应一个函数/方法）
        for i in dgl.unbatch(batch):
            # 为每个子图添加详细预测信息
            preds.append(
                [
                    list(i.ndata["pred"].detach().cpu().numpy()),
                    list(i.ndata["_VULN"].detach().cpu().numpy()),
                    i.ndata["pred_func"].argmax(1).detach().cpu(),
                    list(i.ndata["_LINE"].detach().cpu().numpy()),
                ]
            )
            # 计算方法级预测（使用平均池化）
            logits_f.append(dgl.mean_nodes(i, "pred_func").detach().cpu())
            # 计算方法级标签（使用平均池化）
            labels_f.append(dgl.mean_nodes(i, "_FVULN").detach().cpu())
        # 返回：
        # 1. [行级预测, 方法级预测]
        # 2. [行级标签, 方法级标签]
        # 3. 详细预测结果列表
        return [logits[0], logits_f], [labels, labels_f], preds

    def test_epoch_end(self, outputs):
        """计算整个测试集的评估指标。      
        参数:
            outputs: 测试步骤的输出结果列表
        """
        # 初始化存储预测和真实标签的变量
        all_pred = th.empty((0, 2)).long().cuda()
        # 存储所有行级预测
        all_true = th.empty((0)).long().cuda()# 存储所有行级真实标签
        all_pred_f = [] # 存储所有方法级预测
        all_true_f = [] # 存储所有方法级真实标签
        all_funcs = []  # 存储所有函数的预测信息
        from importlib import reload # 导入模块重新加载功能

        reload(lvdgne)
        reload(ml)
        if self.hparams.methodlevel:
            # 方法级预测模式
            for out in outputs:
                # 收集方法级预测和标签
                all_pred_f += out[0]  # 添加方法级预测
                all_true_f += out[1]  # 添加方法级标签
                # 处理每个图
                for idx, g in enumerate(out[2]):
                    # 收集行级真实标签
                    all_true = th.cat([all_true, g.ndata["_VULN"]])
                    # 初始化 GNN 解释的预测结果
                    gnnelogits = th.zeros((g.number_of_nodes(), 2), device="cuda")
                    gnnelogits[:, 0] = 1
                    if out[1][idx] == 1:
                        # 如果真实标签是漏洞
                        zeros = th.zeros(g.number_of_nodes(), device="cuda")
                        importance = th.ones(g.number_of_nodes(), device="cuda")
                        # 初始化重要性为1
                        try:
                            if out[1][idx] == 1:
                                importance = lvdgne.get_node_importances(self, g)
                            importance = importance.unsqueeze(1)
                            # 增加维度
                            # 构造预测结果，使用重要性作为漏洞概率
                            gnnelogits = th.cat([zeros.unsqueeze(1), importance], dim=1)
                        except Exception as E:
                            print(E)
                            pass
                    # 添加 GNN 解释的预测结果
                    all_pred = th.cat([all_pred, gnnelogits])
                    # 生成函数级预测（重复方法级预测到每个节点）
                    func_pred = out[0][idx].argmax().repeat(g.number_of_nodes())
                    # 添加函数预测信息
                    all_funcs.append(
                        [
                            gnnelogits.detach().cpu().numpy(), # 预测概率
                            g.ndata["_VULN"].detach().cpu().numpy(), # 真实标签
                            func_pred.detach().cpu(),  # 函数级预测
                        ]
                    )
            all_true = all_true.long()  # 确保标签类型为长整型
        else:
            # 非方法级预测模式（行级或多任务）
            for out in outputs:
                # 收集行级预测和标签
                all_pred = th.cat([all_pred, out[0][0]])
                all_true = th.cat([all_true, out[1][0]])
                # 收集方法级预测和标签
                all_pred_f += out[0][1]
                all_true_f += out[1][1]
                # 收集函数预测信息
                all_funcs += out[2]
        # 处理预测结果
        all_pred = F.softmax(all_pred, dim=1) # 对行级预测应用 softmax
        all_pred_f = F.softmax(th.stack(all_pred_f).squeeze(), dim=1) # 对方法级预测应用 softmax
        all_true_f = th.stack(all_true_f).squeeze().long() 
        self.all_funcs = all_funcs
        self.all_true = all_true
        self.all_pred = all_pred
        self.all_pred_f = all_pred_f
        self.all_true_f = all_true_f

        # 自定义排名准确率（包含负样本）
        self.res1 = ivde.eval_statements_list(all_funcs)

        # 自定义排名准确率（仅包含正样本）
        self.res1vo = ivde.eval_statements_list(all_funcs, vo=True, thresh=0)

        # 常规指标
        multitask_pred = []
        multitask_true = []
        for af in all_funcs:
            # 构建行级预测和函数级预测的组合
            line_pred = list(zip(af[0], af[2]))
            # 根据函数级预测调整行级预测
            multitask_pred += [list(i[0]) if i[1] == 1 else [1, 0] for i in line_pred]
            # 添加真实标签
            multitask_true += list(af[1])
        # 保存多任务预测和标签
        self.linevd_pred = multitask_pred
        self.linevd_true = multitask_true
        #multitask_true = th.LongTensor(multitask_true)
        # 转换为张量
        import numpy as np
        multitask_true = th.LongTensor(np.array(multitask_true).astype(int))
        multitask_pred = th.Tensor(multitask_pred)
        # 计算最佳 F1 阈值
        self.f1thresh = ml.best_f1(multitask_true, [i[1] for i in multitask_pred])
        # 计算多任务指标
        self.res2mt = ml.get_metrics_logits(multitask_true, multitask_pred)
        # 计算行级指标
        self.res2 = ml.get_metrics_logits(all_true, all_pred)
        # 计算方法级指标
        self.res2f = ml.get_metrics_logits(all_true_f, all_pred_f)

        # 排名指标
        # 所有样本的排名指标
        rank_metrs = []
        # 仅正样本的排名指标
        rank_metrs_vo = []
        for af in all_funcs:
            # 计算排名指标
            rank_metr_calc = svdr.rank_metr([i[1] for i in af[0]], af[1], 0)
            if max(af[1]) > 0:  # 如果是正样本
                rank_metrs_vo.append(rank_metr_calc)
            rank_metrs.append(rank_metr_calc)
        try:
            # 计算所有样本的平均排名指标
            self.res3 = ml.dict_mean(rank_metrs)
        except Exception as E:
            print(E)
            pass
        # 计算仅正样本的平均排名指标
        self.res3vo = ml.dict_mean(rank_metrs_vo)

        # 从语句级别预测方法级别
        method_level_pred = []
        method_level_true = []
        for af in all_funcs:
            # 计算方法级真实标签（如果有任何漏洞行，则为漏洞）
            method_level_true.append(1 if sum(af[1]) > 0 else 0)
            pred_method = 0  # 默认预测为安全
            # 如果任何行预测为漏洞，则方法级预测为漏洞
            for logit in af[0]:
                if logit[1] > 0.5:
                    pred_method = 1
                    break
            method_level_pred.append(pred_method)
        # 计算从行级预测方法级的指标
        self.res4 = ml.get_metrics(method_level_true, method_level_pred)

        return

    def plot_pr_curve(self):
        """绘制正类的精确率-召回率曲线（测试后调用）。
        该方法使用测试集的预测结果和真实标签来绘制精确率-召回率曲线，
        用于评估模型在不同阈值下的性能表现。
        """
        precision, recall, thresholds = precision_recall_curve(
            self.linevd_true, [i[1] for i in self.linevd_pred]
        )
        disp = PrecisionRecallDisplay(precision, recall)
        disp.plot()
        return

    def configure_optimizers(self):
        """配置模型的优化器。
        """
        return th.optim.AdamW(self.parameters(), lr=self.lr)


def get_relevant_metrics(trial_result):
    """从试验结果中提取相关指标。
    
    参数:
        trial_result: 试验结果，包含模型训练和测试的所有指标
    
    返回:
        dict: 包含关键评估指标的字典
    """
    ret = {}
    ret["trial_id"] = trial_result[0]
    ret["checkpoint"] = trial_result[1]
    ret["acc@5"] = trial_result[2][5]  # 前5名准确率
    ret["stmt_f1"] = trial_result[3]["f1"]  # 语句级F1分数
    ret["stmt_rec"] = trial_result[3]["rec"]  # 语句级召回率
    ret["stmt_prec"] = trial_result[3]["prec"]  # 语句级精确率
    ret["stmt_mcc"] = trial_result[3]["mcc"]  # 语句级Matthews相关系数
    ret["stmt_fpr"] = trial_result[3]["fpr"]  # 语句级假阳性率
    ret["stmt_fnr"] = trial_result[3]["fnr"]  # 语句级假阴性率
    ret["stmt_rocauc"] = trial_result[3]["roc_auc"]  # 语句级ROC曲线下面积
    ret["stmt_prauc"] = trial_result[3]["pr_auc"]  # 语句级PR曲线下面积
    ret["stmt_prauc_pos"] = trial_result[3]["pr_auc_pos"]  # 语句级正样本PR曲线下面积
    ret["func_f1"] = trial_result[4]["f1"]  # 方法级F1分数
    ret["func_rec"] = trial_result[4]["rec"]  # 方法级召回率
    ret["func_prec"] = trial_result[4]["prec"]  # 方法级精确率
    ret["func_mcc"] = trial_result[4]["mcc"]  # 方法级Matthews相关系数
    ret["func_fpr"] = trial_result[4]["fpr"]  # 方法级假阳性率
    ret["func_fnr"] = trial_result[4]["fnr"]  # 方法级假阴性率
    ret["func_rocauc"] = trial_result[4]["roc_auc"]  # 方法级ROC曲线下面积
    ret["func_prauc"] = trial_result[4]["pr_auc"]  # 方法级PR曲线下面积
    ret["MAP@5"] = trial_result[5]["MAP@5"]  # 前5名平均精确率
    ret["nDCG@5"] = trial_result[5]["nDCG@5"]  # 前5名归一化折现累积增益
    ret["MFR"] = trial_result[5]["MFR"]  # 平均故障排名
    ret["MAR"] = trial_result[5]["MAR"]  # 平均绝对排名
    ret["stmtline_f1"] = trial_result[6]["f1"]  # 语句行级F1分数
    ret["stmtline_rec"] = trial_result[6]["rec"]  # 语句行级召回率
    ret["stmtline_prec"] = trial_result[6]["prec"]  # 语句行级精确率
    ret["stmtline_mcc"] = trial_result[6]["mcc"]  # 语句行级Matthews相关系数
    ret["stmtline_fpr"] = trial_result[6]["fpr"]  # 语句行级假阳性率
    ret["stmtline_fnr"] = trial_result[6]["fnr"]  # 语句行级假阴性率
    ret["stmtline_rocauc"] = trial_result[6]["roc_auc"]  # 语句行级ROC曲线下面积
    ret["stmtline_prauc"] = trial_result[6]["pr_auc"]  # 语句行级PR曲线下面积
    ret["stmtline_prauc_pos"] = trial_result[6]["pr_auc_pos"]  # 语句行级正样本PR曲线下面积

    ret = {k: round(v, 3) if isinstance(v, float) else v for k, v in ret.items()}  # 四舍五入到小数点后3位
    ret["learning_rate"] = trial_result[7]  # 学习率
    ret["stmt_loss"] = trial_result[3]["loss"]  # 语句级损失
    ret["func_loss"] = trial_result[4]["loss"]  # 方法级损失
    ret["stmtline_loss"] = trial_result[6]["loss"]  # 语句行级损失
    return ret
