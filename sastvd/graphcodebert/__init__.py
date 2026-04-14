import os

import matplotlib.pyplot as plt
import sastvd as svd
import torch
from transformers import AutoModel, AutoTokenizer
from tsne_torch import TorchTSNE as TSNE


class GraphCodeBert:
    """GraphCodeBert 代码嵌入模型类

    该类封装了 Microsoft GraphCodeBERT 预训练模型，用于将代码片段编码为向量表示
    支持代码-结构对的输入格式，能更好地捕获代码的结构信息
    
    Example:
    gcb = GraphCodeBert()
    code = ["int myfunciscool(float b) { return 1; }", "int main"]
    structure = ["int <line> float <line> return <line>", "int <line>"]
    ret = gcb.encode(code, structure)
    ret.shape
    >>> torch.Size([2, 768])
    """

    def __init__(self):
        """初始化 GraphCodeBert 模型
        
        优先从本地加载模型，如果本地不存在则从 Hugging Face 下载并缓存
        自动选择可用的设备（GPU或CPU）
        """
        # 检查本地是否存在 GraphCodeBert 模型
        graphcodebert_base_path = svd.external_dir() / "graphcodebert-base"
        if os.path.exists(graphcodebert_base_path):
            # 从本地路径加载 tokenizer 和模型
            self.tokenizer = AutoTokenizer.from_pretrained(graphcodebert_base_path)
            self.model = AutoModel.from_pretrained(graphcodebert_base_path)
            
        else:
            # 设置缓存目录
            cache_dir = svd.get_dir(svd.cache_dir() / "graphcodebert_model")
            print("Loading GraphCodeBERT...")
            # 从 Hugging Face 加载预训练模型
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/graphcodebert-base", cache_dir=cache_dir
            )
            self.model = AutoModel.from_pretrained(
                "microsoft/graphcodebert-base", cache_dir=cache_dir
            )
        
        # 选择计算设备（GPU优先）
        #self._dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._dev = torch.device("cpu")
        # 将模型移动到指定设备
        self.model.to(self._dev)

    def generate_structure_info(self, code):
        """为代码生成结构信息
        
        基于代码的缩进和结构生成简单的结构信息
        
        Args:
            code: 代码字符串
            
        Returns:
            str: 结构信息字符串
        """
        lines = code.split('\n')
        structure = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 提取关键字和结构信息
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
        return ' '.join(structure)

    def encode(self, code_sents: list, structure_sents: list = None):
        """将代码片段列表编码为 GraphCodeBERT 向量表示
        
        Args:
            code_sents: 代码片段字符串列表
            structure_sents: 结构信息字符串列表（可选）
            
        Returns:
            torch.Tensor: 形状为 [batch_size, 768] 的代码向量
        """
        # 配置 tokenizer 参数
        tk_args = {
            "padding": True,      # 自动填充到相同长度
            "truncation": True,   # 超过最大长度时截断
            "return_tensors": "pt" # 返回 PyTorch 张量
        }
        
        # 分词并移动到计算设备
        if structure_sents is not None:
            # 使用代码-结构对编码
            tokens = self.tokenizer(
                code_sents, 
                structure_sents, 
                **tk_args
            ).to(self._dev)
        else:
            # 仅使用代码编码
            tokens = self.tokenizer(code_sents, **tk_args).to(self._dev)
        
        # 禁用梯度计算，提高效率
        with torch.no_grad():
            # 获取模型输出，使用 [1] 索引获取池化后的输出 (CLS token 的输出)
            return self.model(tokens["input_ids"], tokens["attention_mask"])[1]


def generate_structure_info(code):
    """为代码生成结构信息
    
    基于代码的缩进和结构生成简单的结构信息
    
    Args:
        code: 代码字符串
        
    Returns:
        str: 结构信息字符串
    """
    lines = code.split('\n')
    structure = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 提取关键字和结构信息
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
    return ' '.join(structure)


def plot_embeddings(embeddings, words):
    """可视化代码向量嵌入
    
    使用 t-SNE 算法将高维向量降至 2 维并可视化
    
    Args:
        embeddings: 代码向量嵌入，形状为 [n, 768]
        words: 与嵌入对应的代码片段列表
    
    Example:
        import sastvd.helpers.datasets as svdd
        gcb = GraphCodeBert()
        df = svdd.bigvul()
        code = df.sample(5).before.tolist()
        structure = [generate_structure_info(c) for c in code]
        embeddings = gcb.encode(code, structure)
        plot_embeddings(embeddings, code)
    """
    # 初始化 t-SNE 模型，降至 2 维空间
    tsne = TSNE(n_components=2, n_iter=2000, verbose=True)
    
    # 将嵌入向量降至 2 维
    Y = tsne.fit_transform(embeddings)
    
    # 提取 x 和 y 坐标
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    
    # 创建散点图
    plt.scatter(x_coords, y_coords)
    
    # 为每个点添加标签
    for label, x, y in zip(words, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
    
    # 设置坐标轴范围，稍微扩大以避免点被截断
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    
    # 显示图形
    plt.show()
