"""金字塔池化（Pyramid Pooling）实现。

该文件实现了两种金字塔池化技术：
1. 空间金字塔池化（Spatial Pyramid Pooling）：用于处理图像等二维数据
2. 时间金字塔池化（Temporal Pyramid Pooling）：用于处理序列等一维时序数据

金字塔池化能够将不同尺寸的输入特征图转换为固定长度的向量表示，
是一种常用的特征融合和降维技术，常用于CNN和RNN等模型中。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPooling(nn.Module):
    """通用金字塔池化类。
    
    默认使用空间金字塔池化，并包含空间和时间池化的静态方法。
    该类是空间金字塔池化和时间金字塔池化的基类。
    """
    
    def __init__(self, levels, mode="max"):
        """初始化金字塔池化层。
        
        参数
        ----------
        levels : list
            定义在宽度和（空间）高度维度上的不同划分级别
        mode : str, 可选
            定义要使用的底层池化模式，可以是 "max"（最大池化）或 "avg"（平均池化），默认为 "max"
        """
        super(PyramidPooling, self).__init__()
        self.levels = levels
        self.mode = mode

    def forward(self, x):
        """前向传播方法。
        
        默认使用空间金字塔池化处理输入特征图。
        
        参数
        ----------
        x : torch.Tensor
            输入特征图张量
        
        返回
        -------
        torch.Tensor
            池化后的特征向量，形状为 [batch x n]
            其中 n 是各层级池化特征的总和
        """
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        """计算输出特征向量的大小。
        
        根据输入过滤器数量和层级计算空间金字塔池化后的输出大小。
        
        参数
        ----------
        filters : int
            输入特征图的过滤器数量（通道数）
        
        返回
        -------
        int
            池化后的特征向量维度大小
        """
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out

    @staticmethod
    def spatial_pyramid_pool(previous_conv, levels, mode):
        """静态空间金字塔池化方法。
        
        根据给定的层级将输入张量在垂直和水平方向（最后2个维度）上进行划分，
        并根据给定的模式对每个划分区域进行池化操作。
        
        参数
        ----------
        previous_conv : torch.Tensor
            前一个卷积层的输出张量，形状为 [batch x channels x height x width]
        levels : list
            定义在宽度和高度维度上的不同划分级别
        mode : str
            定义要使用的池化模式，可以是 "max"（最大池化）或 "avg"（平均池化）
        
        返回
        -------
        torch.Tensor
            池化后的特征向量，形状为 [batch x n]
            其中 n 是 sum(filter_amount*level*level) 对每个 level 在 levels 中的总和
            这是多级池化的浓缩表示
        """
        num_sample = previous_conv.size(0)
        # 获取输入特征图的高度和宽度
        previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
        
        # 遍历每个层级，执行不同级别的池化
        for i in range(len(levels)):
            # 计算当前层级的池化核大小
            h_kernel = int(math.ceil(previous_conv_size[0] / levels[i]))
            w_kernel = int(math.ceil(previous_conv_size[1] / levels[i]))
            
            # 计算需要填充的大小，以确保输入能被均匀划分为 level x level 个区域
            w_pad1 = int(math.floor((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(math.ceil((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            h_pad1 = int(math.floor((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            h_pad2 = int(math.ceil((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            
            # 验证填充大小是否正确
            assert w_pad1 + w_pad2 == (
                w_kernel * levels[i] - previous_conv_size[1]
            ) and h_pad1 + h_pad2 == (h_kernel * levels[i] - previous_conv_size[0])

            # 对输入特征图进行填充
            padded_input = F.pad(
                input=previous_conv,
                pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                mode="constant",
                value=0,
            )
            
            # 根据指定的模式选择池化操作
            if mode == "max":
                pool = nn.MaxPool2d(
                    (h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0)
                )
            elif mode == "avg":
                pool = nn.AvgPool2d(
                    (h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0)
                )
            else:
                raise RuntimeError(
                    '未知的池化类型: %s, 请使用 "max" 或 "avg".'
                )
            
            # 执行池化操作
            x = pool(padded_input)
            
            # 将池化结果展平并拼接
            if i == 0:
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)

        return spp

    @staticmethod
    def temporal_pyramid_pool(previous_conv, out_pool_size, mode):
        """静态时间金字塔池化方法。
        
        根据给定的层级将输入张量在水平方向（最后一个维度）上进行划分，
        并根据给定的模式对每个划分区域进行池化操作。
        
        换句话说：它将输入张量划分为 "level" 个水平条带，每个条带的宽度大致为 (previous_conv.size(3) / level)，
        保持原始高度不变，并对每个条带内的值进行池化。
        
        参数
        ----------
        previous_conv : torch.Tensor
            前一个卷积层的输出张量，形状为 [batch x channels x height x width]
        out_pool_size : list
            定义在宽度维度上的不同划分级别
        mode : str
            定义要使用的池化模式，可以是 "max"（最大池化）或 "avg"（平均池化）
        
        返回
        -------
        torch.Tensor
            池化后的特征向量，形状为 [batch x n]
            其中 n 是 sum(filter_amount*level) 对每个 level 在 levels 中的总和
            这是多级池化的浓缩表示
        """
        num_sample = previous_conv.size(0)
        # 获取输入特征图的高度和宽度
        previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
        
        # 遍历每个层级，执行不同级别的池化
        for i in range(len(out_pool_size)):
            # 时间池化保持高度不变，仅在宽度方向上划分
            h_kernel = previous_conv_size[0]
            w_kernel = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            
            # 计算需要填充的大小，以确保输入能被均匀划分为 level 个区域
            w_pad1 = int(
                math.floor((w_kernel * out_pool_size[i] - previous_conv_size[1]) / 2)
            )
            w_pad2 = int(
                math.ceil((w_kernel * out_pool_size[i] - previous_conv_size[1]) / 2)
            )
            
            # 验证填充大小是否正确
            assert w_pad1 + w_pad2 == (
                w_kernel * out_pool_size[i] - previous_conv_size[1]
            )

            # 对输入特征图进行填充（仅在宽度方向）
            padded_input = F.pad(
                input=previous_conv,
                pad=[w_pad1, w_pad2],
                mode="constant",
                value=0,
            )
            
            # 根据指定的模式选择池化操作
            if mode == "max":
                pool = nn.MaxPool2d(
                    (h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0)
                )
            elif mode == "avg":
                pool = nn.AvgPool2d(
                    (h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0)
                )
            else:
                raise RuntimeError(
                    '未知的池化类型: %s, 请使用 "max" 或 "avg".'
                )
            
            # 执行池化操作
            x = pool(padded_input)
            
            # 将池化结果展平并拼接
            if i == 0:
                tpp = x.view(num_sample, -1)
            else:
                tpp = torch.cat((tpp, x.view(num_sample, -1)), 1)

        return tpp


class SpatialPyramidPooling(PyramidPooling):
    """空间金字塔池化模块。
    
    根据给定的层级将输入张量在水平和垂直方向（最后2个维度）上进行划分，
    并根据给定的模式对每个划分区域进行池化操作。
    
    可以像其他PyTorch模块一样使用，由于是静态池化，因此没有可学习的参数。
    
    换句话说：它将输入张量划分为 level*level 个矩形区域，每个区域的宽度大致为 (previous_conv.size(3) / level)，
    高度大致为 (previous_conv.size(2) / level)，并对每个区域的值进行池化（填充输入以适应大小）。
    """
    
    def __init__(self, levels, mode="max"):
        """初始化空间金字塔池化层。
        
        参数
        ----------
        levels : list
            定义在宽度维度上的不同划分级别
        mode : str, 可选
            定义要使用的底层池化模式，可以是 "max"（最大池化）或 "avg"（平均池化），默认为 "max"
        """
        super(SpatialPyramidPooling, self).__init__(levels, mode=mode)

    def forward(self, x):
        """前向传播方法。
        
        对输入特征图执行空间金字塔池化。
        
        参数
        ----------
        x : torch.Tensor
            输入特征图张量，形状为 [batch x channels x height x width]
        
        返回
        -------
        torch.Tensor
            池化后的特征向量，形状为 [batch x n]
            其中 n 是 sum(filter_amount*level*level) 对每个 level 在 levels 中的总和
            这是多级池化的浓缩表示
        """
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        """计算输出特征向量的大小。
        
        根据输入过滤器数量和层级计算空间金字塔池化后的输出大小。
        可用于将池化后的特征向量调整为全连接层所需的形状。
        
        参数
        ----------
        filters : int
            输入特征图的过滤器数量（通道数）
        
        返回
        -------
        int
            池化后的特征向量维度大小，等于 sum(filter_amount*level*level) 对每个 level 在 levels 中的总和
        """
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out


class TemporalPyramidPooling(PyramidPooling):
    """时间金字塔池化模块。
    
    根据给定的层级将输入张量在水平方向（最后一个维度）上进行划分，
    并根据给定的模式对每个划分区域进行池化操作。
    
    可以像其他PyTorch模块一样使用，由于是静态池化，因此没有可学习的参数。
    
    换句话说：它将输入张量划分为 "level" 个水平条带，每个条带的宽度大致为 (previous_conv.size(3) / level)，
    保持原始高度不变，并对每个条带内的值进行池化。
    """
    
    def __init__(self, levels, mode="max"):
        """初始化时间金字塔池化层。
        
        参数
        ----------
        levels : list
            定义在宽度维度上的不同划分级别
        mode : str, 可选
            定义要使用的底层池化模式，可以是 "max"（最大池化）或 "avg"（平均池化），默认为 "max"
        """
        super(TemporalPyramidPooling, self).__init__(levels, mode=mode)

    def forward(self, x):
        """前向传播方法。
        
        对输入特征图执行时间金字塔池化。
        
        参数
        ----------
        x : torch.Tensor
            输入特征图张量，形状为 [batch x channels x height x width]
        
        返回
        -------
        torch.Tensor
            池化后的特征向量，形状为 [batch x 1 x n]
            其中 n 是 sum(filter_amount*level) 对每个 level 在 levels 中的总和
            这是多级池化的浓缩表示
        """
        return self.temporal_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        """计算输出特征向量的大小。
        
        根据输入过滤器数量和层级计算时间金字塔池化后的输出大小。
        可用于将池化后的特征向量调整为全连接层所需的形状。
        
        参数
        ----------
        filters : int
            输入特征图的过滤器数量（通道数）
        
        返回
        -------
        int
            池化后的特征向量维度大小，等于 sum(filter_amount*level) 对每个 level 在 levels 中的总和
        """
        out = 0
        for level in self.levels:
            out += filters * level
        return out
