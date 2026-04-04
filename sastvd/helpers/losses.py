import torch
import torch.nn.functional as F


class SCELoss(torch.nn.Module):
    """对称交叉熵损失函数(Symmetric Cross Entropy Loss)。

    实现参考：https://github.com/HanxunH/SCELoss-Reproduce/blob/master/loss.py
    """

    def __init__(self, alpha=1, beta=1, num_classes=2):
        """初始化SCELoss类。

        参数:
            alpha (float): 常规交叉熵损失的权重系数
            beta (float): 反向交叉熵损失的权重系数
            num_classes (int): 类别数量，默认为2
        """
        super(SCELoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        """执行损失计算的前向传播。

        参数:
            pred (torch.Tensor): 模型的预测输出，形状为(batch_size, num_classes)
            labels (torch.Tensor): 真实标签，形状为(batch_size,)

        返回:
            torch.Tensor: 计算得到的对称交叉熵损失值
        """
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = (
            torch.nn.functional.one_hot(labels, self.num_classes)
            .float()
            .to(self.device)
        )
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class FocalLoss(torch.nn.Module):
    """Focal Loss 损失函数，用于处理类别不平衡问题。

    Focal Loss 通过降低简单样本的权重，让模型更关注难分类的样本。
    实现参考：Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(self, alpha=None, gamma=2, num_classes=2):
        """初始化FocalLoss类。

        参数:
            alpha (list or None): 类别权重，例如 [1, 5] 表示类别0和类别1的权重
                                如果为None，则不使用类别权重
            gamma (float): 聚焦参数，默认为2。gamma越大，对简单样本的降权越强
                          gamma=0时等同于标准交叉熵损失
            num_classes (int): 类别数量，默认为2
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        
        if alpha is None:
            self.alpha = None
        else:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = torch.tensor([alpha] * num_classes, dtype=torch.float32)

    def forward(self, pred, labels):
        """执行损失计算的前向传播。

        参数:
            pred (torch.Tensor): 模型的预测输出，形状为(batch_size, num_classes)
            labels (torch.Tensor): 真实标签，形状为(batch_size,)

        返回:
            torch.Tensor: 计算得到的Focal Loss值
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(pred, labels, reduction='none')
        
        # 计算预测概率
        p_t = torch.exp(-ce_loss)
        
        # 计算Focal Loss
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        # 应用类别权重
        if self.alpha is not None:
            if self.alpha.device != pred.device:
                self.alpha = self.alpha.to(pred.device)
            at = self.alpha[labels]
            focal_loss = at * focal_loss
        
        return focal_loss.mean()
