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
