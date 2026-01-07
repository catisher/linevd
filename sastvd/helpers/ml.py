import pickle as pkl
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.tensorboard import SummaryWriter


def best_f1(true, pos_logits):
    """找到F1分数的最佳阈值。

    参数:
        true (list): 真实标签列表，例如 [1, 0, 0, 1]
        pos_logits (list): 正类的预测概率列表，例如 [0.27292988, 0.27282527, 0.7942509, 0.20574914]

    返回:
        float: 最佳F1分数对应的阈值
    """
    precision, recall, thresholds = precision_recall_curve(true, pos_logits)
    thresh_scores = []
    for i in range(len(thresholds)):
        if precision[i] + recall[i] == 0:
            continue
        f1 = (2 * (precision[i] * recall[i])) / (precision[i] + recall[i])
        thresh = thresholds[i]
        thresh_scores.append([f1, thresh])
    thresh_scores = sorted(thresh_scores, reverse=True)
    thresh_scores = [i for i in thresh_scores if i[0] > 0]
    return thresh_scores[0][-1]


def get_metrics(true, pred):
    """根据真实标签和预测结果计算相关的评估指标。

    参数:
        true (list): 真实标签列表
        pred (list): 预测结果列表

    返回:
        dict: 包含各项评估指标的字典，包括准确率(acc)、F1分数(f1)、召回率(rec)、精确率(prec)、
              马修斯相关系数(mcc)、误报率(fpr)和漏报率(fnr)
    """
    metrics = {}
    metrics["acc"] = accuracy_score(true, pred)
    metrics["f1"] = f1_score(true, pred, zero_division=0)
    metrics["rec"] = recall_score(true, pred, zero_division=0)
    metrics["prec"] = precision_score(true, pred, zero_division=0)
    metrics["mcc"] = matthews_corrcoef(true, pred)
    metrics["fpr"] = -1
    metrics["fnr"] = -1
    if sum(true + pred) != 0:
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        if fp + tn != 0:
            metrics["fpr"] = fp / (fp + tn)
        if fn + tp != 0:
            metrics["fnr"] = fn / (fn + tp)
    return metrics


def get_metrics_logits(true, logits):
    """从模型的logits计算评估指标，内部调用get_metrics函数。

    参数:
        true (list or torch.Tensor): 真实标签
        logits (torch.Tensor): 模型的logits输出

    返回:
        dict: 包含各项评估指标的字典，包括准确率(acc)、F1分数(f1)、召回率(rec)、精确率(prec)、
              马修斯相关系数(mcc)、误报率(fpr)、漏报率(fnr)、ROC曲线下面积(roc_auc)、
              PR曲线下面积(pr_auc)、正类PR曲线下面积(pr_auc_pos)和损失值(loss)
    """
    loss = F.cross_entropy(logits, true).detach().cpu().item()
    if torch.is_tensor(true):
        true_oh = torch.nn.functional.one_hot(true).detach().cpu().numpy()
        true = true.detach().cpu().numpy()
    if torch.is_tensor(logits):
        sm_logits = torch.nn.functional.softmax(logits, dim=1)
        pos_logits = sm_logits[:, 1].detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
    f1_threshold = best_f1(true, pos_logits)
    pred = [1 if i > f1_threshold else 0 for i in pos_logits]
    try:
        roc_auc = roc_auc_score(true, logits[:, 1])
    except:
        roc_auc = -1
    try:
        pr_auc = average_precision_score(true_oh, logits)
    except:
        pr_auc = -1
    ret = get_metrics(true, pred)
    ret["roc_auc"] = roc_auc
    ret["pr_auc"] = pr_auc
    ret["pr_auc_pos"] = average_precision_score(true, logits[:, 1])
    ret["loss"] = loss
    return ret


def met_dict_to_str(md, prefix="", verbose=1):
    """将指标字典转换为字符串以便打印输出。

    参数:
        md (dict): 包含评估指标的字典
        prefix (str): 打印字符串的前缀，默认为空
        verbose (int): 详细程度，1表示打印输出，0表示不打印

    返回:
        str: 格式化后的指标字符串
    """
    ret_str = prefix
    for k, v in md.items():
        if k == "loss":
            ret_str += k + ": " + "%.5f" % v + " | "
        else:
            ret_str += k + ": " + "%.3f" % v + " | "
    if verbose > 0:
        print("\x1b[40m\x1b[37m" + ret_str[:-1] + "\x1b[0m")
    return ret_str


def met_dict_to_writer(md, step, writer, prefix):
    """将评估指标字典写入Tensorboard日志文件。

    参数:
        md (dict): 包含评估指标的字典
        step (int): 当前训练步数
        writer (SummaryWriter): Tensorboard的SummaryWriter实例
        prefix (str): 写入日志时的前缀
    """
    for k, v in md.items():
        writer.add_scalar(f"{prefix}/{k}", v, step)


def print_seperator(strings: list, max_len: int):
    """在一行中打印带有"="分隔符的文本，总长度不超过指定的最大长度。

    参数:
        strings (list): 要打印的字符串列表
        max_len (int): 输出的最大长度
    """
    midpoints = int(max_len / len(strings))
    strings = [str(i) for i in strings]
    final_str = ""
    cutoff = max_len + (9 * len(strings))
    for s in strings:
        if "\x1b" in s:
            cutoff += 9
        len_s = len(s.replace("\x1b[32m", "").replace("\x1b[39m", ""))
        final_str += "\x1b[40m"
        final_str += "=" * (int((midpoints / 2) - int(len_s / 2)) - 1)
        final_str += f" {s} "
        final_str += "=" * (int((midpoints / 2) - int(len_s / 2)) - 1)
        final_str += "\x1b[0m"
    print(final_str[:cutoff])


def dict_mean(dict_list):
    """计算字典列表中每个键值的平均值。

    参考: https://stackoverflow.com/questions/29027792

    参数:
        dict_list (list): 字典列表，所有字典应具有相同的键

    返回:
        dict: 包含每个键平均值的字典
    """
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list if not np.isnan(d[key])) / len(
            [d[key] for d in dict_list if not np.isnan(d[key])]
        )
    return mean_dict


class LogWriter:
    """用于记录PyTorch模型性能的日志记录器类。"""

    def __init__(
        self,
        model,
        path: str,
        max_patience: int = 100,
        log_every: int = 10,
        val_every: int = 50,
    ):
        """初始化日志记录器。

        参数:
            model: PyTorch模型实例
            path (str): 保存日志文件的路径
            max_patience (int): 早停的最大耐心值，默认为100
            log_every (int): 日志记录的频率，每多少步记录一次，默认为10
            val_every (int): 验证的频率，每多少步验证一次，默认为50
        """
        self._model = model
        self._best_val_loss = 100
        self._patience = 0
        self._max_patience = max_patience
        self._epoch = 0
        self._step = 0
        self._path = Path(path)
        self._writer = SummaryWriter(path)
        self._log_every = log_every
        self._val_every = val_every
        self.save_attrs = ["_best_val_loss", "_patience", "_epoch", "_step"]

    def log(self, train_mets, val_mets):
        """记录训练和验证的指标信息。

        参数:
            train_mets (dict): 训练集上的评估指标
            val_mets (dict): 验证集上的评估指标
        """
        if self._step % self._log_every != 0:
            self.step()
            return

        if not self.log_val():
            met_dict_to_str(train_mets, "TR = ")
            met_dict_to_writer(train_mets, self._step, self._writer, "TRN")
            self.step()
            return

        val_loss = val_mets["loss"]
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            with open(self._path / "best.model", "wb") as f:
                torch.save(self._model.state_dict(), f)
            best_model_string = "Best model saved: %.3f" % val_loss
            best_model_string = f"\x1b[32m{best_model_string}\x1b[39m"
            self._patience = 0
        else:
            self._patience += 1
            best_model_string = "No improvement."
        print_seperator(
            [
                f"Patience: {self._patience:03d}",
                f"Epoch: {self._epoch:03d}",
                f"Step: {self._step:03d}",
                best_model_string,
            ],
            131,
        )
        met_dict_to_str(train_mets, "TR = ")
        met_dict_to_writer(train_mets, self._step, self._writer, "TRN")
        met_dict_to_str(val_mets, "VA = ")
        met_dict_to_writer(val_mets, self._step, self._writer, "VAL")
        self.step()

    def test(self, test_mets):
        """记录测试集上的评估指标。

        参数:
            test_mets (dict): 测试集上的评估指标
        """
        print_seperator(["\x1b[36mTest Set\x1b[39m"], 135)
        met_dict_to_str(test_mets, "TS = ")

    def log_val(self):
        """检查是否应该进行验证。

        返回:
            bool: 如果当前步数是验证步数的倍数，则返回True，否则返回False
        """
        if self._step % self._val_every == 0:
            return True
        return False

    def step(self):
        """增加当前训练步数。"""
        self._step += 1

    def epoch(self):
        """增加当前训练轮数。"""
        self._epoch += 1

    def stop(self):
        """检查是否应该停止训练（早停条件）。

        返回:
            bool: 如果当前耐心值超过最大耐心值，则返回True（停止训练），否则返回False
        """
        return self._patience > self._max_patience

    def load_best_model(self):
        """加载保存的最佳模型权重到当前模型实例。"""
        torch.cuda.empty_cache()
        self._model.load_state_dict(torch.load(self._path / "best.model"))

    def save_logger(self):
        """保存日志记录器的状态和当前模型权重。"""
        with open(self._path / "log.pkl", "wb") as f:
            f.write(pkl.dumps(dict([(i, getattr(self, i)) for i in self.save_attrs])))
        with open(self._path / "current.model", "wb") as f:
            torch.save(self._model.state_dict(), f)

    def load_logger(self):
        """加载之前保存的日志记录器状态和模型权重。"""
        with open(self._path / "log.pkl", "rb") as f:
            attrs = pkl.load(f)
            for k, v in attrs.items():
                setattr(self, k, v)
        torch.cuda.empty_cache()
        self._model.load_state_dict(torch.load(self._path / "current.model"))
