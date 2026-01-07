import numpy as np
from sklearn.metrics import roc_auc_score


def precision_at_k(r, k):
    """计算k位置的精确率（precision @ k）。

    相关性是二值的（非零值表示相关）。
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: 相关性评分长度 < k
    参数:
        r: 按排序顺序的相关性评分（列表或numpy数组）
            （第一个元素是第一个项目）
    返回:
        k位置的精确率
    异常:
        ValueError: 评分长度必须大于等于k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError("相关性评分长度 < k")
    return np.mean(r)


def average_precision(r, limit):
    """计算平均精确率（PR曲线下的面积）。

    相关性是二值的（非零值表示相关）。
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    参数:
        r: 按排序顺序的相关性评分（列表或numpy数组）
            （第一个元素是第一个项目）
        limit: 计算平均精确率的项目数量限制
    返回:
        平均精确率
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(limit) if r[k]]
    if not out:
        return 0.0
    return np.mean(out)


def mean_average_precision(rs, k):
    """计算平均精确率的均值（mean average precision）。

    相关性是二值的（非零值表示相关）。
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    参数:
        rs: 相关性评分迭代器（列表或numpy数组），按排序顺序排列
            （第一个元素是第一个项目）
        k: 计算平均精确率的项目数量限制
    返回:
        平均精确率的均值
    """
    return np.mean([average_precision(r, k) for r in rs])


def dcg_at_k(r, k, method=0):
    """计算折扣累积增益（discounted cumulative gain, DCG）。

    相关性是正实数值。可以像之前的方法一样使用二值相关性。
    示例来自：
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    参数:
        r: 按排序顺序的相关性评分（列表或numpy数组）
            （第一个元素是第一个项目）
        k: 要考虑的结果数量
        method: 折扣计算方法
                如果为0，则权重为 [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                如果为1，则权重为 [1.0, 0.6309, 0.5, 0.4307, ...]
    返回:
        折扣累积增益
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError("method参数必须是0或1")
    return 0.0


def ndcg_at_k(r, k, method=0):
    """计算归一化折扣累积增益（normalized discounted cumulative gain, nDCG）。

    相关性是正实数值。可以像之前的方法一样使用二值相关性。
    示例来自：
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    参数:
        r: 按排序顺序的相关性评分（列表或numpy数组）
            （第一个元素是第一个项目）
        k: 要考虑的结果数量
        method: 折扣计算方法
                如果为0，则权重为 [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                如果为1，则权重为 [1.0, 0.6309, 0.5, 0.4307, ...]
    返回:
        归一化折扣累积增益
    """
    r_ = r[0:k]
    dcg_max = dcg_at_k(sorted(r_, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max


def FR(r, k):
    """计算第一个相关项目的排名（First Ranking）。
    
    参数:
        r: 按排序顺序的相关性评分列表
        k: 要考虑的前k个结果
    
    返回:
        int: 第一个相关项目的排名，如果前k个结果中没有相关项目则返回NaN
    """
    for i in range(k):
        if r[i] != 0:
            return i + 1
    return np.nan


def AR(r, k):
    """计算相关项目的平均排名（Average Ranking）。
    
    参数:
        r: 按排序顺序的相关性评分列表
        k: 要考虑的前k个结果
    
    返回:
        float: 相关项目的平均排名，如果前k个结果中没有相关项目则返回NaN
    """
    count = 0
    total = 0
    for i in range(k):
        if r[i] != 0:
            count = count + 1
            total = total + i + 1
    if total != 0:
        return total / count
    else:
        return np.nan


def MFR(r):
    """计算平均第一个相关项目的排名（Mean First Ranking）。
    
    参数:
        r: 按排序顺序的相关性评分列表
    
    返回:
        float: 所有相关项目的第一个排名的平均值，如果没有相关项目则返回NaN
    """
    ret = [FR(r, i + 1) for i in range(len(r)) if r[i]]
    if len(ret) == 0:
        return np.nan
    return np.mean(ret)


def MAR(r):
    """计算平均相关项目排名（Mean Average Ranking）。
    
    参数:
        r: 按排序顺序的相关性评分列表
    
    返回:
        float: 所有相关项目的平均排名的平均值，如果没有相关项目则返回NaN
    """
    ret = [AR(r, i + 1) for i in range(len(r)) if r[i]]
    if len(ret) == 0:
        return np.nan
    return np.mean(ret)


def get_r(pred, true, r_thresh=0.5, idx=0):
    """根据输出分数对预测值进行排序并生成相关性列表。
    
    参数:
        pred: 预测值列表
        true: 真实标签列表
        r_thresh: 相关性阈值，默认为0.5
        idx: 用于排序的索引位置，默认为0
    
    返回:
        list: 按预测分数排序后的相关性列表，1表示相关，0表示不相关
    """
    zipped = list(zip(pred, true))
    zipped.sort(reverse=True, key=lambda x: x[idx])
    return [1 if i[0] > r_thresh and i[1] == 1 else 0 for i in zipped]


def rank_metr(pred, true, r_thresh=0.5, perfect=False):
    """计算所有排名指标。
    
    参数:
        pred: 预测值列表
        true: 真实标签列表
        r_thresh: 相关性阈值，默认为0.5
        perfect: 是否使用完美排序，默认为False
    
    返回:
        dict: 包含所有计算的排名指标的字典
    """
    if not any([i != 0 and i != 1 for i in pred]):
        print("警告：预测值是二值的，不是连续的。")
    ret = dict()
    kvals = [1, 3, 5, 10, 15, 20]
    r = get_r(pred, true, r_thresh, idx=1 if perfect else 0)
    last_vals = [0, 0, 0, 0]
    for k in kvals:
        if k > len(r):
            ret[f"nDCG@{k}"] = np.nan
            ret[f"MAP@{k}"] = np.nan
            ret[f"FR@{k}"] = np.nan
            ret[f"AR@{k}"] = np.nan
            continue
        ret[f"nDCG@{k}"] = ndcg_at_k(r, k)
        ret[f"MAP@{k}"] = mean_average_precision([r], k)
        ret[f"FR@{k}"] = FR(r, k)
        ret[f"AR@{k}"] = AR(r, k)
        last_vals = [ret[f"nDCG@{k}"], ret[f"MAP@{k}"], ret[f"FR@{k}"], ret[f"AR@{k}"]]

    mean_true = np.mean(true)
    if mean_true == 0 or mean_true == 1:
        ret["AUC"] = np.nan
    else:
        ret["AUC"] = roc_auc_score(true, pred)
    ret["MFR"] = MFR(r)
    ret["MAR"] = MAR(r)
    return ret
