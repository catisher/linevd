import os
import pickle as pkl

import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.ivdetect.helpers as ivdh


def get_dep_add_lines(filepath_before, filepath_after, added_lines):
    """获取依赖于添加行的代码行。

    示例：
    df = svdd.bigvul()
    filepath_before = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/177775.c"
    filepath_after = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/after/177775.c"
    added_lines = df[df.id==177775].added.item()

    参数:
        filepath_before: 漏洞修复前的代码文件路径
        filepath_after: 漏洞修复后的代码文件路径
        added_lines: 修复中添加的行号列表
    返回:
        list: 依赖于添加行的代码行号列表（已排序）
    """
    before_graph = ivdh.feature_extraction(filepath_before)[0]
    after_graph = ivdh.feature_extraction(filepath_after)[0]

    # 获取与添加行对应的图节点
    added_after_lines = after_graph[after_graph.id.isin(added_lines)]

    # 获取添加图中依赖于添加行的行
    dep_add_lines = added_after_lines.data.tolist() + added_after_lines.control.tolist()
    dep_add_lines = set([i for j in dep_add_lines for i in j])

    # 过滤出存在于修复前图中的行
    before_lines = set(before_graph.id.tolist())
    dep_add_lines = sorted([i for i in dep_add_lines if i in before_lines])

    return dep_add_lines


def helper(row):
    """从字典运行get_dep_add_lines函数。

    示例：
    df = svdd.bigvul()
    added = df[df.id==177775].added.item()
    removed = df[df.id==177775].removed.item()
    helper({"id":177775, "removed": removed, "added": added})

    参数:
        row: 包含id、removed和added字段的字典
    返回:
        list: [id, {"removed": removed行号列表, "depadd": 依赖添加行的行号列表}]
    """
    before_path = str(svd.processed_dir() / f"bigvul/before/{row['id']}.c")
    after_path = str(svd.processed_dir() / f"bigvul/after/{row['id']}.c")
    try:
        dep_add_lines = get_dep_add_lines(before_path, after_path, row["added"])
    except Exception:
        dep_add_lines = []
    return [row["id"], {"removed": row["removed"], "depadd": dep_add_lines}]


def get_dep_add_lines_bigvul(cache=True):
    """为BigVul数据集缓存依赖添加行。

    参数:
        cache: 是否使用缓存（默认为True）
    返回:
        dict: 包含每个漏洞ID对应的removed行和depadd行的字典
    """
    saved = svd.get_dir(svd.processed_dir() / "bigvul/eval") / "statement_labels.pkl"
    if os.path.exists(saved) and cache:
        with open(saved, "rb") as f:
            return pkl.load(f)
    df = svdd.bigvul()
    df = df[df.vul == 1]
    desc = "Getting dependent-added lines: "
    lines_dict = svd.dfmp(df, helper, ["id", "removed", "added"], ordr=False, desc=desc)
    lines_dict = dict(lines_dict)
    with open(saved, "wb") as f:
        pkl.dump(lines_dict, f)
    return lines_dict


def eval_statements(sm_logits, labels, thresh=0.5):
    """根据IVDetect评估语句级漏洞检测结果。

    示例：
    sm_logits = [
        [0.5747372, 0.4252628],
        [0.53908646, 0.4609135],
        [0.49043426, 0.5095658],
        [0.65794635, 0.34205365],
        [0.3370166, 0.66298336],
        [0.55573744, 0.4442625],
    ]
    labels = [0, 0, 0, 0, 1, 0]

    参数:
        sm_logits: 模型预测的softmax概率列表，每个元素为[非漏洞概率, 漏洞概率]
        labels: 真实标签列表，0表示非漏洞，1表示漏洞
        thresh: 分类阈值（默认为0.5）
    返回:
        dict: 包含k=1到10的评估结果，值为1表示前k个预测中包含漏洞，0表示不包含
    """
    if sum(labels) == 0:
        # 如果没有漏洞（所有标签都是0）
        preds = [i for i in sm_logits if i[1] > thresh]
        if len(preds) > 0:
            # 如果预测了漏洞，则所有k值的结果都是0（误报）
            ret = {k: 0 for k in range(1, 11)}
        else:
            # 如果没有预测漏洞，则所有k值的结果都是1（正确）
            ret = {k: 1 for k in range(1, 11)}
    else:
        # 如果有漏洞，按漏洞概率降序排序
        zipped = list(zip(sm_logits, labels))
        zipped = sorted(zipped, key=lambda x: x[0][1], reverse=True)
        ret = {}
        for i in range(1, 11):
            # 检查前i个预测中是否包含真实漏洞
            if 1 in [item[1] for item in zipped[:i]]:
                ret[i] = 1
            else:
                ret[i] = 0
    return ret


def eval_statements_inter(stmt_pred_list, thresh=0.5):
    """评估语句级检测结果的中间计算函数。

    参数:
        stmt_pred_list: 语句预测列表，每个元素为[sm_logits, labels]
        thresh: 分类阈值（默认为0.5）
    返回:
        dict: 包含k=1到10的平均评估结果，值为所有样本的平均命中率
    """
    total = len(stmt_pred_list)
    ret = {k: 0 for k in range(1, 11)}
    for item in stmt_pred_list:
        eval_stmt = eval_statements(item[0], item[1], thresh)
        for i in range(1, 11):
            ret[i] += eval_stmt[i]
    ret = {k: v / total for k, v in ret.items()}
    return ret


def eval_statements_list(stmt_pred_list, thresh=0.5, vo=False):
    """对整个预测列表应用语句级评估。

    示例：
    item1 = [[[0.1, 0.9], [0.6, 0.4], [0.4, 0.5]], [0, 1, 1]]
    item2 = [[[0.9, 0.1], [0.6, 0.4]], [0, 0]]
    item3 = [[[0.1, 0.9], [0.6, 0.4]], [1, 1]]
    stmt_pred_list = [item1, item2, item3]

    参数:
        stmt_pred_list: 语句预测列表，每个元素为[sm_logits, labels]
        thresh: 分类阈值（默认为0.5）
        vo: 是否只评估包含漏洞的样本（默认为False）
    返回:
        dict: 包含k=1到10的综合评估结果
            - 如果vo=True: 只返回包含漏洞样本的评估结果
            - 如果vo=False: 返回包含漏洞样本和不包含漏洞样本的乘积结果
    """
    # 筛选出包含漏洞的样本
    vo_list = [i for i in stmt_pred_list if sum(i[1]) > 0]
    vulonly = eval_statements_inter(vo_list, thresh)
    if vo:
        return vulonly
    
    # 筛选出不包含漏洞的样本
    nvo_list = [i for i in stmt_pred_list if sum(i[1]) == 0]
    nonvulnonly = eval_statements_inter(nvo_list, thresh)
    
    # 计算综合结果
    ret = {}
    for i in range(1, 11):
        ret[i] = vulonly[i] * nonvulnonly[i]
    return ret
