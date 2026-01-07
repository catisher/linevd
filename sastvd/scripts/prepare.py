"""数据集准备脚本。

该脚本用于准备BigVul数据集，包括加载数据、生成依赖关系、创建词嵌入等操作。
"""

import sastvd.helpers.datasets as svdd
import sastvd.ivdetect.evaluate as ivde


def bigvul():
    """运行BigVul数据集的准备脚本。
    
    执行以下操作：
    1. 加载BigVul数据集
    2. 计算依赖添加行
    3. 生成GloVe词嵌入
    4. 生成Doc2Vec词嵌入
    """
    svdd.bigvul()  # 加载BigVul数据集
    ivde.get_dep_add_lines_bigvul()  # 计算依赖添加行
    svdd.generate_glove("bigvul")  # 生成GloVe词嵌入
    svdd.generate_d2v("bigvul")  # 生成Doc2Vec词嵌入


if __name__ == "__main__":
    bigvul()  # 执行BigVul数据集准备
