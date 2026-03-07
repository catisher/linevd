import re

"""
代码分词模块

该模块实现了基于IVDetect方法的代码分词功能，用于将代码文本分割成有意义的词元。

主要功能：
1. 将代码字符串分词，处理特殊字符、驼峰命名和下划线命名
2. 按行对代码进行分词处理
3. 过滤掉长度为1的无意义词元

应用场景：
- 代码向量表示
- 代码相似度计算
- 代码漏洞检测
"""


def tokenise(s):
    """根据IVDetect方法对字符串进行分词处理。
    
    参数:
        s (str): 要分词的字符串
    
    返回:
        str: 分词后的字符串，词元之间用空格分隔
    
    分词规则：
    1. 首先根据非字母数字和非空白字符进行分割
    2. 然后根据空格进行分割
    3. 接着将驼峰命名法的标识符分割成多个词元
    4. 最后过滤掉长度为1的无意义词元
    
    示例:
        >>> s = "FooBar fooBar foo bar_blub23/x~y'z"
        >>> tokenise(s)
        'Foo Bar foo Bar foo bar blub'
    """
    # 编译正则表达式：匹配非字母数字和非空白字符
    # 用于识别特殊字符（如 +, -, *, /, _, 等）
    spec_char = re.compile(r"[^a-zA-Z0-9\s]")
    
    # 编译正则表达式：匹配驼峰命名法的分割点
    # (?<=[a-z])(?=[A-Z]): 小写字母后跟大写字母的位置（如 fooBar -> foo|Bar）
    # (?<=[A-Z])(?=[A-Z][a-z]): 大写字母后跟大写字母+小写字母的位置（如 HTTPRequest -> HTTP|Request）
    # $: 字符串结尾
    camelcase = re.compile(r".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    
    # 根据特殊字符分割字符串
    # 例如："foo_bar" -> ["foo", "bar"]
    spec_split = re.split(spec_char, s)
    
    # 将分割结果用空格连接，再按空格分割
    # 例如：["foo", "bar"] -> "foo bar" -> ["foo", "bar"]
    space_split = " ".join(spec_split).split()

    # 定义内部函数：分割驼峰命名法的标识符
    def camel_case_split(identifier):
        # 使用正则表达式找到所有匹配的词元
        # 例如："fooBar" -> ["foo", "Bar"]
        return [i.group(0) for i in re.finditer(camelcase, identifier)]

    # 对每个词元应用驼峰分割，然后展平结果
    # 列表推导式的双重循环：先遍历每个词元的分割结果，再遍历分割后的每个词元
    # 例如：["fooBar", "test"] -> [["foo", "Bar"], ["test"]] -> ["foo", "Bar", "test"]
    camel_split = [i for j in [camel_case_split(i) for i in space_split] for i in j]
    
    # 过滤掉长度为1的词元（通常是无意义的单字符）
    # 例如：["foo", "Bar", "a", "test"] -> ["foo", "Bar", "test"]
    remove_single = [i for i in camel_split if len(i) > 1]
    
    # 将所有词元用空格连接成字符串并返回
    # 例如：["foo", "Bar", "test"] -> "foo Bar test"
    return " ".join(remove_single)


def tokenise_lines(s):
    """根据IVDetect方法按行对字符串进行分词处理。
    
    参数:
        s (str): 要分词的多行字符串
    
    返回:
        list: 分词后的行列表，每行是分词后的字符串
    
    处理过程：
    1. 将输入字符串按行分割
    2. 对每一行调用tokenise函数进行分词
    3. 过滤掉空行
    
    示例:
        >>> s = "line1a line1b\nline2a asdf\nf f f f f\na"
        >>> tokenise_lines(s)
        ['line1a line1b', 'line2a asdf', 'f f f f']
    """
    slines = s.splitlines()
    lines = []
    for sline in slines:
        tokline = tokenise(sline)
        if len(tokline) > 0:
            lines.append(tokline)
    return lines
