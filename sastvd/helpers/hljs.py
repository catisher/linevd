from pathlib import Path

import sastvd as svd
import sastvd.ivdetect.evaluate as svde

"""
代码高亮可视化模块

该模块使用highlight.js库生成带有行号和高亮显示的代码HTML，用于可视化漏洞检测结果。

主要功能：
- 生成带有语法高亮和行号的代码HTML
- 根据预测分数高亮显示可能存在漏洞的行
- 标记真实存在漏洞的行
- 支持不同的代码风格主题
- 将结果保存为HTML文件
"""

html = """<!DOCTYPE html>
<link
  rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.2.0/styles/{}.min.css"
/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.2.0/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/gh/TRSasasusu/highlightjs-highlight-lines.js@1.1.6/highlightjs-highlight-lines.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlightjs-line-numbers.js/2.8.0/highlightjs-line-numbers.min.js"></script>
<script>
  hljs.initHighlightingOnLoad();
  hljs.initLineNumbersOnLoad();
  hljs.initHighlightLinesOnLoad([
    [
        {}
    ],
  ]);
</script>

<style>
  td.hljs-ln-numbers {{
    text-align: center;
    color: #777;
    border-right: 1px solid #999;
    vertical-align: top;
    padding-right: 5px;

    -webkit-touch-callout: none;
    -webkit-user-select: none;
    -khtml-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
  }}
  td.hljs-ln-code {{
    padding-left: 10px;
  }}
  code {{
    white-space: pre-wrap;
    overflow: auto;
  }}
{}
</style>

<pre><code class="language-cpp">{}
</code></pre>"""

lines = svde.get_dep_add_lines_bigvul()


def hljs(code, preds, vulns=[], style="idea", vid=None):
    """生成带有预测结果高亮显示的代码HTML。
    
    参数:
        code (str): 要高亮显示的代码字符串
        preds (dict): 行号到预测分数的映射，格式: {行号: 预测分数}
        vulns (list, optional): 真实漏洞行的列表，默认为空列表
        style (str, optional): 代码高亮风格，默认为"idea"
        vid (int, optional): 样本ID，用于获取真实漏洞信息，默认为None
    
    返回:
        str: 包含语法高亮、行号和预测结果高亮的HTML字符串
    
    示例:
        >>> style = "idea"
        >>> code = '''int main() {
        ...     int a = 1;
        ...     return a;
        ... }'''
        >>> preds = {1: 0.5, 2: 0.3}
        >>> html = hljs(code, preds)
    """
    hl_lines = []
    for k, v in preds.items():
        hl_lines.append(f'{{ start: {k}, end: {k}, color: "rgba(255, 0, 0, {v})" }}')

    removed = set(lines[vid]["removed"])

    vul_lines = []
    for v in vulns:
        color = "darkred"
        if int(v) in removed:
            color = "red"
            print(color)
        vstyle = f'.hljs-ln-numbers[data-line-number="{v}"] {{  font-weight: bold; color: {color}; }}'
        vul_lines.append(vstyle)

    vul_lines.append(".hljs-ln-numbers { background-color: white; }")

    return html.format(style, ",".join(hl_lines), "\n".join(vul_lines), code)


def linevd_to_html(cfile, preds, vulns=[], style="idea"):
    """读取C文件内容，生成带有预测结果高亮的HTML并保存到文件。
    
    参数:
        cfile (str): C文件的路径
        preds (dict): 行号到预测分数的映射，格式: {行号: 预测分数}
        vulns (list, optional): 真实漏洞行的列表，默认为空列表
        style (str, optional): 代码高亮风格，默认为"idea"
    
    返回:
        str: 生成的HTML字符串
    
    示例:
        >>> cfile = svddc.BigVulDataset.itempath(sample.id)
        >>> html = linevd_to_html(cfile, {1: 0.8, 3: 0.6})
    """
    with open(cfile, "r") as f:
        code = f.read()
    ret = hljs(code, preds, vulns, style, int(Path(cfile).stem))
    savedir = svd.get_dir(svd.outputs_dir() / "visualise_preds")
    with open(f"{savedir / Path(cfile).name}.html", "w") as f:
        f.write(ret)
    return ret
