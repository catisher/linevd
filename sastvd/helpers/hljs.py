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
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.2.0/styles/{}.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.2.0/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/gh/TRSasasusu/highlightjs-highlight-lines.js@1.1.6/highlightjs-highlight-lines.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlightjs-line-numbers.js/2.8.0/highlightjs-line-numbers.min.js"></script>
</head>
<body>
<script>
  var preds = {};
  document.addEventListener('DOMContentLoaded', () => {
    hljs.highlightAll();
    hljs.initLineNumbersOnLoad({
      renderLineNumbers: function(lineNum) {
        let score = preds[lineNum-1] ? (preds[lineNum-1] * 100).toFixed(1) + '%' : '';
        return '<span class="line-number">' + lineNum + '</span><span class="confidence-value">' + score + '</span>';
      }
    });
    hljs.initHighlightLinesOnLoad([{}]);
  });
</script>

<style>
  td.hljs-ln-numbers {
    text-align: center;
    color: #777;
    border-right: 1px solid #999;
    vertical-align: top;
    padding-right: 5px;
    width: 80px;
    -webkit-touch-callout: none;
    -webkit-user-select: none;
    -khtml-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
  }
  .confidence-value {
    font-size: 8px;
    color: #999;
    margin-top: 2px;
    display: block;
  }
  td.hljs-ln-code {
    padding-left: 10px;
  }
  code {
    white-space: pre-wrap;
    overflow: auto;
  }
  {}
</style>

<pre><code class="language-cpp">{}
</code></pre>
</body>
</html>"""

lines = svde.get_dep_add_lines_bigvul()

def hljs(code, preds, vulns=[], style="idea", vid=None):
    """生成带有预测结果高亮显示的代码HTML。"""
    hl_lines = []
    for k, v in preds.items():
        alpha = min(max(v, 0.1), 0.8)
        hl_lines.append(f'{{ start: {k}, end: {k}, color: "rgba(255, 0, 0, {alpha})" }}')

    vul_lines = []
    removed = set()
    if vid is not None and vid in lines:
        removed = set(lines[vid]["removed"])

    for v in vulns:
        color = "darkred"
        if int(v) in removed:
            color = "red"
        vstyle = f'.hljs-ln-numbers[data-line-number="{v}"] {{ font-weight: bold; color: {color}; }}'
        vul_lines.append(vstyle)
    vul_lines.append(".hljs-ln-numbers { background-color: white; }")

    preds_js = "{"
    for k, v in preds.items():
        preds_js += f"{k}: {v},"
    preds_js = preds_js.rstrip(",") + "}"

    return html.format(style, preds_js, ",".join(hl_lines), "\n".join(vul_lines), code)

def linevd_to_html(cfile, preds, vulns=[], style="idea"):
    """读取C文件内容，生成带有预测结果高亮的HTML并保存到文件。"""
    with open(cfile, "r", encoding="utf-8") as f:
        code = f.read()
    vid = int(Path(cfile).stem) if Path(cfile).stem.isdigit() else None
    ret = hljs(code, preds, vulns, style, vid)
    savedir = svd.get_dir(svd.outputs_dir() / "visualise_preds")
    out_path = savedir / f"{Path(cfile).name}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(ret)
    print(f"✅ 可视化HTML已保存：{out_path}")
    return ret