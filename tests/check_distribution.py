import sastvd as svd
from pathlib import Path
import glob
import pandas as pd

# 加载数据集
import sastvd.helpers.datasets as svdd
df = svdd.bigvul()

print("=" * 50)
print("数据集整体分布:")
print(f"总样本数: {len(df)}")
print(f"漏洞样本 (vul=1): {len(df[df.vul == 1])}")
print(f"非漏洞样本 (vul=0): {len(df[df.vul == 0])}")

# 获取已完成图构建的样本ID
finished = [
    int(Path(i).name.split(".")[0])
    for i in glob.glob(str(svd.processed_dir() / "bigvul/before/*nodes*"))
]

print("\n" + "=" * 50)
print("已完成图构建的样本分布:")
df_finished = df[df.id.isin(finished)]
print(f"已完成图构建的样本总数: {len(df_finished)}")
print(f"漏洞样本 (vul=1): {len(df_finished[df_finished.vul == 1])}")
print(f"非漏洞样本 (vul=0): {len(df_finished[df_finished.vul == 0])}")

# 检查哪些非漏洞样本没有完成图构建
print("\n" + "=" * 50)
print("未完成的非漏洞样本:")
df_nonvul = df[df.vul == 0]
nonvul_finished = df_nonvul[df_nonvul.id.isin(finished)]
nonvul_not_finished = df_nonvul[~df_nonvul.id.isin(finished)]
print(f"非漏洞样本总数: {len(df_nonvul)}")
print(f"已完成图构建的非漏洞样本: {len(nonvul_finished)}")
print(f"未完成图构建的非漏洞样本: {len(nonvul_not_finished)}")

# 显示部分未完成图构建的非漏洞样本ID
if len(nonvul_not_finished) > 0:
    print(f"\n部分未完成图构建的非漏洞样本ID: {nonvul_not_finished.id.tolist()[:10]}")
