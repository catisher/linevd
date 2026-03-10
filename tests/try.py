import sastvd as svd
from pathlib import Path
import glob

# 获取已完成图构建的样本ID
finished = [
    int(Path(i).name.split(".")[0])
    for i in glob.glob(str(svd.processed_dir() / "bigvul/before/*nodes*"))
]

# 加载数据集
import sastvd.helpers.datasets as svdd
df = svdd.bigvul()

# 检查已完成图构建样本的分布
df_finished = df[df.id.isin(finished)]
print(f"已完成图构建的样本总数: {len(df_finished)}")
print(f"漏洞样本 (vul=1): {len(df_finished[df_finished.vul == 1])}")
print(f"非漏洞样本 (vul=0): {len(df_finished[df_finished.vul == 0])}")