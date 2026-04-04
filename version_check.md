## 验证joern
## 打这个命令就行
## v1.1.1530
joern

## 验证GloVe 
ls -la ~/GloVe/build/ 2>/dev/null || echo "GloVe build 目录不存在"
### 检查各个组件
ls -la ~/GloVe/build/vocab_count 2>/dev/null && echo "✅ vocab_count: 存在"
ls -la ~/GloVe/build/cooccur 2>/dev/null && echo "✅ cooccur: 存在"
ls -la ~/GloVe/build/shuffle 2>/dev/null && echo "✅ shuffle: 存在"
ls -la ~/GloVe/build/glove 2>/dev/null && echo "✅ glove: 存在"
### 查看 glove 帮助信息
cd ~/GloVe
./build/glove 2>&1 | head -10


##  检查 cppcheck 版本
cppcheck --version

## RATS 查看帮助信息（通常包含版本）
./rats-2.4/rats --help 2>&1 | head -10

## 检查 flawfinder 版本
flawfinder --version

# 验证 pygraphviz 安装
python -c "import pygraphviz; print(f'pygraphviz 版本: {pygraphviz.__version__}')"

### 测试 pygraphviz 功能
python -c "
import pygraphviz as pgv
G = pgv.AGraph()
G.add_node('A')
G.add_node('B')
G.add_edge('A', 'B')
print('pygraphviz 功能测试: 成功创建图')
"


## pytorch cuda 版本检查
### 查看 PyTorch 版本
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}')"

# 查看 PyTorch 使用的 CUDA 版本
python -c "import torch; print(f'PyTorch CUDA 版本: {torch.version.cuda}')"

# 查看 CUDA 是否可用
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"

# 验证 DGL 安装
python -c "import dgl; print(f'DGL 版本: {dgl.__version__}')"

# 验证 DGL 是否能正常使用 CUDA
python -c "
import dgl
import torch

# 创建一个简单的图
g = dgl.graph(([0, 1, 2], [1, 2, 3]))

# 检查图是否在 GPU 上
print(f'图是否在 GPU 上: {g.device}')

# 尝试将图移动到 GPU
if torch.cuda.is_available():
    g = g.to('cuda:0')
    print(f'图移动到 GPU 后: {g.device}')
    print('DGL CUDA 支持: True')
else:
    print('DGL CUDA 支持: False (PyTorch CUDA 不可用)')
"

# 测试NLTK                              
python3 -c "
import nltk
print(f'NLTK 版本: {nltk.__version__}')
"

# 检查 NLTK 数据目录
python3 -c "
import nltk
print('NLTK 数据目录:', nltk.data.path)
"

## torch-scatter
python -c "import torch_scatter; print('torch_scatter 已安装，版本:', torch_scatter.__version__)"

## 查看 tsne_torch 的安装信息
pip show tsne_torch




## 代码执行

nohup python sastvd/scripts/getgraphs.py 1 > getgraphs.log 2>&1 &

python sastvd/scripts/getgraphs.py 

export PATH=$PATH:/home/wmy/GloVe/build
python sastvd/scripts/prepare.py 

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python sastvd/scripts/train_best.py

python tests/try.py
python tests/check_distribution.py

python sastvd/ivdetect/main.py

python sastvd/scripts/run_method.py

python sastvd/scripts/test.py
 
scp D:\repositories\codebert-base.zip wmy@10.2.0.11:/home/wmy/linevd/storage/external/

nohup python sastvd/scripts/train_best.py > train.log 2>&1 &

tail -f train.log

python sastvd/linevd/count_bigvul.py

python sastvd/linevd/plot_first_rates.py

scp D:\repositories\graphcodebert-base.zip wmy@10.2.0.11:/home/wmy/linevd/storage/external/

## Baseline 实验
nohup python sastvd/scripts/baseline.py > baseline.log 2>&1 &
tail -f baseline.log

## RQ 实验运行命令

nohup python sastvd/scripts/myrq1.py > myrq1.log 2>&1 &
nohup python sastvd/scripts/myrq2.py > myrq2.log 2>&1 &
nohup python sastvd/scripts/myrq3.py > myrq3.log 2>&1 &
nohup python sastvd/scripts/myrq4.py > myrq4.log 2>&1 &
nohup python sastvd/scripts/myrq5.py > myrq5.log 2>&1 &

## 查看日志
tail -f myrq1.log
tail -f myrq2.log
tail -f myrq3.log
tail -f myrq4.log
tail -f myrq5.log

## 测试实验
nohup python sastvd/scripts/test.py > test.log 2>&1 &
tail -f test.log
