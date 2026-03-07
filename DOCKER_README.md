# 本地配置
# 1.miniconda
cd / 
wget --timeout=60 --tries=3 --no-check-certificate https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh 
bash miniconda.sh -b -u
rm -f miniconda.sh 

# 要权限，还要接受以下这几个，真是有够sb的
# 接受main频道的ToS
sudo /root/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# 接受r频道的ToS
sudo /root/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


# 2.安装GloVe（无Git，压缩包下载）
    wget --timeout=60 --tries=3 --no-check-certificate https://github.com/stanfordnlp/GloVe/archive/refs/heads/master.zip -O GloVe-master.zip 
    unzip -q GloVe-master.zip && mv GloVe-master GloVe && rm -f GloVe-master.zip
    cd GloVe && make -j$(nproc) && cd / && rm -rf GloVe/.git
   

# 安装cppcheck 2.10（兼容GCC 11+）
# 直接拿来的2.5,还没测试改版本会有什么影响

curl -L https://github.com/danmar/cppcheck/archive/refs/tags/2.10.tar.gz > cppcheck2.10.tar.gz    
    mkdir cppcheck
    mv cppcheck2.10.tar.gz cppcheck
    cd cppcheck
    tar -xzvf cppcheck2.10.tar.gz
    cd cppcheck-2.10
    mkdir build
    cd build
    cmake ..
    cmake --build .
# 这句没有权限安装 要加sudo
    make install

# 3.安装RATS（SSL绕过，清理压缩包）
curl -L https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/rough-auditing-tool-for-security/rats-2.4.tgz > rats-2.4.tgz
tar -xzvf rats-2.4.tgz
cd rats-2.4
./configure 
make 


# 4.安装flawfinder（国内源）
pip install flawfinder -i https://pypi.tuna.tsinghua.edu.cn/simple


# 5. 安装PyTorch（cu117，国内源加速）

### 对应版本pytorch 2.0.0 cu117  torchdata 0.6  dgl 0.9.1post1
### 浪费两小时，最后才发现dgl不成功跟这个torchdata版本不匹配
### 中科大源（备选）  成功(还是这个好用)
pip install torch torchvision torchaudio \
  --index-url https://pypi.mirrors.ustc.edu.cn/simple/ \
  --extra-index-url https://download.pytorch.org/whl/cu117 \
  --timeout=300 --retries=10

# 安装与 CUDA 11.7 兼容的 PyTorch
## 后来用的，为了安装与之匹配的dgl
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117

pip install dgl-cu117 -f https://data.dgl.ai/wheels/repo.html

## 官方
## pip会自动解决包冲突 conda不会
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c dglteam/label/th21_cu118 dgl


# 14. 安装pygraphviz和nltk

# 要给conda换源
# 给root的conda添加清华源（对应你要执行的3条配置）
sudo /root/miniconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
sudo /root/miniconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
sudo /root/miniconda3/bin/conda config --set show_channel_urls yes

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

## 太慢了，有待测试
sudo /root/miniconda3/bin/conda install -y pygraphviz

## nltk
## 这个成功
## 乐，nltk项目里根本没用到
pip install nltk -i https://pypi.tuna.tsinghua.edu.cn/simple 
python3 -c 'import nltk; nltk.download("punkt")'
  

# Joern 
# 太慢了
sudo apt install -y openjdk-8-jdk git curl gnupg bash unzip sudo wget 
wget https://github.com/ShiftLeftSecurity/joern/releases/latest/download/joern-install.sh
chmod +x ./joern-install.sh
printf 'Y\n/bin/joern\ny\n/usr/local/bin\n\n' | sudo ./joern-install.sh --interactive
## 直接从官网下载解压
unzip joern-cli.zip -d joern
echo 'export PATH="$HOME/joern:$PATH"' >> ~/.bashrc
source ~/.bashrc



### torch-scatter
## 要指定版本，不然会自动更新pytorch
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

##

