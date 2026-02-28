# Docker使用指南

本项目提供了完整的Docker配置，可以替代Singularity容器来运行LineVD漏洞检测系统。




# 本地配置

## 待处理
chmod u+x /cli.sh \


sudo apt update -y

## 2. 安装LineVD所需系统依赖
sudo apt install -y \
    wget \
    build-essential \
    git \
    graphviz \
    zip \
    unzip \
    curl \
    vim \
    libexpat1-dev \
    cmake

## 3. 清理APT缓存（减小磁盘占用，Docker/本地环境通用）
sudo apt clean
sudo rm -rf /var/lib/apt/lists/*




## 4. miniconda
cd / 
wget --timeout=60 --tries=3 --no-check-certificate https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh 
bash miniconda.sh -b -u
rm -f miniconda.sh 

## 貌似可有可无
sudo /root/miniconda3/bin/conda clean -afy 

# 7. 安装PyTorch（cu118，国内源加速）
pip install torch torchvision torchaudio --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu118

   
# 阿里云源（推荐，稳定性更高）
pip install torch torchvision torchaudio \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  --timeout=300 --retries=10

# 中科大源（备选）  成功
pip install torch torchvision torchaudio \
  --index-url https://pypi.mirrors.ustc.edu.cn/simple/ \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  --timeout=300 --retries=10


# 8. 安装GloVe（无Git，压缩包下载）
    wget --timeout=60 --tries=3 --no-check-certificate https://github.com/stanfordnlp/GloVe/archive/refs/heads/master.zip -O GloVe-master.zip 
    unzip -q GloVe-master.zip && mv GloVe-master GloVe && rm -f GloVe-master.zip
    cd GloVe && make -j$(nproc) && cd / && rm -rf GloVe/.git
   

# 9. 安装cppcheck 2.10（兼容GCC 11+）
# 直接拿来的2.5

curl -L https://github.com/danmar/cppcheck/archive/refs/tags/2.5.tar.gz > cppcheck2.5.tar.gz    
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

# 11. 安装RATS（SSL绕过，清理压缩包）
curl -L https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/rough-auditing-tool-for-security/rats-2.4.tgz > rats-2.4.tgz
tar -xzvf rats-2.4.tgz
cd rats-2.4
./configure 
make 

# 同上，权限问题
    make install


# 12. 安装flawfinder（国内源）
pip install flawfinder -i https://pypi.tuna.tsinghua.edu.cn/simple



# 14. 安装pygraphviz和nltk
# 要权限，还要接受以下这几个，真是有够sb的
# 接受main频道的ToS
sudo /root/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# 接受r频道的ToS
sudo /root/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# 要给conda换源
# 给root的conda添加清华源（对应你要执行的3条配置）
sudo /root/miniconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
sudo /root/miniconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
sudo /root/miniconda3/bin/conda config --set show_channel_urls yes

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

## 太慢了，有待测试
sudo /root/miniconda3/bin/conda install -y pygraphviz

## 这个成功
pip install nltk -i https://pypi.tuna.tsinghua.edu.cn/simple 
python3 -c 'import nltk; nltk.download("punkt")'
  

# Joern 
# 太慢了
sudo apt install -y openjdk-8-jdk git curl gnupg bash unzip sudo wget 
wget https://github.com/ShiftLeftSecurity/joern/releases/latest/download/joern-install.sh
chmod +x ./joern-install.sh
printf 'Y\n/bin/joern\ny\n/usr/local/bin\n\n' | sudo ./joern-install.sh --interactive



## 前置要求

1. **Docker安装**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install docker.io docker-compose

   # 启动Docker服务
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

2. **NVIDIA Docker支持**（用于GPU加速）
   ```bash
   # 安装NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt update
   sudo apt install nvidia-container-toolkit
   sudo systemctl restart docker
   ```

## 构建Docker镜像

### 方法1：使用docker-compose（推荐）
```bash
docker-compose build
```

### 方法2：直接使用docker build
```bash
docker build -t linevd:latest .
```

## 运行容器

### 方法1：使用docker-compose（推荐）
```bash
# 启动容器
docker-compose up -d

# 进入容器
docker-compose exec linevd bash

# 停止容器
docker-compose down
```

### 方法2：直接使用docker run
```bash
# 启动容器（带GPU支持）
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -v $(pwd)/storage:/workspace/storage \
    linevd:latest \
    /bin/bash

# 启动容器（仅CPU）
docker run -it --rm \
    -v $(pwd):/workspace \
    -v $(pwd)/storage:/workspace/storage \
    linevd:latest \
    /bin/bash
```

## 使用说明

### 首次运行 - 初始化环境
```bash
# 进入容器后
bash /cli.sh -p initialise
```

### 特征提取
```bash
# 准备数据
python sastvd/scripts/prepare.py

# 生成图结构
python sastvd/scripts/getgraphs.py
```

### 模型训练
```bash
# 使用GPU训练
python sastvd/scripts/train_best.py
```

### 运行测试
```bash
# 使用CLI运行测试
bash /cli.sh -t

# 或直接使用pytest
pytest tests/
```

### 运行Python程序
```bash
# 使用CLI
bash /cli.sh -p path/to/script.py -a arg1 arg2

# 或直接运行
python path/to/script.py arg1 arg2
```

## HPC环境使用

如果需要在HPC集群上使用Docker，可以参考hpc文件夹中的脚本，将`singularity exec`替换为相应的docker命令：

```bash
# 原始Singularity命令
singularity exec -H /path/to/home --nv main.sif python script.py

# 对应的Docker命令
docker run --gpus all -v /path/to/home:/workspace linevd:latest python script.py
```

## 常用Docker命令

```bash
# 查看运行中的容器
docker ps

# 查看所有容器
docker ps -a

# 停止容器
docker stop <container_id>

# 删除容器
docker rm <container_id>

# 查看镜像
docker images

# 删除镜像
docker rmi <image_id>

# 查看容器日志
docker logs <container_id>

# 进入运行中的容器
docker exec -it <container_id> bash
```

## 注意事项

1. **GPU支持**：确保宿主机安装了NVIDIA驱动和CUDA工具包
2. **内存要求**：训练模型建议至少16GB内存，某些任务可能需要更多
3. **数据持久化**：使用volume挂载确保数据不会在容器删除后丢失
4. **权限问题**：可能需要使用`sudo`运行Docker命令，或将用户添加到docker组

## 故障排除

### GPU不可用
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:10.2-base nvidia-smi
```

### 权限问题
```bash
# 将当前用户添加到docker组
sudo usermod -aG docker $USER

# 重新登录或运行
newgrp docker
```

### 构建失败
```bash
# 清理Docker缓存重新构建
docker system prune -a
docker-compose build --no-cache
```

## 与Singularity的对应关系

| Singularity命令 | Docker命令 |
|----------------|------------|
| `singularity build main.sif Singularity` | `docker build -t linevd:latest .` |
| `singularity exec main.sif python script.py` | `docker run linevd:latest python script.py` |
| `singularity exec --nv main.sif python script.py` | `docker run --gpus all linevd:latest python script.py` |
| `singularity exec -H /path main.sif python script.py` | `docker run -v /path:/workspace linevd:latest python script.py` |