# Docker使用指南

本项目提供了完整的Docker配置，可以替代Singularity容器来运行LineVD漏洞检测系统。

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