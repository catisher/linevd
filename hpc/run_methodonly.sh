#!/bin/bash
# 文件用途：HPC集群上运行方法级 (Method-Only) 实验的SLURM作业脚本
# 该脚本使用Singularity容器执行sastvd/scripts/run_method.py，用于训练仅基于方法级别的LineVD漏洞检测模型

# SLURM作业调度参数
#SBATCH -p batch              # 指定作业队列：batch队列
#SBATCH -N 1                  # 节点数量：1个节点
#SBATCH -n 6                  # 任务数量：6个CPU核心
#SBATCH --time=48:00:00       # 作业运行时间限制：48小时
#SBATCH --mem=128GB           # 内存限制：128GB（方法级实验需要更大内存）
#SBATCH --gres=gpu:1          # GPU资源：1个GPU卡
#SBATCH --err="hpc/logs/mo_%A.info"   # 错误输出文件路径，%A为作业ID
#SBATCH --output="hpc/logs/mo_%A.info" # 标准输出文件路径，%A为作业ID
#SBATCH --job-name="mo"        # 作业名称：mo（Method-Only缩写）

# 环境设置
# 加载Singularity容器模块
module load Singularity
# 加载CUDA 10.2.89模块，用于GPU加速
module load CUDA/10.2.89

# 执行方法级实验脚本
# 使用Singularity容器执行Python脚本，-H指定home目录映射，--nv启用GPU支持，-u启用无缓冲输出
singularity exec -H /g/acvt/a1720858/sastvd --nv main.sif python -u sastvd/scripts/run_method.py
