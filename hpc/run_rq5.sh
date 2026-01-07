#!/bin/bash
# 文件用途：HPC集群上运行研究问题5 (RQ5) 实验的SLURM作业脚本
# 该脚本使用Singularity容器执行sastvd/scripts/rq5.py，用于评估LineVD模型在跨项目场景下的泛化能力

# SLURM作业调度参数
#SBATCH -p batch              # 指定作业队列：batch队列
#SBATCH -N 1                  # 节点数量：1个节点
#SBATCH -n 6                  # 任务数量：6个CPU核心
#SBATCH --time=48:00:00       # 作业运行时间限制：48小时
#SBATCH --mem=32GB            # 内存限制：32GB
#SBATCH --gres=gpu:1          # GPU资源：1个GPU卡
#SBATCH --err="hpc/logs/rq5_%A.info"   # 错误输出文件路径，%A为作业ID
#SBATCH --output="hpc/logs/rq5_%A.info" # 标准输出文件路径，%A为作业ID
#SBATCH --job-name="rq5"       # 作业名称：rq5（研究问题5）

# 环境设置
# 加载Singularity容器模块
module load Singularity
# 加载CUDA 10.2.89模块，用于GPU加速
module load CUDA/10.2.89

# 执行研究问题5实验脚本
# 使用Singularity容器执行Python脚本，-H指定home目录映射，--nv启用GPU支持，-u启用无缓冲输出
singularity exec -H /g/acvt/a1720858/sastvd --nv main.sif python -u sastvd/scripts/rq5.py
