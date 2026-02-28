#!/bin/bash
# 文件用途：HPC集群上运行数据准备任务的SLURM作业脚本
# 该脚本使用Singularity容器执行sastvd/scripts/prepare.py，用于准备LineVD模型的训练数据

# SLURM作业调度参数
#SBATCH -p batch              # 指定作业队列：batch队列
#SBATCH -N 1                  # 节点数量：1个节点
#SBATCH -n 6                  # 任务数量：6个CPU核心
#SBATCH --time=12:00:00       # 作业运行时间限制：12小时
#SBATCH --mem=16GB            # 内存限制：16GB
#SBATCH --err="hpc/logs/prepare.out"   # 错误输出文件路径
#SBATCH --output="hpc/logs/prepare.out" # 标准输出文件路径
#SBATCH --job-name="prepare"   # 作业名称：prepare

# 环境设置
# 加载Singularity容器模块
module load Singularity

# 执行数据准备脚本
# 使用Singularity容器执行Python脚本，-H指定home目录映射，-u启用无缓冲输出
singularity exec -H /g/acvt/a1720858/sastvd main.sif python -u sastvd/scripts/prepare.py