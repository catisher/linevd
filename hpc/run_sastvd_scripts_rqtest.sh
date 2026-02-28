#!/bin/bash
# 文件用途：HPC集群上运行研究问题测试 (RQTest) 实验的SLURM作业脚本
# 该脚本使用Singularity容器执行sastvd/scripts/rqtest.py，用于测试LineVD模型在不同实验配置下的性能

# SLURM作业调度参数
#SBATCH -p batch              # 指定作业队列：batch队列
#SBATCH -N 1                  # 节点数量：1个节点
#SBATCH -n 2                  # 任务数量：2个CPU核心
#SBATCH --time=48:00:00       # 作业运行时间限制：48小时
#SBATCH --mem=64GB            # 内存限制：64GB
#SBATCH --gres=gpu:1          # GPU资源：1个GPU卡
#SBATCH --err="hpc/logs/rqT_%A.info"   # 错误输出文件路径，%A为作业ID
#SBATCH --output="hpc/logs/rqT_%A.info" # 标准输出文件路径，%A为作业ID
#SBATCH --job-name="rqT"       # 作业名称：rqT（研究问题测试）

# 环境设置
# 加载Singularity容器模块
module load Singularity
# 加载CUDA 10.2.89模块，用于GPU加速
module load CUDA/10.2.89

# 执行研究问题测试脚本
# 使用Singularity容器执行Python脚本，-H指定home目录映射，--nv启用GPU支持，-u启用无缓冲输出
singularity exec -H /g/acvt/a1720858/sastvd --nv main.sif python -u sastvd/scripts/rqtest.py
