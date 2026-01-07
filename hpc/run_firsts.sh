#!/bin/bash
# 文件用途：HPC集群上生成LineVD模型first-rank性能指标可视化的SLURM作业脚本
# 该脚本使用Singularity容器执行sastvd/linevd/plot_first_rates.py，用于生成模型预测的first-rank性能可视化图表

# SLURM作业调度参数
#SBATCH -p batch              # 指定作业队列：batch队列
#SBATCH -N 1                  # 节点数量：1个节点
#SBATCH -n 2                  # 任务数量：2个CPU核心
#SBATCH --time=00:20:00       # 作业运行时间限制：20分钟
#SBATCH --mem=16GB            # 内存限制：16GB
#SBATCH --gres=gpu:1          # GPU资源：1个GPU卡
#SBATCH --err="hpc/logs/first_%A.info"   # 错误输出文件路径，%A表示作业ID
#SBATCH --output="hpc/logs/first_%A.info" # 标准输出文件路径，%A表示作业ID
#SBATCH --job-name="first"     # 作业名称：first

# 环境设置
# 加载Singularity容器模块
module load Singularity
# 加载CUDA模块，用于GPU加速
module load CUDA/10.2.89

# 执行可视化生成脚本
# 使用Singularity容器执行Python脚本，-H指定home目录映射，--nv启用GPU支持
singularity exec -H /g/acvt/a1720858/sastvd --nv main.sif python sastvd/linevd/plot_first_rates.py
