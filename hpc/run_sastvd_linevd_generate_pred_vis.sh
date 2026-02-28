#!/bin/bash
# 文件用途：HPC集群上运行LineVD模型预测可视化任务的SLURM作业脚本
# 该脚本使用Singularity容器执行sastvd/linevd/generate_pred_vis.py，用于生成LineVD模型预测结果的可视化

# SLURM作业调度参数
#SBATCH -p batch              # 指定作业队列：batch队列
#SBATCH -N 1                  # 节点数量：1个节点
#SBATCH -n 6                  # 任务数量：6个CPU核心
#SBATCH --time=12:00:00       # 作业运行时间限制：12小时
#SBATCH --mem=16GB            # 内存限制：16GB
#SBATCH --gres=gpu:1          # GPU资源：1个GPU卡
#SBATCH --err="hpc/logs/vis_%A.info"   # 错误输出文件路径，%A为作业ID
#SBATCH --output="hpc/logs/vis_%A.info" # 标准输出文件路径，%A为作业ID
#SBATCH --job-name="vis"       # 作业名称：vis（可视化）

# 环境设置
# 加载Singularity容器模块
module load Singularity
# 加载CUDA 10.2.89模块，用于GPU加速
module load CUDA/10.2.89

# 执行预测可视化脚本
# 使用Singularity容器执行Python脚本，-H指定home目录映射，--nv启用GPU支持
# 生成LineVD模型预测结果的可视化展示
 singularity exec -H /g/acvt/a1720858/sastvd --nv main.sif python sastvd/linevd/generate_pred_vis.py
