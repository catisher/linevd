#!/bin/bash
# 文件用途：HPC集群上运行代码属性图 (CPG) 生成任务的SLURM数组作业脚本
# 该脚本使用Singularity容器执行sastvd/scripts/getgraphs.py，用于并行生成LineVD模型所需的代码属性图
# 通过SLURM数组作业(--array=1-100)将任务分成100个并行子任务，提高处理效率

# SLURM作业调度参数
#SBATCH -p batch              # 指定作业队列：batch队列
#SBATCH -N 1                  # 节点数量：1个节点
#SBATCH -n 8                  # 任务数量：8个CPU核心
#SBATCH --time=48:00:00       # 作业运行时间限制：48小时
#SBATCH --mem=48GB            # 内存限制：48GB
#SBATCH --array=1-100         # 数组作业配置：创建100个并行子任务
#SBATCH --err="hpc/logs/prepros_%a.err"   # 错误输出文件路径，%a为数组任务ID
#SBATCH --output="hpc/logs/prepros_%a.out" # 标准输出文件路径，%a为数组任务ID
#SBATCH --job-name="prepros"    # 作业名称：prepros（预处理）

# 环境设置
# 加载Singularity容器模块
module load Singularity

# 执行图生成脚本
# 使用Singularity容器执行Python脚本，-H指定home目录映射，-u启用无缓冲输出
# $SLURM_ARRAY_TASK_ID作为参数传递给脚本，用于标识当前处理的子任务
 singularity exec -H /g/acvt/a1720858/sastvd main.sif python -u sastvd/scripts/getgraphs.py $SLURM_ARRAY_TASK_ID
