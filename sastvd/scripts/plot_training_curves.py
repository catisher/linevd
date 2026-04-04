"""绘制训练曲线。

该脚本用于从 CSV 日志文件中读取训练数据，并绘制 loss、acc、mcc 等指标随 epoch 变化的曲线。
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import sastvd as svd


def plot_single_trial(csv_path, output_path, trial_name="Trial"):
    """绘制单个试验的训练曲线。
    
    参数:
        csv_path: metrics.csv 文件路径
        output_path: 输出图片路径
        trial_name: 试验名称
    """
    df = pd.read_csv(csv_path)
    
    # 按epoch分组，取每个epoch的最后一个值
    if 'epoch' in df.columns:
        df = df.groupby('epoch').last().reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss 曲线
    ax1 = axes[0, 0]
    if 'train_loss' in df.columns:
        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', markersize=3)
    if 'val_loss' in df.columns:
        ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Curve')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy 曲线
    ax2 = axes[0, 1]
    if 'train_acc' in df.columns:
        ax2.plot(df['epoch'], df['train_acc'], label='Train Acc', marker='o', markersize=3)
    if 'val_acc' in df.columns:
        ax2.plot(df['epoch'], df['val_acc'], label='Val Acc', marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy Curve')
    ax2.grid(True, alpha=0.3)
    
    # MCC 曲线
    ax3 = axes[1, 0]
    if 'train_mcc' in df.columns:
        ax3.plot(df['epoch'], df['train_mcc'], label='Train MCC', marker='o', markersize=3)
    if 'val_mcc' in df.columns:
        ax3.plot(df['epoch'], df['val_mcc'], label='Val MCC', marker='s', markersize=3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MCC')
    ax3.legend()
    ax3.set_title('MCC Curve')
    ax3.grid(True, alpha=0.3)
    
    # AUC-ROC 曲线
    ax4 = axes[1, 1]
    if 'val_auroc' in df.columns:
        ax4.plot(df['epoch'], df['val_auroc'], label='Val AUC-ROC', marker='o', markersize=3, color='green')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('AUC-ROC')
    ax4.legend()
    ax4.set_title('AUC-ROC Curve')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Curves - {trial_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")


def plot_comparison(csv_paths, labels, output_path, title="Comparison"):
    """绘制多个试验的对比曲线。
    
    参数:
        csv_paths: 多个 metrics.csv 文件路径列表
        labels: 每个试验的标签列表
        output_path: 输出图片路径
        title: 图表标题
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10.colors
    
    for i, (csv_path, label) in enumerate(zip(csv_paths, labels)):
        df = pd.read_csv(csv_path)
        if 'epoch' in df.columns:
            df = df.groupby('epoch').last().reset_index()
        
        color = colors[i % len(colors)]
        
        # Loss
        if 'val_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['val_loss'], label=label, 
                           marker='o', markersize=3, color=color)
        
        # Accuracy
        if 'val_acc' in df.columns:
            axes[0, 1].plot(df['epoch'], df['val_acc'], label=label, 
                           marker='o', markersize=3, color=color)
        
        # MCC
        if 'val_mcc' in df.columns:
            axes[1, 0].plot(df['epoch'], df['val_mcc'], label=label, 
                           marker='o', markersize=3, color=color)
        
        # AUC-ROC
        if 'val_auroc' in df.columns:
            axes[1, 1].plot(df['epoch'], df['val_auroc'], label=label, 
                           marker='o', markersize=3, color=color)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Val Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Validation Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Val Accuracy')
    axes[0, 1].legend()
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Val MCC')
    axes[1, 0].legend()
    axes[1, 0].set_title('Validation MCC')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Val AUC-ROC')
    axes[1, 1].legend()
    axes[1, 1].set_title('Validation AUC-ROC')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存对比图: {output_path}")


def find_csv_logs(experiment_name="raytune_myrq1_-1"):
    """查找实验的所有 CSV 日志文件。
    
    参数:
        experiment_name: 实验名称
        
    返回:
        list: [(trial_name, csv_path), ...]
    """
    base_dir = svd.processed_dir() / experiment_name
    csv_files = glob(str(base_dir / "*" / "csv_logs" / "*" / "metrics.csv"))
    
    results = []
    for csv_path in csv_files:
        # 从路径中提取试验名称
        parts = csv_path.split(os.sep)
        trial_name = parts[-3]  # csv_logs 的上一级目录名
        results.append((trial_name, csv_path))
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="绘制训练曲线")
    parser.add_argument("--exp", type=str, default="raytune_myrq1_-1", 
                       help="实验名称，如 raytune_myrq1_-1")
    parser.add_argument("--compare", action="store_true", 
                       help="是否绘制对比图")
    parser.add_argument("--output", type=str, default="training_curves", 
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 查找所有 CSV 日志
    trials = find_csv_logs(args.exp)
    
    if not trials:
        print(f"未找到实验 {args.exp} 的 CSV 日志文件")
        print("请确保训练时使用了 CSVLogger")
        exit(1)
    
    print(f"找到 {len(trials)} 个试验的日志文件")
    
    # 绘制每个试验的曲线
    for trial_name, csv_path in trials:
        safe_name = trial_name.replace("/", "_").replace("\\", "_")
        output_path = os.path.join(args.output, f"{safe_name}.png")
        plot_single_trial(csv_path, output_path, trial_name)
    
    # 绘制对比图
    if args.compare and len(trials) > 1:
        csv_paths = [csv_path for _, csv_path in trials]
        labels = [name for name, _ in trials]
        output_path = os.path.join(args.output, "comparison.png")
        plot_comparison(csv_paths, labels, output_path, title=args.exp)
    
    print(f"\n所有图片已保存到: {args.output}/")
