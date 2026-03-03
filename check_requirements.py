#!/usr/bin/env python3
"""
检查 requirements.txt 中的包是否安装并输出版本信息
"""

import sys
import subprocess
import importlib
from pathlib import Path


def parse_requirements(filepath):
    """解析 requirements.txt 文件，提取包名"""
    packages = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith('#') or line.startswith('-'):
                    continue
                # 处理形如 "package>=version" 或 "package==version" 的情况
                # 提取包名（去除版本号）
                package_name = line.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].split('[')[0].strip()
                if package_name:
                    packages.append(package_name)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filepath}")
        return []
    except Exception as e:
        print(f"错误: 读取文件时出错: {e}")
        return []
    return packages


def check_package(package_name):
    """检查单个包是否安装并返回版本信息"""
    # 处理包名映射（有些导入名和包名不同）
    package_import_map = {
        'scikit-learn': 'sklearn',
        'python-Levenshtein': 'Levenshtein',
        'python-igraph': 'igraph',
        'pygraphviz': 'pygraphviz',
        'torch_scatter': 'torch_scatter',
        'ray[tune]': 'ray',
        'fastparquet': 'fastparquet',
        'imbalanced-learn': 'imblearn',
        'pandarallel': 'pandarallel',
        'unidiff': 'unidiff',
        'fuzzywuzzy': 'fuzzywuzzy',
        'libclang': 'clang',
        'tsne_torch': 'tsne_torch',
        'pytorch-lightning': 'pytorch_lightning',
        'torchsummary': 'torchsummary',
        'torchinfo': 'torchinfo',
        'ujson': 'ujson',
        'unidecode': 'unidecode',
        'dgl': 'dgl',
        'networkx': 'networkx',
        'pydot': 'pydot',
        'graphviz': 'graphviz',
        'tensorboard': 'tensorboard',
        'transformers': 'transformers',
        'torchtext': 'torchtext',
        'nltk': 'nltk',
        'gensim': 'gensim',
        'lightgbm': 'lightgbm',
        'xgboost': 'xgboost',
        'spacy': 'spacy',
        'joblib': 'joblib',
        'tqdm': 'tqdm',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'pytest': 'pytest',
        'jupyterlab': 'jupyterlab',
        'gdown': 'gdown',
        'ipywidgets': 'ipywidgets',
    }
    
    import_name = package_import_map.get(package_name, package_name)
    
    try:
        # 尝试导入包
        module = importlib.import_module(import_name)
        
        # 获取版本信息
        version = getattr(module, '__version__', None)
        if version is None:
            # 尝试其他常见的版本属性名
            version = getattr(module, 'VERSION', None) or getattr(module, 'version', '未知')
        
        return True, version
    except ImportError:
        return False, None
    except Exception as e:
        return False, f"导入错误: {e}"


def main():
    """主函数"""
    # 查找 requirements.txt 文件
    requirements_file = Path('requirements.txt')
    if not requirements_file.exists():
        # 尝试在其他位置查找
        possible_paths = [
            Path('../requirements.txt'),
            Path('~/linevd/requirements.txt').expanduser(),
            Path('/home/wmy/linevd/requirements.txt'),
        ]
        for path in possible_paths:
            if path.exists():
                requirements_file = path
                break
    
    if not requirements_file.exists():
        print("错误: 找不到 requirements.txt 文件")
        print("请确保在正确的目录下运行此脚本")
        sys.exit(1)
    
    print(f"使用 requirements 文件: {requirements_file}")
    print("=" * 80)
    
    # 解析 requirements.txt
    packages = parse_requirements(requirements_file)
    
    if not packages:
        print("没有解析到任何包")
        sys.exit(1)
    
    print(f"共找到 {len(packages)} 个包需要检查\n")
    
    # 检查结果统计
    installed = []
    not_installed = []
    
    # 检查每个包
    for i, package in enumerate(packages, 1):
        is_installed, version = check_package(package)
        status = "✅ 已安装" if is_installed else "❌ 未安装"
        version_str = f" (版本: {version})" if is_installed and version else ""
        
        print(f"{i:3d}. {package:25s} {status:10s}{version_str}")
        
        if is_installed:
            installed.append((package, version))
        else:
            not_installed.append(package)
    
    # 输出总结
    print("\n" + "=" * 80)
    print(f"总结:")
    print(f"  - 已安装: {len(installed)}/{len(packages)} 个包")
    print(f"  - 未安装: {len(not_installed)}/{len(packages)} 个包")
    
    if not_installed:
        print(f"\n未安装的包列表:")
        for pkg in not_installed:
            print(f"  - {pkg}")
    
    # 特别检查关键包
    print("\n" + "=" * 80)
    print("关键包版本信息:")
    key_packages = ['torch', 'dgl', 'numpy', 'scipy', 'pandas', 'sklearn']
    for pkg in key_packages:
        is_installed, version = check_package(pkg)
        if is_installed:
            print(f"  {pkg:15s}: {version}")
        else:
            print(f"  {pkg:15s}: 未安装 ⚠️")
    
    # 返回退出码
    if not_installed:
        print(f"\n⚠️  有 {len(not_installed)} 个包未安装，请安装这些包")
        sys.exit(1)
    else:
        print("\n✅ 所有包都已安装！")
        sys.exit(0)


if __name__ == '__main__':
    main()
