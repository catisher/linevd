print("Hello, World!")

import sys
import os
import platform
import site
import subprocess
from importlib.metadata import version, PackageNotFoundError

def print_separator(title):
    """打印分隔符，让输出更清晰"""
    print(f"\n{'='*20} {title} {'='*20}")

# 1. 基础 Python 信息
print_separator("基础 Python 配置")
print(f"Python 版本: {sys.version}")
print(f"Python 版本号: {sys.version_info}")
print(f"Python 可执行文件路径: {sys.executable}")
print(f"操作系统: {platform.system()} {platform.release()}")
print(f"系统架构: {platform.machine()}")
print(f"当前工作目录: {os.getcwd()}")

# 2. Python 环境变量（关键：虚拟环境/路径）
print_separator("Python 环境变量")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '未设置')}")
print(f"VIRTUAL_ENV (虚拟环境标识): {os.environ.get('VIRTUAL_ENV', '未激活虚拟环境')}")
print(f"PATH 中的 Python 路径:")
path_list = os.environ.get('PATH', '').split(':')
for p in path_list:
    if 'python' in p.lower() or 'venv' in p.lower():
        print(f"  - {p}")

# 3. Python 模块搜索路径（import 时的查找顺序）
print_separator("Python 模块搜索路径 (sys.path)")
for idx, path in enumerate(sys.path):
    print(f"{idx+1}. {path}")

# 4. 已安装的核心依赖包版本（适配 LineVD 项目）
print_separator("LineVD 核心依赖版本")
core_packages = [
    'torch', 'dgl', 'pandas', 'numpy', 'scikit-learn',
    'transformers', 'clang', 'tqdm', 'click', 'tree-sitter'
]
for pkg in core_packages:
    try:
        # 优先用 importlib.metadata（Python 3.8+ 内置）
        ver = version(pkg)
        print(f"{pkg}: {ver}")
    except PackageNotFoundError:
        # 兼容部分特殊包（如 torch 可能需要导入后查版本）
        try:
            mod = __import__(pkg)
            ver = getattr(mod, '__version__', '未知版本')
            print(f"{pkg}: {ver}")
        except ImportError:
            print(f"{pkg}: 未安装")

# 5. 虚拟环境详情（若激活）
print_separator("虚拟环境详情")
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path:
    print(f"虚拟环境路径: {venv_path}")
    print(f"虚拟环境 site-packages 路径: {site.getsitepackages()}")
    # 检查虚拟环境是否为当前 Python 所用
    if sys.prefix == venv_path:
        print("✅ 当前 Python 正在使用该虚拟环境")
    else:
        print("❌ 当前 Python 未使用该虚拟环境（路径不匹配）")
else:
    print("未激活虚拟环境")

# 6. 编译器/构建工具信息（适配 LineVD 的 C/C++ 扩展依赖）
print_separator("编译器/构建工具")
try:
    gcc_version = subprocess.check_output(
        ['gcc', '--version'], stderr=subprocess.STDOUT, text=True
    ).split('\n')[0]
    print(f"GCC: {gcc_version}")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("GCC: 未找到")

try:
    llvm_version = subprocess.check_output(
        ['llvm-config', '--version'], stderr=subprocess.STDOUT, text=True
    ).strip()
    print(f"LLVM: {llvm_version}")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("LLVM: 未找到（可能未加入环境变量）")

# 7. 可选：GPU 可用性（PyTorch）
print_separator("GPU 可用性（PyTorch）")
try:
    import torch
    print(f"PyTorch CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch 未安装，无法检测 GPU")

# 8. 总结
print_separator("配置总结")
if venv_path and sys.prefix == venv_path:
    print("✅ 虚拟环境配置正常")
else:
    print("⚠️  虚拟环境未激活/配置异常（LineVD 建议用虚拟环境运行）")

if 'torch' in sys.modules and not torch.cuda.is_available():
    print("⚠️  PyTorch 未启用 GPU，将使用 CPU 训练 LineVD（耗时更长）")

print("\n📌 关键提示：若依赖包显示'未安装'，需在虚拟环境中执行 pip install 安装")
