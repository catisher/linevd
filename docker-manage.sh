#!/bin/bash

# LineVD Docker快速启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    print_info "Docker已安装: $(docker --version)"
}

# 检查NVIDIA Docker支持
check_nvidia_docker() {
    if docker run --rm --gpus all nvidia/cuda:10.2-base nvidia-smi &> /dev/null; then
        print_info "NVIDIA Docker支持已启用"
        return 0
    else
        print_warn "NVIDIA Docker支持未启用，将使用CPU模式"
        return 1
    fi
}

# 构建Docker镜像
build_image() {
    print_info "开始构建Docker镜像..."
    if [ -f "docker-compose.yml" ]; then
        docker-compose build
    else
        docker build -t linevd:latest .
    fi
    print_info "Docker镜像构建完成"
}

# 启动容器
start_container() {
    print_info "启动LineVD容器..."
    if [ -f "docker-compose.yml" ]; then
        docker-compose up -d
        print_info "容器已启动，使用 'docker-compose exec linevd bash' 进入容器"
    else
        if check_nvidia_docker; then
            docker run -d --name linevd_container \
                --gpus all \
                -v $(pwd):/workspace \
                -v $(pwd)/storage:/workspace/storage \
                linevd:latest \
                tail -f /dev/null
        else
            docker run -d --name linevd_container \
                -v $(pwd):/workspace \
                -v $(pwd)/storage:/workspace/storage \
                linevd:latest \
                tail -f /dev/null
        fi
        print_info "容器已启动，使用 'docker exec -it linevd_container bash' 进入容器"
    fi
}

# 停止容器
stop_container() {
    print_info "停止LineVD容器..."
    if [ -f "docker-compose.yml" ]; then
        docker-compose down
    else
        docker stop linevd_container 2>/dev/null || true
        docker rm linevd_container 2>/dev/null || true
    fi
    print_info "容器已停止"
}

# 进入容器
enter_container() {
    if [ -f "docker-compose.yml" ]; then
        docker-compose exec linevd bash
    else
        docker exec -it linevd_container bash
    fi
}

# 初始化环境
init_environment() {
    print_info "初始化LineVD环境..."
    if [ -f "docker-compose.yml" ]; then
        docker-compose exec linevd bash /cli.sh -p initialise
    else
        docker exec -it linevd_container bash /cli.sh -p initialise
    fi
}

# 运行训练
run_training() {
    print_info "开始训练LineVD模型..."
    if [ -f "docker-compose.yml" ]; then
        docker-compose exec linevd python sastvd/scripts/train_best.py
    else
        docker exec -it linevd_container python sastvd/scripts/train_best.py
    fi
}

# 运行测试
run_tests() {
    print_info "运行测试..."
    if [ -f "docker-compose.yml" ]; then
        docker-compose exec linevd pytest tests/
    else
        docker exec -it linevd_container pytest tests/
    fi
}

# 显示帮助信息
show_help() {
    echo "LineVD Docker管理脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  build       构建Docker镜像"
    echo "  start       启动容器"
    echo "  stop        停止容器"
    echo "  restart     重启容器"
    echo "  enter       进入容器"
    echo "  init        初始化环境（首次运行）"
    echo "  train       运行模型训练"
    echo "  test        运行测试"
    echo "  status      查看容器状态"
    echo "  logs        查看容器日志"
    echo "  clean       清理容器和镜像"
    echo "  help        显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 build && $0 start && $0 enter"
    echo "  $0 init"
    echo "  $0 train"
}

# 查看状态
show_status() {
    print_info "容器状态:"
    if [ -f "docker-compose.yml" ]; then
        docker-compose ps
    else
        docker ps -a | grep linevd_container || echo "容器未运行"
    fi
}

# 查看日志
show_logs() {
    if [ -f "docker-compose.yml" ]; then
        docker-compose logs -f
    else
        docker logs -f linevd_container
    fi
}

# 清理
clean_all() {
    print_warn "这将删除所有LineVD相关的容器和镜像"
    read -p "确定要继续吗? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        stop_container
        docker rmi linevd:latest 2>/dev/null || true
        print_info "清理完成"
    fi
}

# 主函数
main() {
    case "${1:-help}" in
        build)
            check_docker
            build_image
            ;;
        start)
            check_docker
            start_container
            ;;
        stop)
            stop_container
            ;;
        restart)
            stop_container
            start_container
            ;;
        enter)
            enter_container
            ;;
        init)
            init_environment
            ;;
        train)
            run_training
            ;;
        test)
            run_tests
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        clean)
            clean_all
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"