#!/bin/bash

# LineVD Web UI 启动脚本

echo "=========================================="
echo "  LineVD 漏洞检测系统"
echo "=========================================="
echo ""

# 检查是否安装了必要的工具
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "错误: 未找到 npm"
    exit 1
fi

echo "启动后端服务..."
cd backend
pip install -r requirements.txt -q
python main.py &
BACKEND_PID=$!
echo "后端服务已启动 (PID: $BACKEND_PID)"
echo "后端地址: http://localhost:8000"
echo ""

echo "启动前端服务..."
cd ../frontend
npm install -q
npm run serve &
FRONTEND_PID=$!
echo "前端服务已启动 (PID: $FRONTEND_PID)"
echo "前端地址: http://localhost:8080"
echo ""

echo "=========================================="
echo "  服务启动完成!"
echo "  前端: http://localhost:8080"
echo "  后端: http://localhost:8000"
echo "=========================================="
echo ""
echo "按 Ctrl+C 停止服务"

# 等待用户中断
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
