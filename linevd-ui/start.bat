@echo off
chcp 65001 >nul
echo ==========================================
echo   LineVD 漏洞检测系统
echo ==========================================
echo.

echo 启动后端服务...
cd backend
start "LineVD Backend" cmd /k "pip install -r requirements.txt && python main.py"
echo 后端服务已启动
echo 后端地址: http://localhost:8000
echo.

echo 启动前端服务...
cd ../frontend
start "LineVD Frontend" cmd /k "npm install && npm run serve"
echo 前端服务已启动
echo 前端地址: http://localhost:8080
echo.

echo ==========================================
echo   服务启动完成!
echo   前端: http://localhost:8080
echo   后端: http://localhost:8000
echo ==========================================
echo.
echo 按任意键关闭此窗口（服务将在后台继续运行）
pause >nul
