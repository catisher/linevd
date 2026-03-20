# LineVD API 服务

使用 FastAPI 构建的 LineVD 漏洞检测 API 服务，支持前端提交代码并返回漏洞检测结果。

## 目录结构
```
linevd-ui/
├── backend/
│   ├── main.py          # API 主入口
│   ├── models/          # 模型加载和预测逻辑
│   ├── schemas/         # 请求和响应数据模型
│   └── utils/           # 工具函数
└── frontend/            # 前端代码（可选）
```

## 安装依赖

```bash
# 安装 FastAPI 和相关依赖
pip install fastapi uvicorn python-multipart

# 安装 LineVD 依赖
pip install -r ../../requirements.txt
```

## 运行服务

```bash
# 启动 API 服务
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API 端点

### 1. 健康检查
- **路径**: `/health`
- **方法**: GET
- **响应**: `{"status": "ok"}`

### 2. 漏洞检测
- **路径**: `/predict`
- **方法**: POST
- **请求体**:
  ```json
  {
    "code": "#include <stdio.h>\nint main() {\n  char buffer[10];\n  gets(buffer);\n  printf(\"%s\", buffer);\n  return 0;\n}",
    "language": "c"
  }
  ```
- **响应**:
  ```json
  {
    "status": "success",
    "results": [
      {
        "line": 4,
        "prediction": "VULNERABLE",
        "confidence": 0.9876
      },
      {
        "line": 5,
        "prediction": "SAFE",
        "confidence": 0.9999
      }
    ],
    "summary": {
      "total_lines": 6,
      "vulnerable_lines": 1,
      "safe_lines": 5
    }
  }
  ```

## 模型加载

服务启动时会自动加载训练好的模型检查点。默认使用 `storage/processed/raytune_best_-1/` 目录下的最佳模型。

## 注意事项

1. **模型加载时间**: 首次启动服务时，模型加载可能需要几分钟时间
2. **内存要求**: 建议在至少 16GB 内存的环境中运行
3. **GPU 支持**: 如果可用，服务会自动使用 GPU 加速
4. **代码大小限制**: 单次请求的代码大小建议不超过 1MB

## 部署

### Docker 部署

```bash
# 构建镜像
docker build -t linevd-api .

# 运行容器
docker run -p 8000:8000 linevd-api
```

### Kubernetes 部署

参考 `kubernetes/` 目录下的部署配置文件。