from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from schemas.__init__ import CodeRequest, PredictionResponse
from models.predictor import LineVDPredictor

# 创建 FastAPI 应用
app = FastAPI(
    title="LineVD API",
    description="漏洞检测 API 服务",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化模型预测器
predictor = LineVDPredictor()

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    print("正在加载 LineVD 模型...")
    try:
        predictor.load_model()
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: CodeRequest):
    """漏洞检测端点"""
    try:
        # 执行预测
        results = predictor.predict(request.code, request.language)
        
        # 构建响应
        response = PredictionResponse(
            status="success",
            results=results["results"],
            summary=results["summary"]
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)