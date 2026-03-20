from pydantic import BaseModel
from typing import List, Optional

class CodeRequest(BaseModel):
    """代码请求模型"""
    code: str  # 代码内容
    language: str = "c"  # 代码语言，默认为 C

class PredictionResult(BaseModel):
    """预测结果模型"""
    line: int  # 行号
    prediction: str  # 预测结果：VULNERABLE 或 SAFE
    confidence: float  # 置信度

class PredictionSummary(BaseModel):
    """预测摘要模型"""
    total_lines: int  # 总行数
    vulnerable_lines: int  # 漏洞行数
    safe_lines: int  # 安全行数

class PredictionResponse(BaseModel):
    """预测响应模型"""
    status: str  # 状态：success 或 error
    results: List[PredictionResult]  # 预测结果列表
    summary: PredictionSummary  # 预测摘要