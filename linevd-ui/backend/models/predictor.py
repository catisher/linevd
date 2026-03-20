import os
import sys
import tempfile
import torch
from glob import glob
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import sastvd.linevd as lvd
import sastvd as svd
import sastvd.helpers.joern as svdj

class LineVDPredictor:
    """LineVD 模型预测器"""
    
    def __init__(self):
        """初始化预测器"""
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def load_model(self):
        """加载训练好的模型"""
        # 查找检查点文件
        checkpoint_files = self._find_checkpoint_files()
        
        if not checkpoint_files:
            raise Exception("未找到模型检查点文件")
        
        # 使用第一个找到的检查点
        checkpoint_path = checkpoint_files[0]
        print(f"加载模型检查点: {checkpoint_path}")
        
        # 加载模型
        self.model = lvd.LitGNN.load_from_checkpoint(checkpoint_path, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print("模型加载完成")
    
    def _find_checkpoint_files(self):
        """查找模型检查点文件"""
        # 搜索 raytune 目录
        raytune_dirs = glob(str(svd.processed_dir() / "raytune_*_-1"))
        checkpoint_files = []
        
        for base_dir in raytune_dirs:
            # 递归查找所有 checkpoint 文件
            trial_dirs = glob(f"{base_dir}/**/train_linevd_*", recursive=True)
            for trial_dir in trial_dirs:
                # 查找 checkpoint 子目录
                checkpoint_dirs = glob(f"{trial_dir}/checkpoint_*")
                for checkpoint_dir in checkpoint_dirs:
                    checkpoint_file = os.path.join(checkpoint_dir, "checkpoint")
                    if os.path.exists(checkpoint_file):
                        checkpoint_files.append(checkpoint_file)
        
        return checkpoint_files
    
    def predict(self, code: str, language: str = "c"):
        """对代码进行漏洞预测
        
        Args:
            code: 代码内容
            language: 代码语言
            
        Returns:
            预测结果字典
        """
        if self.model is None:
            raise Exception("模型未加载")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
            f.write(code)
            temp_file_path = f.name
        
        try:
            # 处理代码，生成图结构
            print(f"处理代码文件: {temp_file_path}")
            
            # 检查 Joern 输出文件是否存在，如果不存在则运行 Joern
            edges_file = temp_file_path + ".edges.json"
            nodes_file = temp_file_path + ".nodes.json"
            
            if not (os.path.exists(edges_file) and os.path.exists(nodes_file)):
                print("运行 Joern 生成代码属性图...")
                svdj.run_joern(temp_file_path, verbose=0)
                print("Joern 处理完成")
            
            g = lvd.feature_extraction(temp_file_path)
            g = g.to(self.device)
            
            # 进行预测
            with torch.no_grad():
                logits = self.model(g, test=True)
            
            # 处理预测结果
            preds = logits.argmax(dim=1).cpu().numpy()
            confidence = torch.softmax(logits, dim=1).cpu().numpy()
            
            # 获取行号
            line_numbers = g.ndata["_LINE_NUMBER"].cpu().numpy()
            
            # 构建结果
            results = []
            vulnerable_count = 0
            
            for line, pred, conf in zip(line_numbers, preds, confidence):
                pred_label = "VULNERABLE" if pred == 1 else "SAFE"
                conf_score = conf[1] if pred == 1 else conf[0]
                
                results.append({
                    "line": int(line),
                    "prediction": pred_label,
                    "confidence": float(conf_score)
                })
                
                if pred == 1:
                    vulnerable_count += 1
            
            # 构建摘要
            summary = {
                "total_lines": len(results),
                "vulnerable_lines": vulnerable_count,
                "safe_lines": len(results) - vulnerable_count
            }
            
            return {
                "results": results,
                "summary": summary
            }
            
        except Exception as e:
            raise Exception(f"预测失败: {str(e)}")
        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

# 测试代码
if __name__ == "__main__":
    predictor = LineVDPredictor()
    predictor.load_model()
    
    # 测试代码
    test_code = """
    #include <stdio.h>
    int main() {
        char buffer[10];
        gets(buffer);
        printf("%s", buffer);
        return 0;
    }
    """
    
    result = predictor.predict(test_code)
    print("预测结果:")
    print(result)