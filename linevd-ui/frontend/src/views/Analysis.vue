<template>
  <div class="analysis">
    <el-row :gutter="20">
      <!-- 左侧：代码编辑器 -->
      <el-col :span="14">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header">
              <span>代码编辑器</span>
              <div class="header-actions">
                <el-upload
                  action=""
                  :auto-upload="false"
                  :on-change="handleFileChange"
                  :show-file-list="false"
                  accept=".c,.cpp,.h"
                >
                  <el-button type="primary" plain>
                    <el-icon><Upload /></el-icon>
                    上传文件
                  </el-button>
                </el-upload>
                <el-button type="success" @click="analyzeCode" :loading="analyzing">
                  <el-icon><Search /></el-icon>
                  开始分析
                </el-button>
              </div>
            </div>
          </template>
          
          <div class="editor-container">
            <div class="line-numbers">
              <div 
                v-for="line in lineCount" 
                :key="line"
                :class="['line-number', { 'vulnerable': isVulnerableLine(line) }]"
              >
                {{ line }}
              </div>
            </div>
            <textarea
              v-model="code"
              class="code-editor"
              placeholder="请输入 C/C++ 代码..."
              @input="updateLineCount"
              @scroll="syncScroll"
              ref="codeEditor"
            ></textarea>
          </div>
        </el-card>
      </el-col>
      
      <!-- 右侧：分析结果 -->
      <el-col :span="10">
        <el-card shadow="hover" class="results-card">
          <template #header>
            <div class="card-header">
              <span>分析结果</span>
              <el-tag v-if="analysisComplete" type="success">分析完成</el-tag>
            </div>
          </template>
          
          <div v-if="!analysisComplete" class="empty-state">
            <el-icon size="48" color="#909399"><InfoFilled /></el-icon>
            <p>点击"开始分析"检测代码漏洞</p>
          </div>
          
          <div v-else class="results-content">
            <!-- 统计信息 -->
            <el-row :gutter="10" class="stats">
              <el-col :span="8">
                <div class="stat-item high">
                  <div class="stat-number">{{ highRiskCount }}</div>
                  <div class="stat-label">高危</div>
                </div>
              </el-col>
              <el-col :span="8">
                <div class="stat-item medium">
                  <div class="stat-number">{{ mediumRiskCount }}</div>
                  <div class="stat-label">中危</div>
                </div>
              </el-col>
              <el-col :span="8">
                <div class="stat-item low">
                  <div class="stat-number">{{ lowRiskCount }}</div>
                  <div class="stat-label">低危</div>
                </div>
              </el-col>
            </el-row>
            
            <!-- 漏洞列表 -->
            <div class="vulnerability-list">
              <h4>检测到的漏洞</h4>
              <el-timeline>
                <el-timeline-item
                  v-for="(vuln, index) in vulnerabilities"
                  :key="index"
                  :type="getSeverityType(vuln.severity)"
                  :icon="getSeverityIcon(vuln.severity)"
                >
                  <div class="vulnerability-item">
                    <div class="vuln-header">
                      <el-tag :type="getSeverityType(vuln.severity)" size="small">
                        {{ vuln.severity }}
                      </el-tag>
                      <span class="line-number">第 {{ vuln.line }} 行</span>
                      <span class="confidence">置信度: {{ (vuln.confidence * 100).toFixed(1) }}%</span>
                    </div>
                    <div class="vuln-message">{{ vuln.message }}</div>
                    <div class="vuln-code">{{ vuln.code_snippet }}</div>
                  </div>
                </el-timeline-item>
              </el-timeline>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'Analysis',
  data() {
    return {
      code: `#include <stdio.h>
#include <string.h>

void vulnerable_function(char *input) {
    char buffer[10];
    strcpy(buffer, input);  // 高危：缓冲区溢出
    printf("%s\\n", buffer);
}

int main() {
    char user_input[100];
    gets(user_input);  // 高危：不安全的输入
    vulnerable_function(user_input);
    return 0;
}`,
      lineCount: 0,
      analyzing: false,
      analysisComplete: false,
      vulnerabilities: [],
      highRiskCount: 0,
      mediumRiskCount: 0,
      lowRiskCount: 0
    }
  },
  mounted() {
    this.updateLineCount()
  },
  methods: {
    updateLineCount() {
      this.lineCount = this.code.split('\n').length
    },
    syncScroll(e) {
      // 同步行号滚动
      const lineNumbers = this.$el.querySelector('.line-numbers')
      if (lineNumbers) {
        lineNumbers.scrollTop = e.target.scrollTop
      }
    },
    isVulnerableLine(line) {
      return this.vulnerabilities.some(v => v.line === line)
    },
    getSeverityType(severity) {
      const map = {
        'High': 'danger',
        'Medium': 'warning',
        'Low': 'info'
      }
      return map[severity] || 'info'
    },
    getSeverityIcon(severity) {
      const map = {
        'High': 'CircleCloseFilled',
        'Medium': 'WarningFilled',
        'Low': 'InfoFilled'
      }
      return map[severity] || 'InfoFilled'
    },
    handleFileChange(file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        this.code = e.target.result
        this.updateLineCount()
      }
      reader.readAsText(file.raw)
    },
    async analyzeCode() {
      this.analyzing = true
      try {
        const response = await axios.post('http://localhost:8000/analyze', {
          code: this.code,
          filename: 'test.c'
        })
        
        this.vulnerabilities = response.data.vulnerabilities
        this.highRiskCount = this.vulnerabilities.filter(v => v.severity === 'High').length
        this.mediumRiskCount = this.vulnerabilities.filter(v => v.severity === 'Medium').length
        this.lowRiskCount = this.vulnerabilities.filter(v => v.severity === 'Low').length
        this.analysisComplete = true
        
        this.$message.success('分析完成！')
      } catch (error) {
        this.$message.error('分析失败：' + error.message)
        
        // 模拟数据（用于演示）
        this.vulnerabilities = [
          {
            line: 6,
            severity: 'High',
            message: 'Buffer overflow vulnerability: strcpy(buffer, input)',
            confidence: 0.95,
            code_snippet: 'strcpy(buffer, input);'
          },
          {
            line: 13,
            severity: 'High',
            message: 'Unsafe input function: gets(user_input)',
            confidence: 0.92,
            code_snippet: 'gets(user_input);'
          }
        ]
        this.highRiskCount = 2
        this.mediumRiskCount = 0
        this.lowRiskCount = 0
        this.analysisComplete = true
      } finally {
        this.analyzing = false
      }
    }
  }
}
</script>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-actions {
  display: flex;
  gap: 10px;
}

.editor-container {
  display: flex;
  height: 500px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  overflow: hidden;
}

.line-numbers {
  width: 50px;
  background-color: #f5f7fa;
  border-right: 1px solid #dcdfe6;
  padding: 10px 0;
  overflow: hidden;
  text-align: center;
  font-family: 'Courier New', monospace;
  font-size: 14px;
  line-height: 1.5;
}

.line-number {
  height: 21px;
  color: #909399;
}

.line-number.vulnerable {
  background-color: #fde2e2;
  color: #f56c6c;
  font-weight: bold;
}

.code-editor {
  flex: 1;
  border: none;
  outline: none;
  padding: 10px;
  font-family: 'Courier New', monospace;
  font-size: 14px;
  line-height: 1.5;
  resize: none;
  background-color: #fafafa;
}

.results-card {
  height: 100%;
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: #909399;
}

.stats {
  margin-bottom: 20px;
}

.stat-item {
  text-align: center;
  padding: 15px;
  border-radius: 8px;
  color: white;
}

.stat-item.high {
  background: linear-gradient(135deg, #f56c6c 0%, #e6a23c 100%);
}

.stat-item.medium {
  background: linear-gradient(135deg, #e6a23c 0%, #67c23a 100%);
}

.stat-item.low {
  background: linear-gradient(135deg, #67c23a 0%, #409eff 100%);
}

.stat-number {
  font-size: 24px;
  font-weight: bold;
}

.stat-label {
  font-size: 12px;
  margin-top: 5px;
}

.vulnerability-list {
  margin-top: 20px;
}

.vulnerability-list h4 {
  margin-bottom: 15px;
  color: #303133;
}

.vulnerability-item {
  padding: 10px;
  background: #f5f7fa;
  border-radius: 4px;
}

.vuln-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}

.line-number {
  color: #606266;
  font-size: 12px;
}

.confidence {
  color: #909399;
  font-size: 12px;
}

.vuln-message {
  color: #303133;
  font-size: 14px;
  margin-bottom: 5px;
}

.vuln-code {
  font-family: 'Courier New', monospace;
  font-size: 12px;
  color: #606266;
  background: #fff;
  padding: 5px;
  border-radius: 3px;
  border-left: 3px solid #409eff;
}
</style>
