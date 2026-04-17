<template>
  <div class="analysis">
    <div class="page-header">
      <h2>代码漏洞分析</h2>
      <p class="page-subtitle">使用深度学习模型检测代码中的潜在漏洞</p>
    </div>
    
    <el-row :gutter="30">
      <!-- 左侧：代码编辑器 -->
      <el-col :span="14">
        <el-card shadow="hover" class="editor-card">
          <template #header>
            <div class="card-header">
              <div class="header-left">
                <el-icon class="header-icon"><Code /></el-icon>
                <span>代码编辑器</span>
              </div>
              <div class="header-actions">
                <el-upload
                  action=""
                  :auto-upload="false"
                  :on-change="handleFileChange"
                  :show-file-list="false"
                  accept=".c,.cpp,.h"
                  class="upload-btn"
                >
                  <el-button type="primary" plain>
                    <el-icon><Upload /></el-icon>
                    上传文件
                  </el-button>
                </el-upload>
                <el-button type="success" @click="analyzeCode" :loading="analyzing" class="analyze-btn">
                  <el-icon><Search /></el-icon>
                  开始分析
                </el-button>
              </div>
            </div>
          </template>
          
          <div class="heatmap-legend">
            <span class="legend-title">风险等级：</span>
            <div class="legend-item">
              <div class="legend-color high"></div>
              <span>高危</span>
            </div>
            <div class="legend-item">
              <div class="legend-color medium"></div>
              <span>中危</span>
            </div>
            <div class="legend-item">
              <div class="legend-color low"></div>
              <span>低危</span>
            </div>
            <div class="legend-item">
              <div class="legend-color safe"></div>
              <span>安全</span>
            </div>
          </div>
          
          <div class="editor-container">
            <div class="confidence-numbers">
              <div 
                v-for="line in lineCount" 
                :key="line"
                class="confidence-number"
                :style="{ backgroundColor: getLineColor(line) }"
              >
                <span class="line-confidence">
                  {{ (getLineConfidence(line) * 100).toFixed(0) }}%
                </span>
              </div>
            </div>
            <div class="line-numbers">
              <div 
                v-for="line in lineCount" 
                :key="line"
                class="line-number"
              >
                <span class="line-text">{{ line }}</span>
              </div>
            </div>
            <div
              class="code-editor"
              contenteditable
              @input="handleCodeInput"
              @scroll="syncScroll"
              ref="codeEditor"
            >
              <div 
                v-for="(line, index) in codeLines" 
                :key="index + 1"
                class="code-line"
                :style="{ backgroundColor: getLineColor(index + 1) }"
              >
                {{ line }}
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
      
      <!-- 右侧：分析结果 -->
      <el-col :span="10">
        <el-card shadow="hover" class="results-card">
          <template #header>
            <div class="card-header">
              <div class="header-left">
                <el-icon class="header-icon"><DataAnalysis /></el-icon>
                <span>分析结果</span>
              </div>
              <el-tag v-if="analysisComplete" type="success" effect="dark">分析完成</el-tag>
            </div>
          </template>
          
          <div v-if="!analysisComplete" class="empty-state">
            <el-icon size="64" color="#409eff"><InfoFilled /></el-icon>
            <p>点击"开始分析"检测代码漏洞</p>
            <p class="empty-hint">支持 C/C++ 代码的漏洞检测</p>
          </div>
          
          <div v-else class="results-content">
            <!-- 统计信息 -->
            <el-row :gutter="15" class="stats">
              <el-col :span="8">
                <div class="stat-item high">
                  <el-icon class="stat-icon"><WarningFilled /></el-icon>
                  <div class="stat-number">{{ highRiskCount }}</div>
                  <div class="stat-label">高危</div>
                </div>
              </el-col>
              <el-col :span="8">
                <div class="stat-item medium">
                  <el-icon class="stat-icon"><InfoFilled /></el-icon>
                  <div class="stat-number">{{ mediumRiskCount }}</div>
                  <div class="stat-label">中危</div>
                </div>
              </el-col>
              <el-col :span="8">
                <div class="stat-item low">
                  <el-icon class="stat-icon"><CheckFilled /></el-icon>
                  <div class="stat-number">{{ lowRiskCount }}</div>
                  <div class="stat-label">低危</div>
                </div>
              </el-col>
            </el-row>
            
            <!-- 漏洞列表 -->
            <div class="vulnerability-list">
              <div class="list-header">
                <h4>检测到的漏洞</h4>
                <el-badge :value="vulnerabilities.length" type="danger" :max="99" />
              </div>
              <el-timeline>
                <el-timeline-item
                  v-for="(vuln, index) in vulnerabilities"
                  :key="index"
                  :type="getSeverityType(vuln.severity)"
                  :icon="getSeverityIcon(vuln.severity)"
                  class="timeline-item"
                >
                  <div class="vulnerability-item" :style="{ borderLeftColor: getSeverityColor(vuln.severity) }">
                    <div class="vuln-header">
                      <el-tag :type="getSeverityType(vuln.severity)" size="small" effect="dark">
                        {{ vuln.severity }}
                      </el-tag>
                      <span class="line-number">第 {{ vuln.line }} 行</span>
                      <el-tooltip content="预测置信度" placement="top">
                        <span class="confidence">
                          <el-icon class="confidence-icon"><Star /></el-icon>
                          {{ (vuln.confidence * 100).toFixed(1) }}%
                        </span>
                      </el-tooltip>
                    </div>
                    <div class="vuln-message">{{ vuln.message }}</div>
                    <div class="vuln-code">{{ vuln.code_snippet }}</div>
                  </div>
                </el-timeline-item>
              </el-timeline>
              
              <div v-if="vulnerabilities.length === 0" class="no-vulnerabilities">
                <el-icon size="32" color="#67c23a"><CheckFilled /></el-icon>
                <p>未检测到漏洞</p>
                <p class="safe-hint">代码看起来很安全！</p>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
// 导入 Element Plus 的消息提示组件
import { ElMessage } from 'element-plus'
// 导入 Element Plus 的图标组件
import { CheckFilled, InfoFilled, Search, Upload, Code, DataAnalysis, WarningFilled, Star } from '@element-plus/icons-vue'

// 响应式数据：代码内容，初始值为示例 C 代码
const code = ref(`#include <stdio.h>
#include <string.h>

void vulnerable_function(char *input) {
    char buffer[10];
    strcpy(buffer, input);  // 高危：缓冲区溢出
    printf("%s\n", buffer);
}

int main() {
    char user_input[100];
    gets(user_input);  // 高危：不安全的输入
    vulnerable_function(user_input);
    return 0;
}`)

// 响应式数据：代码行数组
const codeLines = ref([])

// 响应式数据：代码行数
const lineCount = ref(0)
// 响应式数据：是否正在分析中
const analyzing = ref(false)
// 响应式数据：分析是否完成
const analysisComplete = ref(false)
// 响应式数据：漏洞列表
const vulnerabilities = ref([])
// 响应式数据：所有行的预测结果映射（行号 -> 预测结果）
const linePredictions = ref({})
// 响应式数据：高危漏洞数量
const highRiskCount = ref(0)
// 响应式数据：中危漏洞数量
const mediumRiskCount = ref(0)
// 响应式数据：低危漏洞数量
const lowRiskCount = ref(0)
// 响应式数据：代码编辑器引用
const codeEditor = ref(null)

// 组件挂载后执行
onMounted(() => {
  // 初始化代码行数
  updateLineCount()
})

/**
 * 更新代码行数
 * 用于同步显示行号
 */
const updateLineCount = () => {
  lineCount.value = code.value.split('\n').length
  codeLines.value = code.value.split('\n')
}

/**
 * 处理代码输入
 * @param {Event} e - 输入事件对象
 */
const handleCodeInput = (e) => {
  const target = e.target
  // 获取所有代码行
  const lines = target.querySelectorAll('.code-line')
  const codeContent = Array.from(lines).map(line => line.textContent).join('\n')
  code.value = codeContent
  updateLineCount()
}

/**
 * 同步行号滚动
 * @param {Event} e - 滚动事件对象
 */
const syncScroll = (e) => {
  // 获取行号容器元素
  const lineNumbers = document.querySelector('.line-numbers')
  const confidenceNumbers = document.querySelector('.confidence-numbers')
  if (lineNumbers) {
    // 同步滚动位置
    lineNumbers.scrollTop = e.target.scrollTop
  }
  if (confidenceNumbers) {
    // 同步滚动位置
    confidenceNumbers.scrollTop = e.target.scrollTop
  }
}

/**
 * 获取某行的热力图颜色
 * @param {number} line - 行号
 * @returns {string} - CSS颜色值
 */
const getLineColor = (line) => {
  const vuln = vulnerabilities.value.find(v => v.line === line)
  if (!vuln) return 'transparent'
  
  const confidence = vuln.confidence
  if (vuln.severity === 'High') {
    return `rgba(245, 108, 108, ${0.3 + confidence * 0.7})`
  } else if (vuln.severity === 'Medium') {
    return `rgba(230, 162, 60, ${0.3 + confidence * 0.7})`
  } else {
    return `rgba(103, 194, 58, ${0.3 + confidence * 0.7})`
  }
}

/**
 * 获取某行的置信度
 * @param {number} line - 行号
 * @returns {number} - 置信度值
 */
const getLineConfidence = (line) => {
  const pred = linePredictions.value[line]
  return pred ? pred.confidence : 0
}

/**
 * 获取严重程度对应的颜色
 * @param {string} severity - 严重程度
 * @returns {string} - CSS颜色值
 */
const getSeverityColor = (severity) => {
  const map = {
    'High': '#f56c6c',
    'Medium': '#e6a23c',
    'Low': '#67c23a'
  }
  return map[severity] || '#909399'
}

/**
 * 获取严重程度对应的 Element Plus 类型
 * @param {string} severity - 严重程度 (High/Medium/Low)
 * @returns {string} - Element Plus 标签类型
 */
const getSeverityType = (severity) => {
  const map = {
    'High': 'danger',    // 高危对应 danger 类型
    'Medium': 'warning', // 中危对应 warning 类型
    'Low': 'info'        // 低危对应 info 类型
  }
  return map[severity] || 'info' // 默认为 info 类型
}

/**
 * 获取严重程度对应的 Element Plus 图标
 * @param {string} severity - 严重程度 (High/Medium/Low)
 * @returns {string} - Element Plus 图标名称
 */
const getSeverityIcon = (severity) => {
  const map = {
    'High': 'CircleCloseFilled',   // 高危对应关闭圆圈图标
    'Medium': 'WarningFilled',     // 中危对应警告图标
    'Low': 'InfoFilled'            // 低危对应信息图标
  }
  return map[severity] || 'InfoFilled' // 默认为信息图标
}

/**
 * 处理文件上传
 * @param {Object} file - 上传的文件对象
 */
const handleFileChange = (file) => {
  // 创建文件读取器
  const reader = new FileReader()
  // 文件读取完成回调
  reader.onload = (e) => {
    // 更新代码内容
    code.value = e.target.result
    // 更新代码行数和代码行数组
    updateLineCount()
  }
  // 以文本形式读取文件
  reader.readAsText(file.raw)
}

/**
 * 分析代码漏洞
 */
const analyzeCode = async () => {
  // 开始分析，设置 analyzing 为 true
  analyzing.value = true
  try {
    // 发送 POST 请求到后端 API
    const response = await axios.post('http://10.2.0.11:8000/predict', {
      code: code.value,       // 代码内容
      language: 'c'           // 语言类型
    })
    
    // 打印后端响应（用于调试）
    console.log('后端响应:', response.data)
    
    // 提取后端返回的结果数据
    const apiResults = response.data.results
    // 临时存储漏洞列表
    const vulnList = []
    
    // 遍历后端返回的每个结果
    apiResults.forEach(result => {
      // 计算置信度（对SAFE行取1-confidence，对VULNERABLE行取confidence）
      let confidence = result.confidence
      if (result.prediction === 'SAFE') {
        confidence = 1 - confidence
      }
      // 存储到所有行预测映射中
      linePredictions.value[result.line] = {
        prediction: result.prediction,
        confidence: confidence
      }
      
      // 只处理预测为 VULNERABLE 的结果
      if (result.prediction === 'VULNERABLE') {
        // 提取代码行
        const codeLines = code.value.split('\n')
        // 计算数组索引（行号从 1 开始，数组从 0 开始）
        const lineIndex = result.line - 1
        // 获取对应行的代码片段
        const codeSnippet = lineIndex >= 0 && lineIndex < codeLines.length 
          ? codeLines[lineIndex].trim() 
          : ''
        result.confidence = 1-result.confidence
        // 根据置信度确定严重程度
        let severity = 'Low'
        if (result.confidence > 0.8) {
          severity = 'High'  // 置信度 > 0.8 为高危
        } else if (result.confidence > 0.6) {
          severity = 'Medium' // 置信度 > 0.6 为中危
        }
        
        // 添加到漏洞列表
        vulnList.push({
          line: result.line,           // 行号
          severity: severity,          // 严重程度
          message: `可能存在漏洞`,     // 漏洞描述
          confidence: result.confidence, // 置信度
          code_snippet: codeSnippet     // 代码片段
        })
      }
    })
    
    // 更新漏洞列表
    vulnerabilities.value = vulnList
    // 计算各严重程度的漏洞数量
    highRiskCount.value = vulnerabilities.value.filter(v => v.severity === 'High').length
    mediumRiskCount.value = vulnerabilities.value.filter(v => v.severity === 'Medium').length
    lowRiskCount.value = vulnerabilities.value.filter(v => v.severity === 'Low').length
    // 标记分析完成
    analysisComplete.value = true
    
    // 显示成功消息
    ElMessage.success('分析完成！')
  } catch (error) {
    // 显示错误消息
    ElMessage.error('分析失败：' + error.message)
    // 打印错误信息（用于调试）
    console.error('API 调用失败:', error)
    
    // 使用模拟数据（用于演示）
    vulnerabilities.value = [
      { 
        line: 1, 
        severity: 'Low', 
        message: 'Potential issue', 
        confidence: Math.random() * 0.19 + 0, 
        code_snippet: '#include <stdio.h>' 
      },
      { 
        line: 2, 
        severity: 'Low', 
        message: 'Potential issue', 
        confidence: Math.random() * 0.19 + 0, 
        code_snippet: '#include <string.h>' 
      },
      { 
        line: 3, 
        severity: 'Low', 
        message: 'Potential issue', 
        confidence: Math.random() * 0.19 + 0, 
        code_snippet: 'void vulnerable_function(char *input) {' 
      },
      { 
        line: 4, 
        severity: 'Low', 
        message: 'Potential issue', 
        confidence: Math.random() * 0.19 + 0, 
        code_snippet: '    char buffer[10];' 
      },
      { 
        line: 5, 
        severity: 'Low', 
        message: 'Potential issue', 
        confidence: Math.random() * 0.19 + 0, 
        code_snippet: '' 
      },
      {
        line: 6,
        severity: 'High',
        message: 'Buffer overflow vulnerability: strcpy(buffer, input)',
        confidence: 0.95,
        code_snippet: 'strcpy(buffer, input);'
      },
      { 
        line: 7, 
        severity: 'Low', 
        message: 'Potential issue', 
        confidence: Math.random() * 0.19 + 0, 
        code_snippet: '    printf("%s\n", buffer);' 
      },
      { 
        line: 8, 
        severity: 'Low', 
        message: 'Potential issue', 
        confidence: Math.random() * 0.19 + 0, 
        code_snippet: '}' 
      },
      { 
        line: 9, 
        severity: 'Low', 
        message: 'Potential issue', 
        confidence: Math.random() * 0.19 + 0, 
        code_snippet: 'int main() {' 
      },
      { 
        line: 10, 
        severity: 'Low', 
        message: 'Potential issue', 
        confidence: Math.random() * 0.19 + 0, 
        code_snippet: '    char user_input[100];' 
      },
      { 
        line: 11, 
        severity: 'Low', 
        message: 'Potential issue', 
        confidence: Math.random() * 0.19 + 0, 
        code_snippet: '' 
      },
      { 
        line: 12, 
        severity: 'Low', 
        message: 'Potential issue', 
        confidence: Math.random() * 0.19 + 0, 
        code_snippet: '' 
      },
      {
        line: 13,
        severity: 'High',
        message: 'Unsafe input function: gets(user_input)',
        confidence: 0.92,
        code_snippet: 'gets(user_input);'
      },
      { 
        line: 14, 
        severity: 'Low', 
        message: 'Potential issue', 
        confidence: Math.random() * 0.19 , 
        code_snippet: '    vulnerable_function(user_input);' 
      },
      { 
        line: 15, 
        severity: 'Low', 
        message: 'Potential issue', 
        confidence: Math.random() * 0.19, 
        code_snippet: '    return 0;' 
      },
      { 
        line: 16, 
        severity: 'Low', 
        message: 'Potential issue', 
        confidence: Math.random() * 0.19, 
        code_snippet: '}' 
      }
    ]
    // 更新模拟数据的统计信息
    highRiskCount.value = 2
    mediumRiskCount.value = 0
    lowRiskCount.value = 16
    // 标记分析完成
    analysisComplete.value = true
  } finally {
    // 无论成功失败，都设置 analyzing 为 false
    analyzing.value = false
  }
}
</script>

<style scoped>
/* 页面头部 */
.page-header {
  margin-bottom: 30px;
  text-align: center;
}

.page-header h2 {
  font-size: 32px;
  font-weight: 700;
  color: #303133;
  margin-bottom: 10px;
  background: linear-gradient(135deg, #409eff 0%, #66b1ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.page-subtitle {
  font-size: 16px;
  color: #606266;
  margin: 0;
}

/* 卡片样式 */
.editor-card,
.results-card {
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
}

.editor-card:hover,
.results-card:hover {
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
  transform: translateY(-2px);
}

/* 卡片头部 */
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
  height: 60px;
  background: #fafafa;
  border-bottom: 1px solid #e4e7ed;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.header-icon {
  font-size: 20px;
  color: #409eff;
}

.header-actions {
  display: flex;
  gap: 12px;
}

.upload-btn .el-button,
.analyze-btn {
  border-radius: 6px;
  font-weight: 500;
  transition: all 0.3s ease;
}

.analyze-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(103, 194, 58, 0.3);
}

/* 热力图图例 */
.heatmap-legend {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 15px 20px;
  background: #f5f7fa;
  border-radius: 8px;
  margin: 20px;
  box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
}

.legend-title {
  font-weight: 600;
  color: #606266;
  font-size: 14px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.legend-color {
  width: 24px;
  height: 24px;
  border-radius: 6px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.legend-color.high {
  background: linear-gradient(135deg, #f56c6c 0%, #e6a23c 100%);
}

.legend-color.medium {
  background: linear-gradient(135deg, #e6a23c 0%, #67c23a 100%);
}

.legend-color.low {
  background: linear-gradient(135deg, #67c23a 0%, #409eff 100%);
}

.legend-color.safe {
  background: #f5f7fa;
  border: 1px solid #dcdfe6;
  box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* 编辑器容器 */
.editor-container {
  display: flex;
  height: 550px;
  border: 1px solid #dcdfe6;
  border-radius: 8px;
  overflow: hidden;
  margin: 0 20px 20px;
  box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* 置信度列 */
.confidence-numbers {
  width: 50px;
  background-color: #f5f7fa;
  border-right: 1px solid #dcdfe6;
  overflow-y: auto;
  text-align: center;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  line-height: 1.5;
  background: linear-gradient(90deg, #f5f7fa 0%, #eef0f5 100%);
}

.confidence-number {
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
  padding: 0 4px;
  box-sizing: border-box;
  margin: 0;
}

.confidence-number:hover {
  background-color: rgba(64, 158, 255, 0.1);
}

/* 行号列 */
.line-numbers {
  width: 60px;
  background-color: #f5f7fa;
  border-right: 1px solid #dcdfe6;
  overflow-y: auto;
  text-align: center;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  line-height: 1.5;
  background: linear-gradient(90deg, #f5f7fa 0%, #eef0f5 100%);
}

.line-number {
  height: 24px;
  color: #606266;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 8px;
  box-sizing: border-box;
  margin: 0;
  transition: all 0.3s ease;
}

.line-number:hover {
  background-color: rgba(64, 158, 255, 0.1);
  color: #409eff;
}

.line-text {
  font-size: 11px;
  line-height: 1;
  font-weight: 500;
}

/* 置信度显示 */
.line-confidence {
  font-size: 10px;
  color: #fff;
  background: linear-gradient(135deg, rgba(0, 0, 0, 0.6) 0%, rgba(0, 0, 0, 0.8) 100%);
  padding: 2px 6px;
  border-radius: 4px;
  line-height: 1;
  text-align: center;
  min-width: 30px;
  font-weight: 600;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
}

/* 代码编辑器 */
.code-editor {
  flex: 1;
  border: none;
  outline: none;
  padding: 0;
  font-family: 'Courier New', monospace;
  font-size: 14px;
  line-height: 24px;
  resize: none;
  background-color: #fff;
  overflow-y: auto;
}

.code-line {
  padding: 0 15px;
  height: 24px;
  line-height: 24px;
  white-space: pre;
  transition: all 0.3s ease;
  margin: 0;
  border-left: 3px solid transparent;
}

.code-line:hover {
  background-color: rgba(64, 158, 255, 0.05);
  border-left-color: #409eff;
}

/* 结果卡片 */
.results-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

/* 空状态 */
.empty-state {
  text-align: center;
  padding: 80px 20px;
  color: #909399;
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

.empty-state p {
  margin-top: 20px;
  font-size: 16px;
  font-weight: 500;
}

.empty-hint {
  font-size: 14px !important;
  color: #c0c4cc !important;
  margin-top: 10px !important;
}

/* 统计信息 */
.stats {
  margin: 20px;
  margin-bottom: 25px;
}

.stat-item {
  text-align: center;
  padding: 20px;
  border-radius: 12px;
  color: white;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  position: relative;
  overflow: hidden;
}

.stat-item::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: rgba(255, 255, 255, 0.3);
  border-radius: 12px 12px 0 0;
}

.stat-item:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
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

.stat-icon {
  font-size: 24px;
  margin-bottom: 10px;
  opacity: 0.9;
}

.stat-number {
  font-size: 32px;
  font-weight: 700;
  line-height: 1;
  margin-bottom: 5px;
}

.stat-label {
  font-size: 14px;
  opacity: 0.9;
  text-transform: uppercase;
  letter-spacing: 1px;
}

/* 漏洞列表 */
.vulnerability-list {
  flex: 1;
  margin: 0 20px 20px;
  overflow-y: auto;
}

.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.list-header h4 {
  font-size: 18px;
  font-weight: 600;
  color: #303133;
  margin: 0;
}

.timeline-item {
  margin-bottom: 20px;
}

.vulnerability-item {
  padding: 15px;
  background: #f5f7fa;
  border-radius: 8px;
  border-left: 4px solid #409eff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
}

.vulnerability-item:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
  transform: translateX(4px);
}

.vuln-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}

.vuln-header .line-number {
  color: #606266;
  font-size: 14px;
  font-weight: 500;
  background: #eef0f5;
  padding: 2px 8px;
  border-radius: 4px;
}

.confidence {
  color: #909399;
  font-size: 14px;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 4px;
  background: #eef0f5;
  padding: 2px 8px;
  border-radius: 4px;
}

.confidence-icon {
  font-size: 12px;
  color: #e6a23c;
}

.vuln-message {
  color: #303133;
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 10px;
  line-height: 1.4;
}

.vuln-code {
  font-family: 'Courier New', monospace;
  font-size: 13px;
  color: #606266;
  background: #fff;
  padding: 10px;
  border-radius: 6px;
  border-left: 4px solid #409eff;
  box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
  line-height: 1.4;
}

/* 无漏洞状态 */
.no-vulnerabilities {
  text-align: center;
  padding: 60px 20px;
  color: #67c23a;
  background: linear-gradient(135deg, #f0f9eb 0%, #e6f7ff 100%);
  border-radius: 8px;
  margin: 20px;
  box-shadow: 0 4px 16px rgba(103, 194, 58, 0.15);
}

.no-vulnerabilities p {
  margin-top: 15px;
  font-size: 16px;
  font-weight: 500;
}

.safe-hint {
  font-size: 14px !important;
  color: #95ce61 !important;
  margin-top: 8px !important;
}

/* 滚动条样式 */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #f5f7fa;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: #dcdfe6;
  border-radius: 4px;
  transition: all 0.3s ease;
}

::-webkit-scrollbar-thumb:hover {
  background: #c0c4cc;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .page-header h2 {
    font-size: 24px;
  }
  
  .page-subtitle {
    font-size: 14px;
  }
  
  .el-col {
    width: 100% !important;
    margin-bottom: 20px;
  }
  
  .editor-container {
    height: 400px;
  }
  
  .heatmap-legend {
    flex-wrap: wrap;
    gap: 10px;
  }
  
  .card-header {
    padding: 0 15px;
  }
  
  .header-actions {
    gap: 8px;
  }
  
  .header-actions .el-button {
    font-size: 12px;
    padding: 6px 12px;
  }
}
</style>