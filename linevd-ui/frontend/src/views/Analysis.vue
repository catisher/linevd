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
              
              <div v-if="vulnerabilities.length === 0" class="no-vulnerabilities">
                <el-icon size="24" color="#67c23a"><CheckFilled /></el-icon>
                <p>未检测到漏洞</p>
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

// 响应式数据：代码内容，初始值为示例 C 代码
const code = ref(`#include <stdio.h>
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
}`)

// 响应式数据：代码行数
const lineCount = ref(0)
// 响应式数据：是否正在分析中
const analyzing = ref(false)
// 响应式数据：分析是否完成
const analysisComplete = ref(false)
// 响应式数据：漏洞列表
const vulnerabilities = ref([])
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
}

/**
 * 同步行号滚动
 * @param {Event} e - 滚动事件对象
 */
const syncScroll = (e) => {
  // 获取行号容器元素
  const lineNumbers = document.querySelector('.line-numbers')
  if (lineNumbers) {
    // 同步滚动位置
    lineNumbers.scrollTop = e.target.scrollTop
  }
}

/**
 * 检查某行是否为漏洞行
 * @param {number} line - 行号
 * @returns {boolean} - 是否为漏洞行
 */
const isVulnerableLine = (line) => {
  return vulnerabilities.value.some(v => v.line === line)
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
    // 更新代码行数
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
    const response = await axios.post('http://localhost:8000/predict', {
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
    // 更新模拟数据的统计信息
    highRiskCount.value = 2
    mediumRiskCount.value = 0
    lowRiskCount.value = 0
    // 标记分析完成
    analysisComplete.value = true
  } finally {
    // 无论成功失败，都设置 analyzing 为 false
    analyzing.value = false
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

.no-vulnerabilities {
  text-align: center;
  padding: 40px 20px;
  color: #67c23a;
}

.no-vulnerabilities p {
  margin-top: 10px;
  font-size: 14px;
}
</style>