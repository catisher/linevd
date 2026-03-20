<template>
  <div class="home">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card class="welcome-card" shadow="hover">
          <div class="welcome-content">
            <el-icon size="64" color="#409EFF"><Monitor /></el-icon>
            <h2>欢迎使用 LineVD 漏洞检测系统</h2>
            <p class="description">
              LineVD 是一个基于图神经网络（GNN）的代码漏洞检测工具，
              能够精确定位到代码行的漏洞位置。
            </p>
            
            <el-row :gutter="20" class="features">
              <el-col :span="8">
                <div class="feature-item">
                  <el-icon size="32" color="#67C23A"><Search /></el-icon>
                  <h3>行级定位</h3>
                  <p>精确到代码行的漏洞定位</p>
                </div>
              </el-col>
              <el-col :span="8">
                <div class="feature-item">
                  <el-icon size="32" color="#E6A23C"><Share /></el-icon>
                  <h3>图神经网络</h3>
                  <p>基于代码属性图的深度学习</p>
                </div>
              </el-col>
              <el-col :span="8">
                <div class="feature-item">
                  <el-icon size="32" color="#F56C6C"><Warning /></el-icon>
                  <h3>多类型检测</h3>
                  <p>支持多种漏洞类型识别</p>
                </div>
              </el-col>
            </el-row>
            
            <el-button 
              type="primary" 
              size="large" 
              @click="$router.push('/analysis')"
              class="start-btn"
            >
              <el-icon><ArrowRight /></el-icon>
              开始分析
            </el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>
    
    <el-row :gutter="20" class="stats-row">
      <el-col :span="6">
        <el-statistic title="已分析文件" :value="128" />
      </el-col>
      <el-col :span="6">
        <el-statistic title="发现漏洞" :value="45" />
      </el-col>
      <el-col :span="6">
        <el-statistic title="检测准确率" :value="92.5" suffix="%" />
      </el-col>
      <el-col :span="6">
        <el-statistic title="平均耗时" :value="2.3" suffix="s" />
      </el-col>
    </el-row>
  </div>
</template>

<script>
export default {
  name: 'Home'
}
</script>

<style scoped>
.welcome-card {
  margin-bottom: 20px;
}

.welcome-content {
  text-align: center;
  padding: 40px 20px;
}

.welcome-content h2 {
  margin: 20px 0;
  font-size: 28px;
  color: #303133;
}

.description {
  font-size: 16px;
  color: #606266;
  max-width: 600px;
  margin: 0 auto 40px;
  line-height: 1.6;
}

.features {
  margin: 40px 0;
}

.feature-item {
  padding: 30px 20px;
  background: #f5f7fa;
  border-radius: 8px;
  transition: all 0.3s;
}

.feature-item:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.feature-item h3 {
  margin: 15px 0 10px;
  font-size: 18px;
  color: #303133;
}

.feature-item p {
  color: #909399;
  font-size: 14px;
}

.start-btn {
  margin-top: 30px;
  padding: 15px 40px;
  font-size: 18px;
}

.stats-row {
  margin-top: 20px;
}

:deep(.el-statistic__content) {
  font-size: 24px;
  font-weight: bold;
  color: #409EFF;
}
</style>
