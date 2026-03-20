<template>
  <div id="app">
    <el-container class="main-container">
      <el-header class="app-header">
        <div class="header-content">
          <el-icon size="28" color="#409EFF"><Warning /></el-icon>
          <h1>LineVD 漏洞检测系统</h1>
          <span class="subtitle">基于深度学习的代码漏洞行级定位</span>
        </div>
      </el-header>
      
      <el-main class="app-main">
        <router-view />
      </el-main>
      
      <el-footer class="app-footer">
        <p>Powered by LineVD | 基于图神经网络的漏洞检测</p>
      </el-footer>
    </el-container>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Hiragino Sans GB',
    'Microsoft YaHei', '微软雅黑', Arial, sans-serif;
  background-color: #f5f7fa;
}

.main-container {
  min-height: 100vh;
}

.app-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.header-content {
  display: flex;
  align-items: center;
  gap: 15px;
}

.header-content h1 {
  font-size: 24px;
  font-weight: 600;
}

.subtitle {
  font-size: 14px;
  opacity: 0.9;
  margin-left: 20px;
}

.app-main {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
}

.app-footer {
  background-color: #fff;
  color: #909399;
  text-align: center;
  padding: 20px;
  border-top: 1px solid #e4e7ed;
}
</style>
