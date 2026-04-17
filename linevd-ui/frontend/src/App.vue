<template>
  <div id="app">
    <el-container class="main-container">
      <el-header class="app-header">
        <div class="header-content">
          <div class="logo">
            <el-icon size="32" color="white"><Warning /></el-icon>
            <h1>LineVD</h1>
          </div>
          <div class="nav">
            <el-menu :default-active="activeMenu" mode="horizontal" background-color="transparent" text-color="white" active-text-color="white">
              <el-menu-item index="home" @click="navigateTo('/')">首页</el-menu-item>
              <el-menu-item index="analysis" @click="navigateTo('/analysis')">漏洞分析</el-menu-item>
              <el-menu-item index="about" @click="navigateTo('/results')">关于</el-menu-item>
            </el-menu>
          </div>
        </div>
      </el-header>
      
      <el-main class="app-main">
        <router-view />
      </el-main>
      
      <el-footer class="app-footer">
        <div class="footer-content">
          <p>Powered by wmy | 基于图神经网络的漏洞检测</p>
          <p class="copyright">© 2026 LineVD. All rights reserved.</p>
        </div>
      </el-footer>
    </el-container>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const activeMenu = ref('analysis')

const navigateTo = (path) => {
  router.push(path)
  activeMenu.value = path === '/' ? 'home' : path
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
  background: linear-gradient(135deg, #f5f7fa 0%, #eef0f5 100%);
  min-height: 100vh;
}

.main-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  background: linear-gradient(135deg, #409eff 0%, #66b1ff 100%);
  color: white;
  box-shadow: 0 4px 16px rgba(64, 158, 255, 0.3);
  position: sticky;
  top: 0;
  z-index: 1000;
}

.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
  padding: 0 30px;
  height: 70px;
}

.logo {
  display: flex;
  align-items: center;
  gap: 15px;
}

.logo h1 {
  font-size: 28px;
  font-weight: 700;
  margin: 0;
  letter-spacing: 1px;
}

.nav {
  flex: 1;
  margin-left: 60px;
}

.nav .el-menu {
  border-bottom: none;
}

.nav .el-menu-item {
  font-size: 16px;
  font-weight: 500;
  height: 70px;
  line-height: 70px;
  margin: 0 10px;
  transition: all 0.3s ease;
}

.nav .el-menu-item:hover {
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
}

.app-main {
  flex: 1;
  padding: 30px;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
}

.app-footer {
  background-color: #fff;
  color: #909399;
  text-align: center;
  padding: 30px;
  border-top: 1px solid #e4e7ed;
  box-shadow: 0 -2px 12px rgba(0, 0, 0, 0.05);
}

.footer-content {
  max-width: 1400px;
  margin: 0 auto;
}

.copyright {
  font-size: 12px;
  margin-top: 8px;
  opacity: 0.8;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .header-content {
    padding: 0 20px;
  }
  
  .logo h1 {
    font-size: 24px;
  }
  
  .nav {
    margin-left: 30px;
  }
  
  .nav .el-menu-item {
    font-size: 14px;
    margin: 0 5px;
  }
  
  .app-main {
    padding: 20px;
  }
}
</style>
