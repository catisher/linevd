<template>
  <div class="results">
    <el-card shadow="hover">
      <template #header>
        <div class="card-header">
          <span>历史分析记录</span>
          <el-button type="primary" @click="$router.push('/analysis')">
            <el-icon><Plus /></el-icon>
            新建分析
          </el-button>
        </div>
      </template>
      
      <el-table :data="analysisHistory" style="width: 100%">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="filename" label="文件名" />
        <el-table-column prop="date" label="分析时间" width="180" />
        <el-table-column prop="totalLines" label="代码行数" width="100" />
        <el-table-column prop="vulnerabilityCount" label="漏洞数" width="100">
          <template #default="scope">
            <el-tag :type="scope.row.vulnerabilityCount > 0 ? 'danger' : 'success'">
              {{ scope.row.vulnerabilityCount }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="150">
          <template #default="scope">
            <el-button type="primary" size="small" @click="viewDetail(scope.row)">
              查看详情
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script>
export default {
  name: 'Results',
  data() {
    return {
      analysisHistory: [
        {
          id: 1,
          filename: 'test.c',
          date: '2026-03-08 14:30:00',
          totalLines: 15,
          vulnerabilityCount: 2
        },
        {
          id: 2,
          filename: 'vulnerable.c',
          date: '2026-03-08 15:45:00',
          totalLines: 32,
          vulnerabilityCount: 5
        },
        {
          id: 3,
          filename: 'secure.c',
          date: '2026-03-08 16:20:00',
          totalLines: 28,
          vulnerabilityCount: 0
        }
      ]
    }
  },
  methods: {
    viewDetail(row) {
      this.$message.info(`查看分析详情: ${row.filename}`)
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
</style>
