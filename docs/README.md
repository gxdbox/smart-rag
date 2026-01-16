# 📚 Smart RAG 文档中心

本目录包含 Smart RAG 系统的所有技术文档，已按类别组织便于查找。

---

## 📂 文档目录结构

```
docs/
├── README.md                    (本文件 - 文档索引)
├── features/                    (功能特性文档)
│   ├── BM25_PRINCIPLE.md       (BM25 算法原理)
│   ├── HYBRID_SEARCH_EXPLAINED.md (混合检索原理)
│   └── RERANK_FEATURE.md       (Rerank 精排功能)
├── implementation/              (实现细节文档)
│   ├── BM25_IMPLEMENTATION_SUMMARY.md (BM25 实现总结)
│   ├── BM25_USAGE.md           (BM25 使用指南)
│   └── DATA_SYNC_ISSUE.md      (数据同步问题)
├── guides/                      (使用指南)
│   ├── UI_INTEGRATION_GUIDE.md (UI 集成指南)
│   ├── multi_turn_test_guide.md (多轮对话测试)
│   └── PRE_PUSH_CHECKLIST.md   (提交前检查清单)
└── archive/                     (历史文档归档)
    ├── chunk_size_optimization.md
    ├── fundamental_design_issue.md
    ├── fundamental_solution.md
    ├── query_cache_requirement.md
    ├── stage1_completion_summary.md
    └── stage1_final_summary.md
```

---

## 🚀 快速导航

### 📖 功能特性 (features/)

**核心检索功能原理和设计**

- **[BM25 算法原理](./features/BM25_PRINCIPLE.md)**
  - BM25 算法公式详解
  - TF、IDF、长度归一化机制
  - 与传统精确匹配的区别
  - 实际案例分析

- **[混合检索原理](./features/HYBRID_SEARCH_EXPLAINED.md)**
  - 向量检索 + BM25 融合机制
  - 分数归一化策略
  - 加权融合算法
  - 权重调优建议

- **[Rerank 精排功能](./features/RERANK_FEATURE.md)**
  - 两阶段检索架构
  - Rerank 模型原理
  - 性能提升分析
  - 使用场景和建议

---

### 🔧 实现细节 (implementation/)

**技术实现和使用文档**

- **[BM25 实现总结](./implementation/BM25_IMPLEMENTATION_SUMMARY.md)**
  - 实现完成情况
  - 核心技术细节
  - 性能指标
  - 优化方案

- **[BM25 使用指南](./implementation/BM25_USAGE.md)**
  - 快速开始
  - API 文档
  - 使用示例
  - 故障排除

- **[数据同步问题](./implementation/DATA_SYNC_ISSUE.md)**
  - 向量库与 BM25 索引同步
  - 问题分析和解决方案
  - 同步工具使用

---

### 📘 使用指南 (guides/)

**操作指南和最佳实践**

- **[UI 集成指南](./guides/UI_INTEGRATION_GUIDE.md)**
  - 功能说明
  - 界面布局
  - 使用流程
  - 配置建议

- **[多轮对话测试指南](./guides/multi_turn_test_guide.md)**
  - 测试场景
  - 测试方法
  - 预期结果

- **[提交前检查清单](./guides/PRE_PUSH_CHECKLIST.md)**
  - 代码检查项
  - 测试要求
  - 文档更新

---

### 🗄️ 历史归档 (archive/)

**已完成阶段的历史文档**

- Stage 1 完成总结
- 早期设计问题和解决方案
- 查询缓存需求分析
- Chunk 大小优化研究

---

## 💡 推荐阅读路径

### 🌟 新手入门
1. [BM25 使用指南](./implementation/BM25_USAGE.md) - 快速上手
2. [UI 集成指南](./guides/UI_INTEGRATION_GUIDE.md) - 了解界面功能
3. [BM25 算法原理](./features/BM25_PRINCIPLE.md) - 理解基础原理

### 🎓 深入学习
1. [BM25 算法原理](./features/BM25_PRINCIPLE.md) - 算法基础
2. [混合检索原理](./features/HYBRID_SEARCH_EXPLAINED.md) - 融合策略
3. [Rerank 精排功能](./features/RERANK_FEATURE.md) - 高级检索
4. [BM25 实现总结](./implementation/BM25_IMPLEMENTATION_SUMMARY.md) - 实现细节

### 👨‍💻 开发者路径
1. [BM25 实现总结](./implementation/BM25_IMPLEMENTATION_SUMMARY.md) - 技术实现
2. [数据同步问题](./implementation/DATA_SYNC_ISSUE.md) - 问题解决
3. [提交前检查清单](./guides/PRE_PUSH_CHECKLIST.md) - 开发规范
4. [混合检索原理](./features/HYBRID_SEARCH_EXPLAINED.md) - 架构设计

---

## 📝 文档更新记录

| 日期 | 更新内容 |
|------|---------|
| 2026-01-16 | 重组文档结构，按类别分类，删除重复文档 |
| 2024-12-26 | 创建功能特性文档（BM25、混合检索） |
| 2024-12-25 | 创建实现和使用指南文档 |

---

## 🔗 相关资源

- **项目主文档**: [`../README.md`](../README.md)
- **可维护性分析**: [`../MAINTAINABILITY_ANALYSIS.md`](../MAINTAINABILITY_ANALYSIS.md)
- **配置管理**: [`../config.py`](../config.py)
- **源代码**: [`../src/`](../src/)

---

**文档维护**: Smart RAG Team  
**最后更新**: 2026-01-16
