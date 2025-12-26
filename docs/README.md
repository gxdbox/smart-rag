# 📚 项目文档目录

本目录包含 RAG 混合检索系统的所有技术文档。

---

## 📖 文档列表

### 1. **BM25 相关文档**

#### [`BM25_USAGE.md`](./BM25_USAGE.md)
**BM25 检索器使用指南**
- 快速开始
- API 文档
- 使用场景和示例
- 故障排除

#### [`BM25_PRINCIPLE.md`](./BM25_PRINCIPLE.md)
**BM25 算法原理详解**
- BM25 算法公式
- 三大核心机制（TF、IDF、长度归一化）
- 与 MySQL 精确查找的区别
- 实际案例分析

#### [`BM25_IMPLEMENTATION_SUMMARY.md`](./BM25_IMPLEMENTATION_SUMMARY.md)
**BM25 实现总结**
- 实现完成情况
- 核心优势
- 技术实现细节
- 性能指标

---

### 2. **混合检索文档**

#### [`HYBRID_SEARCH_EXPLAINED.md`](./HYBRID_SEARCH_EXPLAINED.md)
**混合检索原理详解**
- 6 步工作流程
- 分数归一化机制
- 加权融合策略
- 权重调整建议
- 实际案例对比

---

### 3. **UI 集成文档**

#### [`UI_INTEGRATION_GUIDE.md`](./UI_INTEGRATION_GUIDE.md)
**UI 集成完成指南**
- 新增功能说明
- UI 界面布局
- 使用流程
- 配置建议
- 常见问题

---

## 🗂️ 文档分类

### 按用途分类

| 类型 | 文档 | 适用人群 |
|------|------|---------|
| **使用指南** | `BM25_USAGE.md`<br>`UI_INTEGRATION_GUIDE.md` | 用户、开发者 |
| **原理解释** | `BM25_PRINCIPLE.md`<br>`HYBRID_SEARCH_EXPLAINED.md` | 学习者、研究者 |
| **技术总结** | `BM25_IMPLEMENTATION_SUMMARY.md` | 开发者、维护者 |

### 按主题分类

```
BM25 检索
├── BM25_USAGE.md              (使用)
├── BM25_PRINCIPLE.md          (原理)
└── BM25_IMPLEMENTATION_SUMMARY.md (实现)

混合检索
└── HYBRID_SEARCH_EXPLAINED.md (原理)

UI 集成
└── UI_INTEGRATION_GUIDE.md    (指南)
```

---

## 🚀 快速导航

### 我想了解...

**如何使用 BM25？**
→ [`BM25_USAGE.md`](./BM25_USAGE.md)

**BM25 的工作原理？**
→ [`BM25_PRINCIPLE.md`](./BM25_PRINCIPLE.md)

**混合检索如何工作？**
→ [`HYBRID_SEARCH_EXPLAINED.md`](./HYBRID_SEARCH_EXPLAINED.md)

**如何在 UI 中使用？**
→ [`UI_INTEGRATION_GUIDE.md`](./UI_INTEGRATION_GUIDE.md)

**项目实现了什么？**
→ [`BM25_IMPLEMENTATION_SUMMARY.md`](./BM25_IMPLEMENTATION_SUMMARY.md)

---

## 📝 文档更新记录

| 日期 | 文档 | 更新内容 |
|------|------|---------|
| 2024-12-26 | 所有文档 | 创建并整理到 docs 目录 |
| 2024-12-26 | `BM25_PRINCIPLE.md` | 添加 BM25 原理详解 |
| 2024-12-26 | `HYBRID_SEARCH_EXPLAINED.md` | 添加混合检索原理 |
| 2024-12-25 | `BM25_USAGE.md` | 创建 BM25 使用指南 |
| 2024-12-25 | `BM25_IMPLEMENTATION_SUMMARY.md` | 创建实现总结 |
| 2024-12-25 | `UI_INTEGRATION_GUIDE.md` | 创建 UI 集成指南 |

---

## 💡 阅读建议

### 新手入门路径
1. [`BM25_USAGE.md`](./BM25_USAGE.md) - 快速上手
2. [`UI_INTEGRATION_GUIDE.md`](./UI_INTEGRATION_GUIDE.md) - 了解 UI 功能
3. [`BM25_PRINCIPLE.md`](./BM25_PRINCIPLE.md) - 理解原理

### 深入学习路径
1. [`BM25_PRINCIPLE.md`](./BM25_PRINCIPLE.md) - BM25 算法原理
2. [`HYBRID_SEARCH_EXPLAINED.md`](./HYBRID_SEARCH_EXPLAINED.md) - 混合检索原理
3. [`BM25_IMPLEMENTATION_SUMMARY.md`](./BM25_IMPLEMENTATION_SUMMARY.md) - 实现细节

### 开发者路径
1. [`BM25_IMPLEMENTATION_SUMMARY.md`](./BM25_IMPLEMENTATION_SUMMARY.md) - 实现总结
2. [`HYBRID_SEARCH_EXPLAINED.md`](./HYBRID_SEARCH_EXPLAINED.md) - 混合检索实现
3. [`BM25_USAGE.md`](./BM25_USAGE.md) - API 文档

---

## 🔗 相关资源

- **项目主文档**: [`../README.md`](../README.md)
- **架构重构文档**: [`../REFACTORING.md`](../REFACTORING.md)
- **源代码**: [`../rag/`](../rag/)

---

**文档维护者**: Cascade AI  
**最后更新**: 2024-12-26
