# 查询缓存功能需求文档

## 📋 需求概述

为 smart-rag 添加语义查询缓存功能，避免用户重复提问时的冗余检索，提升响应速度并降低 API 调用成本。

## 🎯 核心目标

1. **性能提升**：缓存命中时，响应时间从 2-3 秒降至 < 100ms
2. **成本节约**：减少 Embedding API 和检索调用
3. **知识积累**：将用户问答对作为新知识源存入向量库

## 🏗️ 技术方案（基于业界最佳实践）

### 方案选择：简单阈值判断法

参考业界成熟产品（GPTCache、LangChain RedisSemanticCache、HuggingFace），采用**简单二分法**：

```python
if similarity >= threshold:
    return cached_answer  # 命中
else:
    return new_search()   # 未命中
```

**核心原则**：
- ✅ 简单 > 复杂
- ✅ 透明 > 智能（让用户看到原问题）
- ✅ 性能 > 完美
- ✅ 监控驱动优化

### 架构设计

```python
class QueryCache:
    """语义查询缓存"""
    
    def __init__(self, 
                 cache_file="query_cache.json",
                 similarity_threshold=0.90,
                 max_cache_size=1000):
        self.threshold = similarity_threshold
        self.cache = self._load_cache(cache_file)
        self.max_size = max_cache_size
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'false_positives': 0  # 用户反馈的错误命中
        }
    
    def query(self, question: str) -> dict:
        """查询缓存"""
        self.stats['total_queries'] += 1
        
        # 1. 生成查询向量
        query_embedding = embed_texts([question])[0]
        
        # 2. 向量相似度搜索
        similarity, cached_item = self._vector_search(query_embedding)
        
        # 3. 简单阈值判断
        if similarity >= self.threshold:
            self.stats['cache_hits'] += 1
            return {
                'hit': True,
                'answer': cached_item['answer'],
                'contexts': cached_item['contexts'],
                'similarity': similarity,
                'original_query': cached_item['query'],
                'timestamp': cached_item['timestamp']
            }
        else:
            self.stats['cache_misses'] += 1
            return {'hit': False}
    
    def add(self, query: str, answer: str, contexts: list):
        """添加到缓存"""
        query_embedding = embed_texts([query])[0]
        
        cache_item = {
            'query': query,
            'query_embedding': query_embedding,
            'answer': answer,
            'contexts': contexts,
            'timestamp': datetime.now().isoformat(),
            'hit_count': 0
        }
        
        self.cache.append(cache_item)
        
        # FIFO 淘汰策略
        if len(self.cache) > self.max_size:
            self.cache.pop(0)
        
        self._save_cache()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        total = self.stats['total_queries']
        if total == 0:
            return {'hit_rate': 0, 'threshold': self.threshold}
        
        return {
            'total_queries': total,
            'cache_hits': self.stats['cache_hits'],
            'hit_rate': self.stats['cache_hits'] / total,
            'false_positive_rate': self.stats['false_positives'] / total,
            'threshold': self.threshold
        }
    
    def mark_false_positive(self, query: str):
        """标记错误命中（用户反馈）"""
        self.stats['false_positives'] += 1
```

## 📊 UI 设计

### 1. 缓存命中提示

```python
if cache_result['hit']:
    st.info(f"""
    💡 **缓存命中**（相似度：{cache_result['similarity']:.1%}）
    
    **原问题**：{cache_result['original_query']}
    **缓存时间**：{cache_result['timestamp']}
    
    如果答案不准确，请点击"重新检索"
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 使用缓存答案"):
            return cache_result['answer']
    with col2:
        if st.button("🔄 重新检索"):
            query_cache.mark_false_positive(query)
            return new_search()
```

### 2. 缓存统计面板（侧边栏）

```python
st.sidebar.subheader("📊 缓存统计")
stats = query_cache.get_stats()

col1, col2 = st.columns(2)
with col1:
    st.metric("总查询", stats['total_queries'])
    st.metric("缓存命中", stats['cache_hits'])
with col2:
    st.metric("命中率", f"{stats['hit_rate']:.1%}")
    st.metric("错误率", f"{stats['false_positive_rate']:.1%}")

# 阈值调整
new_threshold = st.slider(
    "相似度阈值",
    min_value=0.80,
    max_value=0.98,
    value=0.90,
    step=0.02,
    help="阈值越高越严格，命中率越低但准确性越高"
)
query_cache.threshold = new_threshold

# 缓存管理
if st.button("🗑️ 清空缓存"):
    query_cache.clear()
    st.success("缓存已清空")
```

## 🔧 实施步骤

### 阶段 1：基础缓存（1-2 小时）

1. 创建 `query_cache.py` 模块
2. 实现 `QueryCache` 类（核心功能）
3. 在 `app.py` 中集成缓存查询逻辑
4. 添加缓存命中提示 UI

### 阶段 2：统计与优化（1 小时）

1. 添加缓存统计面板
2. 实现阈值动态调整
3. 添加用户反馈机制（标记错误命中）
4. 实现缓存导出/导入功能

### 阶段 3：高级功能（可选，1-2 小时）

1. 将 QA 对存入向量库（知识积累）
2. 实现 LRU 淘汰策略（替代 FIFO）
3. 添加缓存预热功能（导入常见问题）
4. 实现分布式缓存（Redis）

## 📈 性能指标

### 预期效果

| 指标 | 无缓存 | 有缓存（命中） |
|------|--------|---------------|
| 响应时间 | 2-3 秒 | < 100ms |
| API 调用 | 每次都调用 | 命中时 0 次 |
| 命中率 | - | 40-60%（预期） |

### 阈值建议

| 场景 | 阈值 | 说明 |
|------|------|------|
| 严格模式 | 0.95 | 几乎完全相同才复用 |
| 平衡模式 | 0.90 | **推荐**，平衡准确性和命中率 |
| 宽松模式 | 0.85 | 更高命中率，但可能不够精确 |

## ⚠️ 注意事项

### 1. 不做的事情（避免过度工程）

- ❌ 不做复杂的意图判断（LLM 调用太慢）
- ❌ 不做混合策略（增加复杂度，容易污染）
- ❌ 不做多档位处理（简单二分法足够）

### 2. 边界情况处理

```python
# 场景 1：相似但意图相反
query1 = "RAG 的优势是什么？"
query2 = "RAG 的缺点是什么？"
# 相似度可能 0.88，但不应命中
# 解决：阈值设为 0.90+，或让用户自己判断（展示原问题）

# 场景 2：上下文依赖
# 第一次："它有什么优势？"（指代不明）
# 第二次："它有什么优势？"（指代不同对象）
# 解决：多轮对话时，将上下文也纳入缓存 key
```

### 3. 缓存失效策略

```python
# 选项 1：基于时间（推荐）
if (now - cache_timestamp) > 7_days:
    invalidate_cache()

# 选项 2：向量库版本变化时
if vector_db_updated:
    clear_cache()

# 选项 3：手动清理
st.button("清空缓存")
```

## 🔗 参考资料

1. **GPTCache**：https://github.com/zilliztech/GPTCache
   - 7.3k+ 使用者的开源方案
   - 模块化设计，支持多种向量库

2. **LangChain RedisSemanticCache**：
   - 简单阈值判断
   - Redis 向量存储

3. **HuggingFace Semantic Cache**：
   - Faiss 内存向量索引
   - FIFO 淘汰策略

## 📝 数据结构

### 缓存存储格式（JSON）

```json
{
  "cache_version": "1.0",
  "threshold": 0.90,
  "items": [
    {
      "query": "什么是 RAG？",
      "query_embedding": [0.123, 0.456, ...],
      "answer": "RAG 是检索增强生成...",
      "contexts": [
        {"text": "...", "score": 0.95},
        {"text": "...", "score": 0.88}
      ],
      "timestamp": "2026-01-05T13:20:00",
      "hit_count": 3
    }
  ],
  "stats": {
    "total_queries": 100,
    "cache_hits": 45,
    "false_positives": 2
  }
}
```

## ✅ 验收标准

1. ✅ 缓存命中时响应时间 < 100ms
2. ✅ 命中率 > 40%（运行一周后）
3. ✅ 错误命中率 < 5%
4. ✅ 用户可以看到原问题并选择是否使用缓存
5. ✅ 提供完整的统计面板
6. ✅ 支持阈值动态调整

---

**创建时间**：2026-01-05  
**优先级**：中（多轮对话完成后实施）  
**预计工作量**：2-4 小时
