# 分块大小与检索效果优化

## 🔍 问题分析

### 核心矛盾
- **短查询**（1-5个词）vs **长文档块**（几百字）
- 短查询语义信息不足
- 长文档块语义被稀释

### 具体问题

1. **语义稀释**
```
文档块（500字）包含多个主题：
  - RAG 定义（100字）
  - 向量数据库（100字）
  - Embedding 模型（100字）
  - 产品化（100字）
  - 部署架构（100字）

查询："产品"
→ 只关心"产品化"部分
→ 但向量是整个500字的平均语义
→ 相似度被稀释 ❌
```

2. **查询扩展不足**
```
查询："产品"（2个字）
→ 语义信息太少
→ 无法精确匹配
```

## 💡 解决方案

### 方案1：优化分块大小 ⭐ 推荐

**原则**：分块应该是**语义完整的最小单元**

#### 当前问题
```python
# 当前：固定大小分块
chunk_size = 500  # 字符
overlap = 50

# 问题：可能在句子中间切断
chunk = "RAG是检索增强生成技术，它结合了检索和生成两种方法。向量数据库是RAG的核心组件..."
```

#### 改进方案：语义分块

```python
# 方案A：按段落分块
chunks = text.split('\n\n')  # 每个段落一个块

# 方案B：按句子分块
import re
sentences = re.split('[。！？]', text)
chunks = group_sentences(sentences, max_length=200)  # 组合成200字左右

# 方案C：按主题分块（使用LLM）
chunks = llm_semantic_chunking(text)
```

**优势**：
- ✅ 每个块语义完整
- ✅ 减少语义稀释
- ✅ 提高检索精度

**推荐分块大小**：
- **短文档**（< 1000字）：100-200字/块
- **中等文档**（1000-5000字）：200-300字/块
- **长文档**（> 5000字）：300-500字/块

### 方案2：查询扩展（Query Expansion）⭐⭐ 推荐

**核心思想**：将短查询扩展为更丰富的语义表达

#### 实现方式A：使用LLM扩展

```python
def expand_query(query: str, conversation_history: List[Dict]) -> str:
    """使用LLM扩展查询"""
    
    prompt = f"""请将以下简短查询扩展为更详细的搜索表达：

原查询：{query}

对话历史：{format_history(conversation_history)}

任务：
1. 基于对话历史理解查询意图
2. 扩展为3-5个相关的搜索词或短语
3. 保持原意，增加语义丰富度

扩展后的查询："""

    expanded = llm.generate(prompt)
    return expanded

# 示例
query = "产品"
conversation = [
    {"role": "user", "content": "什么是RAG？"},
    {"role": "assistant", "content": "RAG是检索增强生成..."}
]

expanded = expand_query(query, conversation)
# → "RAG产品 检索增强生成产品 RAG商业化产品 RAG开源项目"
```

#### 实现方式B：多查询检索（Multi-Query Retrieval）

```python
def multi_query_retrieval(query: str, topic: str, k: int = 3):
    """生成多个查询变体，分别检索后合并"""
    
    # 生成查询变体
    queries = [
        query,                    # 原查询："产品"
        f"{topic} {query}",       # 主题+查询："RAG 产品"
        f"{topic}相关的{query}",  # 扩展："RAG相关的产品"
        f"{topic}的{query}案例",  # 具体化："RAG的产品案例"
    ]
    
    # 分别检索
    all_results = []
    for q in queries:
        results = search(q, k=k)
        all_results.extend(results)
    
    # 去重 + 重排序
    unique_results = deduplicate(all_results)
    reranked = rerank(unique_results, query)
    
    return reranked[:k]
```

### 方案3：混合检索（Hybrid Search）⭐⭐⭐ 最佳

**结合向量检索和关键词检索**

```python
def hybrid_search(query: str, k: int = 3):
    """混合检索：向量 + BM25"""
    
    # 向量检索（语义匹配）
    vector_results = vector_search(query, k=10)
    
    # BM25检索（关键词匹配）
    bm25_results = bm25_search(query, k=10)
    
    # 融合结果（RRF算法）
    final_results = reciprocal_rank_fusion(
        vector_results, 
        bm25_results,
        k=k
    )
    
    return final_results
```

**优势**：
- ✅ 向量检索：捕获语义相似
- ✅ BM25检索：精确关键词匹配
- ✅ 互补优势，提高召回率

### 方案4：重排序（Reranking）⭐⭐⭐ 最佳

**使用专门的重排序模型精排**

```python
def search_with_rerank(query: str, k: int = 3):
    """检索 + 重排序"""
    
    # 第一阶段：粗排（召回更多候选）
    candidates = vector_search(query, k=20)
    
    # 第二阶段：精排（使用Rerank模型）
    reranked = rerank_model.rerank(
        query=query,
        documents=candidates
    )
    
    return reranked[:k]
```

**Rerank模型优势**：
- ✅ 专门训练用于排序
- ✅ 考虑查询和文档的交互
- ✅ 比单纯的向量相似度更准确

### 方案5：分层检索（Hierarchical Retrieval）

**先粗后细，逐步定位**

```python
# 第一层：文档级检索
relevant_docs = search_documents(query)  # 找到相关文档

# 第二层：段落级检索
relevant_paragraphs = search_within_docs(query, relevant_docs)

# 第三层：句子级检索
relevant_sentences = search_within_paragraphs(query, relevant_paragraphs)
```

## 📊 方案对比

| 方案 | 实现难度 | 效果提升 | 计算成本 | 推荐度 |
|------|---------|---------|---------|--------|
| 优化分块大小 | 低 | ⭐⭐⭐ | 低 | ✅✅ 立即实施 |
| 查询扩展 | 中 | ⭐⭐⭐⭐ | 中 | ✅✅ 推荐 |
| 多查询检索 | 中 | ⭐⭐⭐⭐ | 高 | ✅ 可选 |
| 混合检索 | 中 | ⭐⭐⭐⭐⭐ | 中 | ✅✅✅ 强烈推荐 |
| 重排序 | 中 | ⭐⭐⭐⭐⭐ | 高 | ✅✅✅ 强烈推荐 |
| 分层检索 | 高 | ⭐⭐⭐⭐⭐ | 高 | ✅✅ 长期目标 |

## 🎯 立即可行的改进

### 改进1：检查当前分块大小

```python
# 查看当前分块统计
stats = get_db_stats()
print(f"平均块大小：{stats['avg_chunk_size']} 字符")
print(f"最大块大小：{stats['max_chunk_size']} 字符")
print(f"最小块大小：{stats['min_chunk_size']} 字符")
```

**建议**：
- 如果平均块大小 > 400字 → 考虑减小到200-300字
- 如果块大小差异很大 → 考虑使用语义分块

### 改进2：启用混合检索

您的系统已经支持混合检索！

```python
# 在UI中选择"混合检索"模式
retrieval_mode = "混合检索"
vector_weight = 0.5  # 向量和BM25各占50%
```

### 改进3：启用Rerank精排

您的系统已经支持Rerank！

```python
# 在UI中选择"Rerank 精排"或"混合 + Rerank"
retrieval_mode = "混合 + Rerank（最强）"
recall_k = 20  # 先召回20个候选
top_k = 3      # 精排后返回3个
```

### 改进4：实现查询扩展

这是我们刚刚实现的主题提取功能的扩展版本。

## 🔬 实验建议

### 测试不同分块大小

```python
# 测试1：当前分块（假设500字）
results_500 = test_retrieval(chunk_size=500)

# 测试2：小分块（200字）
results_200 = test_retrieval(chunk_size=200)

# 测试3：大分块（800字）
results_800 = test_retrieval(chunk_size=800)

# 对比效果
compare_results(results_500, results_200, results_800)
```

### 测试不同检索策略

```python
# 测试查询
test_queries = [
    "RAG",
    "产品",
    "RAG 产品",
    "检索增强生成的商业化产品"
]

for query in test_queries:
    # 向量检索
    vector_results = vector_search(query)
    
    # 混合检索
    hybrid_results = hybrid_search(query)
    
    # Rerank
    rerank_results = search_with_rerank(query)
    
    # 对比
    print(f"查询：{query}")
    print(f"向量检索：{evaluate(vector_results)}")
    print(f"混合检索：{evaluate(hybrid_results)}")
    print(f"Rerank：{evaluate(rerank_results)}")
```

## ✅ 总结

**您的担忧是对的**：
- 短查询 vs 长文档块确实存在匹配问题
- 语义稀释会降低检索精度

**解决方案**：
1. **立即实施**：启用混合检索 + Rerank（您的系统已支持）
2. **短期优化**：优化分块大小（200-300字）
3. **中期优化**：实现查询扩展
4. **长期优化**：分层检索 + 语义分块

**关键原则**：
- 分块要语义完整
- 查询要足够丰富
- 检索要多策略融合
- 排序要精确重排

---

**建议下一步**：
1. 检查当前分块大小
2. 测试"混合 + Rerank"模式
3. 如果效果仍不理想，考虑重新分块
