# 多轮对话 RAG 的根本解决方案

## 🔍 问题本质

**当前方案的致命缺陷**：
```
第一轮："什么是 RAG？" → 检索 RAG 内容 ✅
第二轮："目前市场上有哪些成熟的产品" 
  → 查询重写："目前市场上有哪些成熟的 RAG 产品" ✅
  → 重新检索 → 检索到"中间件产品" ❌ 错误！
```

**问题根源**：
1. 向量库中可能没有"RAG 产品"相关内容
2. "成熟的产品"在其他文档中出现更频繁
3. **每次都重新检索，丢失了第一轮的上下文**

## 🎯 业界标准方案：SELF-multi-RAG

参考论文：[Learning When to Retrieve, What to Rewrite, and How to Respond](https://arxiv.org/html/2409.15515v1)

### 核心思想

**不是每次都检索，而是智能判断是否需要检索**

### 三种检索决策

```python
class RetrievalDecision:
    RETRIEVE = "需要检索新内容"           # 新话题、需要新事实
    NO_RETRIEVE = "不需要检索"           # 创意性问题、闲聊
    CONTINUE_USE_EVIDENCE = "复用上次检索结果"  # ⭐ 关键！追问、深入讨论
```

### 工作流程

```
第一轮对话：
  用户："什么是 RAG？"
  ↓
  决策：[Retrieve] - 需要检索新内容
  ↓
  检索 RAG 相关文档
  ↓
  保存检索结果到对话上下文
  ↓
  生成答案："RAG 是检索增强生成..."

第二轮对话：
  用户："目前市场上有哪些成熟的产品"
  ↓
  决策：[Continue to use evidence] - 复用上次检索结果 ⭐
  ↓
  不重新检索！
  ↓
  直接使用第一轮检索到的 RAG 文档
  ↓
  基于 RAG 上下文生成答案："RAG 的成熟产品有..."
```

## 💡 实现方案

### 方案 A：简化版（推荐优先实施）

```python
class ConversationManager:
    def __init__(self):
        self.conversation_history = []
        self.last_retrieved_contexts = []  # ⭐ 缓存上次检索结果
        self.last_query_topic = None       # 记录上次查询的主题
    
    def should_retrieve(self, current_query, conversation_history):
        """判断是否需要重新检索"""
        
        # 1. 没有历史 → 必须检索
        if not conversation_history:
            return True
        
        # 2. 检测是否是追问
        follow_up_patterns = [
            "它", "这个", "那个", "上述", "前面",
            "第一点", "第二点", "详细", "举例",
            "为什么", "怎么", "如何", "能不能",
            "还有", "另外", "其他", "更多"
        ]
        
        is_follow_up = any(pattern in current_query for pattern in follow_up_patterns)
        
        # 3. 追问 + 有缓存 → 复用检索结果
        if is_follow_up and self.last_retrieved_contexts:
            return False  # 不重新检索
        
        # 4. 其他情况 → 重新检索
        return True
    
    def get_contexts(self, query, top_k=3):
        """获取上下文（检索或复用）"""
        
        if self.should_retrieve(query, self.conversation_history):
            # 重新检索
            contexts = search_top_k(query, k=top_k)
            self.last_retrieved_contexts = contexts  # 缓存
            return contexts, "new"
        else:
            # 复用上次检索结果
            return self.last_retrieved_contexts, "reused"
```

### 方案 B：完整版（SELF-multi-RAG）

使用 LLM 进行智能决策：

```python
def decide_retrieval_strategy(query, conversation_history, last_contexts):
    """使用 LLM 判断检索策略"""
    
    prompt = f"""你是一个检索决策助手。请判断是否需要重新检索。

对话历史：
{format_history(conversation_history[-2:])}

上次检索的内容摘要：
{summarize_contexts(last_contexts)}

当前问题：{query}

请判断：
1. [RETRIEVE] - 需要检索新内容（新话题、需要新事实）
2. [NO_RETRIEVE] - 不需要检索（创意性问题、闲聊）
3. [REUSE] - 复用上次检索结果（追问、深入讨论同一话题）

只输出一个标签："""

    response = llm.generate(prompt)
    
    if "[REUSE]" in response:
        return "reuse"
    elif "[RETRIEVE]" in response:
        return "retrieve"
    else:
        return "no_retrieve"
```

## 📊 效果对比

### 当前方案（查询重写）

```
第一轮："什么是 RAG？" 
  → 检索 RAG 文档 ✅

第二轮："目前市场上有哪些成熟的产品"
  → 重写："目前市场上有哪些成熟的 RAG 产品"
  → 重新检索 → 检索到"中间件产品" ❌
```

### 改进方案（智能检索决策）

```
第一轮："什么是 RAG？"
  → 检索 RAG 文档 ✅
  → 缓存检索结果

第二轮："目前市场上有哪些成熟的产品"
  → 判断：追问，复用上次检索结果
  → 不重新检索！
  → 使用缓存的 RAG 文档 ✅
  → 基于 RAG 上下文回答 ✅
```

## 🎯 实施建议

### 阶段 1：简化版（立即可实施）

1. 在 `ConversationManager` 中缓存上次检索结果
2. 简单的追问检测（关键词匹配）
3. 追问时复用缓存，否则重新检索

**优势**：
- 实现简单（约 50 行代码）
- 立即见效
- 无需额外 LLM 调用

**劣势**：
- 关键词匹配不够智能
- 可能误判

### 阶段 2：完整版（长期优化）

1. 使用 LLM 进行智能决策
2. 实现三种检索策略
3. 对话摘要和上下文压缩

**优势**：
- 决策准确
- 适应复杂场景

**劣势**：
- 需要额外 LLM 调用
- 增加延迟

## 🔑 关键洞察

1. **查询重写不是根本解决方案**
   - 重写后仍然会重新检索
   - 向量库中可能没有相关内容
   - 检索算法可能匹配错误

2. **真正的解决方案是上下文复用**
   - 追问时不重新检索
   - 直接使用上次检索的内容
   - 保持话题连贯性

3. **检索决策比查询重写更重要**
   - 先判断是否需要检索
   - 再决定如何检索
   - 最后才是查询重写

## 📝 完整流程

```
用户输入问题
    ↓
检索决策（智能判断）
    ↓
├─ [RETRIEVE] 需要检索
│   ↓
│   查询重写（如有必要）
│   ↓
│   向量检索
│   ↓
│   缓存检索结果
│
├─ [REUSE] 复用上次检索
│   ↓
│   使用缓存的检索结果
│
└─ [NO_RETRIEVE] 不需要检索
    ↓
    直接基于对话历史生成
    ↓
生成答案
```

## ✅ 总结

**问题根源**：每次都重新检索，丢失了对话上下文

**根本解决方案**：
1. ✅ **智能检索决策** - 判断是否需要检索
2. ✅ **上下文复用** - 追问时复用上次检索结果
3. ✅ **查询重写** - 作为辅助手段

**实施优先级**：
1. 先实现上下文复用（解决根本问题）
2. 再优化检索决策（提升准确性）
3. 最后完善查询重写（锦上添花）

---

**创建时间**：2026-01-06  
**状态**：待实施
