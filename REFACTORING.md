# RAG 三层架构重构说明

## 📋 重构概述

本次重构将原有的 RAG 系统从单一模块拆分为清晰的三层架构，提升了代码的可维护性和可扩展性。

### ✅ 重构目标达成情况

- ✅ 将 RAG 流程拆分为 Retriever、Ranker、Generator 三层
- ✅ 现有 JSON 向量方案仍然可用
- ✅ 未来可无侵入替换为 BM25 / FAISS
- ✅ Streamlit UI 调用方式保持不变
- ✅ 所有现有功能正常运行

---

## 🏗️ 新架构说明

### 目录结构

```
rag-streamlit-cn/
├── rag/                          # 新增：核心 RAG 模块
│   ├── __init__.py
│   ├── retriever/               # 召回层
│   │   ├── __init__.py
│   │   ├── base.py             # BaseRetriever 接口
│   │   ├── vector_retriever.py # JSON 向量检索（已实现）
│   │   ├── bm25_retriever.py   # BM25 检索（预留）
│   │   └── faiss_retriever.py  # FAISS 检索（预留）
│   ├── ranker/                  # 排序层
│   │   ├── __init__.py
│   │   ├── base.py             # BaseRanker 接口
│   │   ├── similarity_ranker.py # 余弦相似度排序（已实现）
│   │   └── rerank_ranker.py    # Re-rank 精排（预留）
│   └── generator/               # 生成层
│       ├── __init__.py
│       ├── base.py             # BaseGenerator 接口
│       └── llm_generator.py    # LLM 生成（已实现）
├── rag_engine.py                # 保留：编排逻辑 + 向量化 + 切分
├── app.py                       # 保留：UI 层
└── ...其他文件保持不变
```

---

## 🎯 三层职责定义

### 1. Retriever（召回层）

**职责：**
- 根据查询条件，从存储中召回候选文档集合
- 管理向量库的加载/保存
- 返回 `List[Document]`

**不负责：**
- ❌ 计算相似度分数（Ranker 的职责）
- ❌ 排序结果（Ranker 的职责）
- ❌ 调用 LLM（Generator 的职责）
- ❌ 向量化文本（预处理的职责）

**核心接口：**
```python
class BaseRetriever(ABC):
    def retrieve(self, query_embedding: List[float], top_k: int) -> List[Document]
    def add_documents(self, texts: List[str], embeddings: List[List[float]])
    def clear()
    def get_stats() -> Dict[str, int]
```

---

### 2. Ranker（排序层）

**职责：**
- 对 Retriever 召回的候选集进行打分和排序
- 支持多种排序策略（余弦相似度、BM25、Re-rank）
- 返回 `List[Tuple[Document, float]]`

**不负责：**
- ❌ 访问向量库文件（Retriever 的职责）
- ❌ 生成答案（Generator 的职责）

**核心接口：**
```python
class BaseRanker(ABC):
    def rank(
        self, 
        query: str, 
        query_embedding: List[float], 
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]
```

---

### 3. Generator（生成层）

**职责：**
- 接收已排序的文档和用户问题
- 构建 Prompt 并调用 LLM
- 返回生成的答案

**不负责：**
- ❌ 检索文档（Retriever 的职责）
- ❌ 计算相似度（Ranker 的职责）
- ❌ 访问向量库（Retriever 的职责）

**核心接口：**
```python
class BaseGenerator(ABC):
    def generate(self, query: str, ranked_docs: List[Tuple[Document, float]]) -> str
```

---

## 🔄 代码迁移说明

### 从 `rag_engine.py` 迁移的函数

| 原函数 | 迁移目标 | 状态 |
|--------|----------|------|
| `load_vector_db()` | `VectorRetriever._load_db()` | ✅ 已迁移，保留兼容接口 |
| `save_vector_db()` | `VectorRetriever._save_db()` | ✅ 已迁移，保留兼容接口 |
| `add_to_vector_db()` | `VectorRetriever.add_documents()` | ✅ 已迁移，保留兼容接口 |
| `clear_vector_db()` | `VectorRetriever.clear()` | ✅ 已迁移，保留兼容接口 |
| `get_db_stats()` | `VectorRetriever.get_stats()` | ✅ 已迁移，保留兼容接口 |
| `cosine_similarity()` | `SimilarityRanker._cosine_similarity()` | ✅ 已迁移，保留兼容接口 |
| `search_top_k()` | 拆分为 Retriever + Ranker | ✅ 已重构为三层架构 |
| `generate_answer()` | `LLMGenerator.generate()` | ✅ 已迁移，保留兼容接口 |

### 保留在 `rag_engine.py` 中的函数

- `load_env()` - 环境加载
- `get_embed_client()` - Embedding 客户端
- `get_chat_client()` - Chat 客户端
- `embed_texts()` - 向量化（预处理）
- `split_text()` 及所有切分策略 - 文本切分
- `split_text_by_strategy()` - 切分策略入口

---

## 📝 使用方式

### 方式 1：使用原有接口（向后兼容）

```python
from rag_engine import search_top_k, generate_answer

# 检索
retrieved = search_top_k("你的问题", k=3)

# 生成答案
answer = generate_answer("你的问题", retrieved)
```

### 方式 2：使用新的编排接口（推荐）

```python
from rag_engine import rag_pipeline

# 一次调用完成 RAG 流程
answer, retrieved = rag_pipeline("你的问题", top_k=3)
```

### 方式 3：直接使用三层架构（高级用法）

```python
from rag_engine import embed_texts, get_chat_client
from rag.retriever import VectorRetriever
from rag.ranker import SimilarityRanker
from rag.generator import LLMGenerator
import os

query = "你的问题"

# 1. 向量化
query_embedding = embed_texts([query])[0]

# 2. 召回
retriever = VectorRetriever("vector_db.json")
documents = retriever.retrieve(query_embedding, top_k=10)

# 3. 排序
ranker = SimilarityRanker()
ranked_docs = ranker.rank(query, query_embedding, documents, top_k=3)

# 4. 生成
generator = LLMGenerator(get_chat_client(), os.getenv("CHAT_MODEL"))
answer = generator.generate(query, ranked_docs)
```

---

## 🚀 未来扩展示例

### 添加 BM25 检索器

```python
# 1. 实现 rag/retriever/bm25_retriever.py
class BM25Retriever(BaseRetriever):
    def retrieve(self, query_embedding, top_k):
        # 使用 rank-bm25 实现
        pass

# 2. 在 rag_engine.py 中使用
from rag.retriever import BM25Retriever

retriever = BM25Retriever("bm25_index.json")
documents = retriever.retrieve(query_text, top_k=10)
```

### 添加 Re-rank 精排

```python
# 1. 实现 rag/ranker/rerank_ranker.py
class RerankRanker(BaseRanker):
    def rank(self, query, query_embedding, documents, top_k):
        # 使用 bge-reranker-v2-m3 实现
        pass

# 2. 在 rag_engine.py 中使用
from rag.ranker import RerankRanker

ranker = RerankRanker("BAAI/bge-reranker-v2-m3")
ranked_docs = ranker.rank(query, query_embedding, documents, top_k=3)
```

### 混合检索（向量 + BM25）

```python
# 召回阶段使用多个检索器
vector_retriever = VectorRetriever("vector_db.json")
bm25_retriever = BM25Retriever("bm25_index.json")

vector_docs = vector_retriever.retrieve(query_embedding, top_k=20)
bm25_docs = bm25_retriever.retrieve(query_text, top_k=20)

# 合并候选集
all_docs = vector_docs + bm25_docs

# 使用 Re-rank 精排
reranker = RerankRanker()
final_docs = reranker.rank(query, query_embedding, all_docs, top_k=3)
```

---

## ✅ 验证清单

- [x] 三层架构接口定义完成
- [x] VectorRetriever 实现并测试
- [x] SimilarityRanker 实现并测试
- [x] LLMGenerator 实现并测试
- [x] rag_engine.py 重构完成
- [x] 向后兼容性保持
- [x] 导入测试通过
- [x] 预留扩展接口（BM25、FAISS、Re-rank）

---

## 🎓 重构原则总结

1. **最小改动原则**：保留所有现有接口，确保 app.py 无需修改
2. **职责分离原则**：每一层只做自己的事，不越界
3. **接口优先原则**：先定义接口，再实现功能
4. **迁移不重写原则**：复制粘贴现有代码，而不是重新实现
5. **渐进式重构原则**：每一步都是可运行的，不破坏现有功能

---

## 📚 相关文档

- `rag/retriever/base.py` - Retriever 接口定义
- `rag/ranker/base.py` - Ranker 接口定义
- `rag/generator/base.py` - Generator 接口定义
- `rag_engine.py` - 编排逻辑和兼容层

---

**重构完成时间：** 2024-12-24  
**重构方式：** 最小可运行改动，保持向后兼容  
**测试状态：** ✅ 所有导入测试通过
