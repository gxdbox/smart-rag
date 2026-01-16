"""
RAG 核心引擎模块
负责：环境加载、向量化、RAG 流程编排

重构说明：
- 文本切分逻辑已迁移到 src.rag.chunker.text_splitter
- 混合检索逻辑已迁移到 src.rag.core.hybrid_retrieval
- 向量存储/检索逻辑已迁移到 rag.retriever.VectorRetriever
- 相似度计算/排序逻辑已迁移到 rag.ranker.SimilarityRanker
- LLM 答案生成逻辑已迁移到 rag.generator.LLMGenerator
- 本模块保留：环境加载、向量化、基础检索、BM25 管理
"""

import os
import json
from typing import List, Tuple
from openai import OpenAI
from dotenv import load_dotenv
from src.utils.logger import get_logger

logger = get_logger(__name__)

from rag.retriever import VectorRetriever, BM25Retriever
from rag.ranker import SimilarityRanker, RerankRanker
from rag.generator import LLMGenerator
from rag.retriever.base import Document

# 从新模块导入
from src.rag.chunker import split_text, split_text_by_strategy
from src.rag.core import hybrid_search, hybrid_search_with_rerank

# 向量库文件路径
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "vector_db.json")
BM25_DB_PATH = os.path.join(os.path.dirname(__file__), "bm25_index.pkl")


def load_env():
    """加载 .env 环境变量"""
    load_dotenv()
    required_vars = ["EMBED_BASE_URL", "EMBED_API_KEY", "EMBED_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY", "CHAT_MODEL"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"缺少必要的环境变量: {', '.join(missing)}")


def get_embed_client() -> OpenAI:
    """获取 Embedding API 客户端"""
    client = OpenAI(
        api_key=os.getenv("EMBED_API_KEY"),
        base_url=os.getenv("EMBED_BASE_URL"),
    )
    return client


def get_chat_client() -> OpenAI:
    """获取 Chat API 客户端"""
    client = OpenAI(
        api_key=os.getenv("CHAT_API_KEY"),
        base_url=os.getenv("CHAT_BASE_URL"),
    )
    return client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    将文本列表转换为向量
    
    Args:
        texts: 文本列表
    
    Returns:
        向量列表
    """
    client = get_embed_client()
    model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    
    all_embeddings = []
    batch_size = 50
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Embedding 批次 {i//batch_size + 1} 失败: {e}", exc_info=True)
            raise
    
    return all_embeddings


# ==================== 向量库管理（兼容层） ====================

def load_vector_db() -> dict:
    """加载向量数据库（已废弃，保留用于向后兼容）"""
    retriever = VectorRetriever(VECTOR_DB_PATH)
    return retriever._load_db()


def save_vector_db(db: dict):
    """保存向量数据库（已废弃，保留用于向后兼容）"""
    retriever = VectorRetriever(VECTOR_DB_PATH)
    retriever._save_db(db)


def add_to_vector_db(chunks: List[str], embeddings: List[List[float]]):
    """将 chunks 和对应的 embeddings 添加到向量数据库"""
    retriever = VectorRetriever(VECTOR_DB_PATH)
    retriever.add_documents(chunks, embeddings)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算两个向量的余弦相似度（已废弃，保留用于向后兼容）"""
    ranker = SimilarityRanker()
    return ranker._cosine_similarity(vec1, vec2)


def search_top_k(query: str, k: int = 3) -> List[Tuple[str, float]]:
    """根据查询向量检索最相关的 top-k 个 chunks
    
    Args:
        query: 查询文本
        k: 返回的结果数量
    
    Returns:
        [(chunk, similarity_score), ...] 列表
    """
    query_embedding = embed_texts([query])[0]
    
    retriever = VectorRetriever(VECTOR_DB_PATH)
    documents = retriever.retrieve(query_embedding, top_k=k)
    
    ranker = SimilarityRanker()
    ranked_docs = ranker.rank(query, query_embedding, documents, top_k=k)
    
    return [(doc.text, score) for doc, score in ranked_docs]


def generate_answer(query: str, retrieved_chunks: List[Tuple[str, float]], 
                   conversation_history: List[dict] = None) -> str:
    """根据检索到的 chunks 和用户问题，调用 LLM 生成答案
    
    Args:
        query: 用户问题
        retrieved_chunks: 检索到的 chunks 列表 [(chunk, score), ...]
        conversation_history: 对话历史
    
    Returns:
        LLM 生成的答案
    """
    ranked_docs = [
        (Document(id=i, text=chunk, metadata={}), score)
        for i, (chunk, score) in enumerate(retrieved_chunks)
    ]
    
    client = get_chat_client()
    model = os.getenv("CHAT_MODEL", "deepseek-chat")
    generator = LLMGenerator(client, model)
    
    return generator.generate(query, ranked_docs, conversation_history)


def clear_vector_db():
    """清空向量数据库"""
    retriever = VectorRetriever(VECTOR_DB_PATH)
    retriever.clear()


def get_db_stats() -> dict:
    """获取向量数据库统计信息"""
    retriever = VectorRetriever(VECTOR_DB_PATH)
    return retriever.get_stats()


def rag_pipeline(query: str, top_k: int = 3) -> Tuple[str, List[Tuple[str, float]]]:
    """RAG 完整流程编排
    
    Args:
        query: 用户查询
        top_k: 检索数量
    
    Returns:
        (answer, retrieved_chunks) 元组
    """
    query_embedding = embed_texts([query])[0]
    
    retriever = VectorRetriever(VECTOR_DB_PATH)
    documents = retriever.retrieve(query_embedding, top_k=top_k)
    
    ranker = SimilarityRanker()
    ranked_docs = ranker.rank(query, query_embedding, documents, top_k=top_k)
    
    client = get_chat_client()
    model = os.getenv("CHAT_MODEL", "deepseek-chat")
    generator = LLMGenerator(client, model)
    answer = generator.generate(query, ranked_docs)
    
    retrieved_chunks = [(doc.text, score) for doc, score in ranked_docs]
    
    return answer, retrieved_chunks


# ==================== BM25 检索相关函数 ====================

def add_to_bm25_index(chunks: List[str]):
    """将 chunks 添加到 BM25 索引"""
    retriever = BM25Retriever(BM25_DB_PATH)
    retriever.add_documents(chunks)


def search_bm25(query: str, k: int = 3) -> List[Tuple[str, float]]:
    """使用 BM25 检索最相关的 top-k 个 chunks
    
    Args:
        query: 查询文本
        k: 返回的结果数量
    
    Returns:
        [(chunk, bm25_score), ...] 列表
    """
    retriever = BM25Retriever(BM25_DB_PATH)
    documents = retriever.retrieve_by_text(query, top_k=k)
    
    return [(doc.text, doc.metadata.get('bm25_score', 0.0)) for doc in documents]


def clear_bm25_index():
    """清空 BM25 索引"""
    retriever = BM25Retriever(BM25_DB_PATH)
    retriever.clear()


def get_bm25_stats() -> dict:
    """获取 BM25 索引统计信息"""
    retriever = BM25Retriever(BM25_DB_PATH)
    return retriever.get_stats()


def sync_bm25_from_vector_db():
    """从向量库同步数据到 BM25 索引"""
    clear_bm25_index()
    
    with open(VECTOR_DB_PATH, 'r', encoding='utf-8') as f:
        db = json.load(f)
        chunks = db.get('chunks', [])
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        add_to_bm25_index(batch)
    
    return len(chunks)


def search_with_rerank(query: str, k: int = 3, recall_k: int = 20) -> List[Tuple[str, float]]:
    """使用 Rerank 模型进行精排的检索（两阶段检索）
    
    Args:
        query: 查询文本
        k: 最终返回的结果数量
        recall_k: 第一阶段召回的候选数量
    
    Returns:
        [(chunk, rerank_score), ...] 列表
    """
    query_embedding = embed_texts([query])[0]
    
    retriever = VectorRetriever(VECTOR_DB_PATH)
    documents = retriever.retrieve(query_embedding, top_k=recall_k)
    
    if not documents:
        return []
    
    try:
        rerank_ranker = RerankRanker()
        ranked_docs = rerank_ranker.rank(query, query_embedding, documents, top_k=k)
        
        return [(doc.text, score) for doc, score in ranked_docs]
    
    except Exception as e:
        logger.error(f"Rerank 精排失败，返回向量检索结果: {e}", exc_info=True)
        ranker = SimilarityRanker()
        ranked_docs = ranker.rank(query, query_embedding, documents, top_k=k)
        return [(doc.text, score) for doc, score in ranked_docs]
