"""
知识层构建器
实现 Local、Global、Bridge 三层知识的具体构建逻辑
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.utils.logger import get_logger
from .base import KnowledgeLayer

logger = get_logger(__name__)


class LocalKnowledgeLayer(KnowledgeLayer):
    """
    局部知识层
    复用现有的向量检索系统，不创建新的索引
    """
    
    def __init__(self, vector_retriever):
        """
        初始化局部知识层
        
        Args:
            vector_retriever: 现有的 VectorRetriever 实例
        """
        super().__init__("local")
        self.vector_retriever = vector_retriever
        logger.info("LocalKnowledgeLayer 初始化完成（复用现有向量索引）")
    
    def build(self, documents: List[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        构建局部知识（实际上是复用现有索引，无需重新构建）
        
        Args:
            documents: 文档列表（已经在向量库中）
            metadata: 元数据
        """
        stats = self.vector_retriever.get_stats()
        self._update_stats(
            total_items=stats.get('total_chunks', 0),
            last_updated=datetime.now().isoformat()
        )
        logger.info(f"LocalKnowledgeLayer: 复用现有索引，共 {stats.get('total_chunks', 0)} 个 chunks")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        从局部知识层检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        from rag_engine import embed_texts
        
        query_embedding = embed_texts([query])[0]
        documents = self.vector_retriever.retrieve(query_embedding, top_k=top_k)
        
        results = []
        for doc in documents:
            results.append({
                'text': doc.text,
                'score': 0.0,  # 向量检索的分数在后续计算
                'knowledge_type': 'local',
                'metadata': {
                    'source': 'vector_db',
                    'doc_id': doc.id
                }
            })
        
        return results
    
    def update(self, new_documents: List[str]) -> None:
        """
        增量更新（由 VectorRetriever 负责）
        
        Args:
            new_documents: 新增文档列表
        """
        logger.info(f"LocalKnowledgeLayer: 增量更新由 VectorRetriever 处理")
    
    def clear(self) -> None:
        """清空（由 VectorRetriever 负责）"""
        logger.warning("LocalKnowledgeLayer: 清空操作应通过 VectorRetriever 执行")


class GlobalKnowledgeLayer(KnowledgeLayer):
    """
    全局知识层
    存储文档级别的摘要信息
    """
    
    def __init__(self, storage_path: str = "./hirag_cache/global"):
        """
        初始化全局知识层
        
        Args:
            storage_path: 存储路径
        """
        super().__init__("global")
        self.storage_path = storage_path
        self.summaries: Dict[str, Dict[str, Any]] = {}
        
        os.makedirs(storage_path, exist_ok=True)
        logger.info(f"GlobalKnowledgeLayer 初始化完成，存储路径: {storage_path}")
    
    def build(self, documents: List[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        构建全局知识（生成文档摘要）
        
        Args:
            documents: 文档列表
            metadata: 元数据（包含文档 ID、配置等）
        """
        from rag_engine import get_chat_client
        import hashlib
        
        client = get_chat_client()
        model = os.getenv("CHAT_MODEL", "deepseek-chat")
        max_tokens = metadata.get('max_tokens', 500) if metadata else 500
        
        for i, doc in enumerate(documents):
            doc_id = hashlib.md5(doc.encode()).hexdigest()[:16]
            
            if doc_id in self.summaries:
                logger.info(f"文档 {doc_id} 摘要已存在，跳过")
                continue
            
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的文档摘要助手。请生成简洁、准确的文档摘要。"},
                        {"role": "user", "content": f"请为以下文档生成摘要（不超过{max_tokens}字）：\n\n{doc[:2000]}"}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.3
                )
                
                summary = response.choices[0].message.content
                
                self.summaries[doc_id] = {
                    'summary': summary,
                    'doc_id': doc_id,
                    'created_at': datetime.now().isoformat(),
                    'original_length': len(doc)
                }
                
                logger.info(f"生成文档 {doc_id} 的全局摘要 ({i+1}/{len(documents)})")
                
            except Exception as e:
                logger.error(f"生成文档 {doc_id} 摘要失败: {e}", exc_info=True)
        
        self._update_stats(
            total_items=len(self.summaries),
            last_updated=datetime.now().isoformat()
        )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        从全局知识层检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        from rag_engine import embed_texts
        import numpy as np
        
        if not self.summaries:
            return []
        
        query_embedding = embed_texts([query])[0]
        summary_texts = [s['summary'] for s in self.summaries.values()]
        summary_embeddings = embed_texts(summary_texts)
        
        scores = []
        for emb in summary_embeddings:
            similarity = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            scores.append(similarity)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        summary_list = list(self.summaries.values())
        for idx in top_indices:
            summary_data = summary_list[idx]
            results.append({
                'text': summary_data['summary'],
                'score': float(scores[idx]),
                'knowledge_type': 'global',
                'metadata': {
                    'doc_id': summary_data['doc_id'],
                    'created_at': summary_data['created_at']
                }
            })
        
        return results
    
    def update(self, new_documents: List[str]) -> None:
        """
        增量更新全局知识
        
        Args:
            new_documents: 新增文档列表
        """
        self.build(new_documents)
    
    def clear(self) -> None:
        """清空全局知识"""
        self.summaries.clear()
        self._update_stats(total_items=0)
        logger.info("GlobalKnowledgeLayer 已清空")


class BridgeKnowledgeLayer(KnowledgeLayer):
    """
    桥梁知识层
    连接局部细节和全局概览的中间层
    """
    
    def __init__(self, storage_path: str = "./hirag_cache/bridge"):
        """
        初始化桥梁知识层
        
        Args:
            storage_path: 存储路径
        """
        super().__init__("bridge")
        self.storage_path = storage_path
        self.bridge_chunks: List[Dict[str, Any]] = []
        
        os.makedirs(storage_path, exist_ok=True)
        logger.info(f"BridgeKnowledgeLayer 初始化完成，存储路径: {storage_path}")
    
    def build(self, documents: List[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        构建桥梁知识（生成中等粒度的知识块）
        
        Args:
            documents: 文档列表
            metadata: 元数据（包含 chunk_size、overlap 等）
        """
        chunk_size = metadata.get('chunk_size', 1000) if metadata else 1000
        overlap = metadata.get('overlap', 200) if metadata else 200
        
        for doc in documents:
            chunks = self._split_into_bridge_chunks(doc, chunk_size, overlap)
            
            for chunk in chunks:
                self.bridge_chunks.append({
                    'text': chunk,
                    'created_at': datetime.now().isoformat()
                })
        
        self._update_stats(
            total_items=len(self.bridge_chunks),
            last_updated=datetime.now().isoformat()
        )
        
        logger.info(f"BridgeKnowledgeLayer: 生成 {len(self.bridge_chunks)} 个桥梁知识块")
    
    def _split_into_bridge_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        将文本切分为桥梁知识块
        
        Args:
            text: 文本
            chunk_size: 块大小
            overlap: 重叠大小
            
        Returns:
            切分后的文本块列表
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        
        return chunks
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        从桥梁知识层检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        from rag_engine import embed_texts
        import numpy as np
        
        if not self.bridge_chunks:
            return []
        
        query_embedding = embed_texts([query])[0]
        chunk_texts = [c['text'] for c in self.bridge_chunks]
        chunk_embeddings = embed_texts(chunk_texts)
        
        scores = []
        for emb in chunk_embeddings:
            similarity = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            scores.append(similarity)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk_data = self.bridge_chunks[idx]
            results.append({
                'text': chunk_data['text'],
                'score': float(scores[idx]),
                'knowledge_type': 'bridge',
                'metadata': {
                    'created_at': chunk_data['created_at']
                }
            })
        
        return results
    
    def update(self, new_documents: List[str]) -> None:
        """
        增量更新桥梁知识
        
        Args:
            new_documents: 新增文档列表
        """
        self.build(new_documents)
    
    def clear(self) -> None:
        """清空桥梁知识"""
        self.bridge_chunks.clear()
        self._update_stats(total_items=0)
        logger.info("BridgeKnowledgeLayer 已清空")
