"""
基于余弦相似度的排序器
迁移自 rag_engine.py 的现有实现
"""

import numpy as np
from typing import List, Tuple
from src.rag.retriever.base import Document
from src.rag.ranker.base import BaseRanker


class SimilarityRanker(BaseRanker):
    """基于余弦相似度的排序器
    
    特点：
    - 使用 NumPy 计算余弦相似度
    - 按相似度降序排序
    - 适合向量检索场景
    """
    
    def __init__(self):
        """初始化相似度排序器"""
        pass
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度（迁移自 cosine_similarity）
        
        Args:
            vec1: 向量1
            vec2: 向量2
        
        Returns:
            余弦相似度分数
        """
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    
    def rank(
        self, 
        query: str, 
        query_embedding: List[float], 
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """对文档进行打分和排序（迁移自 search_top_k 的排序部分）
        
        Args:
            query: 查询文本（此实现中未使用，但保留接口一致性）
            query_embedding: 查询向量
            documents: 候选文档列表
            top_k: 返回的文档数量，None 表示返回全部
        
        Returns:
            排序后的 (文档, 分数) 列表，按分数降序排列
        """
        if not documents:
            return []
        
        query_dim = len(query_embedding)
        
        similarities = []
        for doc in documents:
            if len(doc.embedding) != query_dim:
                continue
            
            sim = self._cosine_similarity(query_embedding, doc.embedding)
            similarities.append((doc, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            return similarities[:top_k]
        return similarities
