"""
Ranker 基类定义
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from src.rag.retriever.base import Document


class BaseRanker(ABC):
    """排序层基类
    
    职责：
    - 对 Retriever 召回的候选集进行打分和排序
    - 支持多种排序策略（余弦相似度、BM25、Re-rank 模型等）
    
    不负责：
    - 访问向量库文件（Retriever 的职责）
    - 生成答案（Generator 的职责）
    """
    
    @abstractmethod
    def rank(
        self, 
        query: str, 
        query_embedding: List[float], 
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """对文档进行打分和排序
        
        Args:
            query: 查询文本
            query_embedding: 查询向量
            documents: 候选文档列表
            top_k: 返回的文档数量，None 表示返回全部
        
        Returns:
            排序后的 (文档, 分数) 列表，按分数降序排列
        """
        pass
