"""
Retriever 基类定义
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Document:
    """文档数据结构"""
    id: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)


class BaseRetriever(ABC):
    """召回层基类
    
    职责：
    - 根据查询条件，从存储中召回候选文档集合
    - 管理向量库的加载/保存
    
    不负责：
    - 计算相似度分数（Ranker 的职责）
    - 排序结果（Ranker 的职责）
    - 调用 LLM 生成答案（Generator 的职责）
    - 向量化文本（预处理的职责，在 rag_engine.py 中）
    """
    
    @abstractmethod
    def retrieve(self, query_embedding: List[float], top_k: int) -> List[Document]:
        """根据查询向量召回候选文档
        
        Args:
            query_embedding: 查询向量
            top_k: 召回文档数量（可以召回更多候选，由 Ranker 精排）
        
        Returns:
            候选文档列表
        """
        pass
    
    @abstractmethod
    def add_documents(self, texts: List[str], embeddings: List[List[float]]):
        """添加文档到存储
        
        Args:
            texts: 文本列表
            embeddings: 对应的向量列表
        """
        pass
    
    @abstractmethod
    def clear(self):
        """清空存储"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, int]:
        """获取存储统计信息
        
        Returns:
            统计信息字典，如 {"total_chunks": 100, "total_embeddings": 100}
        """
        pass
