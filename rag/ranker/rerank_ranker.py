"""
基于 Re-rank 模型的排序器（预留接口）

TODO: 未来实现
- 使用 bge-reranker-v2-m3 或其他 Re-rank 模型
- 对检索结果进行精排
- 提升复杂查询的准确率
"""

from typing import List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from rag.retriever.base import Document
from rag.ranker.base import BaseRanker


class RerankRanker(BaseRanker):
    """基于 Re-rank 模型的排序器（预留实现）
    
    特点：
    - 使用深度学习模型进行精排
    - 考虑查询和文档的语义交互
    - 显著提升复杂查询的准确率
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """初始化 Re-rank 排序器
        
        Args:
            model_name: Re-rank 模型名称
        """
        self.model_name = model_name
        raise NotImplementedError("RerankRanker 尚未实现，敬请期待")
    
    def rank(
        self, 
        query: str, 
        query_embedding: List[float], 
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """对文档进行打分和排序
        
        注意：Re-rank 主要使用查询文本和文档文本，不依赖向量
        """
        raise NotImplementedError("RerankRanker 尚未实现")
