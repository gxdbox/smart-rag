"""
Generator 基类定义
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from rag.retriever.base import Document


class BaseGenerator(ABC):
    """生成层基类
    
    职责：
    - 接收已排序的文档和用户问题
    - 构建 Prompt 并调用 LLM
    - 返回生成的答案
    
    不负责：
    - 检索文档（Retriever 的职责）
    - 计算相似度（Ranker 的职责）
    - 访问向量库（Retriever 的职责）
    """
    
    @abstractmethod
    def generate(self, query: str, ranked_docs: List[Tuple[Document, float]]) -> str:
        """根据排序后的文档生成答案
        
        Args:
            query: 用户问题
            ranked_docs: 排序后的 (文档, 分数) 列表
        
        Returns:
            生成的答案文本
        """
        pass
