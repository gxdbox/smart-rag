"""
知识层基类
定义层次化知识的统一接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class KnowledgeStats:
    """知识层统计信息"""
    total_items: int = 0
    total_size_bytes: int = 0
    last_updated: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class KnowledgeLayer(ABC):
    """
    知识层抽象基类
    
    定义了层次化知识系统中每一层的标准接口：
    - Local Layer: 局部细节知识（chunk 级别）
    - Global Layer: 全局概览知识（文档级摘要）
    - Bridge Layer: 桥梁知识（连接局部与全局）
    """
    
    def __init__(self, layer_name: str):
        """
        初始化知识层
        
        Args:
            layer_name: 知识层名称 (local/global/bridge)
        """
        self.layer_name = layer_name
        self._stats = KnowledgeStats()
    
    @abstractmethod
    def build(self, documents: List[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        构建知识层
        
        Args:
            documents: 文档列表
            metadata: 可选的元数据
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        从知识层检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表，每个结果包含：
            {
                'text': str,           # 文本内容
                'score': float,        # 相关度分数
                'knowledge_type': str, # 知识类型
                'metadata': dict       # 元数据
            }
        """
        pass
    
    @abstractmethod
    def update(self, new_documents: List[str]) -> None:
        """
        增量更新知识层
        
        Args:
            new_documents: 新增文档列表
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空知识层"""
        pass
    
    def get_stats(self) -> KnowledgeStats:
        """
        获取知识层统计信息
        
        Returns:
            统计信息对象
        """
        return self._stats
    
    def _update_stats(self, **kwargs):
        """更新统计信息"""
        for key, value in kwargs.items():
            if hasattr(self._stats, key):
                setattr(self._stats, key, value)
