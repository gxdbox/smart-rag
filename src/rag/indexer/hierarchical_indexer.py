"""
层次化索引器
管理 HiRAG 三层知识的构建和检索
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.utils.logger import get_logger
from .knowledge_builder import LocalKnowledgeLayer, GlobalKnowledgeLayer, BridgeKnowledgeLayer

logger = get_logger(__name__)


class HierarchicalIndexer:
    """
    层次化索引器
    
    统一管理 Local、Global、Bridge 三层知识的生命周期
    """
    
    def __init__(self, config, vector_retriever):
        """
        初始化层次化索引器
        
        Args:
            config: HiRAGConfig 配置对象
            vector_retriever: 现有的 VectorRetriever 实例
        """
        self.config = config
        
        # 初始化三层知识
        self.local_layer = LocalKnowledgeLayer(vector_retriever)
        self.global_layer = GlobalKnowledgeLayer(
            storage_path=config.index_cache_path + "/global"
        )
        self.bridge_layer = BridgeKnowledgeLayer(
            storage_path=config.index_cache_path + "/bridge"
        )
        
        self.layers = {
            'local': self.local_layer,
            'global': self.global_layer,
            'bridge': self.bridge_layer
        }
        
        logger.info("HierarchicalIndexer 初始化完成")
    
    def build_index(self, documents: List[str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        构建层次化索引
        
        Args:
            documents: 文档列表
            metadata: 元数据
            
        Returns:
            构建统计信息
        """
        stats = {
            'total_documents': len(documents),
            'layers_built': [],
            'start_time': datetime.now().isoformat()
        }
        
        try:
            # 1. Local 层（复用现有索引，无需构建）
            if self.config.enable_local:
                logger.info("步骤 1/3: 构建 Local 知识层（复用现有向量索引）")
                self.local_layer.build(documents, metadata)
                stats['layers_built'].append('local')
            
            # 2. Global 层（生成文档摘要）
            if self.config.enable_global:
                logger.info("步骤 2/3: 构建 Global 知识层（生成文档摘要）")
                global_metadata = {
                    'max_tokens': self.config.global_summary_max_tokens
                }
                self.global_layer.build(documents, global_metadata)
                stats['layers_built'].append('global')
            
            # 3. Bridge 层（生成中等粒度知识块）
            if self.config.enable_bridge:
                logger.info("步骤 3/3: 构建 Bridge 知识层（生成桥梁知识）")
                bridge_metadata = {
                    'chunk_size': self.config.bridge_chunk_size,
                    'overlap': self.config.bridge_overlap
                }
                self.bridge_layer.build(documents, bridge_metadata)
                stats['layers_built'].append('bridge')
            
            stats['end_time'] = datetime.now().isoformat()
            stats['success'] = True
            
            logger.info(f"层次化索引构建完成: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"层次化索引构建失败: {e}", exc_info=True)
            stats['success'] = False
            stats['error'] = str(e)
            return stats
    
    def update_index(self, new_documents: List[str]) -> Dict[str, Any]:
        """
        增量更新索引
        
        Args:
            new_documents: 新增文档列表
            
        Returns:
            更新统计信息
        """
        stats = {
            'new_documents': len(new_documents),
            'layers_updated': []
        }
        
        try:
            if self.config.enable_local:
                self.local_layer.update(new_documents)
                stats['layers_updated'].append('local')
            
            if self.config.enable_global:
                self.global_layer.update(new_documents)
                stats['layers_updated'].append('global')
            
            if self.config.enable_bridge:
                self.bridge_layer.update(new_documents)
                stats['layers_updated'].append('bridge')
            
            stats['success'] = True
            logger.info(f"增量更新完成: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"增量更新失败: {e}", exc_info=True)
            stats['success'] = False
            stats['error'] = str(e)
            return stats
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'local': self.local_layer.get_stats().__dict__,
            'global': self.global_layer.get_stats().__dict__,
            'bridge': self.bridge_layer.get_stats().__dict__,
            'total_items': (
                self.local_layer.get_stats().total_items +
                self.global_layer.get_stats().total_items +
                self.bridge_layer.get_stats().total_items
            )
        }
        return stats
    
    def clear_index(self) -> None:
        """清空所有索引"""
        self.local_layer.clear()
        self.global_layer.clear()
        self.bridge_layer.clear()
        logger.info("所有层次化索引已清空")
    
    def retrieve_from_layer(self, layer_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        从指定层检索
        
        Args:
            layer_name: 层名称 (local/global/bridge)
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        if layer_name not in self.layers:
            raise ValueError(f"未知的知识层: {layer_name}")
        
        return self.layers[layer_name].retrieve(query, top_k)
    
    def retrieve_hierarchical(self, query: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        从所有启用的层检索
        
        Args:
            query: 查询文本
            top_k: 每层返回的结果数量
            
        Returns:
            分层检索结果
            {
                'local': [...],
                'global': [...],
                'bridge': [...]
            }
        """
        results = {}
        
        if self.config.enable_local:
            results['local'] = self.local_layer.retrieve(query, top_k)
        
        if self.config.enable_global:
            results['global'] = self.global_layer.retrieve(query, top_k)
        
        if self.config.enable_bridge:
            results['bridge'] = self.bridge_layer.retrieve(query, top_k)
        
        return results
