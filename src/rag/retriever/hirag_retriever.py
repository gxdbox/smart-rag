"""
HiRAG 检索器
基于 HierarchicalIndexer 实现层次化检索，提供统一的检索接口
"""

from typing import List, Dict, Any, Optional
import numpy as np
from src.utils.logger import get_logger
from src.rag.indexer import HierarchicalIndexer
from config import HiRAGConfig

logger = get_logger(__name__)


class HiRAGRetriever:
    """
    HiRAG 检索器
    
    提供统一的层次化检索接口，支持：
    - 融合检索（local + global + bridge）
    - 单层检索（local/global/bridge）
    - 多种融合策略（weighted/rrf/max）
    """
    
    def __init__(self, config: HiRAGConfig, vector_retriever):
        """
        初始化 HiRAG 检索器
        
        Args:
            config: HiRAGConfig 配置对象
            vector_retriever: 现有的 VectorRetriever 实例
        """
        self.config = config
        self.indexer = HierarchicalIndexer(config, vector_retriever)
        logger.info("HiRAGRetriever 初始化完成")
    
    def add_documents(self, documents: List[str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        添加文档并构建层次化索引
        
        Args:
            documents: 文档列表
            metadata: 可选的元数据
            
        Returns:
            构建统计信息
        """
        logger.info(f"开始构建层次化索引，文档数量: {len(documents)}")
        stats = self.indexer.build_index(documents, metadata)
        logger.info(f"索引构建完成: {stats}")
        return stats
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        执行层次化检索（融合三层结果）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            融合后的检索结果列表
        """
        if not self.config.enable_hierarchical:
            logger.warning("层次化检索未启用，返回空结果")
            return []
        
        # 从三层检索
        hierarchical_results = self.indexer.retrieve_hierarchical(query, top_k)
        
        # 融合结果
        fused_results = self._fuse_results(hierarchical_results, top_k)
        
        logger.info(f"层次化检索完成，返回 {len(fused_results)} 个结果")
        return fused_results
    
    def retrieve_by_mode(self, query: str, mode: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        按指定模式检索
        
        Args:
            query: 查询文本
            mode: 检索模式
                - "local": 仅局部知识
                - "global": 仅全局知识
                - "bridge": 仅桥梁知识
                - "hierarchical": 融合三层（等同于 retrieve()）
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        if mode == "hierarchical":
            return self.retrieve(query, top_k)
        
        if mode not in ["local", "global", "bridge"]:
            raise ValueError(f"不支持的检索模式: {mode}")
        
        # 检查该层是否启用
        layer_enabled = getattr(self.config, f"enable_{mode}", False)
        if not layer_enabled:
            logger.warning(f"{mode} 层未启用，返回空结果")
            return []
        
        results = self.indexer.retrieve_from_layer(mode, query, top_k)
        logger.info(f"{mode} 模式检索完成，返回 {len(results)} 个结果")
        return results
    
    def _fuse_results(self, hierarchical_results: Dict[str, List[Dict[str, Any]]], top_k: int) -> List[Dict[str, Any]]:
        """
        融合三层检索结果
        
        Args:
            hierarchical_results: 分层检索结果
                {
                    'local': [...],
                    'global': [...],
                    'bridge': [...]
                }
            top_k: 返回结果数量
            
        Returns:
            融合后的结果列表
        """
        fusion_strategy = self.config.fusion_strategy
        
        if fusion_strategy == "weighted":
            return self._weighted_fusion(hierarchical_results, top_k)
        elif fusion_strategy == "rrf":
            return self._rrf_fusion(hierarchical_results, top_k)
        elif fusion_strategy == "max":
            return self._max_fusion(hierarchical_results, top_k)
        else:
            logger.warning(f"未知的融合策略: {fusion_strategy}，使用 weighted")
            return self._weighted_fusion(hierarchical_results, top_k)
    
    def _weighted_fusion(self, hierarchical_results: Dict[str, List[Dict[str, Any]]], top_k: int) -> List[Dict[str, Any]]:
        """
        加权融合策略
        
        根据配置的权重对三层结果进行加权融合
        """
        all_results = []
        
        # Local 层
        if 'local' in hierarchical_results:
            for result in hierarchical_results['local']:
                result['final_score'] = result.get('score', 0.0) * self.config.local_weight
                all_results.append(result)
        
        # Global 层
        if 'global' in hierarchical_results:
            for result in hierarchical_results['global']:
                result['final_score'] = result.get('score', 0.0) * self.config.global_weight
                all_results.append(result)
        
        # Bridge 层
        if 'bridge' in hierarchical_results:
            for result in hierarchical_results['bridge']:
                result['final_score'] = result.get('score', 0.0) * self.config.bridge_weight
                all_results.append(result)
        
        # 按最终分数排序
        all_results.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
        
        return all_results[:top_k]
    
    def _rrf_fusion(self, hierarchical_results: Dict[str, List[Dict[str, Any]]], top_k: int, k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) 融合策略
        
        RRF 公式: score = sum(1 / (k + rank))
        
        Args:
            hierarchical_results: 分层检索结果
            top_k: 返回结果数量
            k: RRF 参数（默认 60）
        """
        # 收集所有文档及其在各层的排名
        doc_scores = {}
        
        for layer_name, results in hierarchical_results.items():
            weight = getattr(self.config, f"{layer_name}_weight", 1.0)
            
            for rank, result in enumerate(results, start=1):
                doc_text = result['text']
                rrf_score = weight / (k + rank)
                
                if doc_text not in doc_scores:
                    doc_scores[doc_text] = {
                        'text': doc_text,
                        'knowledge_type': result.get('knowledge_type', layer_name),
                        'metadata': result.get('metadata', {}),
                        'final_score': 0.0,
                        'layer_scores': {}
                    }
                
                doc_scores[doc_text]['final_score'] += rrf_score
                doc_scores[doc_text]['layer_scores'][layer_name] = rrf_score
        
        # 转换为列表并排序
        all_results = list(doc_scores.values())
        all_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return all_results[:top_k]
    
    def _max_fusion(self, hierarchical_results: Dict[str, List[Dict[str, Any]]], top_k: int) -> List[Dict[str, Any]]:
        """
        最大分数融合策略
        
        对于同一文档，取其在各层中的最高分数
        """
        doc_scores = {}
        
        for layer_name, results in hierarchical_results.items():
            for result in results:
                doc_text = result['text']
                score = result.get('score', 0.0)
                
                if doc_text not in doc_scores or score > doc_scores[doc_text]['final_score']:
                    doc_scores[doc_text] = {
                        'text': doc_text,
                        'knowledge_type': result.get('knowledge_type', layer_name),
                        'metadata': result.get('metadata', {}),
                        'final_score': score,
                        'best_layer': layer_name
                    }
        
        # 转换为列表并排序
        all_results = list(doc_scores.values())
        all_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return all_results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息
        
        Returns:
            统计信息字典
        """
        return self.indexer.get_index_stats()
    
    def clear(self) -> None:
        """清空所有索引"""
        self.indexer.clear_index()
        logger.info("HiRAGRetriever 索引已清空")
