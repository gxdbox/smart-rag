"""
策略路由器
根据查询特征自动选择最佳检索策略
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from .query_analyzer import QueryProfile
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalStrategy:
    """检索策略"""
    mode: str  # vector/bm25/hybrid/hybrid_rerank/hirag/hirag_hybrid
    enable_hyde: bool = False
    enable_multi_variant: bool = False
    enable_query_expansion: bool = False
    enable_hirag: bool = False
    enable_adaptive_filter: bool = True
    enable_rerank: bool = False
    
    # 检索参数
    top_k: int = 5
    recall_k: int = 20
    vector_weight: float = 0.5
    
    # HiRAG 参数
    hirag_mode: str = "hierarchical"  # hierarchical/local/global/bridge
    hirag_weights: Dict[str, float] = field(default_factory=lambda: {
        'local': 0.4,
        'global': 0.3,
        'bridge': 0.3
    })
    
    # 融合策略
    fusion_strategy: str = "weighted"  # weighted/rrf/max
    
    # 路由决策信息
    reason: str = ""
    confidence: float = 1.0


class StrategyRouter:
    """策略路由器"""
    
    def __init__(self):
        logger.info("StrategyRouter 初始化完成")
    
    def route(self, query_profile: QueryProfile) -> RetrievalStrategy:
        """
        根据查询特征路由到最佳策略
        
        Args:
            query_profile: 查询特征画像
            
        Returns:
            RetrievalStrategy: 检索策略
        """
        logger.info(f"开始路由: query_type={query_profile.query_type}, "
                   f"complexity={query_profile.complexity}, "
                   f"domain={query_profile.domain}")
        
        # 规则引擎
        strategy = self._apply_routing_rules(query_profile)
        
        logger.info(f"路由完成: mode={strategy.mode}, reason={strategy.reason}")
        return strategy
    
    def _apply_routing_rules(self, profile: QueryProfile) -> RetrievalStrategy:
        """应用路由规则"""
        
        # 规则 1: 简单事实查询 → 向量检索
        if profile.query_type == 'factual' and profile.complexity == 'simple' and not profile.requires_context:
            return RetrievalStrategy(
                mode='vector',
                enable_hyde=False,
                enable_hirag=False,
                top_k=5,
                reason="简单事实查询，使用快速向量检索"
            )
        
        # 规则 2: 分析型查询 + 需要上下文 → HiRAG 混合检索
        if profile.query_type == 'analytical' and profile.requires_context:
            return RetrievalStrategy(
                mode='hirag_hybrid',
                enable_hirag=True,
                enable_rerank=True,
                hirag_mode='hierarchical',
                hirag_weights={'local': 0.3, 'global': 0.4, 'bridge': 0.3},
                fusion_strategy='weighted',
                vector_weight=0.2,
                recall_k=30,
                reason="分析型查询需要全局视角，启用 HiRAG 层次化检索"
            )
        
        # 规则 3: 流程型查询 → HiRAG Bridge 层
        if profile.query_type == 'procedural':
            return RetrievalStrategy(
                mode='hirag',
                enable_hirag=True,
                hirag_mode='bridge',
                top_k=5,
                reason="流程型查询需要完整步骤，使用 Bridge 层（中等粒度）"
            )
        
        # 规则 4: 高模糊度查询 → HyDE + 多变体
        if profile.ambiguity_score > 0.6:
            return RetrievalStrategy(
                mode='hybrid_rerank',
                enable_hyde=True,
                enable_multi_variant=True,
                enable_rerank=True,
                recall_k=30,
                reason="查询模糊，启用 HyDE 假设文档生成和多变体扩展"
            )
        
        # 规则 5: 多问题查询 → 混合检索 + Rerank
        if profile.has_multiple_questions:
            return RetrievalStrategy(
                mode='hybrid_rerank',
                enable_rerank=True,
                recall_k=30,
                top_k=8,
                reason="多问题查询，使用混合检索 + Rerank 精排"
            )
        
        # 规则 6: 政策领域 + 复杂查询 → HiRAG 政策分析模式
        if profile.domain == 'policy' and profile.complexity in ['medium', 'complex']:
            return RetrievalStrategy(
                mode='hirag_hybrid',
                enable_hirag=True,
                enable_rerank=True,
                hirag_mode='hierarchical',
                hirag_weights={'local': 0.3, 'global': 0.4, 'bridge': 0.3},
                fusion_strategy='weighted',
                vector_weight=0.2,
                recall_k=30,
                reason="政策领域复杂查询，使用 HiRAG 政策分析模式"
            )
        
        # 规则 7: 探索型查询 → HiRAG Global 层
        if profile.query_type == 'exploratory':
            return RetrievalStrategy(
                mode='hirag',
                enable_hirag=True,
                hirag_mode='global',
                top_k=5,
                reason="探索型查询需要宏观视角，使用 Global 层（文档级摘要）"
            )
        
        # 规则 8: 中等复杂度 → 混合检索 + Rerank
        if profile.complexity == 'medium':
            return RetrievalStrategy(
                mode='hybrid_rerank',
                enable_rerank=True,
                vector_weight=0.5,
                recall_k=20,
                reason="中等复杂度查询，使用混合检索 + Rerank"
            )
        
        # 规则 9: 高复杂度 → HiRAG 混合检索
        if profile.complexity == 'complex':
            return RetrievalStrategy(
                mode='hirag_hybrid',
                enable_hirag=True,
                enable_rerank=True,
                hirag_mode='hierarchical',
                fusion_strategy='rrf',
                recall_k=30,
                top_k=8,
                reason="高复杂度查询，使用 HiRAG 混合检索 + RRF 融合"
            )
        
        # 默认规则: 混合检索 + 自适应过滤
        return RetrievalStrategy(
            mode='hybrid',
            enable_adaptive_filter=True,
            vector_weight=0.5,
            top_k=5,
            reason="通用查询，使用平衡的混合检索"
        )
    
    def get_strategy_summary(self, strategy: RetrievalStrategy) -> Dict[str, Any]:
        """获取策略摘要信息"""
        return {
            'mode': strategy.mode,
            'enable_hirag': strategy.enable_hirag,
            'enable_hyde': strategy.enable_hyde,
            'enable_rerank': strategy.enable_rerank,
            'top_k': strategy.top_k,
            'reason': strategy.reason,
            'confidence': strategy.confidence
        }
