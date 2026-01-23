"""
智能路由系统
根据查询特征自动选择最佳检索策略
"""

from .query_analyzer import QueryAnalyzer, QueryProfile
from .strategy_router import StrategyRouter, RetrievalStrategy
from .presets import STRATEGY_PRESETS

__all__ = [
    'QueryAnalyzer',
    'QueryProfile',
    'StrategyRouter',
    'RetrievalStrategy',
    'STRATEGY_PRESETS',
]
