"""
策略预设库
提供预定义的检索策略配置
"""

from typing import Dict, Any

STRATEGY_PRESETS: Dict[str, Dict[str, Any]] = {
    'smart': {
        'name': '智能路由（推荐）',
        'description': '🤖 根据问题自动选择最佳策略',
        'icon': '🎯',
        'config': {
            'use_smart_routing': True
        }
    },
    
    'quick': {
        'name': '快速模式',
        'description': '⚡ 优先响应速度，适合简单查询',
        'icon': '⚡',
        'config': {
            'mode': 'vector',
            'top_k': 3,
            'enable_adaptive_filter': False,
            'enable_hyde': False,
            'enable_hirag': False
        }
    },
    
    'balanced': {
        'name': '平衡模式',
        'description': '⚖️ 速度与质量平衡，适合大多数场景',
        'icon': '⚖️',
        'config': {
            'mode': 'hybrid_rerank',
            'top_k': 5,
            'recall_k': 20,
            'vector_weight': 0.5,
            'enable_adaptive_filter': True,
            'enable_rerank': True,
            'enable_hirag': False
        }
    },
    
    'accurate': {
        'name': '精确模式',
        'description': '🎯 优先准确率，适合复杂查询',
        'icon': '🎯',
        'config': {
            'mode': 'hirag_hybrid',
            'top_k': 8,
            'recall_k': 30,
            'vector_weight': 0.2,
            'bm25_weight': 0.2,
            'hirag_weight': 0.6,
            'enable_hirag': True,
            'enable_rerank': True,
            'enable_adaptive_filter': True,
            'hirag_mode': 'hierarchical',
            'fusion_strategy': 'weighted'
        }
    },
    
    'policy_analysis': {
        'name': '政策分析模式',
        'description': '📜 专为政策文档优化，提供全局视角',
        'icon': '📜',
        'config': {
            'mode': 'hirag_hybrid',
            'top_k': 8,
            'recall_k': 30,
            'vector_weight': 0.2,
            'bm25_weight': 0.2,
            'hirag_weight': 0.6,
            'enable_hirag': True,
            'enable_rerank': True,
            'hirag_mode': 'hierarchical',
            'hirag_weights': {
                'local': 0.3,
                'global': 0.4,
                'bridge': 0.3
            },
            'fusion_strategy': 'weighted'
        }
    },
    
    'deep_search': {
        'name': '深度搜索',
        'description': '🔍 最全面的检索，适合研究型查询',
        'icon': '🔍',
        'config': {
            'mode': 'hirag_hybrid',
            'top_k': 10,
            'recall_k': 50,
            'vector_weight': 0.25,
            'bm25_weight': 0.25,
            'hirag_weight': 0.5,
            'enable_hirag': True,
            'enable_rerank': True,
            'enable_hyde': True,
            'hirag_mode': 'hierarchical',
            'fusion_strategy': 'rrf'
        }
    }
}


def get_preset_config(preset_name: str) -> Dict[str, Any]:
    """
    获取预设配置
    
    Args:
        preset_name: 预设名称
        
    Returns:
        配置字典
    """
    if preset_name not in STRATEGY_PRESETS:
        raise ValueError(f"未知的预设: {preset_name}")
    
    return STRATEGY_PRESETS[preset_name]['config'].copy()


def get_preset_names() -> list:
    """获取所有预设名称"""
    return list(STRATEGY_PRESETS.keys())


def get_preset_display_names() -> list:
    """获取所有预设的显示名称"""
    return [preset['name'] for preset in STRATEGY_PRESETS.values()]


def get_preset_info(preset_name: str) -> Dict[str, str]:
    """
    获取预设信息
    
    Args:
        preset_name: 预设名称
        
    Returns:
        包含 name, description, icon 的字典
    """
    if preset_name not in STRATEGY_PRESETS:
        raise ValueError(f"未知的预设: {preset_name}")
    
    preset = STRATEGY_PRESETS[preset_name]
    return {
        'name': preset['name'],
        'description': preset['description'],
        'icon': preset['icon']
    }
