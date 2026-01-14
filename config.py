"""
RAG 系统配置管理模块
集中管理所有配置参数，避免硬编码和魔法数字
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalConfig:
    """检索配置"""
    # 基础检索参数
    default_top_k: int = 3
    default_recall_k: int = 20
    max_top_k: int = 50
    
    # 混合检索参数
    default_vector_weight: float = 0.5
    min_vector_weight: float = 0.0
    max_vector_weight: float = 1.0
    
    # Embedding 参数
    embedding_batch_size: int = 50
    embedding_model: str = "text-embedding-3-small"
    
    # BM25 参数
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass
class FilterConfig:
    """过滤器配置"""
    # 自适应过滤参数
    min_confidence: float = 0.3
    max_results: int = 10
    min_results: int = 1
    gap_threshold: float = 0.15
    percentile_threshold: float = 0.6
    
    # 相关度阈值
    relevance_threshold: float = 0.5
    min_score: float = 0.0
    max_score: float = 1.0


@dataclass
class ChunkConfig:
    """文本分块配置"""
    # 默认分块参数
    default_chunk_size: int = 500
    default_overlap: int = 50
    
    # 各策略的默认参数
    heading_chunk_size: int = 800
    sliding_window_chunk_size: int = 600
    sliding_window_overlap: int = 150
    sentence_min_len: int = 100
    sentence_max_len: int = 500
    semantic_max_chunk: int = 1000
    
    # 分块限制
    min_chunk_size: int = 50
    max_chunk_size: int = 2000


@dataclass
class QueryOptimizationConfig:
    """查询优化配置"""
    # HyDE 参数
    hyde_temperature: float = 0.7
    hyde_max_tokens: int = 500
    
    # 查询扩展参数
    query_expansion_variants: int = 3
    query_expansion_temperature: float = 0.8
    
    # 多步骤查询参数
    multi_step_max_steps: int = 5
    multi_step_temperature: float = 0.3
    
    # 多变体召回参数
    multi_variant_num: int = 3
    multi_variant_temperature: float = 0.7


@dataclass
class GeneratorConfig:
    """生成器配置"""
    # LLM 生成参数
    default_temperature: float = 0.7
    default_max_tokens: int = 2000
    
    # 上下文参数
    max_context_length: int = 8000
    context_window_buffer: int = 500
    
    # 对话历史参数
    max_history_turns: int = 10
    max_history_tokens: int = 4000


@dataclass
class KnowledgeGraphConfig:
    """知识图谱配置"""
    # 实体抽取参数
    max_entities: int = 20
    entity_confidence_threshold: float = 0.6
    
    # 关系抽取参数
    max_relations: int = 15
    relation_confidence_threshold: float = 0.5
    
    # 缓存参数
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600


@dataclass
class UIConfig:
    """UI 配置"""
    # Streamlit 参数
    page_title: str = "Web RAG Demo"
    page_icon: str = "⚡"
    layout: str = "wide"
    
    # 默认值
    default_retrieval_mode: str = "混合检索"
    default_vector_weight: float = 0.5
    default_recall_k: int = 20
    
    # 限制
    max_file_size_mb: int = 50
    supported_file_types: tuple = ("txt", "pdf", "md", "markdown", "jpg", "jpeg", "png")
    
    # 显示参数
    max_display_chunk_length: int = 300
    conversation_export_format: str = "markdown"


class RAGConfig:
    """
    RAG 系统全局配置
    提供所有子配置的访问接口
    """
    
    def __init__(self):
        self.retrieval = RetrievalConfig()
        self.filter = FilterConfig()
        self.chunk = ChunkConfig()
        self.query_optimization = QueryOptimizationConfig()
        self.generator = GeneratorConfig()
        self.knowledge_graph = KnowledgeGraphConfig()
        self.ui = UIConfig()
    
    @classmethod
    def get_default(cls) -> 'RAGConfig':
        """获取默认配置实例"""
        return cls()
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "retrieval": self.retrieval.__dict__,
            "filter": self.filter.__dict__,
            "chunk": self.chunk.__dict__,
            "query_optimization": self.query_optimization.__dict__,
            "generator": self.generator.__dict__,
            "knowledge_graph": self.knowledge_graph.__dict__,
            "ui": self.ui.__dict__
        }
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"RAGConfig(retrieval={self.retrieval}, filter={self.filter}, ...)"


# 全局默认配置实例
DEFAULT_CONFIG = RAGConfig.get_default()


# 便捷访问函数
def get_retrieval_config() -> RetrievalConfig:
    """获取检索配置"""
    return DEFAULT_CONFIG.retrieval


def get_filter_config() -> FilterConfig:
    """获取过滤器配置"""
    return DEFAULT_CONFIG.filter


def get_chunk_config() -> ChunkConfig:
    """获取分块配置"""
    return DEFAULT_CONFIG.chunk


def get_query_optimization_config() -> QueryOptimizationConfig:
    """获取查询优化配置"""
    return DEFAULT_CONFIG.query_optimization


def get_generator_config() -> GeneratorConfig:
    """获取生成器配置"""
    return DEFAULT_CONFIG.generator


def get_knowledge_graph_config() -> KnowledgeGraphConfig:
    """获取知识图谱配置"""
    return DEFAULT_CONFIG.knowledge_graph


def get_ui_config() -> UIConfig:
    """获取 UI 配置"""
    return DEFAULT_CONFIG.ui
