"""
检索分发器模块
统一管理所有检索模式的调用，避免重复代码
"""

from typing import List, Tuple, Optional, Dict, Any
from openai import OpenAI

from rag_engine import (
    search_top_k,
    search_bm25,
    search_with_rerank
)
from src.rag.core import hybrid_search, hybrid_search_with_rerank


class RetrievalDispatcher:
    """
    统一的检索分发器
    负责根据检索模式调用相应的检索函数
    """
    
    def __init__(self):
        """初始化分发器"""
        self.supported_modes = [
            "向量检索",
            "BM25 检索",
            "混合检索",
            "Rerank 精排",
            "混合 + Rerank"
        ]
    
    def dispatch(
        self,
        query: str,
        mode: str,
        top_k: int = 3,
        vector_weight: float = 0.5,
        recall_k: int = 20,
        use_adaptive_filter: bool = True
    ) -> List[Tuple[str, float]]:
        """
        分发检索请求到对应的检索函数
        
        Args:
            query: 查询文本
            mode: 检索模式
            top_k: 返回的结果数量
            vector_weight: 向量检索权重（仅混合检索使用）
            recall_k: 召回数量（仅 Rerank 使用）
            use_adaptive_filter: 是否使用自适应过滤
        
        Returns:
            [(chunk, score), ...] 列表
        """
        if mode not in self.supported_modes:
            raise ValueError(f"不支持的检索模式: {mode}，支持的模式: {self.supported_modes}")
        
        if mode == "向量检索":
            return search_top_k(query, k=top_k)
        
        elif mode == "BM25 检索":
            return search_bm25(query, k=top_k)
        
        elif mode == "混合检索":
            return hybrid_search(
                query,
                k=top_k,
                vector_weight=vector_weight,
                use_adaptive_filter=use_adaptive_filter
            )
        
        elif mode == "Rerank 精排":
            return search_with_rerank(query, k=top_k, recall_k=recall_k)
        
        elif mode == "混合 + Rerank":
            return hybrid_search_with_rerank(
                query,
                k=top_k,
                vector_weight=vector_weight,
                recall_k=recall_k,
                use_adaptive_filter=use_adaptive_filter
            )
        
        else:
            raise ValueError(f"未实现的检索模式: {mode}")
    
    def get_supported_modes(self) -> List[str]:
        """获取支持的检索模式列表"""
        return self.supported_modes.copy()


class QueryOptimizedDispatcher(RetrievalDispatcher):
    """
    支持查询优化的检索分发器
    在基础检索分发器的基础上，增加查询优化功能
    """
    
    def __init__(self, chat_client: Optional[OpenAI] = None, chat_model: Optional[str] = None):
        """
        初始化查询优化分发器
        
        Args:
            chat_client: OpenAI 客户端（用于查询优化）
            chat_model: 聊天模型名称
        """
        super().__init__()
        self.chat_client = chat_client
        self.chat_model = chat_model
    
    def dispatch_with_optimization(
        self,
        query: str,
        mode: str,
        optimization: Optional[str] = None,
        top_k: int = 3,
        **kwargs
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        """
        使用查询优化进行检索分发
        
        Args:
            query: 原始查询
            mode: 检索模式
            optimization: 查询优化方式 (None, "hyde", "multi_query", "multi_step", "multi_variant")
            top_k: 返回结果数量
            **kwargs: 其他参数
        
        Returns:
            (检索结果, 元数据) 元组
        """
        metadata = {
            "original_query": query,
            "optimization": optimization,
            "mode": mode
        }
        
        if optimization is None:
            # 无优化，直接检索
            results = self.dispatch(query, mode, top_k, **kwargs)
            metadata["optimized_query"] = query
            return results, metadata
        
        elif optimization == "hyde":
            # HyDE: 生成假设文档
            results, hyde_metadata = self._dispatch_with_hyde(query, mode, top_k, **kwargs)
            metadata.update(hyde_metadata)
            return results, metadata
        
        elif optimization == "multi_query":
            # 多查询变体
            results, mq_metadata = self._dispatch_with_multi_query(query, mode, top_k, **kwargs)
            metadata.update(mq_metadata)
            return results, metadata
        
        elif optimization == "multi_step":
            # 多步骤查询
            results, ms_metadata = self._dispatch_with_multi_step(query, mode, top_k, **kwargs)
            metadata.update(ms_metadata)
            return results, metadata
        
        elif optimization == "multi_variant":
            # 多变体召回
            results, mv_metadata = self._dispatch_with_multi_variant(query, mode, top_k, **kwargs)
            metadata.update(mv_metadata)
            return results, metadata
        
        else:
            raise ValueError(f"不支持的查询优化方式: {optimization}")
    
    def _dispatch_with_hyde(
        self,
        query: str,
        mode: str,
        top_k: int,
        **kwargs
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        """使用 HyDE 进行检索"""
        from src.rag.query import HyDERetriever
        
        if not self.chat_client or not self.chat_model:
            raise ValueError("HyDE 需要提供 chat_client 和 chat_model")
        
        hyde = HyDERetriever(self.chat_client, self.chat_model)
        hypothetical_doc = hyde.generate_hypothetical_document(query)
        
        # 第一次检索（使用假设文档）
        initial_results = self.dispatch(hypothetical_doc, mode, top_k * 2, **kwargs)
        
        # 第二次检索（使用原始查询）
        final_results = self.dispatch(query, mode, top_k, **kwargs)
        
        metadata = {
            "hypothetical_doc": hypothetical_doc,
            "optimized_query": query,
            "initial_results_count": len(initial_results)
        }
        
        return final_results, metadata
    
    def _dispatch_with_multi_query(
        self,
        query: str,
        mode: str,
        top_k: int,
        **kwargs
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        """使用多查询变体进行检索"""
        from src.rag.query import QueryExpander, multi_query_retrieval
        
        if not self.chat_client or not self.chat_model:
            raise ValueError("多查询变体需要提供 chat_client 和 chat_model")
        
        expander = QueryExpander(self.chat_client, self.chat_model)
        expanded_queries = expander.expand(query, num_variants=3)
        
        # 使用 multi_query_retrieval 进行检索
        def retrieval_fn(q: str, k: int) -> List[Tuple[str, float]]:
            return self.dispatch(q, mode, k, **kwargs)
        
        results = multi_query_retrieval(
            query,
            expanded_queries,
            retrieval_fn,
            top_k=top_k
        )
        
        metadata = {
            "expanded_queries": expanded_queries,
            "optimized_query": query,
            "num_variants": len(expanded_queries)
        }
        
        return results, metadata
    
    def _dispatch_with_multi_step(
        self,
        query: str,
        mode: str,
        top_k: int,
        **kwargs
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        """使用多步骤查询进行检索"""
        from src.rag.query import MultiStepQueryEngine
        
        if not self.chat_client or not self.chat_model:
            raise ValueError("多步骤查询需要提供 chat_client 和 chat_model")
        
        engine = MultiStepQueryEngine(self.chat_client, self.chat_model)
        
        # 分解查询
        sub_queries = engine.decompose_query(query)
        
        # 逐步检索
        all_results = []
        for sub_q in sub_queries:
            sub_results = self.dispatch(sub_q, mode, top_k, **kwargs)
            all_results.extend(sub_results)
        
        # 去重并排序
        unique_results = {}
        for chunk, score in all_results:
            if chunk not in unique_results or score > unique_results[chunk]:
                unique_results[chunk] = score
        
        results = sorted(unique_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        metadata = {
            "sub_queries": sub_queries,
            "optimized_query": query,
            "num_steps": len(sub_queries)
        }
        
        return results, metadata
    
    def _dispatch_with_multi_variant(
        self,
        query: str,
        mode: str,
        top_k: int,
        **kwargs
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        """使用多变体召回进行检索"""
        from src.rag.query import MultiVariantRecaller
        
        if not self.chat_client or not self.chat_model:
            raise ValueError("多变体召回需要提供 chat_client 和 chat_model")
        
        recaller = MultiVariantRecaller(self.chat_client, self.chat_model)
        
        # 生成查询变体
        variants = recaller.generate_variants(query, num_variants=3)
        
        # 对每个变体进行检索
        all_results = []
        for variant in variants:
            variant_results = self.dispatch(variant, mode, top_k * 2, **kwargs)
            all_results.extend(variant_results)
        
        # 融合结果
        results = recaller.fuse_results(all_results, top_k=top_k)
        
        metadata = {
            "variants": variants,
            "optimized_query": query,
            "num_variants": len(variants)
        }
        
        return results, metadata
