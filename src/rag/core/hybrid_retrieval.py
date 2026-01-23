"""
混合检索编排模块
结合向量检索和 BM25 检索，支持 Rerank 精排和自适应过滤
"""

import os
from typing import List, Tuple
import logging
from src.utils.logger import get_logger

logger = get_logger(__name__)

from src.rag.retriever import VectorRetriever, BM25Retriever
from src.rag.retriever.base import Document
from src.rag.ranker import SimilarityRanker, RerankRanker
from src.rag.filter import AdaptiveFilter, FilterConfig


# 数据库路径
VECTOR_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "vector_db.json")
BM25_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "bm25_index.pkl")


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    将文本列表转换为向量
    注意：这是一个临时导入，实际应该从 rag_engine 导入
    """
    from openai import OpenAI
    
    client = OpenAI(
        api_key=os.getenv("EMBED_API_KEY"),
        base_url=os.getenv("EMBED_BASE_URL")
    )
    model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    
    all_embeddings = []
    batch_size = 50
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            input=batch,
            model=model
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings


def hybrid_search(query: str, k: int = 3, vector_weight: float = 0.5, use_adaptive_filter: bool = True) -> List[Tuple[str, float]]:
    """混合检索：结合向量检索和 BM25 检索
    
    Args:
        query: 查询文本
        k: 期望返回的结果数量（作为 max_results 参考）
        vector_weight: 向量检索的权重（0-1），BM25 权重为 1-vector_weight
        use_adaptive_filter: 是否使用自适应过滤（动态阈值），False 则使用固定 top-k
    
    Returns:
        [(chunk, combined_score), ...] 列表
    """
    # 1. 向量检索
    query_embedding = embed_texts([query])[0]
    vector_retriever = VectorRetriever(VECTOR_DB_PATH)
    vector_docs = vector_retriever.retrieve(query_embedding, top_k=k * 2)
    
    # 2. BM25 检索
    bm25_retriever = BM25Retriever(BM25_DB_PATH)
    bm25_docs = bm25_retriever.retrieve_by_text(query, top_k=k * 2)
    
    # 3. 计算向量相似度分数
    ranker = SimilarityRanker()
    vector_ranked = ranker.rank(query, query_embedding, vector_docs, top_k=None)
    
    # 4. 归一化分数并合并
    vector_scores = {}
    bm25_scores = {}
    
    # 向量分数归一化
    if vector_ranked:
        max_vector_score = max(score for _, score in vector_ranked)
        min_vector_score = min(score for _, score in vector_ranked)
        score_range = max_vector_score - min_vector_score if max_vector_score > min_vector_score else 1.0
        
        for doc, score in vector_ranked:
            normalized_score = (score - min_vector_score) / score_range if score_range > 0 else 0.0
            vector_scores[doc.text] = normalized_score
    
    # BM25 分数归一化
    if bm25_docs:
        bm25_score_list = [doc.metadata.get('bm25_score', 0.0) for doc in bm25_docs]
        max_bm25_score = max(bm25_score_list) if bm25_score_list else 1.0
        min_bm25_score = min(bm25_score_list) if bm25_score_list else 0.0
        score_range = max_bm25_score - min_bm25_score if max_bm25_score > min_bm25_score else 1.0
        
        for doc in bm25_docs:
            raw_score = doc.metadata.get('bm25_score', 0.0)
            normalized_score = (raw_score - min_bm25_score) / score_range if score_range > 0 else 0.0
            bm25_scores[doc.text] = normalized_score
    
    # 5. 合并分数
    all_texts = set(vector_scores.keys()) | set(bm25_scores.keys())
    combined_results = []
    
    for text in all_texts:
        v_score = vector_scores.get(text, 0.0)
        b_score = bm25_scores.get(text, 0.0)
        combined_score = vector_weight * v_score + (1 - vector_weight) * b_score
        combined_results.append((text, combined_score))
    
    # 6. 排序
    combined_results.sort(key=lambda x: x[1], reverse=True)
    
    # 7. 自适应过滤或固定 top-k
    if use_adaptive_filter:
        filter_config = FilterConfig(
            min_confidence=0.3,
            max_results=k,
            min_results=1,
            gap_threshold=0.15,
            percentile_threshold=0.6
        )
        adaptive_filter_obj = AdaptiveFilter(filter_config)
        filtered_results, metadata = adaptive_filter_obj.filter_results(
            combined_results, normalize=False
        )
        logger.info(f"混合检索自适应过滤: {metadata['kept']}/{metadata['total']} 个结果, "
                    f"阈值={metadata['threshold']:.3f} ({metadata['reason']}), "
                    f"平均分={metadata['avg_score']:.3f}")
        return filtered_results
    else:
        return combined_results[:k]


def hybrid_search_with_rerank(
    query: str, 
    k: int = 3, 
    vector_weight: float = 0.5,
    recall_k: int = 20,
    use_adaptive_filter: bool = True
) -> List[Tuple[str, float]]:
    """混合检索 + Rerank 精排（三阶段检索）
    
    工作流程：
    1. 第一阶段（召回）：向量检索 + BM25 检索，召回候选
    2. 第二阶段（融合）：加权融合两种检索结果
    3. 第三阶段（精排）：使用 Rerank 模型对融合结果进行精确排序
    4. 第四阶段（过滤）：自适应阈值过滤低质量结果
    
    Args:
        query: 查询文本
        k: 期望返回的结果数量（作为 max_results 参考）
        vector_weight: 向量检索的权重（0-1）
        recall_k: 第一阶段召回的候选数量
        use_adaptive_filter: 是否使用自适应过滤（动态阈值）
    
    Returns:
        [(chunk, rerank_score), ...] 列表
    """
    # 1. 向量化查询
    query_embedding = embed_texts([query])[0]
    
    # 2. 第一阶段：混合召回
    vector_retriever = VectorRetriever(VECTOR_DB_PATH)
    vector_docs = vector_retriever.retrieve(query_embedding, top_k=recall_k)
    
    bm25_retriever = BM25Retriever(BM25_DB_PATH)
    bm25_docs = bm25_retriever.retrieve_by_text(query, top_k=recall_k)
    
    # 3. 第二阶段：融合分数
    ranker = SimilarityRanker()
    vector_ranked = ranker.rank(query, query_embedding, vector_docs, top_k=None)
    
    vector_scores = {}
    bm25_scores = {}
    
    # 向量分数归一化
    if vector_ranked:
        max_vector_score = max(score for _, score in vector_ranked)
        min_vector_score = min(score for _, score in vector_ranked)
        score_range = max_vector_score - min_vector_score if max_vector_score > min_vector_score else 1.0
        
        for doc, score in vector_ranked:
            normalized_score = (score - min_vector_score) / score_range if score_range > 0 else 0.0
            vector_scores[doc.text] = normalized_score
    
    # BM25 分数归一化
    if bm25_docs:
        bm25_score_list = [doc.metadata.get('bm25_score', 0.0) for doc in bm25_docs]
        max_bm25_score = max(bm25_score_list) if bm25_score_list else 1.0
        min_bm25_score = min(bm25_score_list) if bm25_score_list else 0.0
        score_range = max_bm25_score - min_bm25_score if max_bm25_score > min_bm25_score else 1.0
        
        for doc in bm25_docs:
            raw_score = doc.metadata.get('bm25_score', 0.0)
            normalized_score = (raw_score - min_bm25_score) / score_range if score_range > 0 else 0.0
            bm25_scores[doc.text] = normalized_score
    
    # 合并所有候选文档
    all_texts = set(vector_scores.keys()) | set(bm25_scores.keys())
    
    logger.info(f"混合+Rerank 向量召回: {len(vector_scores)} 个, BM25召回: {len(bm25_scores)} 个, 合并后: {len(all_texts)} 个")
    
    # 构建融合后的文档列表
    fused_documents = []
    for text in all_texts:
        v_score = vector_scores.get(text, 0.0)
        b_score = bm25_scores.get(text, 0.0)
        combined_score = vector_weight * v_score + (1 - vector_weight) * b_score
        
        doc = None
        for d in vector_docs:
            if d.text == text:
                doc = d
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['hybrid_score'] = combined_score
                break
        if doc is None:
            for d in bm25_docs:
                if d.text == text:
                    doc = Document(
                        id=d.id,
                        text=d.text,
                        embedding=query_embedding,
                        metadata={'hybrid_score': combined_score}
                    )
                    break
        
        if doc:
            fused_documents.append(doc)
    
    logger.info(f"混合+Rerank 融合后文档数: {len(fused_documents)}")
    
    # 按融合分数排序
    fused_documents.sort(key=lambda d: d.metadata.get('hybrid_score', 0.0), reverse=True)
    fused_documents = fused_documents[:recall_k]
    
    if not fused_documents:
        return []
    
    # 4. 第三阶段：Rerank 精排
    try:
        rerank_ranker = RerankRanker()
        ranked_docs = rerank_ranker.rank(query, query_embedding, fused_documents, top_k=None)
        
        rerank_results = [(doc.text, score) for doc, score in ranked_docs]
        
        # 5. 第四阶段：自适应过滤
        if use_adaptive_filter:
            filter_config = FilterConfig(
                min_confidence=0.3,
                max_results=k,
                min_results=1,
                gap_threshold=0.15,
                percentile_threshold=0.6
            )
            adaptive_filter_obj = AdaptiveFilter(filter_config)
            filtered_results, metadata = adaptive_filter_obj.filter_results(
                rerank_results, normalize=True
            )
            logger.info(f"混合+Rerank 自适应过滤: {metadata['kept']}/{metadata['total']} 个结果, "
                        f"阈值={metadata['threshold']:.3f} ({metadata['reason']}), "
                        f"平均分={metadata['avg_score']:.3f}")
            return filtered_results
        else:
            return rerank_results[:k]
    
    except Exception as e:
        logger.error(f"Rerank 精排失败，返回混合检索结果: {e}", exc_info=True)
        hybrid_results = [(doc.text, doc.metadata.get('hybrid_score', 0.0)) 
                          for doc in fused_documents]
        
        if use_adaptive_filter:
            filter_config = FilterConfig(
                min_confidence=0.3,
                max_results=k,
                min_results=1
            )
            adaptive_filter_obj = AdaptiveFilter(filter_config)
            filtered_results, metadata = adaptive_filter_obj.filter_results(
                hybrid_results, normalize=False
            )
            logger.info(f"混合+Rerank降级 自适应过滤: {metadata['kept']}/{metadata['total']} 个结果")
            return filtered_results
        else:
            return hybrid_results[:k]


def hirag_search(
    query: str,
    k: int = 3,
    mode: str = "hierarchical",
    use_adaptive_filter: bool = True
) -> List[Tuple[str, float]]:
    """纯 HiRAG 检索
    
    Args:
        query: 查询文本
        k: 期望返回的结果数量
        mode: 检索模式
            - "hierarchical": 融合三层（local + global + bridge）
            - "local": 仅局部知识
            - "global": 仅全局知识
            - "bridge": 仅桥梁知识
        use_adaptive_filter: 是否使用自适应过滤
    
    Returns:
        [(chunk, score), ...] 列表
    """
    from src.rag.retriever import HiRAGRetriever, VectorRetriever
    from config import get_hirag_config
    
    # 初始化 HiRAG 检索器
    config = get_hirag_config()
    vector_retriever = VectorRetriever(VECTOR_DB_PATH)
    hirag_retriever = HiRAGRetriever(config, vector_retriever)
    
    # 执行检索
    if mode == "hierarchical":
        results = hirag_retriever.retrieve(query, top_k=k * 2)
    else:
        results = hirag_retriever.retrieve_by_mode(query, mode, top_k=k * 2)
    
    # 转换为 (text, score) 格式
    hirag_results = [(r['text'], r.get('final_score', r.get('score', 0.0))) for r in results]
    
    logger.info(f"HiRAG 检索 ({mode} 模式): {len(hirag_results)} 个结果")
    
    # 自适应过滤或固定 top-k
    if use_adaptive_filter:
        filter_config = FilterConfig(
            min_confidence=0.3,
            max_results=k,
            min_results=1,
            gap_threshold=0.15,
            percentile_threshold=0.6
        )
        adaptive_filter_obj = AdaptiveFilter(filter_config)
        filtered_results, metadata = adaptive_filter_obj.filter_results(
            hirag_results, normalize=False
        )
        logger.info(f"HiRAG 自适应过滤: {metadata['kept']}/{metadata['total']} 个结果, "
                    f"阈值={metadata['threshold']:.3f} ({metadata['reason']})")
        return filtered_results
    else:
        return hirag_results[:k]


def hirag_hybrid_search(
    query: str,
    k: int = 3,
    vector_weight: float = 0.2,
    bm25_weight: float = 0.2,
    hirag_weight: float = 0.6,
    hirag_mode: str = "hierarchical",
    fusion_strategy: str = "weighted",
    use_adaptive_filter: bool = True
) -> List[Tuple[str, float]]:
    """HiRAG + 传统混合检索
    
    结合三种检索方式：
    1. 向量检索（Vector）
    2. BM25 检索
    3. HiRAG 层次化检索（Local + Global + Bridge）
    
    Args:
        query: 查询文本
        k: 期望返回的结果数量
        vector_weight: 向量检索权重
        bm25_weight: BM25 检索权重
        hirag_weight: HiRAG 检索权重
        hirag_mode: HiRAG 检索模式（hierarchical/local/global/bridge）
        fusion_strategy: 融合策略
            - "weighted": 加权融合（根据权重）
            - "rrf": Reciprocal Rank Fusion（倒数排名融合）
            - "max": 最大分数融合
        use_adaptive_filter: 是否使用自适应过滤
    
    Returns:
        [(chunk, score), ...] 列表
    """
    from src.rag.retriever import HiRAGRetriever, VectorRetriever, BM25Retriever
    from config import get_hirag_config
    
    # 归一化权重
    total_weight = vector_weight + bm25_weight + hirag_weight
    if total_weight > 0:
        vector_weight /= total_weight
        bm25_weight /= total_weight
        hirag_weight /= total_weight
    
    logger.info(f"HiRAG 混合检索 - 权重: Vector={vector_weight:.2f}, BM25={bm25_weight:.2f}, HiRAG={hirag_weight:.2f}")
    
    # 1. 向量检索
    query_embedding = embed_texts([query])[0]
    vector_retriever = VectorRetriever(VECTOR_DB_PATH)
    vector_docs = vector_retriever.retrieve(query_embedding, top_k=k * 2)
    
    # 2. BM25 检索
    bm25_retriever = BM25Retriever(BM25_DB_PATH)
    bm25_docs = bm25_retriever.retrieve_by_text(query, top_k=k * 2)
    
    # 3. HiRAG 检索
    config = get_hirag_config()
    hirag_retriever = HiRAGRetriever(config, vector_retriever)
    
    if hirag_mode == "hierarchical":
        hirag_results = hirag_retriever.retrieve(query, top_k=k * 2)
    else:
        hirag_results = hirag_retriever.retrieve_by_mode(query, hirag_mode, top_k=k * 2)
    
    # 4. 根据融合策略合并结果
    if fusion_strategy == "rrf":
        # RRF 融合：基于排名
        final_results = _rrf_fusion_three_sources(
            vector_docs, bm25_docs, hirag_results,
            vector_weight, bm25_weight, hirag_weight,
            query_embedding
        )
    elif fusion_strategy == "max":
        # Max 融合：取最大分数
        final_results = _max_fusion_three_sources(
            vector_docs, bm25_docs, hirag_results,
            query_embedding
        )
    else:
        # Weighted 融合：加权平均
        final_results = _weighted_fusion_three_sources(
            vector_docs, bm25_docs, hirag_results,
            vector_weight, bm25_weight, hirag_weight,
            query_embedding
        )
    
    logger.info(f"HiRAG 混合检索融合完成: {len(final_results)} 个结果")
    
    # 5. 自适应过滤或固定 top-k
    if use_adaptive_filter:
        filter_config = FilterConfig(
            min_confidence=0.3,
            max_results=k,
            min_results=1,
            gap_threshold=0.15,
            percentile_threshold=0.6
        )
        adaptive_filter_obj = AdaptiveFilter(filter_config)
        filtered_results, metadata = adaptive_filter_obj.filter_results(
            final_results, normalize=False
        )
        logger.info(f"HiRAG 混合检索自适应过滤: {metadata['kept']}/{metadata['total']} 个结果, "
                    f"阈值={metadata['threshold']:.3f} ({metadata['reason']})")
        return filtered_results
    else:
        return final_results[:k]


def _weighted_fusion_three_sources(
    vector_docs, bm25_docs, hirag_results,
    vector_weight, bm25_weight, hirag_weight,
    query_embedding
):
    """加权融合三个检索源"""
    from src.rag.ranker import SimilarityRanker
    
    # 计算向量相似度分数
    ranker = SimilarityRanker()
    vector_ranked = ranker.rank("", query_embedding, vector_docs, top_k=None)
    
    # 收集并归一化各源分数
    vector_scores = {}
    bm25_scores = {}
    hirag_scores = {}
    
    # 向量分数归一化
    if vector_ranked:
        scores = [score for _, score in vector_ranked]
        max_s, min_s = max(scores), min(scores)
        score_range = max_s - min_s if max_s > min_s else 1.0
        for doc, score in vector_ranked:
            vector_scores[doc.text] = (score - min_s) / score_range if score_range > 0 else 0.0
    
    # BM25 分数归一化
    if bm25_docs:
        scores = [doc.metadata.get('bm25_score', 0.0) for doc in bm25_docs]
        max_s, min_s = max(scores), min(scores)
        score_range = max_s - min_s if max_s > min_s else 1.0
        for doc in bm25_docs:
            raw_score = doc.metadata.get('bm25_score', 0.0)
            bm25_scores[doc.text] = (raw_score - min_s) / score_range if score_range > 0 else 0.0
    
    # HiRAG 分数归一化
    if hirag_results:
        scores = [r.get('final_score', r.get('score', 0.0)) for r in hirag_results]
        max_s, min_s = max(scores), min(scores)
        score_range = max_s - min_s if max_s > min_s else 1.0
        for r in hirag_results:
            raw_score = r.get('final_score', r.get('score', 0.0))
            hirag_scores[r['text']] = (raw_score - min_s) / score_range if score_range > 0 else 0.0
    
    # 合并所有文本
    all_texts = set(vector_scores.keys()) | set(bm25_scores.keys()) | set(hirag_scores.keys())
    
    # 加权融合
    combined_results = []
    for text in all_texts:
        v_score = vector_scores.get(text, 0.0)
        b_score = bm25_scores.get(text, 0.0)
        h_score = hirag_scores.get(text, 0.0)
        combined_score = vector_weight * v_score + bm25_weight * b_score + hirag_weight * h_score
        combined_results.append((text, combined_score))
    
    # 排序
    combined_results.sort(key=lambda x: x[1], reverse=True)
    return combined_results


def _rrf_fusion_three_sources(
    vector_docs, bm25_docs, hirag_results,
    vector_weight, bm25_weight, hirag_weight,
    query_embedding, k=60
):
    """RRF 融合三个检索源"""
    from src.rag.ranker import SimilarityRanker
    
    # 计算向量相似度并排序
    ranker = SimilarityRanker()
    vector_ranked = ranker.rank("", query_embedding, vector_docs, top_k=None)
    
    # 收集每个文档的 RRF 分数
    doc_scores = {}
    
    # Vector RRF
    for rank, (doc, _) in enumerate(vector_ranked, start=1):
        text = doc.text
        rrf_score = vector_weight / (k + rank)
        if text not in doc_scores:
            doc_scores[text] = 0.0
        doc_scores[text] += rrf_score
    
    # BM25 RRF
    for rank, doc in enumerate(bm25_docs, start=1):
        text = doc.text
        rrf_score = bm25_weight / (k + rank)
        if text not in doc_scores:
            doc_scores[text] = 0.0
        doc_scores[text] += rrf_score
    
    # HiRAG RRF
    for rank, result in enumerate(hirag_results, start=1):
        text = result['text']
        rrf_score = hirag_weight / (k + rank)
        if text not in doc_scores:
            doc_scores[text] = 0.0
        doc_scores[text] += rrf_score
    
    # 转换为列表并排序
    combined_results = [(text, score) for text, score in doc_scores.items()]
    combined_results.sort(key=lambda x: x[1], reverse=True)
    return combined_results


def _max_fusion_three_sources(
    vector_docs, bm25_docs, hirag_results,
    query_embedding
):
    """最大分数融合三个检索源"""
    from src.rag.ranker import SimilarityRanker
    
    # 计算向量相似度
    ranker = SimilarityRanker()
    vector_ranked = ranker.rank("", query_embedding, vector_docs, top_k=None)
    
    # 收集每个文档的最大分数
    doc_scores = {}
    
    # Vector 分数
    for doc, score in vector_ranked:
        text = doc.text
        if text not in doc_scores or score > doc_scores[text]:
            doc_scores[text] = score
    
    # BM25 分数
    for doc in bm25_docs:
        text = doc.text
        score = doc.metadata.get('bm25_score', 0.0)
        if text not in doc_scores or score > doc_scores[text]:
            doc_scores[text] = score
    
    # HiRAG 分数
    for result in hirag_results:
        text = result['text']
        score = result.get('final_score', result.get('score', 0.0))
        if text not in doc_scores or score > doc_scores[text]:
            doc_scores[text] = score
    
    # 转换为列表并排序
    combined_results = [(text, score) for text, score in doc_scores.items()]
    combined_results.sort(key=lambda x: x[1], reverse=True)
    return combined_results
