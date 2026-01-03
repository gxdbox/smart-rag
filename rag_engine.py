"""
RAG 核心引擎模块
负责：环境加载、文本切分、向量化、RAG 流程编排

重构说明：
- 向量存储/检索逻辑已迁移到 rag.retriever.VectorRetriever
- 相似度计算/排序逻辑已迁移到 rag.ranker.SimilarityRanker
- LLM 答案生成逻辑已迁移到 rag.generator.LLMGenerator
- 本模块保留：环境加载、向量化、切分、编排逻辑
"""

import os
import re
import json
import shutil
import numpy as np
from typing import List, Tuple, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

from rag.retriever import VectorRetriever, BM25Retriever
from rag.ranker import SimilarityRanker, RerankRanker
from rag.generator import LLMGenerator

# 向量库文件路径
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "vector_db.json")
BM25_DB_PATH = os.path.join(os.path.dirname(__file__), "bm25_index.pkl")


def load_env():
    """加载 .env 环境变量"""
    load_dotenv()
    required_vars = ["EMBED_BASE_URL", "EMBED_API_KEY", "EMBED_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY", "CHAT_MODEL"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"缺少必要的环境变量: {', '.join(missing)}")


def get_embed_client() -> OpenAI:
    """获取 Embedding API 客户端"""
    client = OpenAI(
        api_key=os.getenv("EMBED_API_KEY"),
        base_url=os.getenv("EMBED_BASE_URL"),
    )
    return client


def get_chat_client() -> OpenAI:
    """获取 Chat API 客户端"""
    client = OpenAI(
        api_key=os.getenv("CHAT_API_KEY"),
        base_url=os.getenv("CHAT_BASE_URL"),
    )
    return client


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    将文本切分成 chunks
    
    Args:
        text: 原始文本
        chunk_size: 每个 chunk 的最大字符数
        overlap: chunk 之间的重叠字符数
    
    Returns:
        切分后的 chunk 列表
    """
    if not text or not text.strip():
        return []
    
    # 按段落分割
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # 如果当前段落加上已有内容超过 chunk_size
        if len(current_chunk) + len(para) + 1 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # 保留重叠部分
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + " " + para
                else:
                    current_chunk = para
            else:
                # 单个段落就超过了 chunk_size，强制切分
                while len(para) > chunk_size:
                    chunks.append(para[:chunk_size])
                    para = para[chunk_size - overlap:] if overlap > 0 else para[chunk_size:]
                current_chunk = para
        else:
            current_chunk = current_chunk + "\n" + para if current_chunk else para
    
    # 添加最后一个 chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


# ==================== 多种切片策略实现 ====================

def heading_chunk(text: str, chunk_size: int = 800) -> List[str]:
    """
    按标题切分文本
    
    Args:
        text: 原始文本
        chunk_size: 每个 chunk 的最大字符数
    
    Returns:
        切分后的 chunk 列表
    """
    if not text or not text.strip():
        return []
    
    # 标题匹配模式
    heading_pattern = r'^(#{1,6}\s+.+|第[一二三四五六七八九十百千]+[章节条款].+|\d+[\.\、].+|[一二三四五六七八九十]+[\.\、].+)'
    
    lines = text.split('\n')
    chunks = []
    current_chunk = ""
    current_heading = ""
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # 检查是否是标题
        is_heading = bool(re.match(heading_pattern, line_stripped))
        
        if is_heading:
            # 保存之前的 chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_heading = line_stripped
            current_chunk = line_stripped
        else:
            # 检查是否超过 chunk_size
            if len(current_chunk) + len(line_stripped) + 1 > chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # 新 chunk 以当前标题开始（如果有）
                current_chunk = current_heading + "\n" + line_stripped if current_heading else line_stripped
            else:
                current_chunk = current_chunk + "\n" + line_stripped if current_chunk else line_stripped
    
    # 添加最后一个 chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def sliding_window(text: str, chunk_size: int = 600, overlap: int = 150) -> List[str]:
    """
    滑动窗口切分
    
    Args:
        text: 原始文本
        chunk_size: 每个 chunk 的最大字符数
        overlap: 重叠字符数
    
    Returns:
        切分后的 chunk 列表
    """
    if not text or not text.strip():
        return []
    
    # 清理文本，保留段落结构
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 尝试在句子边界切分
        if end < len(text):
            # 向后查找句子结束符
            search_end = min(end + 100, len(text))
            best_break = end
            for i in range(end, search_end):
                if text[i] in '。！？.!?\n':
                    best_break = i + 1
                    break
            end = best_break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 滑动窗口
        start = end - overlap if end < len(text) else len(text)
    
    return chunks


def sentence_chunk(text: str, min_len: int = 100, max_len: int = 500) -> List[str]:
    """
    按句子切分文本
    
    Args:
        text: 原始文本
        min_len: chunk 最小长度
        max_len: chunk 最大长度
    
    Returns:
        切分后的 chunk 列表
    """
    if not text or not text.strip():
        return []
    
    # 按句子分割（支持中英文）
    sentences = re.split(r'([。！？.!?]+)', text)
    
    # 重新组合句子和标点
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i]
        punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
        combined = (sentence + punctuation).strip()
        if combined:
            combined_sentences.append(combined)
    
    # 处理最后可能没有标点的部分
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        combined_sentences.append(sentences[-1].strip())
    
    chunks = []
    current_chunk = ""
    
    for sentence in combined_sentences:
        if len(current_chunk) + len(sentence) + 1 > max_len:
            if current_chunk and len(current_chunk) >= min_len:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            elif current_chunk:
                # 当前 chunk 太短，继续累积
                current_chunk = current_chunk + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence
    
    # 添加最后一个 chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def paragraph_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    按段落切分（与原 split_text 相同）
    
    Args:
        text: 原始文本
        chunk_size: 每个 chunk 的最大字符数
        overlap: 重叠字符数
    
    Returns:
        切分后的 chunk 列表
    """
    return split_text(text, chunk_size, overlap)


def semantic_llm_chunk(text: str, max_chunk: int = 1000) -> List[str]:
    """
    使用 LLM 进行语义切分
    
    Args:
        text: 原始文本
        max_chunk: 每个 chunk 的最大字符数
    
    Returns:
        切分后的 chunk 列表
    """
    if not text or not text.strip():
        return []
    
    # 如果文本较短，直接返回
    if len(text) <= max_chunk:
        return [text.strip()]
    
    try:
        client = get_chat_client()
        model = os.getenv("CHAT_MODEL", "deepseek-chat")
        
        system_prompt = """你是一个文本切分专家。请将用户提供的文本按语义完整性切分成多个片段。

要求：
1. 每个片段应该语义完整，能够独立理解
2. 每个片段长度控制在 200-800 字符之间
3. 保持原文内容，不要修改或总结
4. 用 "---CHUNK---" 作为片段之间的分隔符
5. 直接输出切分后的文本，不要添加任何解释"""

        # 如果文本太长，先粗切
        if len(text) > 4000:
            # 先用滑动窗口粗切，再对每个部分进行语义切分
            coarse_chunks = sliding_window(text, chunk_size=2000, overlap=200)
            all_chunks = []
            for coarse in coarse_chunks:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": coarse}
                    ],
                    temperature=0.3,
                    max_tokens=4000
                )
                result = response.choices[0].message.content
                chunks = [c.strip() for c in result.split("---CHUNK---") if c.strip()]
                all_chunks.extend(chunks)
            return all_chunks
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            result = response.choices[0].message.content
            chunks = [c.strip() for c in result.split("---CHUNK---") if c.strip()]
            return chunks if chunks else [text.strip()]
    
    except Exception as e:
        # 如果 LLM 调用失败，回退到滑动窗口
        print(f"[RAG] 语义切分失败，回退到滑动窗口: {e}")
        return sliding_window(text, chunk_size=max_chunk, overlap=100)


def split_text_by_strategy(text: str, strategy: str, params: Dict[str, Any]) -> List[str]:
    """
    根据策略切分文本的统一入口
    
    Args:
        text: 原始文本
        strategy: 切片策略名称
        params: 策略参数
    
    Returns:
        切分后的 chunk 列表
    """
    if strategy == "heading_chunk":
        return heading_chunk(text, chunk_size=params.get("chunk_size", 800))
    
    elif strategy == "sliding_window":
        return sliding_window(
            text, 
            chunk_size=params.get("chunk_size", 600),
            overlap=params.get("overlap", 150)
        )
    
    elif strategy == "sentence_chunk":
        return sentence_chunk(
            text,
            min_len=params.get("min_len", 100),
            max_len=params.get("max_len", 500)
        )
    
    elif strategy == "semantic_llm_chunk":
        return semantic_llm_chunk(text, max_chunk=params.get("max_chunk", 1000))
    
    elif strategy == "paragraph_chunk":
        return paragraph_chunk(
            text,
            chunk_size=params.get("chunk_size", 500),
            overlap=params.get("overlap", 50)
        )
    
    else:
        # 默认使用段落切分
        return split_text(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    将文本列表转换为向量
    
    Args:
        texts: 文本列表
    
    Returns:
        向量列表
    """
    if not texts:
        return []
    
    client = get_embed_client()
    model = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    
    # 批量处理，每批最多 100 条
    all_embeddings = []
    batch_size = 100
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # 确保文本是有效的 UTF-8 字符串
        batch = [t.encode('utf-8', errors='ignore').decode('utf-8') for t in batch]
        response = client.embeddings.create(
            model=model,
            input=batch,
            encoding_format="float"
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings


def load_vector_db() -> dict:
    """加载向量数据库（已废弃，保留用于向后兼容）
    
    注意：此函数已迁移到 VectorRetriever._load_db()
    保留此函数是为了不破坏现有代码
    """
    retriever = VectorRetriever(VECTOR_DB_PATH)
    return retriever._load_db()


def save_vector_db(db: dict):
    """保存向量数据库（已废弃，保留用于向后兼容）
    
    注意：此函数已迁移到 VectorRetriever._save_db()
    保留此函数是为了不破坏现有代码
    """
    retriever = VectorRetriever(VECTOR_DB_PATH)
    retriever._save_db(db)


def add_to_vector_db(chunks: List[str], embeddings: List[List[float]]):
    """将 chunks 和对应的 embeddings 添加到向量数据库（已重构）
    
    注意：此函数已迁移到 VectorRetriever.add_documents()
    保留此函数是为了不破坏现有代码
    
    Args:
        chunks: 文本片段列表
        embeddings: 对应的向量列表
    """
    retriever = VectorRetriever(VECTOR_DB_PATH)
    retriever.add_documents(chunks, embeddings)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算两个向量的余弦相似度（已废弃，保留用于向后兼容）
    
    注意：此函数已迁移到 SimilarityRanker._cosine_similarity()
    保留此函数是为了不破坏现有代码
    """
    ranker = SimilarityRanker()
    return ranker._cosine_similarity(vec1, vec2)


def search_top_k(query: str, k: int = 3) -> List[Tuple[str, float]]:
    """根据查询向量检索最相关的 top-k 个 chunks（已重构为三层架构）
    
    重构说明：
    - 使用 VectorRetriever 进行召回
    - 使用 SimilarityRanker 进行排序
    - 保持原有返回格式以兼容现有代码
    
    Args:
        query: 查询文本
        k: 返回的结果数量
    
    Returns:
        [(chunk, similarity_score), ...] 列表
    """
    # 1. 向量化查询
    query_embedding = embed_texts([query])[0]
    
    # 2. 召回候选文档（Retriever 层）
    retriever = VectorRetriever(VECTOR_DB_PATH)
    documents = retriever.retrieve(query_embedding, top_k=k * 2)
    
    if not documents:
        return []
    
    # 3. 排序（Ranker 层）
    ranker = SimilarityRanker()
    ranked_docs = ranker.rank(query, query_embedding, documents, top_k=k)
    
    # 4. 转换为原有格式（保持向后兼容）
    return [(doc.text, score) for doc, score in ranked_docs]


def generate_answer(query: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    """根据检索到的 chunks 和用户问题，调用 LLM 生成答案（已重构）
    
    注意：此函数已迁移到 LLMGenerator.generate()
    保留此函数是为了不破坏现有代码
    
    Args:
        query: 用户问题
        retrieved_chunks: 检索到的 (chunk, score) 列表
    
    Returns:
        LLM 生成的答案
    """
    from rag.retriever.base import Document
    
    # 转换为 Document 格式
    ranked_docs = [
        (Document(id=i, text=chunk, metadata={}), score)
        for i, (chunk, score) in enumerate(retrieved_chunks)
    ]
    
    # 使用 Generator 层生成答案
    client = get_chat_client()
    model = os.getenv("CHAT_MODEL", "deepseek-chat")
    generator = LLMGenerator(client, model)
    
    return generator.generate(query, ranked_docs)


def clear_vector_db():
    """清空向量数据库（已重构）
    
    注意：此函数已迁移到 VectorRetriever.clear()
    保留此函数是为了不破坏现有代码
    """
    retriever = VectorRetriever(VECTOR_DB_PATH)
    retriever.clear()


def get_db_stats() -> dict:
    """获取向量数据库统计信息（已重构）
    
    注意：此函数已迁移到 VectorRetriever.get_stats()
    保留此函数是为了不破坏现有代码
    """
    retriever = VectorRetriever(VECTOR_DB_PATH)
    return retriever.get_stats()


def rag_pipeline(query: str, top_k: int = 3) -> Tuple[str, List[Tuple[str, float]]]:
    """RAG 完整流程编排（新增：三层架构清晰版本）
    
    这是推荐使用的新接口，清晰展示了 RAG 的三层架构：
    1. Retriever 层：召回候选文档
    2. Ranker 层：对候选文档进行排序
    3. Generator 层：基于排序后的文档生成答案
    
    Args:
        query: 用户问题
        top_k: 返回的文档数量
    
    Returns:
        (答案, [(chunk, score), ...]) 元组
    """
    from rag.retriever.base import Document
    
    # 1. 向量化查询（预处理）
    query_embedding = embed_texts([query])[0]
    
    # 2. 召回候选文档（Retriever 层）
    retriever = VectorRetriever(VECTOR_DB_PATH)
    documents = retriever.retrieve(query_embedding, top_k=top_k * 2)
    
    if not documents:
        return "向量库为空，请先上传文件。", []
    
    # 3. 排序（Ranker 层）
    ranker = SimilarityRanker()
    ranked_docs = ranker.rank(query, query_embedding, documents, top_k=top_k)
    
    # 4. 生成答案（Generator 层）
    client = get_chat_client()
    model = os.getenv("CHAT_MODEL", "deepseek-chat")
    generator = LLMGenerator(client, model)
    answer = generator.generate(query, ranked_docs)
    
    # 5. 返回答案和检索结果
    retrieved_chunks = [(doc.text, score) for doc, score in ranked_docs]
    return answer, retrieved_chunks


# ==================== BM25 检索相关函数 ====================

def add_to_bm25_index(chunks: List[str]):
    """将 chunks 添加到 BM25 索引
    
    Args:
        chunks: 文本片段列表
    """
    retriever = BM25Retriever(BM25_DB_PATH)
    retriever.add_documents(chunks)


def search_bm25(query: str, k: int = 3) -> List[Tuple[str, float]]:
    """使用 BM25 检索最相关的 top-k 个 chunks
    
    Args:
        query: 查询文本
        k: 返回的结果数量
    
    Returns:
        [(chunk, bm25_score), ...] 列表
    """
    retriever = BM25Retriever(BM25_DB_PATH)
    documents = retriever.retrieve_by_text(query, top_k=k)
    
    return [(doc.text, doc.metadata.get('bm25_score', 0.0)) for doc in documents]


def clear_bm25_index():
    """清空 BM25 索引"""
    retriever = BM25Retriever(BM25_DB_PATH)
    retriever.clear()


def get_bm25_stats() -> dict:
    """获取 BM25 索引统计信息"""
    retriever = BM25Retriever(BM25_DB_PATH)
    return retriever.get_stats()


def hybrid_search(query: str, k: int = 3, vector_weight: float = 0.5) -> List[Tuple[str, float]]:
    """混合检索：结合向量检索和 BM25 检索
    
    Args:
        query: 查询文本
        k: 返回的结果数量
        vector_weight: 向量检索的权重（0-1），BM25 权重为 1-vector_weight
    
    Returns:
        [(chunk, combined_score), ...] 列表
    """
    from rag.retriever.base import Document
    
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
    # 构建文本到分数的映射
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
    
    # 6. 排序并返回 top-k
    combined_results.sort(key=lambda x: x[1], reverse=True)
    return combined_results[:k]


def sync_bm25_from_vector_db():
    """从向量库同步数据到 BM25 索引
    
    用于解决数据导入时 BM25 功能尚未实现导致的索引不同步问题
    """
    import json
    
    # 1. 清空 BM25 索引
    clear_bm25_index()
    
    # 2. 从向量库读取数据
    with open(VECTOR_DB_PATH, 'r', encoding='utf-8') as f:
        db = json.load(f)
        chunks = db.get('chunks', [])
    
    # 3. 批量添加到 BM25
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        add_to_bm25_index(batch)
    
    return len(chunks)


def search_with_rerank(query: str, k: int = 3, recall_k: int = 20) -> List[Tuple[str, float]]:
    """使用 Rerank 模型进行精排的检索（两阶段检索）
    
    工作流程：
    1. 第一阶段（召回）：使用向量检索快速召回 top-N 候选
    2. 第二阶段（精排）：使用 Rerank 模型对候选进行精确排序
    
    优势：
    - 相比余弦相似度提升 20-30% 的准确率
    - 擅长处理复杂查询、长查询、多条件查询
    - 深度理解查询和文档的语义关系
    
    Args:
        query: 查询文本
        k: 最终返回的结果数量
        recall_k: 第一阶段召回的候选数量（建议 3-5 倍于 k）
    
    Returns:
        [(chunk, rerank_score), ...] 列表，按 Rerank 分数降序排列
    """
    # 1. 向量化查询
    query_embedding = embed_texts([query])[0]
    
    # 2. 第一阶段：向量召回（Retriever 层）
    retriever = VectorRetriever(VECTOR_DB_PATH)
    documents = retriever.retrieve(query_embedding, top_k=recall_k)
    
    if not documents:
        return []
    
    # 3. 第二阶段：Rerank 精排（Ranker 层）
    try:
        ranker = RerankRanker()
        ranked_docs = ranker.rank(query, query_embedding, documents, top_k=k)
        
        # 4. 转换为原有格式（保持向后兼容）
        return [(doc.text, score) for doc, score in ranked_docs]
    
    except Exception as e:
        print(f"[Rerank] 精排失败，降级使用余弦相似度: {e}")
        # 降级方案：使用余弦相似度排序
        ranker = SimilarityRanker()
        ranked_docs = ranker.rank(query, query_embedding, documents, top_k=k)
        return [(doc.text, score) for doc, score in ranked_docs]


def hybrid_search_with_rerank(
    query: str, 
    k: int = 3, 
    vector_weight: float = 0.5,
    recall_k: int = 20
) -> List[Tuple[str, float]]:
    """混合检索 + Rerank 精排（三阶段检索）
    
    工作流程：
    1. 第一阶段（召回）：向量检索 + BM25 检索，召回候选
    2. 第二阶段（融合）：加权融合两种检索结果
    3. 第三阶段（精排）：使用 Rerank 模型对融合结果进行精确排序
    
    这是最强的检索方案，结合了：
    - 向量检索的语义理解能力
    - BM25 的精确匹配能力
    - Rerank 的深度语义交互能力
    
    Args:
        query: 查询文本
        k: 最终返回的结果数量
        vector_weight: 向量检索的权重（0-1）
        recall_k: 第一阶段召回的候选数量
    
    Returns:
        [(chunk, rerank_score), ...] 列表
    """
    from rag.retriever.base import Document
    
    # 1. 向量化查询
    query_embedding = embed_texts([query])[0]
    
    # 2. 第一阶段：混合召回
    # 2.1 向量检索
    vector_retriever = VectorRetriever(VECTOR_DB_PATH)
    vector_docs = vector_retriever.retrieve(query_embedding, top_k=recall_k)
    
    # 2.2 BM25 检索
    bm25_retriever = BM25Retriever(BM25_DB_PATH)
    bm25_docs = bm25_retriever.retrieve_by_text(query, top_k=recall_k)
    
    # 3. 第二阶段：融合分数
    ranker = SimilarityRanker()
    vector_ranked = ranker.rank(query, query_embedding, vector_docs, top_k=None)
    
    # 构建文本到分数的映射
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
    
    print(f"[混合+Rerank] 向量召回: {len(vector_scores)} 个, BM25召回: {len(bm25_scores)} 个, 合并后: {len(all_texts)} 个")
    
    # 构建融合后的文档列表
    fused_documents = []
    for text in all_texts:
        v_score = vector_scores.get(text, 0.0)
        b_score = bm25_scores.get(text, 0.0)
        combined_score = vector_weight * v_score + (1 - vector_weight) * b_score
        
        # 创建文档对象（需要找到原始文档以获取 embedding）
        doc = None
        for d in vector_docs:
            if d.text == text:
                doc = d
                # 确保设置 hybrid_score
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['hybrid_score'] = combined_score
                break
        if doc is None:
            for d in bm25_docs:
                if d.text == text:
                    # BM25 文档可能没有 embedding，使用查询向量作为占位
                    doc = Document(
                        id=d.id,
                        text=d.text,
                        embedding=query_embedding,
                        metadata={'hybrid_score': combined_score}
                    )
                    break
        
        if doc:
            fused_documents.append(doc)
    
    print(f"[混合+Rerank] 融合后文档数: {len(fused_documents)}")
    
    # 按融合分数排序，取 top recall_k
    fused_documents.sort(key=lambda d: d.metadata.get('hybrid_score', 0.0), reverse=True)
    fused_documents = fused_documents[:recall_k]
    
    if not fused_documents:
        return []
    
    # 4. 第三阶段：Rerank 精排
    try:
        rerank_ranker = RerankRanker()
        ranked_docs = rerank_ranker.rank(query, query_embedding, fused_documents, top_k=k)
        
        return [(doc.text, score) for doc, score in ranked_docs]
    
    except Exception as e:
        print(f"[Rerank] 精排失败，返回混合检索结果: {e}")
        # 降级方案：返回混合检索的结果
        return [(doc.text, doc.metadata.get('hybrid_score', 0.0)) 
                for doc in fused_documents[:k]]
