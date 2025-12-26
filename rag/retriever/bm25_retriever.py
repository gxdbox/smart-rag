"""
基于 BM25 的检索器

实现说明：
- 使用 rank-bm25 库进行 BM25 检索
- 使用 jieba 进行中文分词
- 支持与向量检索混合使用
- 使用 pickle 进行持久化存储
"""

import os
import pickle
import string
import jieba
from rank_bm25 import BM25Okapi
from typing import List, Dict
from .base import BaseRetriever, Document


class BM25Retriever(BaseRetriever):
    """基于 BM25 的检索器（已实现）
    
    特点：
    - 基于词频的稀疏检索
    - 对专业术语（如 ABSD）命中率高
    - 可与向量检索混合使用
    - 支持中文分词
    """
    
    def __init__(self, db_path: str):
        """初始化 BM25 检索器
        
        Args:
            db_path: 数据库文件路径（pickle 格式）
        """
        self.db_path = db_path
        self.corpus = []
        self.tokenized_corpus = []
        self.bm25 = None
        
        if os.path.exists(db_path):
            self._load_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """中文分词
        
        Args:
            text: 原始文本
        
        Returns:
            分词后的词列表（已过滤标点符号）
        """
        # 中英文标点符号集合
        punctuation = set(string.punctuation + '，。！？；：""''（）【】《》、·…—')
        
        # 分词并过滤空白和标点符号
        tokens = [
            token.strip() 
            for token in jieba.cut(text) 
            if token.strip() and token.strip() not in punctuation
        ]
        return tokens
    
    def _build_index(self):
        """构建 BM25 索引"""
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def _load_index(self):
        """从文件加载索引"""
        try:
            with open(self.db_path, 'rb') as f:
                data = pickle.load(f)
                self.corpus = data.get('corpus', [])
                self.tokenized_corpus = data.get('tokenized_corpus', [])
                if self.tokenized_corpus:
                    self._build_index()
        except Exception as e:
            print(f"[BM25] 加载索引失败: {e}，将创建新索引")
            self.corpus = []
            self.tokenized_corpus = []
    
    def _save_index(self):
        """保存索引到文件"""
        data = {
            'corpus': self.corpus,
            'tokenized_corpus': self.tokenized_corpus
        }
        with open(self.db_path, 'wb') as f:
            pickle.dump(data, f)
    
    def retrieve(self, query_embedding: List[float], top_k: int) -> List[Document]:
        """根据查询召回候选文档
        
        注意：BM25 不使用向量，此参数保留用于接口一致性
        实际使用时需要传入查询文本（通过 metadata）
        
        Args:
            query_embedding: 查询向量（BM25 不使用，保留接口一致性）
            top_k: 召回文档数量
        
        Returns:
            候选文档列表
        """
        raise NotImplementedError(
            "BM25Retriever.retrieve() 不支持向量查询，请使用 retrieve_by_text() 方法"
        )
    
    def retrieve_by_text(self, query: str, top_k: int) -> List[Document]:
        """根据查询文本召回候选文档
        
        Args:
            query: 查询文本
            top_k: 召回文档数量
        
        Returns:
            候选文档列表（按 BM25 分数排序）
        """
        if not self.bm25 or not self.corpus:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        documents = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = Document(
                    id=idx,
                    text=self.corpus[idx],
                    metadata={'bm25_score': float(scores[idx])}
                )
                documents.append(doc)
        
        return documents
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]] = None):
        """添加文档到存储
        
        Args:
            texts: 文本列表
            embeddings: 向量列表（BM25 不使用，保留接口一致性）
        """
        if not texts:
            return
        
        self.corpus.extend(texts)
        
        for text in texts:
            tokens = self._tokenize(text)
            self.tokenized_corpus.append(tokens)
        
        self._build_index()
        self._save_index()
    
    def clear(self):
        """清空存储"""
        self.corpus = []
        self.tokenized_corpus = []
        self.bm25 = None
        
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
    
    def get_stats(self) -> Dict[str, int]:
        """获取存储统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_chunks': len(self.corpus),
            'total_tokens': sum(len(tokens) for tokens in self.tokenized_corpus)
        }
