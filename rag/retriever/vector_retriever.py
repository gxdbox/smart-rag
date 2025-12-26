"""
基于 JSON + NumPy 的向量检索器
迁移自 rag_engine.py 的现有实现
"""

import os
import json
import shutil
from typing import List, Dict
from .base import BaseRetriever, Document


class VectorRetriever(BaseRetriever):
    """基于 JSON + NumPy 的向量检索器
    
    特点：
    - 使用 JSON 文件存储向量和文本
    - 使用 NumPy 进行暴力遍历检索
    - 适合小规模数据集（< 10000 条）
    """
    
    def __init__(self, db_path: str):
        """初始化向量检索器
        
        Args:
            db_path: 向量库 JSON 文件路径
        """
        self.db_path = db_path
    
    def _load_db(self) -> dict:
        """加载向量数据库（迁移自 load_vector_db）"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError as exc:
                backup_path = f"{self.db_path}.corrupt"
                try:
                    shutil.copyfile(self.db_path, backup_path)
                except OSError:
                    pass
                print(
                    f"[RAG] 向量库文件损坏，已备份至 {backup_path}. 错误: {exc}. 将重置为默认空库。"
                )
                data = {"chunks": [], "embeddings": []}
                self._save_db(data)
                return data
        return {"chunks": [], "embeddings": []}
    
    def _save_db(self, db: dict):
        """保存向量数据库（迁移自 save_vector_db）"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    
    def retrieve(self, query_embedding: List[float], top_k: int) -> List[Document]:
        """根据查询向量召回候选文档
        
        注意：此方法只负责召回，不计算相似度分数
        相似度计算和排序由 Ranker 负责
        
        Args:
            query_embedding: 查询向量
            top_k: 召回文档数量
        
        Returns:
            候选文档列表（包含 embedding）
        """
        db = self._load_db()
        
        if not db["chunks"]:
            return []
        
        query_dim = len(query_embedding)
        documents = []
        
        for i, (chunk, emb) in enumerate(zip(db["chunks"], db["embeddings"])):
            if len(emb) != query_dim:
                continue
            
            doc = Document(
                id=i,
                text=chunk,
                embedding=emb,
                metadata={}
            )
            documents.append(doc)
        
        return documents
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]]):
        """添加文档到存储（迁移自 add_to_vector_db）
        
        Args:
            texts: 文本列表
            embeddings: 对应的向量列表
        """
        if len(texts) != len(embeddings):
            raise ValueError("chunks 和 embeddings 数量不匹配")

        if not texts:
            return

        target_dim = len(embeddings[0]) if embeddings and embeddings[0] else None
        if target_dim is None:
            raise ValueError("无法获取向量维度，请检查嵌入结果")

        if any(len(vec) != target_dim for vec in embeddings):
            raise ValueError("新生成的向量维度不一致，请重试上传")

        db = self._load_db()

        existing_dims = sorted({len(vec) for vec in db["embeddings"] if vec})
        if existing_dims and any(dim != target_dim for dim in existing_dims):
            raise ValueError(
                "向量库中存在与当前 Embedding 模型维度不一致的数据。请通过左侧按钮清空向量库后重新上传文件。"
            )

        db["chunks"].extend(texts)
        db["embeddings"].extend(embeddings)
        self._save_db(db)
    
    def clear(self):
        """清空存储（迁移自 clear_vector_db）"""
        self._save_db({"chunks": [], "embeddings": []})
    
    def get_stats(self) -> Dict[str, int]:
        """获取存储统计信息（迁移自 get_db_stats）
        
        Returns:
            统计信息字典
        """
        db = self._load_db()
        return {
            "total_chunks": len(db["chunks"]),
            "total_embeddings": len(db["embeddings"])
        }
