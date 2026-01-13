"""
HyDE (Hypothetical Document Embeddings) - 通过生成假设文档增强查询

核心思想：
不直接用查询检索，而是先让 LLM 生成一个"假设的答案文档"，
然后用这个假设文档的 Embedding 去检索真实文档。

优势：
1. 文档 vs 文档的匹配（而非查询 vs 文档）
2. 假设文档包含更丰富的语义信息
3. 特别适合模糊查询或信息不足的场景

参考论文：
Precise Zero-Shot Dense Retrieval without Relevance Labels (Gao et al., 2022)
"""

from typing import List, Tuple


class HyDERetriever:
    """HyDE 检索器 - 通过生成假设文档增强查询"""
    
    def __init__(self, llm_client, model: str, use_enhanced: bool = False):
        """初始化 HyDE 检索器
        
        Args:
            llm_client: OpenAI 客户端
            model: 模型名称
            use_enhanced: 是否使用增强模式（结合真实数据）
        """
        self.client = llm_client
        self.model = model
        self.use_enhanced = use_enhanced
    
    def generate_hypothetical_document(self, query: str, 
                                      conversation_history: List[dict] = None,
                                      reference_context: str = None) -> str:
        """生成假设文档
        
        Args:
            query: 用户查询
            conversation_history: 对话历史（用于理解上下文）
            reference_context: 参考上下文（增强模式下的真实数据）
        
        Returns:
            假设文档文本
        """
        # 构建上下文
        context_text = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-4:]
            context_text = "\n对话历史：\n" + self._format_history(recent_history) + "\n"
        
        # 根据是否有参考上下文，选择不同的 Prompt
        if reference_context:
            # 增强模式：基于真实数据生成
            hyde_prompt = f"""你是一个知识库专家。请基于提供的参考信息，生成一个详细的答案文档。

{context_text}
参考信息（来自知识库的真实数据）：
{reference_context}

用户问题：{query}

要求：
1. **优先使用参考信息中的内容**（这是真实数据，准确度高）
2. 可以基于参考信息进行合理的扩展和推理
3. 生成一个详细、专业的答案文档（200-300字）
4. 包含问题相关的关键概念、术语、细节
5. 直接输出答案内容，不要有前缀或解释

请生成增强的假设文档："""
            print(f"[HyDE] 使用增强模式（有参考数据）")
        else:
            # 标准模式：纯 LLM 生成
            hyde_prompt = f"""你是一个知识库专家。请根据用户的问题，生成一个假设的答案文档。

{context_text}
用户问题：{query}

要求：
1. 生成一个详细、专业的答案文档（200-300字）
2. 包含问题相关的关键概念、术语、细节
3. 即使不确定答案，也要基于常识和推理生成合理的内容
4. 不要说"我不知道"或"需要查询资料"
5. 直接输出答案内容，不要有前缀或解释

示例：
问题：RAG的优势是什么？
输出：
RAG（检索增强生成）的主要优势包括：首先，它能够有效减少大语言模型的幻觉问题，通过从知识库中检索真实信息来支撑生成的答案。其次，RAG具有知识可更新性，无需重新训练模型就能获取最新信息。第三，它降低了模型对参数记忆的依赖，使得较小的模型也能回答专业问题。此外，RAG还提供了答案的可追溯性，用户可以查看信息来源。在企业应用中，RAG能够整合私有知识库，实现领域定制化，同时保持较低的部署成本。

请生成假设文档："""
            print(f"[HyDE] 使用标准模式（无参考数据）")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的知识库专家，擅长生成详细、准确的答案文档。"},
                    {"role": "user", "content": hyde_prompt}
                ],
                temperature=0.5,  # 适度的创造性
                max_tokens=500
            )
            
            hypothetical_doc = response.choices[0].message.content.strip()
            
            print(f"[HyDE] 原查询: {query}")
            print(f"[HyDE] 假设文档长度: {len(hypothetical_doc)} 字符")
            print(f"[HyDE] 假设文档预览: {hypothetical_doc[:100]}...")
            
            return hypothetical_doc
        
        except Exception as e:
            print(f"[HyDE] 生成假设文档失败: {e}，使用原查询")
            return query
    
    def generate_multiple_hypothetical_documents(self, query: str, 
                                                conversation_history: List[dict] = None,
                                                num_docs: int = 3) -> List[str]:
        """生成多个假设文档（增加多样性）
        
        Args:
            query: 用户查询
            conversation_history: 对话历史
            num_docs: 生成的假设文档数量
        
        Returns:
            假设文档列表
        """
        hypothetical_docs = []
        
        for i in range(num_docs):
            # 调整 temperature 增加多样性
            context_text = ""
            if conversation_history and len(conversation_history) > 0:
                recent_history = conversation_history[-4:]
                context_text = "\n对话历史：\n" + self._format_history(recent_history) + "\n"
            
            hyde_prompt = f"""你是一个知识库专家。请根据用户的问题，生成一个假设的答案文档。

{context_text}
用户问题：{query}

要求：
1. 生成一个详细、专业的答案文档（200-300字）
2. 从不同角度回答问题（这是第 {i+1} 个版本）
3. 包含问题相关的关键概念、术语、细节
4. 直接输出答案内容，不要有前缀或解释

请生成假设文档："""

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的知识库专家。"},
                        {"role": "user", "content": hyde_prompt}
                    ],
                    temperature=0.3 + i * 0.2,  # 逐渐增加多样性
                    max_tokens=500
                )
                
                hypothetical_doc = response.choices[0].message.content.strip()
                hypothetical_docs.append(hypothetical_doc)
                
                print(f"[HyDE] 生成假设文档 {i+1}/{num_docs}: {len(hypothetical_doc)} 字符")
            
            except Exception as e:
                print(f"[HyDE] 生成假设文档 {i+1} 失败: {e}")
                continue
        
        # 如果全部失败，返回原查询
        if not hypothetical_docs:
            print(f"[HyDE] 所有假设文档生成失败，使用原查询")
            return [query]
        
        return hypothetical_docs
    
    def _format_history(self, history: List[dict]) -> str:
        """格式化对话历史"""
        formatted = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)


def hyde_search(hyde_retriever: HyDERetriever, embed_func, search_func,
                query: str, conversation_history: List[dict] = None,
                k: int = 3, use_multiple: bool = False) -> List[Tuple[str, float]]:
    """使用 HyDE 进行检索
    
    Args:
        hyde_retriever: HyDE 检索器
        embed_func: Embedding 函数
        search_func: 检索函数（接受 embedding 向量）
        query: 用户查询
        conversation_history: 对话历史
        k: 返回的结果数量
        use_multiple: 是否生成多个假设文档
    
    Returns:
        检索结果 [(chunk, score), ...]
    """
    if use_multiple:
        # 生成多个假设文档
        hypothetical_docs = hyde_retriever.generate_multiple_hypothetical_documents(
            query, conversation_history, num_docs=3
        )
        
        # 对每个假设文档检索并融合结果
        all_results = {}
        
        for doc in hypothetical_docs:
            # 使用假设文档的 Embedding 检索
            doc_embedding = embed_func(doc)
            results = search_func(doc_embedding, k=k*2)
            
            # 融合结果
            for chunk, score in results:
                if chunk not in all_results or score > all_results[chunk]:
                    all_results[chunk] = score
        
        # 排序并返回
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]
    
    else:
        # 生成单个假设文档
        hypothetical_doc = hyde_retriever.generate_hypothetical_document(
            query, conversation_history
        )
        
        # 使用假设文档的 Embedding 检索
        doc_embedding = embed_func(hypothetical_doc)
        results = search_func(doc_embedding, k=k)
        
        return results


def test_hyde():
    """测试 HyDE"""
    import os
    from openai import OpenAI
    from dotenv import load_dotenv
    
    load_dotenv()
    
    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY")
    )
    
    hyde_retriever = HyDERetriever(client, os.getenv("CHAT_MODEL", "deepseek-chat"))
    
    # 测试场景 1：模糊查询
    print("\n=== 场景1：模糊查询 ===")
    query1 = "RAG的优势"
    hypothetical_doc1 = hyde_retriever.generate_hypothetical_document(query1)
    print(f"\n原查询: {query1}")
    print(f"假设文档:\n{hypothetical_doc1}")
    
    # 测试场景 2：结合对话历史
    print("\n=== 场景2：结合对话历史 ===")
    query2 = "它有什么劣势？"
    history2 = [
        {"role": "user", "content": "什么是RAG？"},
        {"role": "assistant", "content": "RAG是检索增强生成..."}
    ]
    hypothetical_doc2 = hyde_retriever.generate_hypothetical_document(query2, history2)
    print(f"\n原查询: {query2}")
    print(f"对话历史: {history2}")
    print(f"假设文档:\n{hypothetical_doc2}")
    
    # 测试场景 3：生成多个假设文档
    print("\n=== 场景3：生成多个假设文档 ===")
    query3 = "如何实践RAG？"
    hypothetical_docs3 = hyde_retriever.generate_multiple_hypothetical_documents(query3, num_docs=3)
    print(f"\n原查询: {query3}")
    for i, doc in enumerate(hypothetical_docs3, 1):
        print(f"\n假设文档 {i}:\n{doc[:150]}...")


if __name__ == "__main__":
    test_hyde()
