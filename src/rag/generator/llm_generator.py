"""
基于 LLM 的答案生成器
迁移自 rag_engine.py 的现有实现
"""

from typing import List, Tuple, Optional
from openai import OpenAI
from src.rag.retriever.base import Document
from src.rag.generator.base import BaseGenerator


class LLMGenerator(BaseGenerator):
    """基于 LLM 的答案生成器
    
    特点：
    - 使用 OpenAI 兼容的 API 接口
    - 支持国内大模型（DeepSeek、Moonshot、通义千问等）
    - 基于检索到的上下文生成答案
    """
    
    def __init__(self, client, model: str):
        """初始化 LLM 生成器
        
        Args:
            client: OpenAI 客户端实例
            model: 模型名称
        """
        self.client = client
        self.model = model
    
    def generate(self, query: str, ranked_docs: List[Tuple[Document, float]], 
                 conversation_history: List[dict] = None) -> str:
        """根据排序后的文档生成答案（支持多轮对话）
        
        Args:
            query: 用户问题
            ranked_docs: 排序后的 (文档, 分数) 列表
            conversation_history: 对话历史 [{"role": "user/assistant", "content": "..."}]
        
        Returns:
            生成的答案文本
        """
        context = "\n\n---\n\n".join([doc.text for doc, _ in ranked_docs])
        
        system_prompt = """你是一个专业的知识问答助手。请根据提供的参考资料回答用户的问题。

要求：
1. 优先使用参考资料中的内容回答问题
2. 可以基于参考资料进行合理的推理、综合和分析
3. 如果用户询问多个概念之间的关系，应该：
   - 先分别介绍各个概念（基于参考资料）
   - 再分析它们之间的联系、区别或相互关系（可以进行合理推理）
4. 只有当参考资料完全不相关时，才说明"根据提供的资料，无法回答该问题"
5. 回答要准确、全面、有条理，充分利用参考资料的信息价值
6. 在多轮对话中，注意理解上下文和代词指代（如"它"、"这个"等）"""

        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history[-6:])
        
        user_prompt = f"""参考资料：
{context}

用户问题：{query}

请根据以上参考资料回答问题："""

        messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def generate_stream(self, query: str, ranked_docs: List[Tuple[Document, float]], 
                       conversation_history: List[dict] = None):
        """流式生成答案（支持多轮对话）
        
        Args:
            query: 用户问题
            ranked_docs: 排序后的 (文档, 分数) 列表
            conversation_history: 对话历史 [{"role": "user/assistant", "content": "..."}]
        
        Yields:
            生成的答案文本片段
        """
        context = "\n\n---\n\n".join([doc.text for doc, _ in ranked_docs])
        
        system_prompt = """你是一个专业的知识问答助手。请根据提供的参考资料回答用户的问题。

要求：
1. 优先使用参考资料中的内容回答问题
2. 可以基于参考资料进行合理的推理、综合和分析
3. 如果用户询问多个概念之间的关系，应该：
   - 先分别介绍各个概念（基于参考资料）
   - 再分析它们之间的联系、区别或相互关系（可以进行合理推理）
4. 只有当参考资料完全不相关时，才说明"根据提供的资料，无法回答该问题"
5. 回答要准确、全面、有条理，充分利用参考资料的信息价值
6. 在多轮对话中，注意理解上下文和代词指代（如"它"、"这个"等）"""

        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history[-6:])
        
        user_prompt = f"""参考资料：
{context}

用户问题：{query}

请根据以上参考资料回答问题："""

        messages.append({"role": "user", "content": user_prompt})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            stream=True  # 启用流式输出
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
