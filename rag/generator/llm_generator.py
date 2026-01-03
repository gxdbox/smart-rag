"""
基于 LLM 的答案生成器
迁移自 rag_engine.py 的现有实现
"""

from typing import List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from rag.retriever.base import Document
from rag.generator.base import BaseGenerator


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
    
    def generate(self, query: str, ranked_docs: List[Tuple[Document, float]]) -> str:
        """根据排序后的文档生成答案（迁移自 generate_answer）
        
        Args:
            query: 用户问题
            ranked_docs: 排序后的 (文档, 分数) 列表
        
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
5. 回答要准确、全面、有条理，充分利用参考资料的信息价值"""

        user_prompt = f"""参考资料：
{context}

用户问题：{query}

请根据以上参考资料回答问题："""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
