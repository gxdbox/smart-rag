"""
查询重写模块 - 解决多轮对话中的指代消解问题

核心功能：
1. 检测查询是否包含指代词（"它"、"这个"、"那个"等）
2. 结合对话历史，将查询重写为独立的、自包含的问题
3. 重写后的查询用于检索，确保检索准确性

参考：
- IBM Granite Query Rewrite
- LangChain ConversationalRetrievalChain
"""

from typing import List, Dict, Optional
import re


class QueryRewriter:
    """查询重写器 - 处理多轮对话中的指代消解"""
    
    # 常见的指代词
    PRONOUNS = [
        "它", "这个", "那个", "这些", "那些",
        "他", "她", "它们", "其", "此",
        "前者", "后者", "上述", "以上"
    ]
    
    # 常见的追问模式
    FOLLOW_UP_PATTERNS = [
        r"^(能|可以|请).*(详细|具体|解释|说明|举例)",
        r"^(为什么|怎么|如何)",
        r"(第\d+[点个条]|第一|第二|第三)",
        r"(优势|缺点|好处|坏处|区别|联系|关系)",
        r"(更|还有|另外|其他)"
    ]
    
    def __init__(self, llm_client, model: str):
        """初始化查询重写器
        
        Args:
            llm_client: OpenAI 客户端
            model: 模型名称
        """
        self.client = llm_client
        self.model = model
    
    def needs_rewrite(self, query: str, conversation_history: List[Dict]) -> bool:
        """判断查询是否需要重写
        
        Args:
            query: 用户查询
            conversation_history: 对话历史
        
        Returns:
            True 如果需要重写，False 否则
        """
        # 没有对话历史，不需要重写
        if not conversation_history or len(conversation_history) == 0:
            return False
        
        # 检查是否包含指代词
        for pronoun in self.PRONOUNS:
            if pronoun in query:
                return True
        
        # 检查是否是追问模式
        for pattern in self.FOLLOW_UP_PATTERNS:
            if re.search(pattern, query):
                return True
        
        # 查询很短（< 8 个字），很可能是追问
        if len(query.strip()) < 8:
            return True
        
        return False
    
    def rewrite(self, query: str, conversation_history: List[Dict]) -> str:
        """重写查询为独立的、自包含的问题
        
        Args:
            query: 原始查询
            conversation_history: 对话历史
        
        Returns:
            重写后的查询
        """
        # 不需要重写，直接返回
        if not self.needs_rewrite(query, conversation_history):
            return query
        
        # 构建重写 Prompt
        history_text = self._format_history(conversation_history[-4:])  # 最近2轮
        
        rewrite_prompt = f"""你是一个查询重写助手。你的任务是将用户的追问改写为独立的、自包含的问题。

对话历史：
{history_text}

当前问题：{query}

任务：
1. 如果问题中包含代词（如"它"、"这个"、"那个"），请替换为具体的实体名称
2. 如果问题是追问（如"第一点是什么"），请补全完整的上下文
3. 如果问题已经是独立的、清晰的，请直接输出原问题

要求：
- 只输出重写后的问题，不要解释
- 保持问题的原意不变
- 确保重写后的问题可以独立理解，不需要对话历史

重写后的问题："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的查询重写助手。"},
                    {"role": "user", "content": rewrite_prompt}
                ],
                temperature=0.0,  # 使用 0 温度，确保稳定性
                max_tokens=200
            )
            
            rewritten = response.choices[0].message.content.strip()
            
            # 移除可能的引号
            rewritten = rewritten.strip('"\'""''')
            
            print(f"[查询重写] 原问题: {query}")
            print(f"[查询重写] 重写后: {rewritten}")
            
            return rewritten
        
        except Exception as e:
            print(f"[查询重写] 重写失败: {e}，使用原问题")
            return query
    
    def _format_history(self, history: List[Dict]) -> str:
        """格式化对话历史
        
        Args:
            history: 对话历史
        
        Returns:
            格式化后的历史文本
        """
        formatted = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            # 限制每条消息的长度
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)


def test_query_rewriter():
    """测试查询重写器"""
    import os
    from openai import OpenAI
    from dotenv import load_dotenv
    
    load_dotenv()
    
    client = OpenAI(
        base_url=os.getenv("CHAT_BASE_URL"),
        api_key=os.getenv("CHAT_API_KEY")
    )
    
    rewriter = QueryRewriter(client, os.getenv("CHAT_MODEL", "deepseek-chat"))
    
    # 测试场景 1：指代消解
    history1 = [
        {"role": "user", "content": "什么是 RAG？"},
        {"role": "assistant", "content": "RAG 是检索增强生成（Retrieval-Augmented Generation）的缩写..."}
    ]
    query1 = "它有什么优势？"
    rewritten1 = rewriter.rewrite(query1, history1)
    print(f"\n场景1 - 原问题: {query1}")
    print(f"场景1 - 重写后: {rewritten1}")
    
    # 测试场景 2：追问
    history2 = [
        {"role": "user", "content": "如何优化 RAG 系统？"},
        {"role": "assistant", "content": "优化 RAG 系统可以从以下几个方面入手：\n1. 改进检索质量\n2. 优化分块策略\n3. 使用重排序模型"}
    ]
    query2 = "第一点能详细说明吗？"
    rewritten2 = rewriter.rewrite(query2, history2)
    print(f"\n场景2 - 原问题: {query2}")
    print(f"场景2 - 重写后: {rewritten2}")
    
    # 测试场景 3：独立问题（不需要重写）
    history3 = []
    query3 = "什么是向量数据库？"
    rewritten3 = rewriter.rewrite(query3, history3)
    print(f"\n场景3 - 原问题: {query3}")
    print(f"场景3 - 重写后: {rewritten3}")


if __name__ == "__main__":
    test_query_rewriter()
