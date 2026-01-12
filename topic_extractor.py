"""
主题提取模块 - 从对话历史中提取当前话题

用于解决多轮对话中的话题连贯性问题
"""

from typing import List, Dict, Optional


class TopicExtractor:
    """主题提取器 - 从对话历史中提取当前讨论的主题"""
    
    def __init__(self, llm_client, model: str):
        """初始化主题提取器
        
        Args:
            llm_client: OpenAI 客户端
            model: 模型名称
        """
        self.client = llm_client
        self.model = model
    
    def extract_topic(self, conversation_history: List[Dict]) -> Optional[str]:
        """从对话历史中提取当前话题
        
        Args:
            conversation_history: 对话历史
        
        Returns:
            提取的主题，如果无法提取则返回 None
        """
        if not conversation_history or len(conversation_history) == 0:
            return None
        
        # 只看最近2轮对话
        recent_history = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
        
        # 构建提取 Prompt
        history_text = self._format_history(recent_history)
        
        extract_prompt = f"""你是一个主题提取助手。请从对话历史中提取当前讨论的主要话题。

对话历史：
{history_text}

任务：
1. 提取对话中的核心主题（通常是一个名词或名词短语）
2. 只输出主题本身，不要解释
3. 如果有多个主题，输出最主要的一个
4. 主题应该简洁（1-5个词）

示例：
对话："什么是 RAG？" "RAG 是检索增强生成..."
主题：RAG

对话："向量数据库有哪些？" "常见的有 Milvus、Weaviate..."
主题：向量数据库

当前对话的主题："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的主题提取助手。"},
                    {"role": "user", "content": extract_prompt}
                ],
                temperature=0.0,
                max_tokens=50
            )
            
            topic = response.choices[0].message.content.strip()
            
            # 清理可能的引号和标点
            topic = topic.strip('"\'""''。，、')
            
            print(f"[主题提取] 提取的主题: {topic}")
            
            return topic if topic else None
        
        except Exception as e:
            print(f"[主题提取] 提取失败: {e}")
            return None
    
    def enhance_query_with_topic(self, query: str, topic: str) -> str:
        """将主题注入到查询中
        
        Args:
            query: 原始查询
            topic: 提取的主题
        
        Returns:
            增强后的查询
        """
        # 如果查询已经包含主题，不重复添加
        if topic in query:
            return query
        
        # 将主题添加到查询前面
        enhanced_query = f"{topic} {query}"
        
        print(f"[主题注入] 原查询: {query}")
        print(f"[主题注入] 增强后: {enhanced_query}")
        
        return enhanced_query
    
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
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)


def test_topic_extractor():
    """测试主题提取器"""
    import os
    from openai import OpenAI
    from dotenv import load_dotenv
    
    load_dotenv()
    
    client = OpenAI(
        base_url=os.getenv("CHAT_BASE_URL"),
        api_key=os.getenv("CHAT_API_KEY")
    )
    
    extractor = TopicExtractor(client, os.getenv("CHAT_MODEL", "deepseek-chat"))
    
    # 测试场景 1
    history1 = [
        {"role": "user", "content": "什么是 RAG？"},
        {"role": "assistant", "content": "RAG 是检索增强生成（Retrieval-Augmented Generation）的缩写..."}
    ]
    topic1 = extractor.extract_topic(history1)
    print(f"\n场景1 - 提取的主题: {topic1}")
    
    query1 = "产品"
    enhanced1 = extractor.enhance_query_with_topic(query1, topic1)
    print(f"场景1 - 增强后的查询: {enhanced1}")
    
    # 测试场景 2
    history2 = [
        {"role": "user", "content": "向量数据库有哪些？"},
        {"role": "assistant", "content": "常见的向量数据库包括 Milvus、Weaviate、Pinecone..."}
    ]
    topic2 = extractor.extract_topic(history2)
    print(f"\n场景2 - 提取的主题: {topic2}")
    
    query2 = "如何选择"
    enhanced2 = extractor.enhance_query_with_topic(query2, topic2)
    print(f"场景2 - 增强后的查询: {enhanced2}")


if __name__ == "__main__":
    test_topic_extractor()
