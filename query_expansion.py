"""
查询扩展模块 - 提升模糊查询的语义匹配能力

核心功能：
1. 将模糊查询扩展为多个具体的查询变体
2. 增加同义词、相关词、上下文信息
3. 提高召回率和检索精度

应用场景：
- 模糊查询："产品" → "RAG产品 检索增强生成产品 RAG商业化方案"
- 口语化查询："好玩的地方" → "旅游景点 娱乐场所 休闲活动 观光地点"
- 不完整查询："适合亲子" → "适合亲子游的地方 亲子活动 家庭旅游景点"
"""

from typing import List, Dict, Optional


class QueryExpander:
    """查询扩展器 - 将模糊查询扩展为多个具体查询"""
    
    def __init__(self, llm_client, model: str):
        """初始化查询扩展器
        
        Args:
            llm_client: OpenAI 客户端
            model: 模型名称
        """
        self.client = llm_client
        self.model = model
    
    def expand(self, query: str, conversation_history: List[Dict] = None,
               num_variants: int = 3) -> List[str]:
        """扩展查询为多个变体
        
        Args:
            query: 原始查询
            conversation_history: 对话历史（用于理解上下文）
            num_variants: 生成的查询变体数量
        
        Returns:
            查询变体列表（包含原查询）
        """
        # 构建扩展 Prompt
        context_text = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-4:]
            context_text = "\n对话历史：\n" + self._format_history(recent_history)
        
        expansion_prompt = f"""你是一个查询扩展助手。请将用户的模糊查询扩展为多个具体的查询变体。

{context_text}

原始查询：{query}

任务：
1. 分析查询意图和上下文
2. 生成 {num_variants} 个不同角度的查询变体
3. 每个变体应该：
   - 更具体、更清晰
   - 包含同义词、相关词
   - 结合上下文信息（如果有）
   - 保持原意不变

输出格式：
每行一个查询变体，不要编号，不要解释

示例1：
原始查询：产品
对话历史：用户问了"什么是RAG"
输出：
RAG产品
检索增强生成的商业化产品
RAG开源项目和工具

示例2：
原始查询：好玩的地方
输出：
旅游景点推荐
娱乐休闲场所
适合游玩的观光地点

请生成查询变体："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的查询扩展助手。"},
                    {"role": "user", "content": expansion_prompt}
                ],
                temperature=0.3,  # 适度的随机性
                max_tokens=200
            )
            
            expanded_text = response.choices[0].message.content.strip()
            
            # 解析变体
            variants = [line.strip() for line in expanded_text.split('\n') if line.strip()]
            
            # 清理可能的编号
            variants = [self._clean_variant(v) for v in variants]
            
            # 去重
            variants = list(dict.fromkeys(variants))
            
            # 确保包含原查询
            if query not in variants:
                variants.insert(0, query)
            
            print(f"[查询扩展] 原查询: {query}")
            print(f"[查询扩展] 扩展后: {variants}")
            
            return variants[:num_variants + 1]  # 返回原查询 + N个变体
        
        except Exception as e:
            print(f"[查询扩展] 扩展失败: {e}，使用原查询")
            return [query]
    
    def expand_with_topic(self, query: str, topic: str, num_variants: int = 3) -> List[str]:
        """基于主题扩展查询
        
        Args:
            query: 原始查询
            topic: 当前话题
            num_variants: 生成的查询变体数量
        
        Returns:
            查询变体列表
        """
        expansion_prompt = f"""你是一个查询扩展助手。请基于当前话题扩展用户的查询。

当前话题：{topic}
用户查询：{query}

任务：
1. 结合话题和查询，生成 {num_variants} 个具体的查询变体
2. 每个变体都应该明确包含话题信息
3. 从不同角度表达查询意图

输出格式：
每行一个查询变体，不要编号

示例：
话题：RAG
查询：产品
输出：
RAG产品有哪些
检索增强生成的商业化产品
RAG技术的实际应用案例

请生成查询变体："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的查询扩展助手。"},
                    {"role": "user", "content": expansion_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            expanded_text = response.choices[0].message.content.strip()
            variants = [line.strip() for line in expanded_text.split('\n') if line.strip()]
            variants = [self._clean_variant(v) for v in variants]
            variants = list(dict.fromkeys(variants))
            
            # 添加简单的主题+查询组合
            simple_variant = f"{topic} {query}"
            if simple_variant not in variants:
                variants.insert(0, simple_variant)
            
            print(f"[查询扩展] 原查询: {query}")
            print(f"[查询扩展] 话题: {topic}")
            print(f"[查询扩展] 扩展后: {variants}")
            
            return variants[:num_variants + 1]
        
        except Exception as e:
            print(f"[查询扩展] 扩展失败: {e}，使用主题+查询")
            return [f"{topic} {query}", query]
    
    def _clean_variant(self, variant: str) -> str:
        """清理查询变体，移除编号等"""
        import re
        # 移除开头的编号（如 "1. "、"1) "、"- "等）
        variant = re.sub(r'^[\d\-\*\•]+[\.\)]\s*', '', variant)
        # 移除引号
        variant = variant.strip('"\'""''')
        return variant.strip()
    
    def _format_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        formatted = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)


def multi_query_retrieval(expander: QueryExpander, search_func, 
                          query: str, conversation_history: List[Dict] = None,
                          k: int = 3) -> List:
    """多查询检索 - 使用扩展的查询变体进行检索并融合结果
    
    Args:
        expander: 查询扩展器
        search_func: 检索函数
        query: 原始查询
        conversation_history: 对话历史
        k: 返回的结果数量
    
    Returns:
        融合后的检索结果
    """
    # 1. 扩展查询
    query_variants = expander.expand(query, conversation_history, num_variants=2)
    
    # 2. 对每个变体进行检索
    all_results = {}  # {chunk: max_score}
    
    for variant in query_variants:
        results = search_func(variant, k=k*2)  # 每个变体多召回一些
        
        for chunk, score in results:
            # 保留最高分数
            if chunk not in all_results or score > all_results[chunk]:
                all_results[chunk] = score
    
    # 3. 按分数排序
    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    
    # 4. 返回 top-k
    return sorted_results[:k]


def test_query_expander():
    """测试查询扩展器"""
    import os
    from openai import OpenAI
    from dotenv import load_dotenv
    
    load_dotenv()
    
    client = OpenAI(
        base_url=os.getenv("CHAT_BASE_URL"),
        api_key=os.getenv("CHAT_API_KEY")
    )
    
    expander = QueryExpander(client, os.getenv("CHAT_MODEL", "deepseek-chat"))
    
    # 测试场景 1：模糊查询
    print("\n=== 场景1：模糊查询 ===")
    query1 = "产品"
    history1 = [
        {"role": "user", "content": "什么是 RAG？"},
        {"role": "assistant", "content": "RAG 是检索增强生成..."}
    ]
    variants1 = expander.expand(query1, history1, num_variants=3)
    print(f"原查询: {query1}")
    print(f"扩展后: {variants1}")
    
    # 测试场景 2：基于主题扩展
    print("\n=== 场景2：基于主题扩展 ===")
    query2 = "产品"
    topic2 = "RAG"
    variants2 = expander.expand_with_topic(query2, topic2, num_variants=3)
    print(f"原查询: {query2}")
    print(f"话题: {topic2}")
    print(f"扩展后: {variants2}")
    
    # 测试场景 3：口语化查询
    print("\n=== 场景3：口语化查询 ===")
    query3 = "好玩的地方"
    variants3 = expander.expand(query3, num_variants=3)
    print(f"原查询: {query3}")
    print(f"扩展后: {variants3}")


if __name__ == "__main__":
    test_query_expander()
