"""
多步骤检索模块 - 将复杂问题拆分为多个子问题，逐步检索

核心功能：
1. 使用 LLM 将复杂问题拆分为多个子问题
2. 对每个子问题独立检索
3. 整合所有检索结果
4. 基于完整上下文生成答案

应用场景：
- 复杂问题："什么是RAG？它有什么优势和劣势？如何实践？"
  → 子问题1："什么是RAG？"
  → 子问题2："RAG有什么优势？"
  → 子问题3："RAG有什么劣势？"
  → 子问题4："如何在企业中实践RAG？"
"""

from typing import List, Dict, Tuple, Callable


class MultiStepQueryEngine:
    """多步骤查询引擎 - 拆分复杂问题并逐步检索"""
    
    def __init__(self, llm_client, model: str):
        """初始化多步骤查询引擎
        
        Args:
            llm_client: OpenAI 客户端
            model: 模型名称
        """
        self.client = llm_client
        self.model = model
    
    def decompose_query(self, query: str) -> List[str]:
        """将复杂问题拆分为多个子问题
        
        Args:
            query: 复杂问题
        
        Returns:
            子问题列表
        """
        decompose_prompt = f"""你是一个问题分析助手。请判断用户的问题是否复杂，如果复杂则拆分为多个子问题。

用户问题：{query}

判断规则：
1. **简单问题**（不需要拆分）：
   - 只问一个具体的事情
   - 例如："什么是RAG？"、"向量数据库有哪些？"
   
2. **复杂问题**（需要拆分）：
   - 包含多个疑问词（什么、为什么、如何、有哪些等）
   - 包含"和"、"以及"连接的多个问题
   - 例如："什么是RAG？它有什么优势和劣势？"

任务：
- 如果是简单问题，输出："SIMPLE"
- 如果是复杂问题，拆分为多个子问题，每行一个，不要编号

示例1：
问题：什么是RAG？
输出：
SIMPLE

示例2：
问题：什么是RAG？它有什么优势和劣势？如何在企业中实践？
输出：
什么是RAG？
RAG有什么优势？
RAG有什么劣势？
如何在企业中实践RAG？

示例3：
问题：介绍一下向量数据库的原理和应用场景
输出：
向量数据库的工作原理是什么？
向量数据库有哪些应用场景？

请分析并输出："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的问题分析助手。"},
                    {"role": "user", "content": decompose_prompt}
                ],
                temperature=0.0,
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            
            # 判断是否为简单问题
            if "SIMPLE" in result.upper():
                print(f"[多步骤检索] 简单问题，不需要拆分")
                return [query]
            
            # 解析子问题
            sub_queries = [line.strip() for line in result.split('\n') if line.strip()]
            
            # 清理可能的编号
            sub_queries = [self._clean_query(q) for q in sub_queries]
            
            # 过滤掉 "SIMPLE" 等非问题文本
            sub_queries = [q for q in sub_queries if len(q) > 3 and '?' in q or '？' in q or any(kw in q for kw in ['什么', '如何', '为什么', '有哪些', '怎么'])]
            
            if not sub_queries:
                print(f"[多步骤检索] 拆分失败，使用原问题")
                return [query]
            
            print(f"[多步骤检索] 复杂问题，拆分为 {len(sub_queries)} 个子问题:")
            for i, sq in enumerate(sub_queries, 1):
                print(f"  {i}. {sq}")
            
            return sub_queries
        
        except Exception as e:
            print(f"[多步骤检索] 拆分失败: {e}，使用原问题")
            return [query]
    
    def multi_step_retrieve(self, query: str, search_func: Callable, 
                           k_per_query: int = 2) -> List[Tuple[str, float]]:
        """多步骤检索：拆分问题 → 逐个检索 → 整合结果
        
        Args:
            query: 用户问题
            search_func: 检索函数，接受 (query, k) 参数
            k_per_query: 每个子问题检索的文档数
        
        Returns:
            整合后的检索结果 [(chunk, score), ...]
        """
        # 1. 拆分问题
        sub_queries = self.decompose_query(query)
        
        # 2. 对每个子问题检索
        all_results = {}  # {chunk: max_score}
        
        for i, sub_query in enumerate(sub_queries, 1):
            print(f"[多步骤检索] 检索子问题 {i}/{len(sub_queries)}: {sub_query}")
            
            try:
                results = search_func(sub_query, k=k_per_query)
                
                # 融合结果（保留最高分数）
                for chunk, score in results:
                    if chunk not in all_results or score > all_results[chunk]:
                        all_results[chunk] = score
            
            except Exception as e:
                print(f"[多步骤检索] 子问题检索失败: {e}")
                continue
        
        # 3. 排序并返回
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        
        print(f"[多步骤检索] 共检索到 {len(sorted_results)} 个不重复的文档")
        
        return sorted_results
    
    def _clean_query(self, query: str) -> str:
        """清理子问题，移除编号等"""
        import re
        # 移除开头的编号
        query = re.sub(r'^[\d\-\*\•]+[\.\)]\s*', '', query)
        # 移除引号
        query = query.strip('"\'""''')
        return query.strip()


def test_multi_step_query():
    """测试多步骤检索"""
    import os
    from openai import OpenAI
    from dotenv import load_dotenv
    
    load_dotenv()
    
    client = OpenAI(
        base_url=os.getenv("CHAT_BASE_URL"),
        api_key=os.getenv("CHAT_API_KEY")
    )
    
    engine = MultiStepQueryEngine(client, os.getenv("CHAT_MODEL", "deepseek-chat"))
    
    # 测试场景 1：简单问题
    print("\n=== 场景1：简单问题 ===")
    query1 = "什么是 RAG？"
    sub_queries1 = engine.decompose_query(query1)
    print(f"原问题: {query1}")
    print(f"拆分结果: {sub_queries1}")
    
    # 测试场景 2：复杂问题（多个疑问）
    print("\n=== 场景2：复杂问题 ===")
    query2 = "什么是 RAG？它有什么优势和劣势？如何在企业中更好地实践 RAG？"
    sub_queries2 = engine.decompose_query(query2)
    print(f"原问题: {query2}")
    print(f"拆分结果: {sub_queries2}")
    
    # 测试场景 3：包含"和"的问题
    print("\n=== 场景3：包含'和'的问题 ===")
    query3 = "介绍一下向量数据库的原理和应用场景"
    sub_queries3 = engine.decompose_query(query3)
    print(f"原问题: {query3}")
    print(f"拆分结果: {sub_queries3}")
    
    # 测试场景 4：多步骤检索（模拟）
    print("\n=== 场景4：多步骤检索（模拟）===")
    
    def mock_search(query: str, k: int = 3):
        """模拟检索函数"""
        print(f"  → 检索: {query}")
        # 模拟返回结果
        return [
            (f"文档A关于{query[:10]}", 0.8),
            (f"文档B关于{query[:10]}", 0.7),
        ]
    
    query4 = "什么是 RAG？它有什么优势？"
    results = engine.multi_step_retrieve(query4, mock_search, k_per_query=2)
    print(f"\n最终检索结果: {len(results)} 个文档")
    for chunk, score in results[:5]:
        print(f"  - {chunk[:50]}... (分数: {score:.2f})")


if __name__ == "__main__":
    test_multi_step_query()
