"""
多变体召回 - 通过生成多种查询变体提升召回率

核心思想：
为每个查询生成多个不同的表达形式（变体），包括：
1. 同义词变体：不同词汇表示相同概念
2. 语义扩展：相关概念和领域术语
3. 表达方式：不同的语法结构和表述形式

优势：
- 提升召回率：捕捉更多潜在相关信息
- 弥补查询不精确：适应模糊或不标准的查询
- 适应不同表达习惯：覆盖不同用户的搜索方式
"""

from typing import List, Dict, Tuple


class MultiVariantRecaller:
    """多变体召回器 - 生成多种查询变体提升召回率"""
    
    def __init__(self, llm_client, model: str):
        """初始化多变体召回器
        
        Args:
            llm_client: OpenAI 客户端
            model: 模型名称
        """
        self.client = llm_client
        self.model = model
    
    def generate_variants(self, query: str, conversation_history: List[Dict] = None,
                         num_variants: int = 5) -> Dict[str, List[str]]:
        """生成多种类型的查询变体
        
        Args:
            query: 原始查询
            conversation_history: 对话历史
            num_variants: 每种类型生成的变体数量
        
        Returns:
            变体字典 {
                "synonyms": [...],      # 同义词变体
                "semantic": [...],      # 语义扩展
                "expressions": [...]    # 不同表达方式
            }
        """
        # 构建上下文
        context_text = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-4:]
            context_text = "\n对话历史：\n" + self._format_history(recent_history) + "\n"
        
        variant_prompt = f"""你是一个查询优化专家。请为用户的查询生成多种变体，以提升信息检索的召回率。

{context_text}
原始查询：{query}

任务：生成三种类型的查询变体，每种类型 {num_variants} 个

1. **同义词变体**（用不同词汇表达相同意思）
   - 替换关键词为同义词
   - 保持原意不变
   - 例如："汽车修理" → "车辆修理"、"汽车维修"、"车子修理"

2. **语义扩展**（相关概念和领域术语）
   - 扩展到相关概念
   - 包含领域专业术语
   - 例如："汽车修理" → "修车"、"汽车保养"、"车辆维护"

3. **不同表达方式**（改变语法结构）
   - 改变句式结构
   - 使用不同的提问方式
   - 例如："汽车修理" → "如何修理汽车"、"汽车修理方法"、"修车服务"

输出格式（严格按照此格式）：
同义词变体：
- 变体1
- 变体2
...

语义扩展：
- 变体1
- 变体2
...

不同表达方式：
- 变体1
- 变体2
...

请生成变体："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的查询优化专家。"},
                    {"role": "user", "content": variant_prompt}
                ],
                temperature=0.5,
                max_tokens=600
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 解析变体
            variants = self._parse_variants(result_text)
            
            print(f"[多变体召回] 原查询: {query}")
            print(f"[多变体召回] 同义词变体: {len(variants.get('synonyms', []))} 个")
            print(f"[多变体召回] 语义扩展: {len(variants.get('semantic', []))} 个")
            print(f"[多变体召回] 不同表达: {len(variants.get('expressions', []))} 个")
            
            return variants
        
        except Exception as e:
            print(f"[多变体召回] 生成失败: {e}，使用原查询")
            return {
                "synonyms": [query],
                "semantic": [],
                "expressions": []
            }
    
    def multi_variant_search(self, query: str, search_func, 
                            conversation_history: List[Dict] = None,
                            k: int = 3, strategy: str = "balanced") -> List[Tuple[str, float]]:
        """多变体召回检索
        
        Args:
            query: 原始查询
            search_func: 检索函数
            conversation_history: 对话历史
            k: 返回的结果数量
            strategy: 召回策略
                - "aggressive": 激进（使用所有变体，最大化召回）
                - "balanced": 平衡（使用部分变体，平衡召回和精度）
                - "conservative": 保守（只使用高质量变体，优先精度）
        
        Returns:
            检索结果 [(chunk, score), ...]
        """
        # 1. 生成变体
        variants_dict = self.generate_variants(query, conversation_history, num_variants=3)
        
        # 2. 根据策略选择变体
        all_variants = [query]  # 始终包含原查询
        
        if strategy == "aggressive":
            # 使用所有变体
            all_variants.extend(variants_dict.get("synonyms", []))
            all_variants.extend(variants_dict.get("semantic", []))
            all_variants.extend(variants_dict.get("expressions", []))
        elif strategy == "balanced":
            # 使用同义词 + 部分语义扩展
            all_variants.extend(variants_dict.get("synonyms", [])[:2])
            all_variants.extend(variants_dict.get("semantic", [])[:2])
            all_variants.extend(variants_dict.get("expressions", [])[:1])
        else:  # conservative
            # 只使用同义词变体
            all_variants.extend(variants_dict.get("synonyms", [])[:2])
        
        # 去重
        all_variants = list(dict.fromkeys(all_variants))
        
        print(f"[多变体召回] 策略: {strategy}")
        print(f"[多变体召回] 使用 {len(all_variants)} 个变体进行检索")
        
        # 3. 多路召回
        all_results = {}
        
        for i, variant in enumerate(all_variants, 1):
            print(f"[多变体召回] 检索变体 {i}/{len(all_variants)}: {variant}")
            
            try:
                results = search_func(variant, k=k*2)  # 每个变体多召回一些
                
                # 融合结果（保留最高分数）
                for chunk, score in results:
                    if chunk not in all_results or score > all_results[chunk]:
                        all_results[chunk] = score
            
            except Exception as e:
                print(f"[多变体召回] 变体检索失败: {e}")
                continue
        
        # 4. 排序并返回
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        
        print(f"[多变体召回] 共召回 {len(sorted_results)} 个不重复文档")
        
        return sorted_results[:k]
    
    def _parse_variants(self, text: str) -> Dict[str, List[str]]:
        """解析 LLM 生成的变体文本"""
        variants = {
            "synonyms": [],
            "semantic": [],
            "expressions": []
        }
        
        current_type = None
        
        for line in text.split('\n'):
            line = line.strip()
            
            if not line:
                continue
            
            # 识别类型标题
            if "同义词" in line:
                current_type = "synonyms"
                continue
            elif "语义扩展" in line or "语义" in line:
                current_type = "semantic"
                continue
            elif "表达方式" in line or "表达" in line:
                current_type = "expressions"
                continue
            
            # 提取变体
            if current_type and line.startswith('-'):
                variant = line[1:].strip()
                variant = self._clean_variant(variant)
                if variant and len(variant) > 1:
                    variants[current_type].append(variant)
        
        return variants
    
    def _clean_variant(self, variant: str) -> str:
        """清理变体文本"""
        import re
        # 移除编号
        variant = re.sub(r'^[\d\.\)]+\s*', '', variant)
        # 移除引号
        variant = variant.strip('"\'""''')
        # 移除"变体"等标签
        variant = re.sub(r'^变体\d+[:：]?\s*', '', variant)
        return variant.strip()
    
    def _format_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        formatted = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)


def test_multi_variant_recall():
    """测试多变体召回"""
    import os
    from openai import OpenAI
    from dotenv import load_dotenv
    
    load_dotenv()
    
    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY")
    )
    
    recaller = MultiVariantRecaller(client, "deepseek-chat")
    
    # 测试场景 1：生成变体
    print("\n=== 场景1：生成查询变体 ===")
    query1 = "汽车修理"
    variants1 = recaller.generate_variants(query1, num_variants=3)
    print(f"\n原查询: {query1}")
    print(f"\n同义词变体:")
    for v in variants1.get("synonyms", []):
        print(f"  - {v}")
    print(f"\n语义扩展:")
    for v in variants1.get("semantic", []):
        print(f"  - {v}")
    print(f"\n不同表达方式:")
    for v in variants1.get("expressions", []):
        print(f"  - {v}")
    
    # 测试场景 2：结合对话历史
    print("\n=== 场景2：结合对话历史 ===")
    query2 = "产品"
    history2 = [
        {"role": "user", "content": "什么是RAG？"},
        {"role": "assistant", "content": "RAG是检索增强生成..."}
    ]
    variants2 = recaller.generate_variants(query2, history2, num_variants=3)
    print(f"\n原查询: {query2}")
    print(f"对话历史: {history2[0]['content']}")
    print(f"\n同义词变体:")
    for v in variants2.get("synonyms", []):
        print(f"  - {v}")
    
    # 测试场景 3：多变体召回（模拟）
    print("\n=== 场景3：多变体召回（模拟）===")
    
    def mock_search(query: str, k: int = 3):
        """模拟检索函数"""
        print(f"  → 检索: {query}")
        return [
            (f"文档A关于{query[:10]}", 0.8),
            (f"文档B关于{query[:10]}", 0.7),
        ]
    
    query3 = "智能手机"
    results = recaller.multi_variant_search(
        query3, 
        mock_search, 
        k=5, 
        strategy="balanced"
    )
    print(f"\n最终召回结果: {len(results)} 个文档")
    for chunk, score in results[:5]:
        print(f"  - {chunk[:50]}... (分数: {score:.2f})")


if __name__ == "__main__":
    test_multi_variant_recall()
