"""
知识图谱抽取模块
对检索到的 chunks 进行实时知识图谱抽取
"""
import json
import os
import hashlib
from typing import List, Dict, Optional
from pathlib import Path

# 图谱缓存文件
GRAPH_CACHE_PATH = "knowledge_graph_cache.json"

# 抽取 Prompt（使用双大括号转义 JSON 示例）
EXTRACTION_PROMPT = '''从以下文本中抽取软件架构领域的实体和关系。

实体类型: Method, Model, ArchitectureView, DesignStage, QualityAttribute, Concept
关系类型: includes, consists_of, applies_to, affects, defines, compares_with, belongs_to, supports

规则:
1. 只抽取架构相关的核心知识
2. 不确定就不要输出
3. 如果没有可抽取内容，返回空数组

直接输出JSON，不要任何解释:
{{"entities":[{{"name":"名称","type":"类型","description":"描述"}}],"relations":[{{"source":"源","relation":"关系","target":"目标"}}]}}

文本:
{text}

JSON:'''


def get_chunk_hash(text: str) -> str:
    """计算 chunk 的哈希值，用于缓存"""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def load_graph_cache() -> Dict:
    """加载图谱缓存"""
    if os.path.exists(GRAPH_CACHE_PATH):
        try:
            with open(GRAPH_CACHE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"chunks": {}, "entities": {}, "relations": []}


def save_graph_cache(cache: Dict):
    """保存图谱缓存"""
    with open(GRAPH_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def extract_from_chunk_with_llm(text: str, llm_client, model: str) -> Optional[Dict]:
    """使用 LLM 从单个 chunk 抽取实体和关系"""
    prompt = EXTRACTION_PROMPT.format(text=text[:2000])  # 限制长度
    
    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content
        print(f"[KG] LLM 响应: {content[:200]}...")
        
        # 尝试解析 JSON
        import re
        
        # 先尝试直接解析
        try:
            result = json.loads(content)
            if isinstance(result, dict) and ("entities" in result or "relations" in result):
                return result
        except:
            pass
        
        # 尝试提取 JSON 块（可能被 markdown 包裹）
        # 移除 markdown 代码块标记
        content_clean = re.sub(r'```json\s*', '', content)
        content_clean = re.sub(r'```\s*', '', content_clean)
        
        json_match = re.search(r'\{[\s\S]*\}', content_clean)
        if json_match:
            try:
                result = json.loads(json_match.group())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError as e:
                print(f"[KG] JSON 解析失败: {e}")
        
    except Exception as e:
        print(f"[KG] 抽取失败: {e}")
    
    # 返回空结果而不是 None，避免后续报错
    return {"entities": [], "relations": []}


def extract_knowledge_graph(
    chunks: List[str],
    llm_client,
    model: str,
    use_cache: bool = True
) -> Dict:
    """
    从检索到的 chunks 中抽取知识图谱
    
    Args:
        chunks: 检索到的文本列表
        llm_client: OpenAI 兼容的 LLM 客户端
        model: 模型名称
        use_cache: 是否使用缓存
    
    Returns:
        合并后的知识图谱 {"entities": [...], "relations": [...]}
    """
    cache = load_graph_cache() if use_cache else {"chunks": {}, "entities": {}, "relations": []}
    
    all_entities = {}
    all_relations = []
    extracted_count = 0
    cached_count = 0
    
    for chunk in chunks:
        chunk_hash = get_chunk_hash(chunk)
        
        # 检查缓存
        if use_cache and chunk_hash in cache["chunks"]:
            result = cache["chunks"][chunk_hash]
            cached_count += 1
        else:
            # 调用 LLM 抽取
            result = extract_from_chunk_with_llm(chunk, llm_client, model)
            if result and use_cache:
                cache["chunks"][chunk_hash] = result
                save_graph_cache(cache)
            extracted_count += 1
        
        # 合并结果
        if result and isinstance(result, dict):
            entities = result.get("entities", [])
            relations = result.get("relations", [])
            
            # 确保是列表
            if isinstance(entities, list):
                for entity in entities:
                    if isinstance(entity, dict):
                        name = entity.get("name", "").strip()
                        if name and name not in all_entities and name != "名称":  # 过滤模板
                            all_entities[name] = entity
            
            if isinstance(relations, list):
                for relation in relations:
                    if isinstance(relation, dict) and relation.get("source") and relation.get("target"):
                        # 过滤模板内容
                        if relation.get("source") == "源" or relation.get("target") == "目标":
                            continue
                        # 简单去重
                        exists = any(
                            r.get("source") == relation.get("source") and
                            r.get("relation") == relation.get("relation") and
                            r.get("target") == relation.get("target")
                            for r in all_relations
                        )
                        if not exists:
                            all_relations.append(relation)
    
    print(f"[KG] 抽取完成: {extracted_count} 个新抽取, {cached_count} 个来自缓存")
    print(f"[KG] 结果: {len(all_entities)} 个实体, {len(all_relations)} 个关系")
    
    return {
        "entities": list(all_entities.values()),
        "relations": all_relations
    }


def format_graph_for_prompt(graph: Dict) -> str:
    """将知识图谱格式化为可加入 prompt 的文本"""
    if not graph["entities"] and not graph["relations"]:
        return ""
    
    lines = ["【相关知识图谱】"]
    
    if graph["entities"]:
        lines.append("实体:")
        for e in graph["entities"][:10]:  # 限制数量
            lines.append(f"  - {e.get('name')} ({e.get('type')}): {e.get('description', '')}")
    
    if graph["relations"]:
        lines.append("关系:")
        for r in graph["relations"][:10]:  # 限制数量
            lines.append(f"  - {r.get('source')} --[{r.get('relation')}]--> {r.get('target')}")
    
    return "\n".join(lines)


def get_graph_stats() -> Dict:
    """获取图谱缓存统计"""
    cache = load_graph_cache()
    
    all_entities = set()
    all_relations = 0
    
    for chunk_data in cache.get("chunks", {}).values():
        if chunk_data:
            for e in chunk_data.get("entities", []):
                all_entities.add(e.get("name", ""))
            all_relations += len(chunk_data.get("relations", []))
    
    return {
        "cached_chunks": len(cache.get("chunks", {})),
        "total_entities": len(all_entities),
        "total_relations": all_relations
    }
