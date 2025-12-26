"""
自动切片策略选择器
根据文本结构自动选择最佳切片策略
"""

import re
from typing import Dict, Tuple, Any


def analyze_structure(text: str) -> Dict[str, Any]:
    """
    分析文本结构特征
    
    Args:
        text: 原始文本
    
    Returns:
        包含各项特征的字典
    """
    if not text or not text.strip():
        return {
            "title_ratio": 0,
            "avg_para_len": 0,
            "para_std": 0,
            "sentence_count": 0,
            "line_count": 0,
            "sentence_density": 0,
            "total_chars": 0
        }
    
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    line_count = len(lines)
    
    if line_count == 0:
        return {
            "title_ratio": 0,
            "avg_para_len": 0,
            "para_std": 0,
            "sentence_count": 0,
            "line_count": 0,
            "sentence_density": 0,
            "total_chars": 0
        }
    
    # 标题检测模式（Markdown 标题、数字编号标题、中文标题等）
    title_patterns = [
        r'^#{1,6}\s+',           # Markdown 标题
        r'^\d+[\.\、]\s*\S',     # 数字编号 (1. 或 1、)
        r'^第[一二三四五六七八九十百千]+[章节条款]',  # 中文章节
        r'^[一二三四五六七八九十]+[\.\、\s]',        # 中文数字编号
        r'^[A-Z][A-Z\s]{2,}$',   # 全大写标题
        r'^\([一二三四五六七八九十\d]+\)',           # 括号编号
    ]
    
    title_count = 0
    for line in lines:
        for pattern in title_patterns:
            if re.match(pattern, line):
                title_count += 1
                break
    
    title_ratio = title_count / line_count if line_count > 0 else 0
    
    # 段落长度统计
    para_lengths = [len(line) for line in lines]
    avg_para_len = sum(para_lengths) / len(para_lengths) if para_lengths else 0
    
    # 段落长度标准差
    if len(para_lengths) > 1:
        mean = avg_para_len
        variance = sum((x - mean) ** 2 for x in para_lengths) / len(para_lengths)
        para_std = variance ** 0.5
    else:
        para_std = 0
    
    # 句子统计（中英文句子结束符）
    sentence_endings = re.findall(r'[。！？.!?]', text)
    sentence_count = len(sentence_endings)
    
    # 句子密度
    sentence_density = sentence_count / line_count if line_count > 0 else 0
    
    # 总字符数
    total_chars = len(text)
    
    return {
        "title_ratio": title_ratio,
        "avg_para_len": avg_para_len,
        "para_std": para_std,
        "sentence_count": sentence_count,
        "line_count": line_count,
        "sentence_density": sentence_density,
        "total_chars": total_chars
    }


def detect_type(features: Dict[str, Any]) -> str:
    """
    根据特征识别文体类型
    
    Args:
        features: analyze_structure 返回的特征字典
    
    Returns:
        文体类型: structured, long_form, fragment, legal, normal
    """
    title_ratio = features.get("title_ratio", 0)
    avg_para_len = features.get("avg_para_len", 0)
    para_std = features.get("para_std", 0)
    sentence_density = features.get("sentence_density", 0)
    total_chars = features.get("total_chars", 0)
    
    # 法律/合同文档特征：段落较长、结构化程度高、句子密度高
    if title_ratio > 0.1 and avg_para_len > 100 and sentence_density > 2:
        return "legal"
    
    # 结构化文档（技术文档/标题多）：标题密度高
    if title_ratio > 0.15:
        return "structured"
    
    # 碎片化文档（FAQ/对话/短句多）：平均段落短、句子密度低
    if avg_para_len < 50 and sentence_density < 1.5:
        return "fragment"
    
    # 长文档（论文/书籍）：总字符多、段落长、标准差大
    if total_chars > 5000 and avg_para_len > 150:
        return "long_form"
    
    # 默认普通文档
    return "normal"


def choose_chunk_strategy(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    根据文本自动选择最佳切片策略
    
    Args:
        text: 原始文本
    
    Returns:
        (strategy_name, params) 元组
        strategy_name: heading_chunk, sliding_window, sentence_chunk, 
                      semantic_llm_chunk, paragraph_chunk
        params: 策略参数字典
    """
    features = analyze_structure(text)
    doc_type = detect_type(features)
    
    # 根据文体类型选择策略
    if doc_type == "structured":
        # 结构化文档：按标题切分
        return ("heading_chunk", {"chunk_size": 800})
    
    elif doc_type == "long_form":
        # 长文档：滑动窗口，较大 chunk，较多重叠
        return ("sliding_window", {"chunk_size": 600, "overlap": 150})
    
    elif doc_type == "fragment":
        # 碎片化文档：按句子切分
        return ("sentence_chunk", {"min_len": 100, "max_len": 500})
    
    elif doc_type == "legal":
        # 法律文档：语义切分（调用 LLM）
        return ("semantic_llm_chunk", {"max_chunk": 1000})
    
    else:
        # 普通文档：段落切分
        return ("paragraph_chunk", {"chunk_size": 500, "overlap": 50})


def get_strategy_description(strategy: str) -> str:
    """
    获取策略的中文描述
    
    Args:
        strategy: 策略名称
    
    Returns:
        策略的中文描述
    """
    descriptions = {
        "heading_chunk": "📑 标题切分（适合技术文档）",
        "sliding_window": "🪟 滑动窗口（适合长文档）",
        "sentence_chunk": "📝 句子切分（适合FAQ/对话）",
        "semantic_llm_chunk": "🧠 语义切分（适合法律文档）",
        "paragraph_chunk": "📄 段落切分（通用策略）"
    }
    return descriptions.get(strategy, strategy)
