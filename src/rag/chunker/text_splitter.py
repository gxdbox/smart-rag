"""
文本切分模块
提供多种文本切分策略
"""

import os
import re
from typing import List, Dict, Any
from openai import OpenAI


def get_chat_client() -> OpenAI:
    """获取 Chat API 客户端"""
    client = OpenAI(
        api_key=os.getenv("CHAT_API_KEY"),
        base_url=os.getenv("CHAT_BASE_URL")
    )
    return client


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    将文本切分成 chunks
    
    Args:
        text: 原始文本
        chunk_size: 每个 chunk 的最大字符数
        overlap: chunk 之间的重叠字符数
    
    Returns:
        切分后的 chunk 列表
    """
    if not text or not text.strip():
        return []
    
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        if len(current_chunk) + len(para) + 1 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + " " + para
                else:
                    current_chunk = para
            else:
                while len(para) > chunk_size:
                    chunks.append(para[:chunk_size])
                    para = para[chunk_size - overlap:] if overlap > 0 else para[chunk_size:]
                current_chunk = para
        else:
            current_chunk = current_chunk + "\n" + para if current_chunk else para
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def heading_chunk(text: str, chunk_size: int = 800) -> List[str]:
    """
    按标题切分文本
    
    Args:
        text: 原始文本
        chunk_size: 每个 chunk 的最大字符数
    
    Returns:
        切分后的 chunk 列表
    """
    if not text or not text.strip():
        return []
    
    heading_pattern = r'^(#{1,6}\s+.+|第[一二三四五六七八九十百千]+[章节条款].+|\d+[\.\、].+|[一二三四五六七八九十]+[\.\、].+)'
    
    lines = text.split('\n')
    chunks = []
    current_chunk = ""
    current_heading = ""
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        is_heading = bool(re.match(heading_pattern, line_stripped))
        
        if is_heading:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_heading = line_stripped
            current_chunk = line_stripped
        else:
            if len(current_chunk) + len(line_stripped) + 1 > chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = current_heading + "\n" + line_stripped if current_heading else line_stripped
            else:
                current_chunk = current_chunk + "\n" + line_stripped if current_chunk else line_stripped
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def sliding_window(text: str, chunk_size: int = 600, overlap: int = 150) -> List[str]:
    """
    滑动窗口切分
    
    Args:
        text: 原始文本
        chunk_size: 每个 chunk 的最大字符数
        overlap: 重叠字符数
    
    Returns:
        切分后的 chunk 列表
    """
    if not text or not text.strip():
        return []
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            search_end = min(end + 100, len(text))
            best_break = end
            for i in range(end, search_end):
                if text[i] in '。！？.!?\n':
                    best_break = i + 1
                    break
            end = best_break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < len(text) else len(text)
    
    return chunks


def sentence_chunk(text: str, min_len: int = 100, max_len: int = 500) -> List[str]:
    """
    按句子切分文本
    
    Args:
        text: 原始文本
        min_len: chunk 最小长度
        max_len: chunk 最大长度
    
    Returns:
        切分后的 chunk 列表
    """
    if not text or not text.strip():
        return []
    
    sentences = re.split(r'([。！？.!?]+)', text)
    
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i]
        punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
        combined = (sentence + punctuation).strip()
        if combined:
            combined_sentences.append(combined)
    
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        combined_sentences.append(sentences[-1].strip())
    
    chunks = []
    current_chunk = ""
    
    for sentence in combined_sentences:
        if len(current_chunk) + len(sentence) + 1 > max_len:
            if current_chunk and len(current_chunk) >= min_len:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            elif current_chunk:
                current_chunk = current_chunk + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def paragraph_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    按段落切分（与原 split_text 相同）
    
    Args:
        text: 原始文本
        chunk_size: 每个 chunk 的最大字符数
        overlap: 重叠字符数
    
    Returns:
        切分后的 chunk 列表
    """
    return split_text(text, chunk_size, overlap)


def semantic_llm_chunk(text: str, max_chunk: int = 1000) -> List[str]:
    """
    使用 LLM 进行语义切分
    
    Args:
        text: 原始文本
        max_chunk: 每个 chunk 的最大字符数
    
    Returns:
        切分后的 chunk 列表
    """
    if not text or not text.strip():
        return []
    
    if len(text) <= max_chunk:
        return [text.strip()]
    
    try:
        client = get_chat_client()
        model = os.getenv("CHAT_MODEL", "deepseek-chat")
        
        system_prompt = """你是一个文本切分专家。请将用户提供的文本按语义完整性切分成多个片段。

要求：
1. 每个片段应该语义完整，能够独立理解
2. 每个片段长度控制在 200-800 字符之间
3. 保持原文内容，不要修改或总结
4. 用 "---CHUNK---" 作为片段之间的分隔符
5. 直接输出切分后的文本，不要添加任何解释"""

        if len(text) > 4000:
            coarse_chunks = sliding_window(text, chunk_size=2000, overlap=200)
            all_chunks = []
            for coarse in coarse_chunks:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": coarse}
                    ],
                    temperature=0.3,
                    max_tokens=4000
                )
                result = response.choices[0].message.content
                chunks = [c.strip() for c in result.split("---CHUNK---") if c.strip()]
                all_chunks.extend(chunks)
            return all_chunks
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            result = response.choices[0].message.content
            chunks = [c.strip() for c in result.split("---CHUNK---") if c.strip()]
            return chunks if chunks else [text.strip()]
    
    except Exception as e:
        print(f"[RAG] 语义切分失败，回退到滑动窗口: {e}")
        return sliding_window(text, chunk_size=max_chunk, overlap=100)


def split_text_by_strategy(text: str, strategy: str, params: Dict[str, Any]) -> List[str]:
    """
    根据策略切分文本的统一入口
    
    Args:
        text: 原始文本
        strategy: 切片策略名称
        params: 策略参数
    
    Returns:
        切分后的 chunk 列表
    """
    if strategy == "heading_chunk":
        return heading_chunk(text, chunk_size=params.get("chunk_size", 800))
    
    elif strategy == "sliding_window":
        return sliding_window(
            text, 
            chunk_size=params.get("chunk_size", 600),
            overlap=params.get("overlap", 150)
        )
    
    elif strategy == "sentence_chunk":
        return sentence_chunk(
            text,
            min_len=params.get("min_len", 100),
            max_len=params.get("max_len", 500)
        )
    
    elif strategy == "semantic_llm_chunk":
        return semantic_llm_chunk(text, max_chunk=params.get("max_chunk", 1000))
    
    elif strategy == "paragraph_chunk":
        return paragraph_chunk(
            text,
            chunk_size=params.get("chunk_size", 500),
            overlap=params.get("overlap", 50)
        )
    
    else:
        return split_text(text)
