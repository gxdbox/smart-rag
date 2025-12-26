"""
文件处理工具模块
负责：读取并解析上传的文件（txt、pdf、markdown、图片）
"""

import io
from typing import Optional


def read_txt(file_content: bytes) -> str:
    """
    读取 txt 文件内容
    
    Args:
        file_content: 文件的二进制内容
    
    Returns:
        文本内容
    """
    # 尝试多种编码
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    
    for encoding in encodings:
        try:
            return file_content.decode(encoding)
        except UnicodeDecodeError:
            continue
    
    # 如果都失败，使用 utf-8 忽略错误
    return file_content.decode('utf-8', errors='ignore')


def read_pdf(file_content: bytes) -> str:
    """
    读取 PDF 文件内容
    
    解析策略：
    1. 优先使用 pypdf 提取文本（文本型 PDF）
    2. 若文本长度 < 50 字，则视为图片 PDF → 自动 OCR
    
    Args:
        file_content: 文件的二进制内容
    
    Returns:
        提取的文本内容
    """
    from pypdf import PdfReader
    
    try:
        pdf_file = io.BytesIO(file_content)
        reader = PdfReader(pdf_file)
        
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                # 清理文本，移除无法编码的字符
                page_text = page_text.encode('utf-8', errors='ignore').decode('utf-8')
                text_parts.append(page_text)
        
        text = "\n\n".join(text_parts)
        
        # 判断是否为图片 PDF（文本太短）
        if len(text.strip()) >= 50:
            return text
        
        # 文本太短 → 图片 PDF → 自动 OCR
        print("[PDF] 检测到图片型 PDF，启用 OCR...")
        try:
            from ocr_utils import pdf_to_text_with_ocr
            ocr_text = pdf_to_text_with_ocr(file_content)
            return ocr_text
        except Exception as ocr_e:
            print(f"[PDF] OCR 失败: {ocr_e}")
            raise RuntimeError(f"图片 PDF OCR 失败: {ocr_e}")
    
    except Exception as e:
        # pypdf 失败，强制走 OCR
        print(f"[PDF] pypdf 解析失败 ({e})，尝试 OCR...")
        try:
            from ocr_utils import pdf_to_text_with_ocr
            return pdf_to_text_with_ocr(file_content)
        except Exception as ocr_e:
            print(f"[PDF] OCR 也失败了: {ocr_e}")
            raise RuntimeError(f"PDF 解析失败: {ocr_e}")


def read_markdown(file_content: bytes) -> str:
    """
    读取 Markdown 文件内容
    
    Args:
        file_content: 文件的二进制内容
    
    Returns:
        文本内容（保留 Markdown 格式）
    """
    # Markdown 本质上就是文本，直接读取
    return read_txt(file_content)


def read_image(file_content: bytes) -> str:
    """
    读取图片文件内容（通过 OCR）
    
    Args:
        file_content: 图片的二进制内容
    
    Returns:
        OCR 识别出的文本内容
    """
    from ocr_utils import ocr_image_bytes
    
    try:
        text = ocr_image_bytes(file_content)
        return text.strip()
    except Exception as e:
        print(f"[Image] OCR 失败: {e}")
        raise RuntimeError(f"图片 OCR 失败: {e}")


def read_file(file_name: str, file_content: bytes) -> Optional[str]:
    """
    根据文件扩展名读取文件内容
    
    Args:
        file_name: 文件名
        file_content: 文件的二进制内容
    
    Returns:
        提取的文本内容，如果不支持该文件类型则返回 None
    """
    file_name_lower = file_name.lower()
    
    if file_name_lower.endswith('.txt'):
        return read_txt(file_content)
    elif file_name_lower.endswith('.pdf'):
        return read_pdf(file_content)
    elif file_name_lower.endswith('.md') or file_name_lower.endswith('.markdown'):
        return read_markdown(file_content)
    elif file_name_lower.endswith(('.jpg', '.jpeg', '.png')):
        return read_image(file_content)
    else:
        return None


def get_supported_extensions() -> list:
    """获取支持的文件扩展名列表"""
    return ['.txt', '.pdf', '.md', '.markdown', '.jpg', '.jpeg', '.png']
