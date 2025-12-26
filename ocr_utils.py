"""
OCR 工具模块
负责：对图片型 PDF 和图片进行 OCR 文字识别
"""

import io
import os
from typing import Optional

# 延迟导入，避免启动时加载 PaddleOCR
_ocr_instance = None


def get_ocr():
    """
    获取 PaddleOCR 实例（单例模式，延迟加载）
    """
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        # 使用 PP-OCRv4 模型，更稳定
        _ocr_instance = PaddleOCR(lang="ch")
    return _ocr_instance


def pdf_page_to_image(page) -> bytes:
    """
    将 PDF 页面转换为 PNG 图像（二进制）
    
    Args:
        page: PyMuPDF 的页面对象
    
    Returns:
        PNG 图像的二进制数据
    """
    import fitz
    # 使用 2x 缩放获得高清图像
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    return pix.tobytes("png")


def ocr_image_bytes(image_bytes: bytes) -> str:
    """
    对单张图片（bytes）执行 OCR，返回识别文本
    
    Args:
        image_bytes: 图片的二进制数据
    
    Returns:
        识别出的文本
    """
    import tempfile
    from PIL import Image
    
    # 将 bytes 保存为临时文件（PaddleOCR 对文件路径支持更稳定）
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        # 转换为 PNG 格式保存
        image = Image.open(io.BytesIO(image_bytes))
        # 确保是 RGB 模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(tmp_path, format='PNG')
    
    try:
        ocr = get_ocr()
        # 使用文件路径调用 predict
        results = ocr.predict(tmp_path)
        
        text = ""
        # 新版返回格式: [{'rec_texts': [...], ...}]
        if results and len(results) > 0:
            result = results[0]
            if isinstance(result, dict) and 'rec_texts' in result:
                # 新版 API 格式
                texts = result.get('rec_texts', [])
                text = '\n'.join(texts)
            elif isinstance(result, list):
                # 兼容旧版格式
                for line in result:
                    if line and len(line) >= 2:
                        text += line[1][0] + "\n"
        
        return text
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def pdf_to_text_with_ocr(pdf_content: bytes) -> str:
    """
    对整份 PDF 执行 OCR（图片 PDF 专用）
    
    Args:
        pdf_content: PDF 文件的二进制内容
    
    Returns:
        OCR 识别出的全部文本
    """
    import fitz
    
    # 从二进制内容打开 PDF
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    all_text = ""
    
    for page_num, page in enumerate(doc):
        try:
            img_bytes = pdf_page_to_image(page)
            page_text = ocr_image_bytes(img_bytes)
            all_text += page_text + "\n"
        except Exception as e:
            print(f"[OCR] 第 {page_num + 1} 页 OCR 失败: {e}")
            continue
    
    doc.close()
    return all_text.strip()
