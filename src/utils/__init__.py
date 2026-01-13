"""
工具模块
"""

from .file_utils import read_file, get_supported_extensions
from .ocr_utils import extract_text_from_image

__all__ = ['read_file', 'get_supported_extensions', 'extract_text_from_image']
