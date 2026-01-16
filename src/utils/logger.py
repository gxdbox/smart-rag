"""
日志系统模块
提供统一的日志配置和管理
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class RAGLogger:
    """RAG 系统日志管理器"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not RAGLogger._initialized:
            self._setup_logging()
            RAGLogger._initialized = True
    
    def _setup_logging(self):
        """配置日志系统"""
        # 创建日志目录
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # 日志文件路径
        log_file = log_dir / f"rag_{datetime.now().strftime('%Y%m%d')}.log"
        
        # 配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # 清除现有的处理器
        root_logger.handlers.clear()
        
        # 控制台处理器（INFO 及以上）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # 文件处理器（DEBUG 及以上）
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # 添加处理器
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        # 设置第三方库的日志级别
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('jieba').setLevel(logging.WARNING)
        logging.getLogger('paddleocr').setLevel(logging.WARNING)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        获取指定名称的日志记录器
        
        Args:
            name: 日志记录器名称（通常使用 __name__）
        
        Returns:
            日志记录器实例
        """
        return logging.getLogger(name)
    
    @staticmethod
    def set_level(level: str):
        """
        设置日志级别
        
        Args:
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {level}')
        logging.getLogger().setLevel(numeric_level)


# 初始化日志系统
_logger_instance = RAGLogger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    便捷函数：获取日志记录器
    
    Args:
        name: 日志记录器名称，默认为调用模块的 __name__
    
    Returns:
        日志记录器实例
    
    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'rag')
    
    return RAGLogger.get_logger(name)


def set_log_level(level: str):
    """
    便捷函数：设置全局日志级别
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    RAGLogger.set_level(level)
