"""
RAG Web 应用主程序
基于 Streamlit 构建的问答界面
"""

import sys
import os

# 确保使用 UTF-8 编码
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import streamlit as st

# 导入 UI 模块
from src.ui import render_sidebar, render_file_upload, render_chat_interface


def init_session_state():
    """初始化 session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    if 'chunk_strategy' not in st.session_state:
        st.session_state.chunk_strategy = None
    if 'chunk_params' not in st.session_state:
        st.session_state.chunk_params = None
    if 'retrieval_mode' not in st.session_state:
        st.session_state.retrieval_mode = '混合检索'
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_contexts' not in st.session_state:
        st.session_state.current_contexts = []
    if 'last_retrieved_contexts' not in st.session_state:
        st.session_state.last_retrieved_contexts = []


def main():
    # 页面配置
    st.set_page_config(
        page_title="Web RAG Demo",
        page_icon="⚡",
        layout="wide"
    )
    
    init_session_state()
    
    # 主标题
    st.title("⚡ Web 版 RAG（支持国内大模型）")
    st.markdown("---")
    
    # 创建两列布局
    col_left, col_right = st.columns([1, 3])
    
    # 左侧栏：模型配置和检索设置
    with col_left:
        render_sidebar()
    
    # 右侧栏：主功能区
    with col_right:
        render_file_upload()
        render_chat_interface()


if __name__ == "__main__":
    main()
