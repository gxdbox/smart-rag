"""
侧边栏 UI 组件
包含模型配置、向量库状态、检索模式设置、查询优化选项等
"""

import streamlit as st
import os
from dotenv import load_dotenv

from rag_engine import (
    get_db_stats,
    get_bm25_stats,
    sync_bm25_from_vector_db,
    clear_vector_db,
    clear_bm25_index
)
from file_utils import get_supported_extensions
from chunk_strategy import get_strategy_description


def load_config():
    """加载配置"""
    load_dotenv()
    return {
        "embed_base_url": os.getenv("EMBED_BASE_URL", ""),
        "embed_api_key": os.getenv("EMBED_API_KEY", ""),
        "embed_model": os.getenv("EMBED_MODEL", ""),
        "chat_base_url": os.getenv("CHAT_BASE_URL", ""),
        "chat_api_key": os.getenv("CHAT_API_KEY", ""),
        "chat_model": os.getenv("CHAT_MODEL", "")
    }


def render_sidebar():
    """渲染侧边栏"""
    st.subheader("🔧 模型配置")
    
    config = load_config()
    
    # 显示当前配置状态
    with st.expander("Embedding 配置", expanded=True):
        st.text_input(
            "Embed Base URL",
            value=config["embed_base_url"],
            disabled=True
        )
        embed_key_display = "已配置 ✅" if config["embed_api_key"] else "未配置 ❌"
        st.text_input("Embed API Key", value=embed_key_display, disabled=True)
        st.text_input("Embed Model", value=config["embed_model"], disabled=True)
    
    with st.expander("Chat 配置", expanded=True):
        st.text_input(
            "Chat Base URL",
            value=config["chat_base_url"],
            disabled=True
        )
        chat_key_display = "已配置 ✅" if config["chat_api_key"] else "未配置 ❌"
        st.text_input("Chat API Key", value=chat_key_display, disabled=True)
        st.text_input("Chat Model", value=config["chat_model"], disabled=True)
    
    # 向量库状态
    st.subheader("📊 向量库状态")
    stats = get_db_stats()
    bm25_stats = get_bm25_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("向量库", stats["total_chunks"])
    with col2:
        st.metric("BM25 索引", bm25_stats["total_chunks"])
    
    # 检查同步状态
    if stats["total_chunks"] != bm25_stats["total_chunks"]:
        diff = abs(stats["total_chunks"] - bm25_stats["total_chunks"])
        st.warning(f"⚠️ 索引不同步（差异: {diff} 个文档）")
        if st.button("🔄 同步 BM25 索引", use_container_width=True, type="primary"):
            with st.spinner("正在从向量库同步到 BM25..."):
                synced_count = sync_bm25_from_vector_db()
            st.success(f"✅ 同步完成！已同步 {synced_count} 个文档")
            st.rerun()
    else:
        st.success("✅ 索引已同步")
    
    # 清空按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空向量库", use_container_width=True):
            clear_vector_db()
            st.success("向量库已清空！")
            st.rerun()
    with col2:
        if st.button("🗑️ 清空 BM25", use_container_width=True):
            clear_bm25_index()
            st.success("BM25 索引已清空！")
            st.rerun()
    
    # 检索模式选择
    st.subheader("🔍 检索模式")
    retrieval_mode = st.radio(
        "选择检索方式",
        ["向量检索", "BM25 检索", "混合检索", "Rerank 精排", "混合 + Rerank（最强）"],
        index=2,
        help="向量检索：语义理解\nBM25：精确匹配\n混合检索：综合最优\nRerank 精排：深度语义理解，准确率提升 20-30%\n混合 + Rerank：最强检索方案（推荐）"
    )
    st.session_state.retrieval_mode = retrieval_mode
    
    # 混合检索权重设置
    if retrieval_mode in ["混合检索", "混合 + Rerank（最强）"]:
        vector_weight = st.slider(
            "向量检索权重",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="权重越高，越依赖语义理解；权重越低，越依赖精确匹配"
        )
        st.session_state.vector_weight = vector_weight
        st.caption(f"BM25 权重: {1-vector_weight:.1f}")
    
    # Rerank 召回数量设置
    if retrieval_mode in ["Rerank 精排", "混合 + Rerank（最强）"]:
        recall_k = st.slider(
            "召回候选数量",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="第一阶段召回的候选数量，建议为最终结果数的 3-5 倍"
        )
        st.session_state.recall_k = recall_k
        st.info("💡 Rerank 模型首次使用时会自动下载，请耐心等待")
    
    # 自适应过滤选项
    if retrieval_mode in ["混合检索", "混合 + Rerank（最强）"]:
        enable_adaptive_filter = st.checkbox(
            "🎯 启用自适应过滤（动态阈值）",
            value=True,
            help="根据分数分布自动确定过滤阈值，避免盲目截断。推荐开启以提升召回质量。"
        )
        st.session_state.enable_adaptive_filter = enable_adaptive_filter
        
        if enable_adaptive_filter:
            st.caption("✅ 将使用肘部法则、断崖检测等策略动态过滤低质量结果")
        else:
            st.caption("⚠️ 将使用固定 Top-K 截断（可能引入噪声或遗漏高质量结果）")
    
    # 查询优化选项
    st.subheader("🔎 查询优化")
    
    enable_hyde = st.checkbox(
        "启用 HyDE",
        value=False,
        help="生成假设文档增强查询语义。特别适合模糊查询或信息不足的场景。"
    )
    st.session_state.enable_hyde = enable_hyde
    
    if enable_hyde:
        hyde_mode = st.radio(
            "HyDE 模式",
            ["standard", "enhanced"],
            index=1,
            format_func=lambda x: {
                "standard": "标准模式（纯 LLM 生成）",
                "enhanced": "增强模式（结合真实数据）⭐"
            }[x],
            help="标准：完全基于 LLM 知识；增强：先检索真实数据再生成"
        )
        st.session_state.hyde_mode = hyde_mode
    
    enable_multi_variant = st.checkbox(
        "启用多变体召回",
        value=False,
        help="生成同义词、语义扩展、不同表达方式等多种变体，最大化召回率。"
    )
    st.session_state.enable_multi_variant = enable_multi_variant
    
    if enable_multi_variant:
        recall_strategy = st.radio(
            "召回策略",
            ["aggressive", "balanced", "conservative"],
            index=1,
            format_func=lambda x: {
                "aggressive": "激进（最大召回）",
                "balanced": "平衡（推荐）",
                "conservative": "保守（优先精度）"
            }[x],
            help="激进：使用所有变体；平衡：使用部分变体；保守：只使用同义词"
        )
        st.session_state.recall_strategy = recall_strategy
    
    enable_query_expansion = st.checkbox(
        "启用查询扩展",
        value=False,
        help="将模糊查询扩展为多个具体查询，提高召回率和精度。适用于短查询、模糊查询。"
    )
    st.session_state.enable_query_expansion = enable_query_expansion
    
    enable_multi_step = st.checkbox(
        "启用多步骤检索",
        value=False,
        help="将复杂问题拆分为多个子问题，逐步检索并整合结果。适用于包含多个疑问的复杂问题。"
    )
    st.session_state.enable_multi_step = enable_multi_step
    
    if enable_hyde:
        st.caption("💡 HyDE：生成假设答案文档 → 用文档检索文档（语义更丰富）")
    
    if enable_multi_variant:
        st.caption("💡 多变体召回：'汽车修理' → 同义词+语义扩展+不同表达（提升召回率）")
    
    if enable_query_expansion:
        st.caption("💡 查询扩展：'产品' → 'RAG产品' | '检索增强生成产品'")
    
    if enable_multi_step:
        st.caption("💡 多步骤检索：'什么是RAG？它有什么优势？' → 拆分为2个子问题分别检索")
    
    # 当前切片策略
    if st.session_state.chunk_strategy:
        st.subheader("🔀 当前切片策略")
        strategy_desc = get_strategy_description(st.session_state.chunk_strategy)
        st.info(strategy_desc)
        if st.session_state.chunk_params:
            with st.expander("策略参数"):
                for key, value in st.session_state.chunk_params.items():
                    st.write(f"- **{key}**: {value}")
    
    # 支持的文件类型
    st.subheader("📁 支持的文件类型")
    for ext in get_supported_extensions():
        st.markdown(f"- `{ext}`")
