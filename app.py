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
from dotenv import load_dotenv

# 导入自定义模块
from rag_engine import (
    load_env,
    split_text,
    split_text_by_strategy,
    embed_texts,
    add_to_vector_db,
    search_top_k,
    generate_answer,
    clear_vector_db,
    get_db_stats,
    add_to_bm25_index,
    search_bm25,
    hybrid_search,
    clear_bm25_index,
    get_bm25_stats,
    sync_bm25_from_vector_db
)
from file_utils import read_file, get_supported_extensions
from chunk_strategy import choose_chunk_strategy, get_strategy_description
from knowledge_graph import extract_knowledge_graph, format_graph_for_prompt, get_graph_stats


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
    
    # ===== 左侧栏：模型配置 =====
    with col_left:
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
            ["向量检索", "BM25 检索", "混合检索"],
            index=2,
            help="向量检索：语义理解\nBM25：精确匹配\n混合检索：综合最优（推荐）"
        )
        st.session_state.retrieval_mode = retrieval_mode
        
        # 混合检索权重设置
        if retrieval_mode == "混合检索":
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
    
    # ===== 右侧栏：主功能区 =====
    with col_right:
        # 文件上传区域
        st.subheader("📤 上传文件")
        
        uploaded_files = st.file_uploader(
            "选择要上传的文件（支持多选）",
            type=["txt", "pdf", "md", "markdown", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="上传的文件将自动切分并存入向量库（图片将通过 OCR 识别）"
        )
        
        if uploaded_files:
            load_env()  # 确保加载环境变量
            with st.spinner("正在处理上传的文件..."):
                for uploaded_file in uploaded_files:
                    try:
                        # 读取文件内容
                        file_content = uploaded_file.read()
                        text = read_file(uploaded_file.name, file_content)
                        
                        if text:
                            # 自动选择切片策略
                            strategy, params = choose_chunk_strategy(text)
                            st.session_state.chunk_strategy = strategy
                            st.session_state.chunk_params = params
                            
                            # 使用策略切分文本
                            chunks = split_text_by_strategy(text, strategy, params)
                            
                            if chunks:
                                # 生成 embeddings
                                embeddings = embed_texts(chunks)
                                
                                # 存入向量库
                                add_to_vector_db(chunks, embeddings)
                                
                                # 同步添加到 BM25 索引
                                add_to_bm25_index(chunks)
                                
                                strategy_desc = get_strategy_description(strategy)
                                st.success(f"✅ {uploaded_file.name}: 成功添加 {len(chunks)} 个 chunks（{strategy_desc}）")
                                st.info(f"📊 已同步到向量库和 BM25 索引")
                            else:
                                st.warning(f"⚠️ {uploaded_file.name}: 文件内容为空")
                        else:
                            st.error(f"❌ {uploaded_file.name}: 不支持的文件类型")
                    
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: 处理失败 - {str(e)}")
        
        # 导入预处理数据
        st.subheader("📥 导入预处理数据")
        imported_file = st.file_uploader(
            "导入 JSON 格式的 chunks",
            type=["json"],
            key="import_chunks_file",
            help="支持从 pdf-rag-pipeline 导出的 chunks_for_streamlit.json"
        )
        
        if imported_file:
            import json
            # 读取并缓存文件内容
            if 'imported_chunks' not in st.session_state or st.session_state.get('imported_file_name') != imported_file.name:
                try:
                    content = imported_file.read().decode('utf-8')
                    chunks = json.loads(content)
                    if isinstance(chunks, list) and chunks:
                        st.session_state.imported_chunks = chunks
                        st.session_state.imported_file_name = imported_file.name
                        st.info(f"📄 已加载 {len(chunks)} 个 chunks，点击下方按钮开始导入")
                    else:
                        st.error("❌ JSON 格式错误，需要是字符串列表")
                except Exception as e:
                    st.error(f"❌ 读取文件失败: {str(e)}")
            else:
                st.info(f"📄 已加载 {len(st.session_state.imported_chunks)} 个 chunks，点击下方按钮开始导入")
            
            if st.button("🚀 开始导入到向量库", key="start_import"):
                import time
                chunks = st.session_state.imported_chunks
                load_env()
                
                # 分批处理（实名认证后无限流）
                batch_size = 50
                total_added = 0
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i+batch_size]
                        status_text.text(f"处理中: {min(i+batch_size, len(chunks))}/{len(chunks)} chunks...")
                        
                        embeddings = embed_texts(batch)
                        add_to_vector_db(batch, embeddings)
                        add_to_bm25_index(batch)
                        total_added += len(batch)
                        
                        progress_bar.progress(min(total_added / len(chunks), 1.0))
                    
                    progress_bar.empty()
                    status_text.empty()
                    # 清除缓存
                    del st.session_state.imported_chunks
                    del st.session_state.imported_file_name
                    st.success(f"✅ 成功导入 {total_added} 个 chunks（已同步到向量库和 BM25 索引）")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 导入失败: {str(e)}（已导入 {total_added} 个）")
        
        st.markdown("---")
        
        # 问答区域
        st.subheader("💬 问答区域")
        
        # 用户问题输入
        user_query = st.text_area(
            "请输入您的问题",
            placeholder="例如：这个文档主要讲了什么内容？",
            height=100
        )
        
        # 检索数量设置
        top_k = st.slider("检索 Top-K 数量", min_value=1, max_value=10, value=3)
        
        # 知识图谱选项
        enable_kg = st.checkbox("🔗 启用知识图谱增强", value=False, 
                                help="对检索结果进行实时知识图谱抽取，提升复杂问题的回答质量")
        
        # 生成回答按钮
        if st.button("🚀 生成回答", type="primary", use_container_width=True):
            if not user_query.strip():
                st.warning("请先输入问题！")
            elif stats["total_chunks"] == 0:
                st.warning("向量库为空，请先上传文件！")
            else:
                try:
                    # 加载环境变量
                    load_env()
                    
                    with st.spinner("正在检索相关内容..."):
                        # 根据检索模式选择不同的检索方法
                        retrieval_mode = st.session_state.retrieval_mode
                        
                        if retrieval_mode == "向量检索":
                            retrieved = search_top_k(user_query, k=top_k)
                        elif retrieval_mode == "BM25 检索":
                            retrieved = search_bm25(user_query, k=top_k)
                        else:  # 混合检索
                            vector_weight = st.session_state.get('vector_weight', 0.5)
                            retrieved = hybrid_search(user_query, k=top_k, vector_weight=vector_weight)
                    
                    if not retrieved:
                        st.warning("未找到相关内容")
                    else:
                        # 显示检索结果
                        st.subheader("📚 检索到的知识片段")
                        
                        chunks_text = [chunk for chunk, score in retrieved]
                        
                        for i, (chunk, score) in enumerate(retrieved, 1):
                            score_label = "分数" if retrieval_mode == "BM25 检索" else "相似度"
                            with st.expander(f"片段 {i} ({score_label}: {score:.4f})", expanded=(i == 1)):
                                st.markdown(chunk)
                        
                        # 知识图谱抽取
                        kg_context = ""
                        if enable_kg:
                            try:
                                with st.spinner("正在抽取知识图谱..."):
                                    from openai import OpenAI
                                    chat_client = OpenAI(
                                        base_url=os.getenv("CHAT_BASE_URL"),
                                        api_key=os.getenv("CHAT_API_KEY")
                                    )
                                    graph = extract_knowledge_graph(
                                        chunks_text,
                                        chat_client,
                                        os.getenv("CHAT_MODEL", "deepseek-chat")
                                    )
                                    
                                    entities = graph.get("entities", [])
                                    relations = graph.get("relations", [])
                                    
                                    if entities or relations:
                                        kg_context = format_graph_for_prompt(graph)
                                        
                                        # 显示知识图谱
                                        st.subheader("🔗 抽取的知识图谱")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown("**实体**")
                                            for e in entities[:8]:
                                                if isinstance(e, dict):
                                                    st.markdown(f"- `{e.get('name', '')}` ({e.get('type', '')})")
                                        with col2:
                                            st.markdown("**关系**")
                                            for r in relations[:8]:
                                                if isinstance(r, dict):
                                                    st.markdown(f"- {r.get('source', '')} → {r.get('target', '')}")
                            except Exception as kg_error:
                                st.warning(f"知识图谱抽取失败: {kg_error}，继续使用普通检索")
                        
                        st.markdown("---")
                        
                        # 生成答案（带知识图谱上下文）
                        with st.spinner("正在生成答案..."):
                            if kg_context:
                                # 将知识图谱加入上下文
                                enhanced_retrieved = [(f"{chunk}\n\n{kg_context}", score) 
                                                     for chunk, score in retrieved[:1]]
                                enhanced_retrieved.extend(retrieved[1:])
                                answer = generate_answer(user_query, enhanced_retrieved)
                            else:
                                answer = generate_answer(user_query, retrieved)
                        
                        # 显示最终答案
                        st.subheader("✨ 最终回答")
                        st.markdown(answer)
                        
                        # 显示问题回顾
                        with st.expander("查看原问题"):
                            st.info(user_query)
                
                except ValueError as e:
                    st.error(f"配置错误: {str(e)}")
                except Exception as e:
                    st.error(f"生成答案时出错: {str(e)}")
        
        # 使用说明
        with st.expander("📖 使用说明"):
            st.markdown("""
            ### 使用步骤
            
            1. **配置环境变量**：复制 `.env.example` 为 `.env`，填入您的 API 配置
            2. **上传文件**：支持 txt、pdf、markdown 格式
            3. **输入问题**：在问答区域输入您想了解的问题
            4. **生成回答**：点击按钮，系统将检索相关内容并生成答案
            
            ### 支持的国内大模型
            
            - **DeepSeek**: `https://api.deepseek.com`
            - **Moonshot (Kimi)**: `https://api.moonshot.cn/v1`
            - **通义千问**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
            - **智谱 GLM4**: `https://open.bigmodel.cn/api/paas/v4`
            
            ### 注意事项
            
            - 请确保 API Key 配置正确
            - 文件上传后会自动进入向量库
            - 可以通过左侧栏清空向量库
            """)


if __name__ == "__main__":
    main()
