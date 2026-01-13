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
    sync_bm25_from_vector_db,
    search_with_rerank,
    hybrid_search_with_rerank
)
from file_utils import read_file, get_supported_extensions
from chunk_strategy import choose_chunk_strategy, get_strategy_description
from knowledge_graph import extract_knowledge_graph, format_graph_for_prompt, get_graph_stats
from query_rewriter import QueryRewriter
from topic_extractor import TopicExtractor
from query_expansion import QueryExpander, multi_query_retrieval
from multi_step_query import MultiStepQueryEngine
from hyde import HyDERetriever
from multi_variant_recall import MultiVariantRecaller


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
        st.session_state.last_retrieved_contexts = []  # 缓存上次检索结果


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
        
        # 对话历史展示区
        if st.session_state.conversation_history:
            st.subheader("💬 对话历史")
            
            # 显示对话轮数
            num_turns = len(st.session_state.conversation_history) // 2
            st.caption(f"共 {num_turns} 轮对话")
            
            # 展示对话历史
            with st.expander("查看完整对话历史", expanded=False):
                for i, msg in enumerate(st.session_state.conversation_history):
                    if msg["role"] == "user":
                        st.markdown(f"**🙋 用户**: {msg['content']}")
                    else:
                        st.markdown(f"**🤖 助手**: {msg['content']}")
                    if i < len(st.session_state.conversation_history) - 1:
                        st.markdown("---")
            
            # 对话管理按钮
            col1, col2, col3 = st.columns([2, 1, 1])
            with col2:
                # 导出对话为 Markdown
                if st.button("📥 导出对话", use_container_width=True):
                    import json
                    from datetime import datetime
                    
                    # 生成 Markdown 格式
                    md_content = f"# 对话记录\n\n**导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
                    for i, msg in enumerate(st.session_state.conversation_history):
                        if msg["role"] == "user":
                            md_content += f"## 🙋 用户\n\n{msg['content']}\n\n"
                        else:
                            md_content += f"## 🤖 助手\n\n{msg['content']}\n\n---\n\n"
                    
                    st.download_button(
                        label="💾 下载 Markdown",
                        data=md_content,
                        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
            with col3:
                if st.button("🗑️ 清空对话", use_container_width=True):
                    st.session_state.conversation_history = []
                    st.success("对话历史已清空！")
                    st.rerun()
            
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
                    
                    # 智能检索决策（使用 LLM 判断）
                    should_retrieve = True
                    actual_query = user_query
                    
                    # 如果有对话历史和缓存，使用 LLM 判断是否需要重新检索
                    if st.session_state.conversation_history and len(st.session_state.last_retrieved_contexts) > 0:
                        from openai import OpenAI
                        chat_client = OpenAI(
                            base_url=os.getenv("CHAT_BASE_URL"),
                            api_key=os.getenv("CHAT_API_KEY")
                        )
                        
                        # 构建判断 Prompt
                        last_turn = st.session_state.conversation_history[-2:] if len(st.session_state.conversation_history) >= 2 else st.session_state.conversation_history
                        history_text = "\n".join([f"{msg['role']}: {msg['content'][:100]}" for msg in last_turn])
                        
                        decision_prompt = f"""你是一个检索决策助手。请判断用户的新问题是否需要重新检索。

对话历史：
{history_text}

当前问题：{user_query}

判断规则：
1. 如果当前问题是对上一轮话题的追问、深入讨论、举例说明，输出：REUSE
2. 如果当前问题是全新的话题，与之前无关，输出：RETRIEVE
3. 如果不确定，优先输出：REUSE（保持话题连贯）

只输出一个词：REUSE 或 RETRIEVE"""

                        try:
                            with st.spinner("正在分析问题..."):
                                response = chat_client.chat.completions.create(
                                    model=os.getenv("CHAT_MODEL", "deepseek-chat"),
                                    messages=[{"role": "user", "content": decision_prompt}],
                                    temperature=0.0,
                                    max_tokens=10
                                )
                                decision = response.choices[0].message.content.strip().upper()
                                
                                if "REUSE" in decision:
                                    should_retrieve = False
                                    retrieved = st.session_state.last_retrieved_contexts
                                    st.success("💡 检测到追问，复用上次检索的内容（保持话题连贯）")
                        except Exception as e:
                            print(f"[检索决策] LLM 判断失败: {e}，默认重新检索")
                    
                    if should_retrieve:
                        # 需要重新检索
                        # 主题提取 + 上下文注入（根本解决方案）
                        if st.session_state.conversation_history:
                            from openai import OpenAI
                            chat_client = OpenAI(
                                base_url=os.getenv("CHAT_BASE_URL"),
                                api_key=os.getenv("CHAT_API_KEY")
                            )
                            
                            # 提取当前话题
                            extractor = TopicExtractor(chat_client, os.getenv("CHAT_MODEL", "deepseek-chat"))
                            topic = extractor.extract_topic(st.session_state.conversation_history)
                            
                            if topic:
                                # 将话题注入到查询中
                                actual_query = extractor.enhance_query_with_topic(user_query, topic)
                                
                                # 显示增强结果
                                if actual_query != user_query:
                                    st.info(f"💡 基于话题「{topic}」增强查询：{actual_query}")
                            else:
                                # 降级到查询重写
                                rewriter = QueryRewriter(chat_client, os.getenv("CHAT_MODEL", "deepseek-chat"))
                                if rewriter.needs_rewrite(user_query, st.session_state.conversation_history):
                                    with st.spinner("正在理解您的问题..."):
                                        actual_query = rewriter.rewrite(user_query, st.session_state.conversation_history)
                                    if actual_query != user_query:
                                        st.info(f"💡 理解为：{actual_query}")
                        
                        with st.spinner("正在检索相关内容..."):
                            # 检查是否启用 HyDE、多步骤检索或查询扩展
                            enable_hyde = st.session_state.get('enable_hyde', False)
                            enable_multi_step = st.session_state.get('enable_multi_step', False)
                            enable_expansion = st.session_state.get('enable_query_expansion', False)
                            
                            if enable_hyde:
                                # 使用 HyDE 检索（优先级最高）
                                from openai import OpenAI
                                chat_client = OpenAI(
                                    base_url=os.getenv("CHAT_BASE_URL"),
                                    api_key=os.getenv("CHAT_API_KEY")
                                )
                                
                                hyde_mode = st.session_state.get('hyde_mode', 'standard')
                                retrieval_mode = st.session_state.retrieval_mode
                                
                                # 增强模式：先检索真实数据
                                reference_context = None
                                if hyde_mode == 'enhanced':
                                    st.info("🔍 第一阶段：检索真实数据...")
                                    # 初步检索（获取真实数据）
                                    if retrieval_mode == "向量检索":
                                        initial_results = search_top_k(actual_query, k=3)
                                    elif retrieval_mode == "BM25 检索":
                                        initial_results = search_bm25(actual_query, k=3)
                                    elif retrieval_mode == "混合检索":
                                        vector_weight = st.session_state.get('vector_weight', 0.5)
                                        use_adaptive = st.session_state.get('enable_adaptive_filter', True)
                                        initial_results = hybrid_search(actual_query, k=3, vector_weight=vector_weight, use_adaptive_filter=use_adaptive)
                                    else:  # Rerank 或混合+Rerank
                                        initial_results = search_top_k(actual_query, k=3)
                                    
                                    # 提取参考上下文（取前150字）
                                    if initial_results:
                                        reference_snippets = [chunk[:150] for chunk, _ in initial_results[:2]]
                                        reference_context = "\n\n".join(reference_snippets)
                                        
                                        with st.expander("📚 查看参考数据（用于增强 HyDE）"):
                                            st.text(reference_context)
                                
                                hyde_retriever = HyDERetriever(chat_client, os.getenv("CHAT_MODEL", "deepseek-chat"))
                                
                                # 生成假设文档（可能包含参考上下文）
                                st.info("🔮 第二阶段：生成假设文档...")
                                hypothetical_doc = hyde_retriever.generate_hypothetical_document(
                                    actual_query,
                                    st.session_state.conversation_history,
                                    reference_context=reference_context
                                )
                                
                                # 显示假设文档
                                with st.expander("🔮 查看假设文档（HyDE）"):
                                    st.info(hypothetical_doc)
                                    if hyde_mode == 'enhanced':
                                        st.success("✅ 此假设文档基于真实数据生成，准确度更高")
                                
                                # 使用假设文档进行检索（二次检索）
                                st.info("🎯 第三阶段：使用假设文档检索...")
                                
                                if retrieval_mode == "向量检索":
                                    retrieved = search_top_k(hypothetical_doc, k=top_k)
                                elif retrieval_mode == "BM25 检索":
                                    retrieved = search_bm25(hypothetical_doc, k=top_k)
                                elif retrieval_mode == "混合检索":
                                    vector_weight = st.session_state.get('vector_weight', 0.5)
                                    use_adaptive = st.session_state.get('enable_adaptive_filter', True)
                                    retrieved = hybrid_search(hypothetical_doc, k=top_k, vector_weight=vector_weight, use_adaptive_filter=use_adaptive)
                                elif retrieval_mode == "Rerank 精排":
                                    recall_k = st.session_state.get('recall_k', 20)
                                    retrieved = search_with_rerank(hypothetical_doc, k=top_k, recall_k=recall_k)
                                else:  # 混合 + Rerank（最强）
                                    vector_weight = st.session_state.get('vector_weight', 0.5)
                                    recall_k = st.session_state.get('recall_k', 20)
                                    use_adaptive = st.session_state.get('enable_adaptive_filter', True)
                                    retrieved = hybrid_search_with_rerank(
                                        hypothetical_doc, k=top_k, vector_weight=vector_weight, recall_k=recall_k, use_adaptive_filter=use_adaptive
                                    )
                            
                            elif enable_multi_variant:
                                # 使用多变体召回
                                from openai import OpenAI
                                chat_client = OpenAI(
                                    base_url=os.getenv("CHAT_BASE_URL"),
                                    api_key=os.getenv("CHAT_API_KEY")
                                )
                                recaller = MultiVariantRecaller(chat_client, os.getenv("CHAT_MODEL", "deepseek-chat"))
                                
                                # 定义检索函数
                                def search_func(query: str, k: int):
                                    retrieval_mode = st.session_state.retrieval_mode
                                    if retrieval_mode == "向量检索":
                                        return search_top_k(query, k=k)
                                    elif retrieval_mode == "BM25 检索":
                                        return search_bm25(query, k=k)
                                    elif retrieval_mode == "混合检索":
                                        vector_weight = st.session_state.get('vector_weight', 0.5)
                                        use_adaptive = st.session_state.get('enable_adaptive_filter', True)
                                        return hybrid_search(query, k=k, vector_weight=vector_weight, use_adaptive_filter=use_adaptive)
                                    elif retrieval_mode == "Rerank 精排":
                                        recall_k = st.session_state.get('recall_k', 20)
                                        return search_with_rerank(query, k=k, recall_k=recall_k)
                                    else:  # 混合 + Rerank（最强）
                                        vector_weight = st.session_state.get('vector_weight', 0.5)
                                        recall_k = st.session_state.get('recall_k', 20)
                                        use_adaptive = st.session_state.get('enable_adaptive_filter', True)
                                        return hybrid_search_with_rerank(query, k=k, vector_weight=vector_weight, recall_k=recall_k, use_adaptive_filter=use_adaptive)
                                
                                # 多变体召回
                                strategy = st.session_state.get('recall_strategy', 'balanced')
                                retrieved = recaller.multi_variant_search(
                                    actual_query,
                                    search_func,
                                    conversation_history=st.session_state.conversation_history,
                                    k=top_k,
                                    strategy=strategy
                                )
                            
                            elif enable_multi_step:
                                # 使用多步骤检索（优先级最高）
                                from openai import OpenAI
                                chat_client = OpenAI(
                                    base_url=os.getenv("CHAT_BASE_URL"),
                                    api_key=os.getenv("CHAT_API_KEY")
                                )
                                multi_step_engine = MultiStepQueryEngine(chat_client, os.getenv("CHAT_MODEL", "deepseek-chat"))
                                
                                # 定义检索函数
                                def search_func(query: str, k: int):
                                    retrieval_mode = st.session_state.retrieval_mode
                                    if retrieval_mode == "向量检索":
                                        return search_top_k(query, k=k)
                                    elif retrieval_mode == "BM25 检索":
                                        return search_bm25(query, k=k)
                                    elif retrieval_mode == "混合检索":
                                        vector_weight = st.session_state.get('vector_weight', 0.5)
                                        use_adaptive = st.session_state.get('enable_adaptive_filter', True)
                                        return hybrid_search(query, k=k, vector_weight=vector_weight, use_adaptive_filter=use_adaptive)
                                    elif retrieval_mode == "Rerank 精排":
                                        recall_k = st.session_state.get('recall_k', 20)
                                        return search_with_rerank(query, k=k, recall_k=recall_k)
                                    else:  # 混合 + Rerank（最强）
                                        vector_weight = st.session_state.get('vector_weight', 0.5)
                                        recall_k = st.session_state.get('recall_k', 20)
                                        use_adaptive = st.session_state.get('enable_adaptive_filter', True)
                                        return hybrid_search_with_rerank(query, k=k, vector_weight=vector_weight, recall_k=recall_k, use_adaptive_filter=use_adaptive)
                                
                                # 多步骤检索
                                retrieved = multi_step_engine.multi_step_retrieve(
                                    actual_query, 
                                    search_func,
                                    k_per_query=2  # 每个子问题检索2个文档
                                )
                                
                                # 限制总数
                                retrieved = retrieved[:top_k]
                            
                            elif enable_expansion:
                                # 使用查询扩展进行多查询检索
                                from openai import OpenAI
                                chat_client = OpenAI(
                                    base_url=os.getenv("CHAT_BASE_URL"),
                                    api_key=os.getenv("CHAT_API_KEY")
                                )
                                expander = QueryExpander(chat_client, os.getenv("CHAT_MODEL", "deepseek-chat"))
                                
                                # 扩展查询
                                query_variants = expander.expand(
                                    actual_query, 
                                    st.session_state.conversation_history,
                                    num_variants=2
                                )
                                
                                # 显示扩展结果
                                if len(query_variants) > 1:
                                    st.info(f"🔎 查询扩展：{' | '.join(query_variants)}")
                                
                                # 多查询检索并融合结果
                                all_results = {}
                                retrieval_mode = st.session_state.retrieval_mode
                                
                                for variant in query_variants:
                                    if retrieval_mode == "向量检索":
                                        results = search_top_k(variant, k=top_k*2)
                                    elif retrieval_mode == "BM25 检索":
                                        results = search_bm25(variant, k=top_k*2)
                                    elif retrieval_mode == "混合检索":
                                        vector_weight = st.session_state.get('vector_weight', 0.5)
                                        use_adaptive = st.session_state.get('enable_adaptive_filter', True)
                                        results = hybrid_search(variant, k=top_k*2, vector_weight=vector_weight, use_adaptive_filter=use_adaptive)
                                    elif retrieval_mode == "Rerank 精排":
                                        recall_k = st.session_state.get('recall_k', 20)
                                        results = search_with_rerank(variant, k=top_k*2, recall_k=recall_k)
                                    else:  # 混合 + Rerank（最强）
                                        vector_weight = st.session_state.get('vector_weight', 0.5)
                                        recall_k = st.session_state.get('recall_k', 20)
                                        use_adaptive = st.session_state.get('enable_adaptive_filter', True)
                                        results = hybrid_search_with_rerank(
                                            variant, k=top_k*2, vector_weight=vector_weight, recall_k=recall_k, use_adaptive_filter=use_adaptive
                                        )
                                    
                                    # 融合结果（保留最高分数）
                                    for chunk, score in results:
                                        if chunk not in all_results or score > all_results[chunk]:
                                            all_results[chunk] = score
                                
                                # 排序并返回 top-k
                                retrieved = sorted(all_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
                            else:
                                # 标准检索（不使用查询扩展）
                                retrieval_mode = st.session_state.retrieval_mode
                                
                                if retrieval_mode == "向量检索":
                                    retrieved = search_top_k(actual_query, k=top_k)
                                elif retrieval_mode == "BM25 检索":
                                    retrieved = search_bm25(actual_query, k=top_k)
                                elif retrieval_mode == "混合检索":
                                    vector_weight = st.session_state.get('vector_weight', 0.5)
                                    use_adaptive = st.session_state.get('enable_adaptive_filter', True)
                                    retrieved = hybrid_search(actual_query, k=top_k, vector_weight=vector_weight, use_adaptive_filter=use_adaptive)
                                elif retrieval_mode == "Rerank 精排":
                                    recall_k = st.session_state.get('recall_k', 20)
                                    retrieved = search_with_rerank(actual_query, k=top_k, recall_k=recall_k)
                                else:  # 混合 + Rerank（最强）
                                    vector_weight = st.session_state.get('vector_weight', 0.5)
                                    recall_k = st.session_state.get('recall_k', 20)
                                    use_adaptive = st.session_state.get('enable_adaptive_filter', True)
                                    retrieved = hybrid_search_with_rerank(
                                        actual_query, k=top_k, vector_weight=vector_weight, recall_k=recall_k, use_adaptive_filter=use_adaptive
                                    )
                            
                            # 缓存检索结果
                            st.session_state.last_retrieved_contexts = retrieved
                    
                    if not retrieved:
                        st.warning("未找到相关内容")
                    else:
                        # 相关性过滤（关键改进）
                        relevance_threshold = 0.5  # 相似度阈值
                        relevant_results = [(chunk, score) for chunk, score in retrieved if score >= relevance_threshold]
                        
                        if not relevant_results:
                            # 所有结果相似度都太低，不使用检索结果
                            st.warning(f"⚠️ 检索到的内容相关性较低（最高相似度：{retrieved[0][1]:.2f}），将基于对话历史直接回答")
                            
                            # 降级：不使用检索结果，只基于对话历史生成答案
                            with st.spinner("正在生成答案..."):
                                from openai import OpenAI
                                chat_client = OpenAI(
                                    base_url=os.getenv("CHAT_BASE_URL"),
                                    api_key=os.getenv("CHAT_API_KEY")
                                )
                                
                                # 构建不依赖检索的 Prompt
                                messages = [{"role": "system", "content": "你是一个专业的知识问答助手。"}]
                                if st.session_state.conversation_history:
                                    messages.extend(st.session_state.conversation_history[-6:])
                                messages.append({"role": "user", "content": user_query})
                                
                                response = chat_client.chat.completions.create(
                                    model=os.getenv("CHAT_MODEL", "deepseek-chat"),
                                    messages=messages,
                                    temperature=0.7,
                                    max_tokens=2000
                                )
                                answer = response.choices[0].message.content
                            
                            # 保存对话
                            st.session_state.conversation_history.append({"role": "user", "content": user_query})
                            st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                            if len(st.session_state.conversation_history) > 20:
                                st.session_state.conversation_history = st.session_state.conversation_history[-20:]
                            
                            # 显示答案
                            st.subheader("✨ 最终回答")
                            st.markdown(answer)
                            
                            num_turns = len(st.session_state.conversation_history) // 2
                            if num_turns > 0:
                                st.info(f"💬 当前对话已进行 {num_turns} 轮，您可以继续追问相关问题")
                            
                            with st.expander("查看原问题"):
                                st.info(user_query)
                        else:
                            # 有相关结果，使用过滤后的结果
                            original_count = len(retrieved)
                            retrieved = relevant_results
                            if len(relevant_results) < original_count:
                                st.info(f"📊 过滤掉 {original_count - len(relevant_results)} 个低相关性结果，保留 {len(relevant_results)} 个高质量结果")
                            
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
                        # 生成答案（流式输出）
                        st.subheader("✨ 最终回答")
                        answer_placeholder = st.empty()
                        full_answer = ""
                        
                        # 准备检索结果
                        if enable_kg and kg_context:
                            # 将知识图谱加入上下文
                            enhanced_retrieved = [(f"{chunk}\n\n{kg_context}", score) 
                                                 for chunk, score in retrieved[:1]]
                            enhanced_retrieved.extend(retrieved[1:])
                            final_retrieved = enhanced_retrieved
                        else:
                            final_retrieved = retrieved
                        
                        # 使用流式生成
                        from rag.retriever.base import Document
                        from rag.generator.llm_generator import LLMGenerator
                        from openai import OpenAI
                        
                        chat_client = OpenAI(
                            base_url=os.getenv("CHAT_BASE_URL"),
                            api_key=os.getenv("CHAT_API_KEY")
                        )
                        
                        # 转换为 Document 格式
                        ranked_docs = [
                            (Document(id=i, text=chunk, metadata={}), score)
                            for i, (chunk, score) in enumerate(final_retrieved)
                        ]
                        
                        generator = LLMGenerator(chat_client, os.getenv("CHAT_MODEL", "deepseek-chat"))
                        
                        # 流式生成答案
                        for chunk in generator.generate_stream(
                            user_query, 
                            ranked_docs,
                            conversation_history=st.session_state.conversation_history
                        ):
                            full_answer += chunk
                            answer_placeholder.markdown(full_answer + "▌")
                        
                        # 移除光标
                        answer_placeholder.markdown(full_answer)
                        
                        # 保存对话到历史
                        st.session_state.conversation_history.append({
                            "role": "user",
                            "content": user_query
                        })
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": full_answer
                        })
                        
                        # 限制历史轮数（保留最近10轮，即20条消息）
                        if len(st.session_state.conversation_history) > 20:
                            st.session_state.conversation_history = st.session_state.conversation_history[-20:]
                        
                        # 显示对话轮数提示
                        num_turns = len(st.session_state.conversation_history) // 2
                        if num_turns > 0:
                            st.info(f"💬 当前对话已进行 {num_turns} 轮，您可以继续追问相关问题")
                        
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
