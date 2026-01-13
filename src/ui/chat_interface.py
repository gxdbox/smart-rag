"""
对话界面 UI 组件
包含对话历史展示、问答区域、检索和生成逻辑
"""

import streamlit as st
import os
from datetime import datetime
from openai import OpenAI

from rag_engine import (
    load_env,
    search_top_k,
    search_bm25,
    hybrid_search,
    search_with_rerank,
    hybrid_search_with_rerank,
    generate_answer,
    get_db_stats
)
from knowledge_graph import extract_knowledge_graph, format_graph_for_prompt
from query_rewriter import QueryRewriter
from topic_extractor import TopicExtractor
from query_expansion import QueryExpander, multi_query_retrieval
from multi_step_query import MultiStepQueryEngine
from hyde import HyDERetriever
from multi_variant_recall import MultiVariantRecaller


def render_conversation_history():
    """渲染对话历史"""
    if st.session_state.conversation_history:
        st.subheader("💬 对话历史")
        
        num_turns = len(st.session_state.conversation_history) // 2
        st.caption(f"共 {num_turns} 轮对话")
        
        with st.expander("查看完整对话历史", expanded=False):
            for i, msg in enumerate(st.session_state.conversation_history):
                if msg["role"] == "user":
                    st.markdown(f"**🙋 用户**: {msg['content']}")
                else:
                    st.markdown(f"**🤖 助手**: {msg['content']}")
                if i < len(st.session_state.conversation_history) - 1:
                    st.markdown("---")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            if st.button("📥 导出对话", use_container_width=True):
                md_content = f"# 对话记录\n\n**导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
                for msg in st.session_state.conversation_history:
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


def perform_retrieval(actual_query, retrieval_mode, top_k):
    """执行检索（统一入口）"""
    if retrieval_mode == "向量检索":
        return search_top_k(actual_query, k=top_k)
    elif retrieval_mode == "BM25 检索":
        return search_bm25(actual_query, k=top_k)
    elif retrieval_mode == "混合检索":
        vector_weight = st.session_state.get('vector_weight', 0.5)
        use_adaptive = st.session_state.get('enable_adaptive_filter', True)
        return hybrid_search(actual_query, k=top_k, vector_weight=vector_weight, use_adaptive_filter=use_adaptive)
    elif retrieval_mode == "Rerank 精排":
        recall_k = st.session_state.get('recall_k', 20)
        return search_with_rerank(actual_query, k=top_k, recall_k=recall_k)
    else:  # 混合 + Rerank（最强）
        vector_weight = st.session_state.get('vector_weight', 0.5)
        recall_k = st.session_state.get('recall_k', 20)
        use_adaptive = st.session_state.get('enable_adaptive_filter', True)
        return hybrid_search_with_rerank(
            actual_query, k=top_k, vector_weight=vector_weight, 
            recall_k=recall_k, use_adaptive_filter=use_adaptive
        )


def render_chat_interface():
    """渲染对话界面"""
    render_conversation_history()
    
    st.subheader("💬 问答区域")
    
    user_query = st.text_area(
        "请输入您的问题",
        placeholder="例如：这个文档主要讲了什么内容？",
        height=100
    )
    
    top_k = st.slider("检索 Top-K 数量", min_value=1, max_value=10, value=3)
    
    enable_kg = st.checkbox("🔗 启用知识图谱增强", value=False, 
                            help="对检索结果进行实时知识图谱抽取，提升复杂问题的回答质量")
    
    if st.button("🚀 生成回答", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("请先输入问题！")
            return
        
        stats = get_db_stats()
        if stats["total_chunks"] == 0:
            st.warning("向量库为空，请先上传文件！")
            return
        
        try:
            load_env()
            
            # 智能检索决策
            should_retrieve = True
            actual_query = user_query
            
            if st.session_state.conversation_history and len(st.session_state.last_retrieved_contexts) > 0:
                chat_client = OpenAI(
                    base_url=os.getenv("CHAT_BASE_URL"),
                    api_key=os.getenv("CHAT_API_KEY")
                )
                
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
                # 查询增强
                if st.session_state.conversation_history:
                    chat_client = OpenAI(
                        base_url=os.getenv("CHAT_BASE_URL"),
                        api_key=os.getenv("CHAT_API_KEY")
                    )
                    
                    extractor = TopicExtractor(chat_client, os.getenv("CHAT_MODEL", "deepseek-chat"))
                    topic = extractor.extract_topic(st.session_state.conversation_history)
                    
                    if topic:
                        actual_query = extractor.enhance_query_with_topic(user_query, topic)
                        if actual_query != user_query:
                            st.info(f"💡 基于话题「{topic}」增强查询：{actual_query}")
                    else:
                        rewriter = QueryRewriter(chat_client, os.getenv("CHAT_MODEL", "deepseek-chat"))
                        if rewriter.needs_rewrite(user_query, st.session_state.conversation_history):
                            with st.spinner("正在理解您的问题..."):
                                actual_query = rewriter.rewrite(user_query, st.session_state.conversation_history)
                            if actual_query != user_query:
                                st.info(f"💡 理解为：{actual_query}")
                
                # 执行检索（根据查询优化选项）
                with st.spinner("正在检索相关内容..."):
                    retrieval_mode = st.session_state.retrieval_mode
                    
                    # 注意：这里只实现了基础检索，完整的 HyDE/多变体等逻辑保留在 app.py
                    # 后续可以继续重构
                    retrieved = perform_retrieval(actual_query, retrieval_mode, top_k)
                
                st.session_state.last_retrieved_contexts = retrieved
            
            # 显示检索结果
            with st.expander("📚 检索到的相关内容", expanded=True):
                for i, (chunk, score) in enumerate(retrieved, 1):
                    st.markdown(f"**[{i}] 相关度: {score:.4f}**")
                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    st.markdown("---")
            
            # 知识图谱增强
            if enable_kg and retrieved:
                with st.spinner("正在抽取知识图谱..."):
                    chat_client = OpenAI(
                        base_url=os.getenv("CHAT_BASE_URL"),
                        api_key=os.getenv("CHAT_API_KEY")
                    )
                    
                    top_chunks = [chunk for chunk, _ in retrieved[:3]]
                    graph = extract_knowledge_graph(top_chunks, chat_client, os.getenv("CHAT_MODEL"))
                    
                    if graph and (graph.get("entities") or graph.get("relations")):
                        with st.expander("🔗 知识图谱", expanded=False):
                            st.json(graph)
                        
                        graph_context = format_graph_for_prompt(graph)
                        first_chunk = retrieved[0][0]
                        enhanced_chunk = f"{first_chunk}\n\n【知识图谱增强】\n{graph_context}"
                        retrieved[0] = (enhanced_chunk, retrieved[0][1])
                        st.success("✅ 已将知识图谱信息注入到上下文")
            
            # 生成回答
            with st.spinner("正在生成回答..."):
                context = "\n\n".join([chunk for chunk, _ in retrieved])
                answer = generate_answer(user_query, context, st.session_state.conversation_history)
            
            st.markdown("### 🤖 回答")
            st.markdown(answer)
            
            # 更新对话历史
            st.session_state.conversation_history.append({"role": "user", "content": user_query})
            st.session_state.conversation_history.append({"role": "assistant", "content": answer})
            st.session_state.current_contexts = retrieved
            
        except Exception as e:
            st.error(f"❌ 生成回答失败: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
