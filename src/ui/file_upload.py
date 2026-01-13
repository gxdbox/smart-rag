"""
文件上传 UI 组件
包含文件上传和 JSON 数据导入功能
"""

import streamlit as st
import json

from rag_engine import (
    load_env,
    split_text_by_strategy,
    embed_texts,
    add_to_vector_db,
    add_to_bm25_index
)
from src.utils import read_file
from src.rag.chunker import choose_chunk_strategy, get_strategy_description


def render_file_upload():
    """渲染文件上传区域"""
    st.subheader("📤 上传文件")
    
    uploaded_files = st.file_uploader(
        "选择要上传的文件（支持多选）",
        type=["txt", "pdf", "md", "markdown", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="上传的文件将自动切分并存入向量库（图片将通过 OCR 识别）"
    )
    
    if uploaded_files:
        load_env()
        with st.spinner("正在处理上传的文件..."):
            for uploaded_file in uploaded_files:
                try:
                    file_content = uploaded_file.read()
                    text = read_file(uploaded_file.name, file_content)
                    
                    if text:
                        strategy, params = choose_chunk_strategy(text)
                        st.session_state.chunk_strategy = strategy
                        st.session_state.chunk_params = params
                        
                        chunks = split_text_by_strategy(text, strategy, params)
                        
                        if chunks:
                            embeddings = embed_texts(chunks)
                            add_to_vector_db(chunks, embeddings)
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
            chunks = st.session_state.imported_chunks
            load_env()
            
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
                del st.session_state.imported_chunks
                del st.session_state.imported_file_name
                st.success(f"✅ 成功导入 {total_added} 个 chunks（已同步到向量库和 BM25 索引）")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 导入失败: {str(e)}（已导入 {total_added} 个）")
    
    st.markdown("---")
