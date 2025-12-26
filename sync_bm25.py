"""
BM25 索引同步工具

用于将向量库中的文档同步到 BM25 索引
"""

import json
from rag_engine import add_to_bm25_index, clear_bm25_index, get_bm25_stats, get_db_stats

def sync_bm25_from_vector_db():
    """从向量库同步数据到 BM25 索引"""
    
    print("=" * 60)
    print("BM25 索引同步工具")
    print("=" * 60)
    
    # 1. 检查当前状态
    print("\n📊 当前状态:")
    vector_stats = get_db_stats()
    bm25_stats = get_bm25_stats()
    print(f"  向量库文档数: {vector_stats['total_chunks']}")
    print(f"  BM25 索引文档数: {bm25_stats['total_chunks']}")
    
    if vector_stats['total_chunks'] == 0:
        print("\n❌ 向量库为空，无需同步")
        return
    
    if vector_stats['total_chunks'] == bm25_stats['total_chunks']:
        print("\n✅ 索引已同步，文档数量一致")
        choice = input("\n是否强制重新同步？(y/n): ")
        if choice.lower() != 'y':
            return
    
    # 2. 读取向量库数据
    print("\n📖 读取向量库数据...")
    with open('vector_db.json', 'r', encoding='utf-8') as f:
        db = json.load(f)
        chunks = db.get('chunks', [])
    
    print(f"  读取到 {len(chunks)} 个文档")
    
    # 3. 清空 BM25 索引
    print("\n🗑️  清空旧的 BM25 索引...")
    clear_bm25_index()
    
    # 4. 批量添加到 BM25
    print("\n📥 同步到 BM25 索引...")
    batch_size = 100
    total = len(chunks)
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        add_to_bm25_index(batch)
        progress = min(i + batch_size, total)
        print(f"  进度: {progress}/{total} ({progress*100//total}%)")
    
    # 5. 验证同步结果
    print("\n✅ 同步完成！")
    new_bm25_stats = get_bm25_stats()
    print(f"\n📊 同步后状态:")
    print(f"  向量库文档数: {vector_stats['total_chunks']}")
    print(f"  BM25 索引文档数: {new_bm25_stats['total_chunks']}")
    print(f"  BM25 总词数: {new_bm25_stats['total_tokens']}")
    
    if vector_stats['total_chunks'] == new_bm25_stats['total_chunks']:
        print("\n🎉 同步成功！两个索引文档数量一致")
    else:
        print("\n⚠️  警告：文档数量不一致，请检查")

if __name__ == "__main__":
    try:
        sync_bm25_from_vector_db()
    except Exception as e:
        print(f"\n❌ 同步失败: {e}")
        import traceback
        traceback.print_exc()
