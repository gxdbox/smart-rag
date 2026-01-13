"""
自适应过滤器使用示例

演示如何使用动态阈值过滤检索结果
"""

from src.rag.filter import AdaptiveFilter, FilterConfig
from src.rag.filter.adaptive_filter import adaptive_filter


def example_1_basic_usage():
    """示例1：基础用法"""
    print("\n=== 示例1：基础用法 ===")
    
    # 模拟检索结果（分数从高到低）
    results = [
        ("文档1：完全匹配的内容", 0.95),
        ("文档2：高度相关的内容", 0.88),
        ("文档3：相关内容", 0.75),
        ("文档4：部分相关", 0.45),
        ("文档5：低相关", 0.32),
        ("文档6：几乎无关", 0.15),
    ]
    
    # 使用默认配置过滤
    filtered, metadata = adaptive_filter(results, normalize=False)
    
    print(f"原始结果数: {metadata['total']}")
    print(f"保留结果数: {metadata['kept']}")
    print(f"过滤阈值: {metadata['threshold']:.3f}")
    print(f"阈值策略: {metadata['reason']}")
    print(f"平均分数: {metadata['avg_score']:.3f}")
    print("\n保留的结果:")
    for text, score in filtered:
        print(f"  - {text}: {score:.3f}")


def example_2_quality_scenarios():
    """示例2：不同质量场景"""
    print("\n=== 示例2：不同质量场景 ===")
    
    scenarios = {
        "高质量召回": [
            ("doc1", 0.92), ("doc2", 0.89), ("doc3", 0.85),
            ("doc4", 0.82), ("doc5", 0.78)
        ],
        "中等质量召回": [
            ("doc1", 0.75), ("doc2", 0.68), ("doc3", 0.55),
            ("doc4", 0.42), ("doc5", 0.35)
        ],
        "低质量召回": [
            ("doc1", 0.45), ("doc2", 0.38), ("doc3", 0.32),
            ("doc4", 0.28), ("doc5", 0.25)
        ],
        "断崖式分布": [
            ("doc1", 0.95), ("doc2", 0.92), ("doc3", 0.35),
            ("doc4", 0.32), ("doc5", 0.30)
        ]
    }
    
    filter_obj = AdaptiveFilter()
    
    for scenario_name, results in scenarios.items():
        filtered, metadata = filter_obj.filter_results(results, normalize=False)
        quality = filter_obj.get_quality_level(metadata)
        
        print(f"\n{scenario_name}:")
        print(f"  保留: {metadata['kept']}/{metadata['total']} 个")
        print(f"  阈值: {metadata['threshold']:.3f} ({metadata['reason']})")
        print(f"  质量: {quality}")


def example_3_custom_config():
    """示例3：自定义配置"""
    print("\n=== 示例3：自定义配置 ===")
    
    results = [
        ("doc1", 0.88), ("doc2", 0.75), ("doc3", 0.62),
        ("doc4", 0.48), ("doc5", 0.35), ("doc6", 0.22)
    ]
    
    # 严格配置（高阈值）
    strict_config = FilterConfig(
        min_confidence=0.6,  # 更高的最低阈值
        max_results=3,
        gap_threshold=0.2,
        percentile_threshold=0.7
    )
    
    # 宽松配置（低阈值）
    loose_config = FilterConfig(
        min_confidence=0.2,  # 更低的最低阈值
        max_results=10,
        gap_threshold=0.1,
        percentile_threshold=0.5
    )
    
    print("\n严格配置:")
    strict_filter = AdaptiveFilter(strict_config)
    filtered, metadata = strict_filter.filter_results(results, normalize=False)
    print(f"  保留: {metadata['kept']}/{metadata['total']} 个, 阈值: {metadata['threshold']:.3f}")
    
    print("\n宽松配置:")
    loose_filter = AdaptiveFilter(loose_config)
    filtered, metadata = loose_filter.filter_results(results, normalize=False)
    print(f"  保留: {metadata['kept']}/{metadata['total']} 个, 阈值: {metadata['threshold']:.3f}")


def example_4_compare_with_topk():
    """示例4：对比固定 Top-K 和自适应过滤"""
    print("\n=== 示例4：固定 Top-K vs 自适应过滤 ===")
    
    # 模拟一个质量参差不齐的检索结果
    results = [
        ("高质量1", 0.92),
        ("高质量2", 0.88),
        ("高质量3", 0.85),
        ("噪声1", 0.35),  # 明显的质量断崖
        ("噪声2", 0.32),
        ("噪声3", 0.28),
    ]
    
    k = 5  # 假设要取 Top-5
    
    # 方法1：固定 Top-K（盲目截断）
    topk_results = results[:k]
    print(f"\n固定 Top-{k}:")
    print(f"  保留了 {len(topk_results)} 个结果")
    print(f"  包含噪声: {sum(1 for _, s in topk_results if s < 0.5)} 个")
    for text, score in topk_results:
        marker = "⚠️" if score < 0.5 else "✅"
        print(f"  {marker} {text}: {score:.3f}")
    
    # 方法2：自适应过滤（智能截断）
    filtered, metadata = adaptive_filter(results, normalize=False)
    print(f"\n自适应过滤:")
    print(f"  保留了 {metadata['kept']} 个结果")
    print(f"  阈值: {metadata['threshold']:.3f} ({metadata['reason']})")
    print(f"  包含噪声: {sum(1 for _, s in filtered if s < 0.5)} 个")
    for text, score in filtered:
        marker = "⚠️" if score < 0.5 else "✅"
        print(f"  {marker} {text}: {score:.3f}")


def example_5_real_world_simulation():
    """示例5：真实场景模拟"""
    print("\n=== 示例5：真实混合检索场景 ===")
    
    # 模拟混合检索：向量检索 + BM25 检索
    # 向量检索结果（语义相关）
    vector_results = {
        "深度学习在自然语言处理中的应用": 0.89,
        "Transformer 模型架构详解": 0.85,
        "BERT 预训练技术": 0.78,
        "神经网络优化方法": 0.65,
    }
    
    # BM25 检索结果（关键词匹配）
    bm25_results = {
        "Transformer 模型架构详解": 0.92,  # 重复
        "注意力机制原理": 0.88,
        "深度学习在自然语言处理中的应用": 0.75,  # 重复
        "Python NLP 工具包": 0.45,
    }
    
    # 融合（已去重）
    all_texts = set(vector_results.keys()) | set(bm25_results.keys())
    fused_results = []
    for text in all_texts:
        v_score = vector_results.get(text, 0.0)
        b_score = bm25_results.get(text, 0.0)
        combined = 0.5 * v_score + 0.5 * b_score
        fused_results.append((text, combined))
    
    fused_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n融合后结果数: {len(fused_results)}")
    print("\n融合分数:")
    for text, score in fused_results:
        print(f"  {text}: {score:.3f}")
    
    # 应用自适应过滤
    filtered, metadata = adaptive_filter(fused_results, normalize=False)
    
    print(f"\n自适应过滤后:")
    print(f"  保留: {metadata['kept']}/{metadata['total']} 个")
    print(f"  阈值: {metadata['threshold']:.3f}")
    print(f"  策略: {metadata['reason']}")
    print("\n最终结果:")
    for text, score in filtered:
        print(f"  ✅ {text}: {score:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("自适应过滤器使用示例")
    print("=" * 60)
    
    example_1_basic_usage()
    example_2_quality_scenarios()
    example_3_custom_config()
    example_4_compare_with_topk()
    example_5_real_world_simulation()
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)
