"""
自适应过滤器 - 动态阈值与置信度评估

核心功能：
1. 统一置信度归一化（不同召回源分数映射到 [0,1]）
2. 自适应阈值计算（基于分数分布动态确定过滤线）
3. 质量感知过滤（避免盲目截断，保证召回质量）
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class FilterConfig:
    """过滤器配置"""
    min_confidence: float = 0.3  # 最低置信度阈值（硬下限）
    max_results: int = 10  # 最大返回数量（性能上限）
    min_results: int = 1  # 最小返回数量（保底）
    gap_threshold: float = 0.15  # 分数断崖阈值（相邻分数差异）
    percentile_threshold: float = 0.6  # 百分位阈值（取前 60% 分位以上）
    use_elbow: bool = True  # 是否启用肘部法则
    use_gap_detection: bool = True  # 是否启用断崖检测


class ConfidenceNormalizer:
    """置信度归一化器 - 将不同召回源的分数统一到 [0,1]"""
    
    @staticmethod
    def normalize_scores(scores: List[float], method: str = "minmax") -> List[float]:
        """归一化分数到 [0,1]
        
        Args:
            scores: 原始分数列表
            method: 归一化方法 (minmax/zscore/sigmoid)
        
        Returns:
            归一化后的分数列表
        """
        if not scores:
            return []
        
        scores_array = np.array(scores)
        
        if method == "minmax":
            # Min-Max 归一化
            min_score = scores_array.min()
            max_score = scores_array.max()
            if max_score == min_score:
                return [1.0] * len(scores)  # 所有分数相同，视为高置信度
            return ((scores_array - min_score) / (max_score - min_score)).tolist()
        
        elif method == "zscore":
            # Z-Score 归一化 + Sigmoid 映射
            mean = scores_array.mean()
            std = scores_array.std()
            if std == 0:
                return [1.0] * len(scores)
            z_scores = (scores_array - mean) / std
            # Sigmoid 映射到 [0,1]
            return (1 / (1 + np.exp(-z_scores))).tolist()
        
        elif method == "sigmoid":
            # 直接 Sigmoid（假设原始分数已中心化）
            return (1 / (1 + np.exp(-scores_array))).tolist()
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def normalize_hybrid_scores(
        vector_scores: Dict[str, float],
        bm25_scores: Dict[str, float],
        vector_weight: float = 0.5
    ) -> Dict[str, float]:
        """归一化混合检索分数（已在 rag_engine.py 中实现，这里提供统一接口）
        
        Args:
            vector_scores: 向量检索分数字典 {text: score}
            bm25_scores: BM25 检索分数字典 {text: score}
            vector_weight: 向量权重
        
        Returns:
            归一化后的混合分数字典
        """
        # 归一化向量分数
        if vector_scores:
            v_vals = list(vector_scores.values())
            v_norm = ConfidenceNormalizer.normalize_scores(v_vals, method="minmax")
            vector_scores = {k: v_norm[i] for i, k in enumerate(vector_scores.keys())}
        
        # 归一化 BM25 分数
        if bm25_scores:
            b_vals = list(bm25_scores.values())
            b_norm = ConfidenceNormalizer.normalize_scores(b_vals, method="minmax")
            bm25_scores = {k: b_norm[i] for i, k in enumerate(bm25_scores.keys())}
        
        # 融合
        all_texts = set(vector_scores.keys()) | set(bm25_scores.keys())
        combined = {}
        for text in all_texts:
            v_score = vector_scores.get(text, 0.0)
            b_score = bm25_scores.get(text, 0.0)
            combined[text] = vector_weight * v_score + (1 - vector_weight) * b_score
        
        return combined


class AdaptiveThresholdCalculator:
    """自适应阈值计算器 - 基于分数分布动态确定过滤阈值"""
    
    @staticmethod
    def calculate_threshold(
        scores: List[float],
        config: FilterConfig
    ) -> Tuple[float, str]:
        """计算自适应阈值
        
        Args:
            scores: 归一化后的置信度分数（降序排列）
            config: 过滤器配置
        
        Returns:
            (threshold, reason) - 阈值和选择原因
        """
        if not scores:
            return config.min_confidence, "empty_scores"
        
        scores_array = np.array(scores)
        
        # 策略1: 肘部法则（Elbow Method）- 找拐点
        if config.use_elbow and len(scores) >= 3:
            elbow_threshold = AdaptiveThresholdCalculator._find_elbow(scores_array)
            if elbow_threshold is not None and elbow_threshold >= config.min_confidence:
                return elbow_threshold, "elbow_method"
        
        # 策略2: 断崖检测（Gap Detection）- 找分数突降点
        if config.use_gap_detection and len(scores) >= 2:
            gap_threshold = AdaptiveThresholdCalculator._find_gap(
                scores_array, config.gap_threshold
            )
            if gap_threshold is not None and gap_threshold >= config.min_confidence:
                return gap_threshold, "gap_detection"
        
        # 策略3: 百分位阈值（Percentile）- 保留高分段
        percentile_threshold = np.percentile(scores_array, (1 - config.percentile_threshold) * 100)
        if percentile_threshold >= config.min_confidence:
            return float(percentile_threshold), "percentile"
        
        # 策略4: 硬下限（Fallback）
        return config.min_confidence, "min_confidence_fallback"
    
    @staticmethod
    def _find_elbow(scores: np.ndarray) -> Optional[float]:
        """肘部法则：找分数曲线的拐点
        
        原理：计算每个点到首尾连线的距离，距离最大的点即为拐点
        """
        if len(scores) < 3:
            return None
        
        # 构建点集 (index, score)
        points = np.array([[i, score] for i, score in enumerate(scores)])
        
        # 首尾连线
        line_vec = points[-1] - points[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        
        # 计算每个点到连线的距离
        vec_from_first = points - points[0]
        scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
        vec_to_line = vec_from_first - np.outer(scalar_product, line_vec_norm)
        distances = np.sqrt(np.sum(vec_to_line**2, axis=1))
        
        # 找最大距离点（拐点）
        elbow_idx = np.argmax(distances)
        
        # 拐点后的分数作为阈值（保留拐点及之前的高分）
        if elbow_idx < len(scores) - 1:
            return float(scores[elbow_idx + 1])
        
        return None
    
    @staticmethod
    def _find_gap(scores: np.ndarray, gap_threshold: float) -> Optional[float]:
        """断崖检测：找相邻分数差异最大的位置
        
        Args:
            scores: 降序排列的分数
            gap_threshold: 最小间隙阈值
        
        Returns:
            断崖后的分数（作为阈值）
        """
        if len(scores) < 2:
            return None
        
        # 计算相邻分数差
        gaps = np.diff(scores)
        
        # 找最大间隙（注意是负数，因为降序）
        max_gap_idx = np.argmax(-gaps)  # 最大下降
        max_gap = -gaps[max_gap_idx]
        
        # 如果间隙足够大，使用断崖后的分数
        if max_gap >= gap_threshold:
            return float(scores[max_gap_idx + 1])
        
        return None


class AdaptiveFilter:
    """自适应过滤器 - 整合归一化和动态阈值"""
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """初始化过滤器
        
        Args:
            config: 过滤器配置，None 则使用默认配置
        """
        self.config = config or FilterConfig()
        self.normalizer = ConfidenceNormalizer()
        self.threshold_calculator = AdaptiveThresholdCalculator()
    
    def filter_results(
        self,
        results: List[Tuple[str, float]],
        normalize: bool = True
    ) -> Tuple[List[Tuple[str, float]], Dict[str, any]]:
        """过滤检索结果
        
        Args:
            results: [(text, score), ...] 检索结果（可能未归一化）
            normalize: 是否先归一化分数
        
        Returns:
            (filtered_results, metadata) - 过滤后的结果和元数据
        """
        if not results:
            return [], {"reason": "empty_input", "threshold": 0.0, "total": 0, "kept": 0}
        
        # 1. 归一化分数
        texts, scores = zip(*results)
        if normalize:
            scores = self.normalizer.normalize_scores(list(scores), method="minmax")
            results = list(zip(texts, scores))
        
        # 按分数降序排序
        results = sorted(results, key=lambda x: x[1], reverse=True)
        texts, scores = zip(*results)
        
        # 2. 计算自适应阈值
        threshold, reason = self.threshold_calculator.calculate_threshold(
            list(scores), self.config
        )
        
        # 3. 应用阈值过滤
        filtered = [(t, s) for t, s in results if s >= threshold]
        
        # 4. 应用数量限制
        if len(filtered) > self.config.max_results:
            filtered = filtered[:self.config.max_results]
        
        # 5. 保底机制：至少返回 min_results 个（如果有的话）
        if len(filtered) < self.config.min_results and len(results) >= self.config.min_results:
            filtered = results[:self.config.min_results]
        
        # 6. 元数据
        metadata = {
            "threshold": threshold,
            "reason": reason,
            "total": len(results),
            "kept": len(filtered),
            "min_score": filtered[-1][1] if filtered else 0.0,
            "max_score": filtered[0][1] if filtered else 0.0,
            "avg_score": np.mean([s for _, s in filtered]) if filtered else 0.0
        }
        
        return filtered, metadata
    
    def get_quality_level(self, metadata: Dict[str, any]) -> str:
        """评估召回质量等级
        
        Args:
            metadata: filter_results 返回的元数据
        
        Returns:
            质量等级: "excellent" / "good" / "fair" / "poor"
        """
        avg_score = metadata.get("avg_score", 0.0)
        kept_ratio = metadata.get("kept", 0) / max(metadata.get("total", 1), 1)
        
        if avg_score >= 0.8 and kept_ratio >= 0.5:
            return "excellent"
        elif avg_score >= 0.6 and kept_ratio >= 0.3:
            return "good"
        elif avg_score >= 0.4 or kept_ratio >= 0.2:
            return "fair"
        else:
            return "poor"


# 便捷函数
def adaptive_filter(
    results: List[Tuple[str, float]],
    config: Optional[FilterConfig] = None,
    normalize: bool = True
) -> Tuple[List[Tuple[str, float]], Dict[str, any]]:
    """便捷函数：自适应过滤检索结果
    
    Args:
        results: [(text, score), ...] 检索结果
        config: 过滤器配置
        normalize: 是否归一化分数
    
    Returns:
        (filtered_results, metadata)
    
    Example:
        >>> results = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.3)]
        >>> filtered, meta = adaptive_filter(results)
        >>> print(f"保留 {meta['kept']}/{meta['total']} 个结果，阈值: {meta['threshold']:.2f}")
    """
    filter_obj = AdaptiveFilter(config)
    return filter_obj.filter_results(results, normalize=normalize)
