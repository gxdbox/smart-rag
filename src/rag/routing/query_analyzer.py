"""
查询分析器
分析用户查询的特征，为策略路由提供依据
"""

from dataclasses import dataclass
from typing import Optional
import re
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryProfile:
    """查询特征画像"""
    query: str
    query_type: str  # factual/analytical/exploratory/procedural
    complexity: str  # simple/medium/complex
    length: int
    has_multiple_questions: bool
    domain: str  # policy/financial/technical/general
    ambiguity_score: float  # 0-1，越高越模糊
    requires_context: bool
    keywords: list


class QueryAnalyzer:
    """查询分析器"""
    
    def __init__(self):
        self.factual_keywords = ['什么', '哪些', '多少', '何时', '谁', '是否', '有没有']
        self.analytical_keywords = ['为什么', '如何', '怎样', '原因', '分析', '比较', '区别', '影响']
        self.exploratory_keywords = ['可能', '也许', '大概', '估计', '趋势', '未来', '预测']
        self.procedural_keywords = ['步骤', '流程', '如何办理', '怎么做', '申请', '操作']
        
        self.policy_keywords = ['政策', '规定', '条例', '办法', '通知', '文件', '规范', '标准']
        self.financial_keywords = ['资金', '费用', '补贴', '税', '财政', '预算', '成本']
        self.technical_keywords = ['技术', '系统', '平台', '接口', '算法', '模型']
        
        logger.info("QueryAnalyzer 初始化完成")
    
    def analyze(self, query: str) -> QueryProfile:
        """
        分析查询特征
        
        Args:
            query: 用户查询文本
            
        Returns:
            QueryProfile: 查询特征画像
        """
        query = query.strip()
        
        # 1. 查询类型
        query_type = self._detect_query_type(query)
        
        # 2. 复杂度
        complexity = self._detect_complexity(query)
        
        # 3. 是否包含多个问题
        has_multiple_questions = self._has_multiple_questions(query)
        
        # 4. 领域
        domain = self._detect_domain(query)
        
        # 5. 模糊度
        ambiguity_score = self._calculate_ambiguity(query)
        
        # 6. 是否需要上下文
        requires_context = self._requires_context(query, query_type)
        
        # 7. 关键词提取
        keywords = self._extract_keywords(query)
        
        profile = QueryProfile(
            query=query,
            query_type=query_type,
            complexity=complexity,
            length=len(query),
            has_multiple_questions=has_multiple_questions,
            domain=domain,
            ambiguity_score=ambiguity_score,
            requires_context=requires_context,
            keywords=keywords
        )
        
        logger.info(f"查询分析完成: type={query_type}, complexity={complexity}, domain={domain}")
        return profile
    
    def _detect_query_type(self, query: str) -> str:
        """检测查询类型"""
        # 事实型查询
        if any(kw in query for kw in self.factual_keywords):
            return 'factual'
        
        # 分析型查询
        if any(kw in query for kw in self.analytical_keywords):
            return 'analytical'
        
        # 流程型查询
        if any(kw in query for kw in self.procedural_keywords):
            return 'procedural'
        
        # 探索型查询
        if any(kw in query for kw in self.exploratory_keywords):
            return 'exploratory'
        
        # 默认为事实型
        return 'factual'
    
    def _detect_complexity(self, query: str) -> str:
        """检测查询复杂度"""
        length = len(query)
        
        # 长度判断
        if length < 10:
            return 'simple'
        elif length > 50:
            complexity_score = 2
        else:
            complexity_score = 1
        
        # 包含多个条件
        if '并且' in query or '同时' in query or '以及' in query:
            complexity_score += 1
        
        # 包含否定
        if '不' in query or '没有' in query or '非' in query:
            complexity_score += 0.5
        
        # 包含比较
        if '比较' in query or '区别' in query or '差异' in query:
            complexity_score += 1
        
        if complexity_score <= 1:
            return 'simple'
        elif complexity_score <= 2.5:
            return 'medium'
        else:
            return 'complex'
    
    def _has_multiple_questions(self, query: str) -> bool:
        """检测是否包含多个问题"""
        question_marks = query.count('？') + query.count('?')
        
        # 多个问号
        if question_marks > 1:
            return True
        
        # 包含连接词
        if any(word in query for word in ['另外', '还有', '以及', '同时', '并且']):
            return True
        
        return False
    
    def _detect_domain(self, query: str) -> str:
        """检测查询领域"""
        policy_score = sum(1 for kw in self.policy_keywords if kw in query)
        financial_score = sum(1 for kw in self.financial_keywords if kw in query)
        technical_score = sum(1 for kw in self.technical_keywords if kw in query)
        
        max_score = max(policy_score, financial_score, technical_score)
        
        if max_score == 0:
            return 'general'
        elif policy_score == max_score:
            return 'policy'
        elif financial_score == max_score:
            return 'financial'
        else:
            return 'technical'
    
    def _calculate_ambiguity(self, query: str) -> float:
        """计算查询模糊度"""
        ambiguity_score = 0.0
        
        # 包含模糊词
        ambiguous_words = ['可能', '也许', '大概', '估计', '大约', '左右', '差不多', '类似']
        ambiguity_score += sum(0.15 for word in ambiguous_words if word in query)
        
        # 查询过短
        if len(query) < 5:
            ambiguity_score += 0.3
        
        # 缺少具体名词
        if not re.search(r'[政策|规定|文件|办法|通知|条例]', query):
            ambiguity_score += 0.2
        
        # 包含代词
        pronouns = ['这个', '那个', '它', '他们', '我们']
        if any(p in query for p in pronouns):
            ambiguity_score += 0.2
        
        return min(ambiguity_score, 1.0)
    
    def _requires_context(self, query: str, query_type: str) -> bool:
        """判断是否需要上下文"""
        # 分析型和流程型查询通常需要上下文
        if query_type in ['analytical', 'procedural']:
            return True
        
        # 包含"为什么"、"如何"等词
        if any(word in query for word in ['为什么', '如何', '怎样', '原因', '背景']):
            return True
        
        # 查询较长，可能需要更多上下文
        if len(query) > 30:
            return True
        
        return False
    
    def _extract_keywords(self, query: str) -> list:
        """提取关键词（简单实现）"""
        # 移除常见停用词
        stopwords = ['的', '了', '是', '在', '有', '和', '与', '或', '等', '吗', '呢', '吧']
        
        # 简单分词（按字符）
        words = []
        for i in range(len(query)):
            for length in [2, 3, 4]:
                if i + length <= len(query):
                    word = query[i:i+length]
                    if word not in stopwords and len(word) > 1:
                        words.append(word)
        
        # 去重并返回前10个
        keywords = list(set(words))[:10]
        return keywords
