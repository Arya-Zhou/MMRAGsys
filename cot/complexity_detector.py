"""
复杂度检测器
检测问题复杂度并进行问题分类
"""

import re
import jieba
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplexityDetector:
    """复杂度检测器"""
    
    def __init__(self):
        """初始化复杂度检测器"""
        
        # 复杂度权重配置
        self.complexity_weights = {
            # 强复杂度指示词 (0.8-1.0)
            "分析": 0.9,
            "对比": 0.85,
            "比较": 0.85,
            "评估": 0.9,
            "推理": 0.95,
            "推导": 0.9,
            "论证": 0.9,
            "解释": 0.85,
            "为什么": 0.85,
            "如何": 0.8,
            "怎么": 0.8,
            "原因": 0.85,
            "影响": 0.8,
            "关系": 0.8,
            "区别": 0.85,
            "优缺点": 0.9,
            "建议": 0.8,
            "方案": 0.85,
            "策略": 0.85,
            "规划": 0.9,
            
            # 中等复杂度指示词 (0.4-0.7)
            "解释": 0.6,
            "描述": 0.5,
            "说明": 0.5,
            "列出": 0.4,
            "列举": 0.4,
            "介绍": 0.5,
            "概述": 0.6,
            "总结": 0.6,
            "归纳": 0.7,
            "包括": 0.4,
            "特点": 0.5,
            "功能": 0.5,
            "作用": 0.6,
            
            # 简单指示词 (0.1-0.3)
            "是什么": 0.2,
            "定义": 0.1,
            "查找": 0.1,
            "名称": 0.1,
            "时间": 0.2,
            "地点": 0.2,
            "数量": 0.2,
            "是否": 0.1,
            "有没有": 0.1,
            "多少": 0.2
        }
        
        # 问题类型关键词
        self.question_type_keywords = {
            "事实查询类": ["是什么", "定义", "名称", "时间", "地点", "数量", "是否"],
            "分析推理类": ["分析", "推理", "推导", "论证", "为什么", "原因"],
            "对比评估类": ["对比", "比较", "区别", "优缺点", "评估", "选择"],
            "因果解释类": ["为什么", "原因", "导致", "影响", "结果", "因为"],
            "流程规划类": ["如何", "怎么", "步骤", "流程", "规划", "方案", "策略"],
            "综合论述类": ["总结", "归纳", "综述", "观点", "论述", "阐述"]
        }
        
        # 认知复杂度等级
        self.cognitive_levels = {
            "记忆": 0.1,
            "理解": 0.3,
            "应用": 0.5,
            "分析": 0.7,
            "评估": 0.85,
            "创造": 0.95
        }
        
        logger.info("初始化复杂度检测器")
        
    def detect_complexity(self, query: str) -> str:
        """
        检测问题复杂度
        Args:
            query: 查询文本
        Returns:
            "simple" 或 "complex"
        """
        # 计算复杂度分数
        complexity_score = self._calculate_complexity_score(query)
        
        # 获取简单特征比例
        simple_ratio = self._get_simple_feature_ratio(query)
        
        # 决策逻辑
        if complexity_score > 0.7:
            # 强复杂度信号
            result = "complex"
        elif simple_ratio > 0.6:
            # 简单特征占主导
            result = "simple"
        else:
            # 混合情况，使用加权决策
            result = self._weighted_decision(complexity_score, simple_ratio)
            
        logger.info(f"复杂度检测: query='{query[:50]}...', score={complexity_score:.2f}, result={result}")
        
        return result
        
    def classify_question_type(self, query: str) -> str:
        """
        问题类型分类
        Args:
            query: 查询文本
        Returns:
            问题类型
        """
        # 分词
        words = list(jieba.cut(query))
        
        # 计算每个类型的匹配分数
        type_scores = {}
        
        for qtype, keywords in self.question_type_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query:
                    # 关键词匹配加分
                    score += 1
                    # 如果关键词在开头，额外加分
                    if query.startswith(keyword):
                        score += 0.5
                        
            type_scores[qtype] = score
            
        # 如果没有明显匹配，使用认知层次分析
        if max(type_scores.values()) == 0:
            qtype = self._classify_by_cognitive_level(query)
        else:
            # 选择得分最高的类型
            qtype = max(type_scores, key=type_scores.get)
            
        logger.info(f"问题分类: '{query[:50]}...' -> {qtype}")
        
        return qtype
        
    def _calculate_complexity_score(self, query: str) -> float:
        """
        计算复杂度分数
        Args:
            query: 查询文本
        Returns:
            复杂度分数(0-1)
        """
        # 分词
        words = list(jieba.cut(query))
        
        # 收集所有匹配的权重
        weights = []
        matched_keywords = []
        
        for word in words:
            if word in self.complexity_weights:
                weights.append(self.complexity_weights[word])
                matched_keywords.append(word)
                
        if not weights:
            # 没有匹配的关键词，检查其他特征
            return self._analyze_structural_complexity(query)
            
        # 多关键词处理：加权累积方法
        if len(weights) == 1:
            score = weights[0]
        else:
            # 取最高权重作为主要分数
            max_weight = max(weights)
            # 其他权重作为补充（衰减）
            other_weights = [w for w in weights if w != max_weight]
            supplement = sum(w * 0.3 for w in other_weights)  # 30%权重
            score = min(max_weight + supplement, 1.0)
            
        # 同类关键词聚集增强
        if self._check_keyword_clustering(matched_keywords):
            score = min(score * 1.2, 1.0)
            
        return score
        
    def _get_simple_feature_ratio(self, query: str) -> float:
        """
        获取简单特征比例
        Args:
            query: 查询文本
        Returns:
            简单特征比例(0-1)
        """
        simple_features = 0
        total_features = 0
        
        # 检查简单关键词
        for keyword, weight in self.complexity_weights.items():
            if keyword in query:
                total_features += 1
                if weight < 0.3:  # 简单关键词
                    simple_features += 1
                    
        # 检查问题长度（简单问题通常较短）
        if len(query) < 20:
            simple_features += 1
            total_features += 1
            
        # 检查是否是是非题
        if any(word in query for word in ["是否", "是不是", "有没有", "对不对"]):
            simple_features += 2
            total_features += 2
            
        # 检查是否包含具体实体（简单查询通常有明确目标）
        if self._contains_specific_entity(query):
            simple_features += 1
            total_features += 1
            
        if total_features == 0:
            return 0.5  # 默认中等
            
        return simple_features / total_features
        
    def _weighted_decision(self, complexity_score: float, simple_ratio: float) -> str:
        """
        加权决策
        Args:
            complexity_score: 复杂度分数
            simple_ratio: 简单特征比例
        Returns:
            决策结果
        """
        # 综合分数
        final_score = complexity_score * 0.7 - simple_ratio * 0.3
        
        # 阈值判断
        if final_score > 0.4:
            return "complex"
        else:
            return "simple"
            
    def _analyze_structural_complexity(self, query: str) -> float:
        """
        分析结构复杂度
        Args:
            query: 查询文本
        Returns:
            结构复杂度分数
        """
        score = 0.3  # 基础分数
        
        # 检查句子长度
        if len(query) > 50:
            score += 0.2
        elif len(query) < 15:
            score -= 0.1
            
        # 检查是否包含多个子句
        if "，" in query or "；" in query:
            score += 0.15
            
        # 检查是否包含条件语句
        if any(word in query for word in ["如果", "假如", "当", "若"]):
            score += 0.2
            
        # 检查是否包含多个问题
        if query.count("？") > 1 or query.count("?") > 1:
            score += 0.25
            
        return min(max(score, 0), 1)
        
    def _check_keyword_clustering(self, keywords: List[str]) -> bool:
        """
        检查关键词聚集
        Args:
            keywords: 关键词列表
        Returns:
            是否存在聚集
        """
        # 定义同类关键词组
        keyword_groups = [
            {"分析", "推理", "推导", "论证"},
            {"对比", "比较", "区别", "差异"},
            {"原因", "为什么", "导致", "因为"},
            {"如何", "怎么", "步骤", "方法"}
        ]
        
        for group in keyword_groups:
            matched = [kw for kw in keywords if kw in group]
            if len(matched) >= 2:
                return True
                
        return False
        
    def _contains_specific_entity(self, query: str) -> bool:
        """
        检查是否包含具体实体
        Args:
            query: 查询文本
        Returns:
            是否包含具体实体
        """
        # 检查是否包含引号（通常标记具体实体）
        if '"' in query or "'" in query or """ in query:
            return True
            
        # 检查是否包含专有名词特征
        # 这里简化处理，检查是否有大写字母开头的词
        words = query.split()
        for word in words:
            if word and word[0].isupper():
                return True
                
        # 检查是否包含数字（具体数值）
        if any(char.isdigit() for char in query):
            return True
            
        return False
        
    def _classify_by_cognitive_level(self, query: str) -> str:
        """
        基于认知层次分类
        Args:
            query: 查询文本
        Returns:
            问题类型
        """
        # 分析认知层次
        cognitive_score = 0
        
        # 检查动词指示认知层次
        cognitive_verbs = {
            "记住": 0.1, "知道": 0.1, "识别": 0.1,
            "理解": 0.3, "解释": 0.3, "描述": 0.3,
            "应用": 0.5, "使用": 0.5, "实施": 0.5,
            "分析": 0.7, "比较": 0.7, "检验": 0.7,
            "评估": 0.85, "判断": 0.85, "批判": 0.85,
            "创造": 0.95, "设计": 0.95, "构建": 0.95
        }
        
        for verb, level in cognitive_verbs.items():
            if verb in query:
                cognitive_score = max(cognitive_score, level)
                
        # 根据认知层次映射到问题类型
        if cognitive_score < 0.3:
            return "事实查询类"
        elif cognitive_score < 0.5:
            return "因果解释类"
        elif cognitive_score < 0.7:
            return "流程规划类"
        elif cognitive_score < 0.85:
            return "分析推理类"
        else:
            return "综合论述类"
            
    def analyze_question(self, query: str) -> Dict[str, Any]:
        """
        全面分析问题
        Args:
            query: 查询文本
        Returns:
            分析结果字典
        """
        complexity = self.detect_complexity(query)
        question_type = self.classify_question_type(query)
        complexity_score = self._calculate_complexity_score(query)
        simple_ratio = self._get_simple_feature_ratio(query)
        
        # 提取关键词
        words = list(jieba.cut(query))
        detected_keywords = [w for w in words if w in self.complexity_weights]
        
        analysis = {
            "query": query,
            "complexity": complexity,
            "question_type": question_type,
            "complexity_score": complexity_score,
            "simple_ratio": simple_ratio,
            "detected_keywords": detected_keywords,
            "query_length": len(query),
            "recommended_cot": complexity == "complex"
        }
        
        return analysis
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        Returns:
            统计信息字典
        """
        return {
            "total_complexity_keywords": len(self.complexity_weights),
            "question_types": list(self.question_type_keywords.keys()),
            "cognitive_levels": list(self.cognitive_levels.keys())
        }