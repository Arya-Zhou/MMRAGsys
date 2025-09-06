"""
质量监控器
监控CoT推理质量并进行评估
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from collections import deque
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """质量指标"""
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    relevance_score: float = 0.0
    total_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """质量报告"""
    query: str
    question_type: str
    metrics: QualityMetrics
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class QualityMonitor:
    """质量监控器"""
    
    def __init__(self,
                 quality_threshold: float = 0.6,
                 history_size: int = 1000):
        """
        初始化质量监控器
        Args:
            quality_threshold: 质量阈值
            history_size: 历史记录大小
        """
        self.quality_threshold = quality_threshold
        self.history_size = history_size
        
        # 质量历史记录
        self.quality_history = deque(maxlen=history_size)
        
        # 质量权重配置
        self.quality_weights = {
            "completeness": 0.3,
            "consistency": 0.4,
            "relevance": 0.3
        }
        
        # 问题类型质量基准
        self.type_benchmarks = {
            "事实查询类": {"completeness": 0.7, "consistency": 0.8, "relevance": 0.9},
            "分析推理类": {"completeness": 0.8, "consistency": 0.7, "relevance": 0.7},
            "对比评估类": {"completeness": 0.85, "consistency": 0.75, "relevance": 0.7},
            "因果解释类": {"completeness": 0.75, "consistency": 0.8, "relevance": 0.75},
            "流程规划类": {"completeness": 0.9, "consistency": 0.7, "relevance": 0.7},
            "综合论述类": {"completeness": 0.85, "consistency": 0.75, "relevance": 0.8}
        }
        
        # 统计数据
        self.stats = {
            "total_evaluations": 0,
            "passed": 0,
            "failed": 0,
            "average_score": 0.0
        }
        
        logger.info(f"初始化质量监控器: threshold={quality_threshold}")
        
    def evaluate_quality(self,
                        reasoning_chain: Any,
                        answer: str,
                        query: str,
                        question_type: str) -> QualityReport:
        """
        评估推理质量
        Args:
            reasoning_chain: 推理链
            answer: 答案
            query: 查询
            question_type: 问题类型
        Returns:
            质量报告
        """
        # 计算各维度分数
        completeness = self._evaluate_completeness(reasoning_chain, question_type)
        consistency = self._evaluate_consistency(reasoning_chain)
        relevance = self._evaluate_relevance(answer, query, reasoning_chain)
        
        # 计算总分
        total_score = (
            completeness * self.quality_weights["completeness"] +
            consistency * self.quality_weights["consistency"] +
            relevance * self.quality_weights["relevance"]
        )
        
        # 创建质量指标
        metrics = QualityMetrics(
            completeness_score=completeness,
            consistency_score=consistency,
            relevance_score=relevance,
            total_score=total_score
        )
        
        # 判断是否通过
        passed = total_score >= self.quality_threshold
        
        # 生成报告
        report = QualityReport(
            query=query,
            question_type=question_type,
            metrics=metrics,
            passed=passed
        )
        
        # 分析失败原因
        if not passed:
            report.failure_reasons = self._analyze_failure_reasons(metrics, question_type)
            report.suggestions = self._generate_improvement_suggestions(metrics, question_type)
            
        # 更新历史和统计
        self._update_history(report)
        
        logger.info(f"质量评估完成: score={total_score:.2f}, passed={passed}")
        
        return report
        
    def _evaluate_completeness(self, reasoning_chain: Any, question_type: str) -> float:
        """
        评估完整性
        Args:
            reasoning_chain: 推理链
            question_type: 问题类型
        Returns:
            完整性分数
        """
        if not hasattr(reasoning_chain, 'steps') or not reasoning_chain.steps:
            return 0.0
            
        # 步骤数量评估 (30%)
        expected_steps = self._get_expected_steps(question_type)
        actual_steps = len(reasoning_chain.steps)
        step_ratio = min(actual_steps / expected_steps, 1.0) if expected_steps > 0 else 0
        
        # 内容覆盖度评估 (70%)
        coverage_score = self._evaluate_content_coverage(reasoning_chain)
        
        completeness = step_ratio * 0.3 + coverage_score * 0.7
        
        return completeness
        
    def _evaluate_content_coverage(self, reasoning_chain: Any) -> float:
        """
        评估内容覆盖度
        Args:
            reasoning_chain: 推理链
        Returns:
            覆盖度分数
        """
        if not hasattr(reasoning_chain, 'steps'):
            return 0.0
            
        # 检查关键要素
        key_elements = {
            "问题分解": False,
            "信息检索": False,
            "推理过程": False,
            "结论支撑": False
        }
        
        for step in reasoning_chain.steps:
            content = step.content.lower() if hasattr(step, 'content') else ""
            
            if any(word in content for word in ["分析", "理解", "拆解", "问题"]):
                key_elements["问题分解"] = True
                
            if any(word in content for word in ["文档", "信息", "提取", "发现"]):
                key_elements["信息检索"] = True
                
            if any(word in content for word in ["推理", "推导", "因此", "所以"]):
                key_elements["推理过程"] = True
                
            if any(word in content for word in ["结论", "答案", "综合", "总结"]):
                key_elements["结论支撑"] = True
                
        # 计算覆盖率
        coverage = sum(key_elements.values()) / len(key_elements)
        
        # 检查内容长度
        total_content = sum(len(step.content) if hasattr(step, 'content') else 0 
                          for step in reasoning_chain.steps)
        length_score = min(total_content / 500, 1.0)  # 假设500字符为适当长度
        
        return coverage * 0.6 + length_score * 0.4
        
    def _evaluate_consistency(self, reasoning_chain: Any) -> float:
        """
        评估一致性
        Args:
            reasoning_chain: 推理链
        Returns:
            一致性分数
        """
        if not hasattr(reasoning_chain, 'steps') or len(reasoning_chain.steps) < 2:
            return 1.0  # 单步或无步骤时认为一致
            
        # 语义相似度评估 (40%)
        semantic_score = self._evaluate_semantic_consistency(reasoning_chain)
        
        # 逻辑关联度评估 (60%)
        logical_score = self._evaluate_logical_consistency(reasoning_chain)
        
        consistency = semantic_score * 0.4 + logical_score * 0.6
        
        return consistency
        
    def _evaluate_semantic_consistency(self, reasoning_chain: Any) -> float:
        """
        评估语义一致性
        Args:
            reasoning_chain: 推理链
        Returns:
            语义一致性分数
        """
        consistency_scores = []
        
        for i in range(1, len(reasoning_chain.steps)):
            prev_step = reasoning_chain.steps[i-1]
            curr_step = reasoning_chain.steps[i]
            
            # 简化的语义相似度计算
            prev_words = set(prev_step.content.lower().split() if hasattr(prev_step, 'content') else [])
            curr_words = set(curr_step.content.lower().split() if hasattr(curr_step, 'content') else [])
            
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words)
                total = len(prev_words | curr_words)
                similarity = overlap / total if total > 0 else 0
                consistency_scores.append(similarity)
                
        return np.mean(consistency_scores) if consistency_scores else 0.5
        
    def _evaluate_logical_consistency(self, reasoning_chain: Any) -> float:
        """
        评估逻辑一致性
        Args:
            reasoning_chain: 推理链
        Returns:
            逻辑一致性分数
        """
        # 检查因果关系
        causal_score = self._check_causal_consistency(reasoning_chain)
        
        # 检查论据支撑
        support_score = self._check_argument_support(reasoning_chain)
        
        # 检查概念一致性
        concept_score = self._check_concept_consistency(reasoning_chain)
        
        return (causal_score + support_score + concept_score) / 3
        
    def _check_causal_consistency(self, reasoning_chain: Any) -> float:
        """
        检查因果关系一致性
        Args:
            reasoning_chain: 推理链
        Returns:
            因果一致性分数
        """
        causal_markers = ["因此", "所以", "导致", "因为", "由于", "结果"]
        score = 0.5  # 基础分
        
        for i in range(1, len(reasoning_chain.steps)):
            curr_content = reasoning_chain.steps[i].content if hasattr(reasoning_chain.steps[i], 'content') else ""
            
            # 检查是否有因果标记
            has_causal = any(marker in curr_content for marker in causal_markers)
            if has_causal:
                score += 0.1
                
        return min(score, 1.0)
        
    def _check_argument_support(self, reasoning_chain: Any) -> float:
        """
        检查论据支撑强度
        Args:
            reasoning_chain: 推理链
        Returns:
            支撑强度分数
        """
        # 检查是否有证据引用
        evidence_markers = ["根据", "显示", "表明", "证明", "文档", "数据"]
        evidence_count = 0
        
        for step in reasoning_chain.steps:
            content = step.content if hasattr(step, 'content') else ""
            if any(marker in content for marker in evidence_markers):
                evidence_count += 1
                
        return min(evidence_count / max(len(reasoning_chain.steps), 1), 1.0)
        
    def _check_concept_consistency(self, reasoning_chain: Any) -> float:
        """
        检查概念使用一致性
        Args:
            reasoning_chain: 推理链
        Returns:
            概念一致性分数
        """
        # 简化实现：检查关键词在多个步骤中的出现
        all_content = " ".join(step.content if hasattr(step, 'content') else "" 
                              for step in reasoning_chain.steps)
        
        # 提取可能的关键概念（长度>2的词）
        words = all_content.split()
        key_concepts = [w for w in words if len(w) > 2]
        
        if not key_concepts:
            return 0.5
            
        # 统计概念重复使用
        concept_counts = {}
        for concept in key_concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
            
        # 概念重用表示一致性
        reused_concepts = sum(1 for count in concept_counts.values() if count > 1)
        consistency = reused_concepts / len(set(key_concepts))
        
        return min(consistency * 2, 1.0)  # 放大效果
        
    def _evaluate_relevance(self, answer: str, query: str, reasoning_chain: Any) -> float:
        """
        评估相关性
        Args:
            answer: 答案
            query: 查询
            reasoning_chain: 推理链
        Returns:
            相关性分数
        """
        # 答案与问题的相关性
        answer_relevance = self._calculate_answer_relevance(answer, query)
        
        # 推理步骤与问题的相关性
        steps_relevance = self._calculate_steps_relevance(reasoning_chain, query)
        
        return answer_relevance * 0.6 + steps_relevance * 0.4
        
    def _calculate_answer_relevance(self, answer: str, query: str) -> float:
        """
        计算答案相关性
        Args:
            answer: 答案
            query: 查询
        Returns:
            相关性分数
        """
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        if not query_words:
            return 0.5
            
        # 计算词重叠度
        overlap = len(query_words & answer_words)
        relevance = overlap / len(query_words)
        
        # 检查是否直接回答了问题
        question_markers = ["什么", "为什么", "如何", "怎么", "是否", "多少"]
        for marker in question_markers:
            if marker in query:
                # 检查答案是否包含相应的回答模式
                if marker == "是否" and any(word in answer for word in ["是", "否", "不"]):
                    relevance += 0.2
                elif marker in ["什么", "多少"] and len(answer) > 10:
                    relevance += 0.1
                    
        return min(relevance, 1.0)
        
    def _calculate_steps_relevance(self, reasoning_chain: Any, query: str) -> float:
        """
        计算推理步骤相关性
        Args:
            reasoning_chain: 推理链
            query: 查询
        Returns:
            相关性分数
        """
        if not hasattr(reasoning_chain, 'steps') or not reasoning_chain.steps:
            return 0.0
            
        query_words = set(query.lower().split())
        relevance_scores = []
        
        for step in reasoning_chain.steps:
            content = step.content if hasattr(step, 'content') else ""
            step_words = set(content.lower().split())
            
            if step_words:
                overlap = len(query_words & step_words)
                relevance = overlap / (len(query_words) + 1)
                relevance_scores.append(relevance)
                
        return np.mean(relevance_scores) if relevance_scores else 0.0
        
    def _get_expected_steps(self, question_type: str) -> int:
        """
        获取预期步骤数
        Args:
            question_type: 问题类型
        Returns:
            预期步骤数
        """
        step_expectations = {
            "事实查询类": 3,
            "分析推理类": 5,
            "对比评估类": 6,
            "因果解释类": 5,
            "流程规划类": 7,
            "综合论述类": 8
        }
        
        return step_expectations.get(question_type, 5)
        
    def _analyze_failure_reasons(self, metrics: QualityMetrics, question_type: str) -> List[str]:
        """
        分析失败原因
        Args:
            metrics: 质量指标
            question_type: 问题类型
        Returns:
            失败原因列表
        """
        reasons = []
        benchmarks = self.type_benchmarks.get(question_type, {})
        
        if metrics.completeness_score < benchmarks.get("completeness", 0.7):
            reasons.append(f"完整性不足 ({metrics.completeness_score:.2f} < {benchmarks.get('completeness', 0.7)})")
            
        if metrics.consistency_score < benchmarks.get("consistency", 0.7):
            reasons.append(f"一致性较差 ({metrics.consistency_score:.2f} < {benchmarks.get('consistency', 0.7)})")
            
        if metrics.relevance_score < benchmarks.get("relevance", 0.7):
            reasons.append(f"相关性偏低 ({metrics.relevance_score:.2f} < {benchmarks.get('relevance', 0.7)})")
            
        if metrics.total_score < self.quality_threshold:
            reasons.append(f"总体质量未达标 ({metrics.total_score:.2f} < {self.quality_threshold})")
            
        return reasons
        
    def _generate_improvement_suggestions(self, metrics: QualityMetrics, question_type: str) -> List[str]:
        """
        生成改进建议
        Args:
            metrics: 质量指标
            question_type: 问题类型
        Returns:
            改进建议列表
        """
        suggestions = []
        
        if metrics.completeness_score < 0.7:
            suggestions.append("增加推理步骤，确保覆盖所有关键要素")
            suggestions.append("扩展每个步骤的内容，提供更详细的分析")
            
        if metrics.consistency_score < 0.7:
            suggestions.append("加强步骤间的逻辑关联")
            suggestions.append("保持概念和术语使用的一致性")
            
        if metrics.relevance_score < 0.7:
            suggestions.append("确保答案直接回应问题")
            suggestions.append("在推理步骤中保持与问题的相关性")
            
        if question_type == "分析推理类" and metrics.consistency_score < 0.8:
            suggestions.append("加强因果关系的论证")
            
        if question_type == "对比评估类" and metrics.completeness_score < 0.8:
            suggestions.append("确保对比的各个维度都有覆盖")
            
        return suggestions
        
    def _update_history(self, report: QualityReport):
        """
        更新历史记录
        Args:
            report: 质量报告
        """
        self.quality_history.append(report)
        
        # 更新统计
        self.stats["total_evaluations"] += 1
        if report.passed:
            self.stats["passed"] += 1
        else:
            self.stats["failed"] += 1
            
        # 更新平均分数
        all_scores = [r.metrics.total_score for r in self.quality_history]
        self.stats["average_score"] = np.mean(all_scores) if all_scores else 0.0
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        Returns:
            统计信息字典
        """
        type_stats = {}
        for report in self.quality_history:
            qtype = report.question_type
            if qtype not in type_stats:
                type_stats[qtype] = {"count": 0, "passed": 0, "avg_score": []}
                
            type_stats[qtype]["count"] += 1
            if report.passed:
                type_stats[qtype]["passed"] += 1
            type_stats[qtype]["avg_score"].append(report.metrics.total_score)
            
        # 计算每种类型的平均分
        for qtype in type_stats:
            scores = type_stats[qtype]["avg_score"]
            type_stats[qtype]["avg_score"] = np.mean(scores) if scores else 0.0
            type_stats[qtype]["pass_rate"] = type_stats[qtype]["passed"] / type_stats[qtype]["count"]
            
        return {
            "overall": self.stats,
            "by_type": type_stats,
            "quality_threshold": self.quality_threshold,
            "history_size": len(self.quality_history)
        }
        
    def export_report(self, filepath: str):
        """
        导出质量报告
        Args:
            filepath: 输出文件路径
        """
        reports_data = []
        for report in self.quality_history:
            reports_data.append({
                "query": report.query[:100],
                "question_type": report.question_type,
                "completeness": report.metrics.completeness_score,
                "consistency": report.metrics.consistency_score,
                "relevance": report.metrics.relevance_score,
                "total_score": report.metrics.total_score,
                "passed": report.passed,
                "failure_reasons": report.failure_reasons,
                "suggestions": report.suggestions,
                "timestamp": report.metrics.timestamp
            })
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "statistics": self.get_statistics(),
                "reports": reports_data
            }, f, ensure_ascii=False, indent=2)
            
        logger.info(f"导出质量报告到 {filepath}")