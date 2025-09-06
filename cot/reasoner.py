"""
CoT推理引擎
实现Chain-of-Thought推理链，提升复杂问题的答案质量
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_number: int
    description: str
    content: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningChain:
    """推理链"""
    query: str
    question_type: str
    steps: List[ReasoningStep] = field(default_factory=list)
    answer: str = ""
    quality_score: float = 0.0
    tracking_marks: List[str] = field(default_factory=list)
    total_time: float = 0.0
    retries: int = 0


class CoTReasoner:
    """CoT推理引擎"""
    
    def __init__(self,
                 min_steps: int = 3,
                 max_steps: int = 8,
                 quality_threshold: float = 0.6,
                 max_retries: int = 3):
        """
        初始化CoT推理引擎
        Args:
            min_steps: 最小推理步数
            max_steps: 最大推理步数
            quality_threshold: 质量阈值
            max_retries: 最大重试次数
        """
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.quality_threshold = quality_threshold
        self.max_retries = max_retries
        
        # 超时配置
        self.timeout_config = {
            3: 10,   # 3步10秒
            5: 20,   # 5步20秒
            8: 35    # 8步35秒
        }
        
        # Temperature配置
        self.temperature_schedule = [0.7, 0.5, 0.3, 0.9]
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"初始化CoT推理引擎: steps={min_steps}-{max_steps}, quality_threshold={quality_threshold}")
        
    def process_with_cot(self,
                        query: str,
                        docs_with_tracking: Dict[str, Any],
                        question_type: str,
                        data_type: str = 'text') -> Tuple[str, List[str]]:
        """
        使用CoT处理查询
        Args:
            query: 查询文本
            docs_with_tracking: 带追踪信息的文档
            question_type: 问题类型
            data_type: 数据类型
        Returns:
            (答案, 追踪标记列表)
        """
        start_time = time.time()
        
        # 确定推理步数
        num_steps = self._determine_steps(question_type)
        timeout = self._get_timeout(num_steps)
        
        # 创建推理链
        reasoning_chain = ReasoningChain(
            query=query,
            question_type=question_type,
            tracking_marks=docs_with_tracking.get('tracking', [])
        )
        
        # 执行推理
        for retry in range(self.max_retries):
            try:
                # 设置temperature
                temperature = self.temperature_schedule[min(retry, len(self.temperature_schedule)-1)]
                
                # 执行推理步骤
                reasoning_chain = self._execute_reasoning(
                    reasoning_chain,
                    docs_with_tracking,
                    num_steps,
                    temperature,
                    timeout,
                    data_type
                )
                
                # 评估质量
                quality_score = self.evaluate_quality(reasoning_chain)
                reasoning_chain.quality_score = quality_score
                
                if quality_score >= self.quality_threshold:
                    break
                else:
                    logger.warning(f"推理质量不足 ({quality_score:.2f}), 重试 {retry+1}/{self.max_retries}")
                    reasoning_chain.retries = retry + 1
                    
            except TimeoutError:
                logger.error(f"推理超时 (>{timeout}秒)")
                reasoning_chain.answer = self._generate_fallback_answer(query, docs_with_tracking)
                break
                
            except Exception as e:
                logger.error(f"推理失败: {e}")
                reasoning_chain.answer = self._generate_fallback_answer(query, docs_with_tracking)
                break
                
        # 记录总时间
        reasoning_chain.total_time = time.time() - start_time
        
        # 格式化输出
        formatted_answer = self._format_cot_output(reasoning_chain)
        
        logger.info(f"CoT推理完成: steps={len(reasoning_chain.steps)}, quality={reasoning_chain.quality_score:.2f}, time={reasoning_chain.total_time:.2f}s")
        
        return formatted_answer, reasoning_chain.tracking_marks
        
    def _determine_steps(self, question_type: str) -> int:
        """
        根据问题类型确定推理步数
        Args:
            question_type: 问题类型
        Returns:
            推理步数
        """
        step_mapping = {
            "事实查询类": 3,
            "分析推理类": 5,
            "对比评估类": 6,
            "因果解释类": 5,
            "流程规划类": 7,
            "综合论述类": 8
        }
        
        return step_mapping.get(question_type, 5)
        
    def _get_timeout(self, num_steps: int) -> int:
        """
        获取超时时间
        Args:
            num_steps: 推理步数
        Returns:
            超时时间（秒）
        """
        # 找到最接近的配置
        for steps, timeout in sorted(self.timeout_config.items()):
            if num_steps <= steps:
                return timeout
        return 35  # 默认最大超时
        
    def _execute_reasoning(self,
                          reasoning_chain: ReasoningChain,
                          docs_with_tracking: Dict[str, Any],
                          num_steps: int,
                          temperature: float,
                          timeout: int,
                          data_type: str) -> ReasoningChain:
        """
        执行推理步骤
        Args:
            reasoning_chain: 推理链
            docs_with_tracking: 文档内容
            num_steps: 推理步数
            temperature: 生成温度
            timeout: 超时时间
            data_type: 数据类型
        Returns:
            更新后的推理链
        """
        # 获取文档内容
        documents = docs_with_tracking.get('content', [])
        
        # 根据数据类型选择模板
        template = self._get_template(data_type)
        
        # 执行每个推理步骤
        for step_num in range(1, min(num_steps + 1, self.max_steps + 1)):
            step = self._generate_step(
                step_num,
                reasoning_chain.query,
                documents,
                reasoning_chain.steps,
                template,
                temperature
            )
            reasoning_chain.steps.append(step)
            
            # 检查是否可以提前结束
            if self._should_stop_early(reasoning_chain):
                break
                
        # 生成最终答案
        reasoning_chain.answer = self._synthesize_answer(
            reasoning_chain.query,
            reasoning_chain.steps,
            documents
        )
        
        return reasoning_chain
        
    def _generate_step(self,
                      step_num: int,
                      query: str,
                      documents: List[str],
                      previous_steps: List[ReasoningStep],
                      template: Dict[str, str],
                      temperature: float) -> ReasoningStep:
        """
        生成单个推理步骤
        Args:
            step_num: 步骤编号
            query: 查询
            documents: 文档列表
            previous_steps: 之前的步骤
            template: 模板
            temperature: 生成温度
        Returns:
            推理步骤
        """
        # 获取步骤描述
        step_descriptions = template.get('steps', [])
        if step_num <= len(step_descriptions):
            description = step_descriptions[step_num - 1]
        else:
            description = f"Step{step_num}: 深入分析"
            
        # 生成步骤内容（这里简化处理，实际应调用LLM）
        content = self._simulate_step_generation(
            query,
            documents,
            previous_steps,
            description,
            temperature
        )
        
        # 计算置信度
        confidence = self._calculate_confidence(content, documents)
        
        return ReasoningStep(
            step_number=step_num,
            description=description,
            content=content,
            confidence=confidence
        )
        
    def _simulate_step_generation(self,
                                 query: str,
                                 documents: List[str],
                                 previous_steps: List[ReasoningStep],
                                 description: str,
                                 temperature: float) -> str:
        """
        模拟步骤生成（实际应调用LLM）
        Args:
            query: 查询
            documents: 文档
            previous_steps: 之前的步骤
            description: 步骤描述
            temperature: 温度
        Returns:
            步骤内容
        """
        # 这里是模拟实现，实际应该调用语言模型
        if "问题理解" in description or "识别" in description:
            return f"分析问题：{query[:50]}... 需要从文档中提取相关信息。"
            
        elif "证据提取" in description or "特征" in description:
            if documents:
                return f"从文档中发现关键信息：{documents[0][:100]}..."
            return "正在分析相关文档内容..."
            
        elif "推理" in description or "分析" in description:
            return "基于提取的信息进行逻辑推理，建立因果关系..."
            
        elif "结论" in description or "综合" in description:
            return "综合所有信息，形成最终结论..."
            
        else:
            return f"执行{description}..."
            
    def _synthesize_answer(self,
                          query: str,
                          steps: List[ReasoningStep],
                          documents: List[str]) -> str:
        """
        综合推理步骤生成答案
        Args:
            query: 查询
            steps: 推理步骤
            documents: 文档
        Returns:
            最终答案
        """
        # 这里是简化实现，实际应该基于推理链生成答案
        if not steps:
            return "无法生成答案"
            
        # 提取关键信息
        key_points = []
        for step in steps:
            if step.confidence > 0.5:
                key_points.append(step.content)
                
        # 生成答案
        if key_points:
            answer = "基于分析，" + " ".join(key_points[:2])
            if documents:
                answer += f" 相关文档显示：{documents[0][:100]}..."
        else:
            answer = "根据可用信息，" + (documents[0][:200] if documents else "无法找到相关内容")
            
        return answer
        
    def _should_stop_early(self, reasoning_chain: ReasoningChain) -> bool:
        """
        判断是否可以提前结束推理
        Args:
            reasoning_chain: 推理链
        Returns:
            是否停止
        """
        if len(reasoning_chain.steps) < self.min_steps:
            return False
            
        # 检查最近步骤的置信度
        recent_confidences = [s.confidence for s in reasoning_chain.steps[-3:]]
        if all(c > 0.9 for c in recent_confidences):
            logger.info("高置信度，提前结束推理")
            return True
            
        return False
        
    def _get_template(self, data_type: str) -> Dict[str, str]:
        """
        获取推理模板
        Args:
            data_type: 数据类型
        Returns:
            模板字典
        """
        templates = {
            'text': {
                'intro': "让我们一步步分析：",
                'steps': [
                    "Step1: 问题理解与拆解",
                    "Step2: 证据提取与分析",
                    "Step3: 逻辑推理与综合",
                    "Step4: 结论生成与验证",
                    "Step5: 补充分析",
                    "Step6: 深度推理",
                    "Step7: 全面总结",
                    "Step8: 最终验证"
                ]
            },
            'image': {
                'intro': "让我们逐步分析：",
                'steps': [
                    "Step1: 视觉内容识别",
                    "Step2: 关键特征分析",
                    "Step3: 语义理解推理",
                    "Step4: 综合描述生成",
                    "Step5: 细节补充",
                    "Step6: 关联分析",
                    "Step7: 完整描述",
                    "Step8: 质量验证"
                ]
            },
            'table': {
                'intro': "让我们系统分析：",
                'steps': [
                    "Step1: 数据结构理解",
                    "Step2: 关键指标提取",
                    "Step3: 趋势模式分析",
                    "Step4: 洞察总结输出",
                    "Step5: 相关性分析",
                    "Step6: 异常检测",
                    "Step7: 预测推断",
                    "Step8: 综合报告"
                ]
            }
        }
        
        return templates.get(data_type, templates['text'])
        
    def _calculate_confidence(self, content: str, documents: List[str]) -> float:
        """
        计算步骤置信度
        Args:
            content: 步骤内容
            documents: 文档列表
        Returns:
            置信度分数
        """
        # 简化的置信度计算
        confidence = 0.5
        
        # 检查内容长度
        if len(content) > 50:
            confidence += 0.1
            
        # 检查是否引用文档
        if documents and any(doc[:20] in content for doc in documents):
            confidence += 0.2
            
        # 检查关键词
        keywords = ["因此", "所以", "根据", "基于", "分析", "显示"]
        if any(kw in content for kw in keywords):
            confidence += 0.1
            
        return min(confidence, 1.0)
        
    def evaluate_quality(self, reasoning_chain: ReasoningChain) -> float:
        """
        评估推理质量
        Args:
            reasoning_chain: 推理链
        Returns:
            质量分数(0-1)
        """
        if not reasoning_chain.steps:
            return 0.0
            
        # 完整性评估 (0.3)
        completeness_score = self._evaluate_completeness(reasoning_chain)
        
        # 一致性评估 (0.4)
        consistency_score = self._evaluate_consistency(reasoning_chain)
        
        # 相关性评估 (0.3)
        relevance_score = self._evaluate_relevance(reasoning_chain)
        
        # 加权总分
        total_score = (
            completeness_score * 0.3 +
            consistency_score * 0.4 +
            relevance_score * 0.3
        )
        
        return total_score
        
    def _evaluate_completeness(self, reasoning_chain: ReasoningChain) -> float:
        """
        评估完整性
        Args:
            reasoning_chain: 推理链
        Returns:
            完整性分数
        """
        # 步骤数量评估
        step_count = len(reasoning_chain.steps)
        expected_steps = self._determine_steps(reasoning_chain.question_type)
        step_ratio = min(step_count / expected_steps, 1.0)
        
        # 内容覆盖度评估
        total_content_length = sum(len(s.content) for s in reasoning_chain.steps)
        coverage = min(total_content_length / 500, 1.0)  # 假设500字符为完整
        
        return step_ratio * 0.3 + coverage * 0.7
        
    def _evaluate_consistency(self, reasoning_chain: ReasoningChain) -> float:
        """
        评估一致性
        Args:
            reasoning_chain: 推理链
        Returns:
            一致性分数
        """
        if len(reasoning_chain.steps) < 2:
            return 1.0
            
        # 检查步骤间的逻辑关联
        consistency_scores = []
        
        for i in range(1, len(reasoning_chain.steps)):
            prev_step = reasoning_chain.steps[i-1]
            curr_step = reasoning_chain.steps[i]
            
            # 简化的一致性检查
            score = 0.5
            
            # 检查是否有引用关系
            if any(word in curr_step.content for word in prev_step.content.split()[:5]):
                score += 0.3
                
            # 检查置信度变化
            if abs(curr_step.confidence - prev_step.confidence) < 0.3:
                score += 0.2
                
            consistency_scores.append(score)
            
        return np.mean(consistency_scores) if consistency_scores else 0.5
        
    def _evaluate_relevance(self, reasoning_chain: ReasoningChain) -> float:
        """
        评估相关性
        Args:
            reasoning_chain: 推理链
        Returns:
            相关性分数
        """
        query_words = set(reasoning_chain.query.lower().split())
        
        # 检查答案相关性
        answer_words = set(reasoning_chain.answer.lower().split())
        answer_relevance = len(query_words & answer_words) / (len(query_words) + 1)
        
        # 检查步骤相关性
        step_relevance_scores = []
        for step in reasoning_chain.steps:
            step_words = set(step.content.lower().split())
            relevance = len(query_words & step_words) / (len(query_words) + 1)
            step_relevance_scores.append(relevance)
            
        avg_step_relevance = np.mean(step_relevance_scores) if step_relevance_scores else 0
        
        return answer_relevance * 0.6 + avg_step_relevance * 0.4
        
    def _format_cot_output(self, reasoning_chain: ReasoningChain) -> str:
        """
        格式化CoT输出
        Args:
            reasoning_chain: 推理链
        Returns:
            格式化的输出
        """
        if reasoning_chain.quality_score < self.quality_threshold:
            # 质量不足，返回简单答案
            return reasoning_chain.answer + " (NoUseCoT)"
            
        # 构建完整的推理输出
        output_parts = []
        
        # 添加推理过程
        template = self._get_template('text')
        output_parts.append(template['intro'])
        
        for step in reasoning_chain.steps:
            output_parts.append(f"\n{step.description}")
            output_parts.append(step.content)
            
        # 添加最终答案
        output_parts.append(f"\n\n最终答案：{reasoning_chain.answer}")
        
        return "\n".join(output_parts)
        
    def _generate_fallback_answer(self, query: str, docs_with_tracking: Dict[str, Any]) -> str:
        """
        生成降级答案
        Args:
            query: 查询
            docs_with_tracking: 文档
        Returns:
            降级答案
        """
        documents = docs_with_tracking.get('content', [])
        
        if documents:
            # 使用第一个文档生成简单答案
            return f"根据相关文档：{documents[0][:200]}... (NoUseCoT)"
        else:
            return "无法生成详细答案，请提供更多信息。(NoUseCoT)"
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        Returns:
            统计信息字典
        """
        return {
            "min_steps": self.min_steps,
            "max_steps": self.max_steps,
            "quality_threshold": self.quality_threshold,
            "max_retries": self.max_retries,
            "timeout_config": self.timeout_config
        }