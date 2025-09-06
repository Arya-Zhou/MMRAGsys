"""
模板管理器
管理不同数据类型的CoT推理模板
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReasoningTemplate:
    """推理模板"""
    name: str
    data_type: str
    intro: str
    steps: List[str]
    conclusion_format: str = "最终答案：{answer}"
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemplateManager:
    """模板管理器"""
    
    def __init__(self):
        """初始化模板管理器"""
        self.templates = {}
        self._init_default_templates()
        logger.info("初始化模板管理器")
        
    def _init_default_templates(self):
        """初始化默认模板"""
        
        # Text类型模板
        self.templates['text'] = ReasoningTemplate(
            name="文本推理模板",
            data_type="text",
            intro="让我们一步步分析：",
            steps=[
                "Step1: 问题理解与拆解\n仔细分析问题的核心需求，识别关键概念和约束条件。",
                "Step2: 证据提取与分析\n从相关文档中提取支持性证据，评估信息的可靠性和相关性。",
                "Step3: 逻辑推理与综合\n基于提取的证据进行逻辑推导，建立因果关系和推理链条。",
                "Step4: 结论生成与验证\n综合所有信息形成结论，并验证结论的合理性和完整性。",
                "Step5: 补充分析\n识别可能遗漏的要点，补充必要的细节和解释。",
                "Step6: 深度推理\n探索更深层的含义和隐含的关联，提供更全面的理解。",
                "Step7: 全面总结\n整合所有分析结果，形成系统性的总结。",
                "Step8: 最终验证\n回顾整个推理过程，确保逻辑一致性和答案的准确性。"
            ],
            conclusion_format="基于以上分析，最终答案是：{answer}"
        )
        
        # Image类型模板
        self.templates['image'] = ReasoningTemplate(
            name="图像推理模板",
            data_type="image",
            intro="让我们逐步分析这个图像：",
            steps=[
                "Step1: 视觉内容识别\n识别图像中的主要对象、场景和视觉元素。",
                "Step2: 关键特征分析\n分析颜色、形状、纹理、空间关系等关键视觉特征。",
                "Step3: 语义理解推理\n理解图像传达的含义，推断场景的上下文和目的。",
                "Step4: 综合描述生成\n基于视觉分析生成完整的图像描述。",
                "Step5: 细节补充\n注意图像中的细微特征和可能被忽略的细节。",
                "Step6: 关联分析\n分析图像元素之间的关系和相互作用。",
                "Step7: 完整描述\n提供包含所有重要信息的全面描述。",
                "Step8: 质量验证\n确保描述的准确性和完整性。"
            ],
            conclusion_format="综合视觉分析，得出：{answer}"
        )
        
        # Table类型模板
        self.templates['table'] = ReasoningTemplate(
            name="表格推理模板",
            data_type="table",
            intro="让我们系统分析这个表格数据：",
            steps=[
                "Step1: 数据结构理解\n识别表格的列名、行标签和数据类型。",
                "Step2: 关键指标提取\n识别并提取最重要的数据指标和度量。",
                "Step3: 趋势模式分析\n分析数据中的趋势、模式和规律。",
                "Step4: 洞察总结输出\n基于数据分析生成关键洞察和发现。",
                "Step5: 相关性分析\n探索不同数据维度之间的相关关系。",
                "Step6: 异常检测\n识别数据中的异常值和特殊模式。",
                "Step7: 预测推断\n基于当前数据推断可能的趋势和结果。",
                "Step8: 综合报告\n生成包含所有重要发现的综合分析报告。"
            ],
            conclusion_format="数据分析结论：{answer}"
        )
        
        # 问题类型特定模板
        self._init_question_type_templates()
        
    def _init_question_type_templates(self):
        """初始化问题类型特定模板"""
        
        # 事实查询类模板
        self.templates['fact_query'] = ReasoningTemplate(
            name="事实查询模板",
            data_type="text",
            intro="查找相关事实信息：",
            steps=[
                "Step1: 识别查询目标\n明确需要查找的具体信息。",
                "Step2: 定位信息源\n在文档中定位包含目标信息的部分。",
                "Step3: 提取核心事实\n准确提取所需的事实信息。",
                "Step4: 验证准确性\n确认提取信息的准确性和完整性。"
            ],
            conclusion_format="查询结果：{answer}"
        )
        
        # 分析推理类模板
        self.templates['analysis'] = ReasoningTemplate(
            name="分析推理模板",
            data_type="text",
            intro="进行深入分析和推理：",
            steps=[
                "Step1: 问题分解\n将复杂问题分解为可管理的子问题。",
                "Step2: 假设建立\n基于已知信息建立工作假设。",
                "Step3: 逻辑推导\n通过逻辑推理验证或否定假设。",
                "Step4: 证据评估\n评估支持和反对的证据。",
                "Step5: 综合判断\n基于所有分析形成综合判断。",
                "Step6: 结论验证\n验证结论的逻辑一致性。"
            ],
            conclusion_format="分析结论：{answer}"
        )
        
        # 对比评估类模板
        self.templates['comparison'] = ReasoningTemplate(
            name="对比评估模板",
            data_type="text",
            intro="进行系统性对比评估：",
            steps=[
                "Step1: 识别比较对象\n明确需要比较的各个对象或方案。",
                "Step2: 确定比较维度\n建立比较的标准和维度。",
                "Step3: 逐项对比\n在每个维度上进行详细对比。",
                "Step4: 优劣分析\n分析各对象的优势和劣势。",
                "Step5: 权重考量\n考虑不同维度的重要性权重。",
                "Step6: 综合评估\n形成综合的评估结论。"
            ],
            conclusion_format="对比评估结果：{answer}"
        )
        
    def get_template(self, 
                    data_type: str = 'text',
                    question_type: Optional[str] = None) -> ReasoningTemplate:
        """
        获取推理模板
        Args:
            data_type: 数据类型
            question_type: 问题类型（可选）
        Returns:
            推理模板
        """
        # 优先使用问题类型特定模板
        if question_type:
            type_mapping = {
                "事实查询类": "fact_query",
                "分析推理类": "analysis",
                "对比评估类": "comparison"
            }
            
            template_key = type_mapping.get(question_type)
            if template_key and template_key in self.templates:
                return self.templates[template_key]
                
        # 使用数据类型模板
        if data_type in self.templates:
            return self.templates[data_type]
            
        # 默认使用text模板
        return self.templates['text']
        
    def customize_template(self,
                          name: str,
                          data_type: str,
                          intro: str,
                          steps: List[str],
                          conclusion_format: str = "最终答案：{answer}"):
        """
        自定义模板
        Args:
            name: 模板名称
            data_type: 数据类型
            intro: 介绍语
            steps: 步骤列表
            conclusion_format: 结论格式
        """
        custom_template = ReasoningTemplate(
            name=name,
            data_type=data_type,
            intro=intro,
            steps=steps,
            conclusion_format=conclusion_format
        )
        
        # 使用名称作为键存储
        template_key = f"custom_{name.replace(' ', '_').lower()}"
        self.templates[template_key] = custom_template
        
        logger.info(f"添加自定义模板: {template_key}")
        
    def format_reasoning_output(self,
                               template: ReasoningTemplate,
                               reasoning_steps: List[Dict[str, str]],
                               answer: str) -> str:
        """
        格式化推理输出
        Args:
            template: 推理模板
            reasoning_steps: 推理步骤列表
            answer: 最终答案
        Returns:
            格式化的输出
        """
        output_parts = []
        
        # 添加介绍语
        output_parts.append(template.intro)
        
        # 添加推理步骤
        for i, step_content in enumerate(reasoning_steps):
            if i < len(template.steps):
                step_header = template.steps[i].split('\\n')[0]  # 获取步骤标题
                output_parts.append(f"\n{step_header}")
                output_parts.append(step_content.get('content', ''))
                
        # 添加结论
        output_parts.append(f"\n\n{template.conclusion_format.format(answer=answer)}")
        
        return "\n".join(output_parts)
        
    def get_step_prompt(self,
                       template: ReasoningTemplate,
                       step_number: int) -> str:
        """
        获取特定步骤的提示
        Args:
            template: 推理模板
            step_number: 步骤编号（从1开始）
        Returns:
            步骤提示
        """
        if step_number <= 0 or step_number > len(template.steps):
            return f"Step{step_number}: 继续分析"
            
        return template.steps[step_number - 1]
        
    def list_templates(self) -> List[str]:
        """
        列出所有可用模板
        Returns:
            模板名称列表
        """
        return list(self.templates.keys())
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        Returns:
            统计信息字典
        """
        stats = {
            "total_templates": len(self.templates),
            "template_types": {},
            "average_steps": 0
        }
        
        total_steps = 0
        for key, template in self.templates.items():
            data_type = template.data_type
            if data_type not in stats["template_types"]:
                stats["template_types"][data_type] = 0
            stats["template_types"][data_type] += 1
            total_steps += len(template.steps)
            
        if self.templates:
            stats["average_steps"] = total_steps / len(self.templates)
            
        return stats