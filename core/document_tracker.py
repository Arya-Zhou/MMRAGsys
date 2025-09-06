"""
文档追踪系统
负责全流程的文档来源追踪和标记管理
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrackingMark:
    """追踪标记数据结构"""
    source_file: str
    page_number: int
    doc_id: str = ""
    content_snippet: str = ""
    relevance_score: float = 0.0
    
    def to_string(self) -> str:
        """转换为标准格式字符串"""
        return f"{self.source_file}-p{self.page_number}"
    
    @classmethod
    def from_string(cls, mark_str: str) -> 'TrackingMark':
        """从标准格式字符串解析"""
        pattern = r"(.+)-p(\d+)"
        match = re.match(pattern, mark_str)
        if match:
            source_file = match.group(1)
            page_number = int(match.group(2))
            return cls(source_file=source_file, page_number=page_number)
        else:
            raise ValueError(f"无效的追踪标记格式: {mark_str}")


@dataclass
class TrackedDocument:
    """带追踪信息的文档"""
    content: str
    tracking_mark: TrackingMark
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[Any] = None


class DocumentTracker:
    """文档追踪管理器"""
    
    def __init__(self):
        """初始化文档追踪器"""
        self.tracking_history: List[TrackingMark] = []
        self.document_registry: Dict[str, TrackedDocument] = {}
        self.active_tracking_marks: List[str] = []
        
        logger.info("初始化文档追踪系统")
        
    def create_tracking_marks(self, retrieved_docs: List[Any]) -> Tuple[List[str], Dict]:
        """
        为检索到的文档创建追踪标记
        Args:
            retrieved_docs: 检索到的文档列表
        Returns:
            (追踪标记列表, 带追踪信息的文档字典)
        """
        tracking_marks = []
        docs_with_tracking = {
            'content': [],
            'tracking': []
        }
        
        for doc in retrieved_docs:
            # 提取文档信息
            source_file = None
            page_number = None
            
            # 尝试从不同的属性中获取信息
            if hasattr(doc, 'source_file'):
                source_file = doc.source_file
            elif hasattr(doc, 'metadata') and doc.metadata:
                source_file = doc.metadata.get('source_file')
                page_number = doc.metadata.get('page_number')
                
            if hasattr(doc, 'page_number'):
                page_number = doc.page_number
                
            # 如果没有明确的信息，尝试从内容中提取
            if not source_file:
                source_file = self._extract_source_from_content(doc)
                
            if not page_number:
                page_number = self._extract_page_from_content(doc)
                
            # 创建追踪标记
            if source_file and page_number is not None:
                mark = TrackingMark(
                    source_file=source_file,
                    page_number=page_number,
                    doc_id=getattr(doc, 'id', ''),
                    content_snippet=self._get_content_snippet(doc),
                    relevance_score=getattr(doc, 'score', 0.0)
                )
                
                mark_str = mark.to_string()
                tracking_marks.append(mark_str)
                self.tracking_history.append(mark)
                
                # 创建追踪文档
                tracked_doc = TrackedDocument(
                    content=self._get_content(doc),
                    tracking_mark=mark,
                    metadata=getattr(doc, 'metadata', {}),
                    embedding=getattr(doc, 'embedding', None)
                )
                
                # 注册文档
                self.document_registry[mark_str] = tracked_doc
                
                # 添加到返回结果
                docs_with_tracking['content'].append(self._get_content(doc))
                docs_with_tracking['tracking'].append(mark_str)
                
        self.active_tracking_marks = tracking_marks
        logger.info(f"创建了 {len(tracking_marks)} 个追踪标记")
        
        return tracking_marks, docs_with_tracking
    
    def extract_final_sources(self, tracking_marks: List[str]) -> Dict[str, Any]:
        """
        从追踪标记中提取最终的来源信息
        Args:
            tracking_marks: 追踪标记列表
        Returns:
            包含source_file和page_number的字典
        """
        if not tracking_marks:
            tracking_marks = self.active_tracking_marks
            
        source_files = set()
        page_numbers = set()
        
        for mark_str in tracking_marks:
            try:
                mark = TrackingMark.from_string(mark_str)
                source_files.add(mark.source_file)
                page_numbers.add(mark.page_number)
            except ValueError as e:
                logger.warning(f"解析追踪标记失败: {e}")
                continue
                
        # 构建输出格式
        result = {
            "source_file": ",".join(sorted(source_files)),
            "page_number": sorted(list(page_numbers))
        }
        
        logger.info(f"提取来源信息: {len(source_files)} 个文件, {len(page_numbers)} 个页面")
        
        return result
    
    def merge_tracking_marks(self, marks_list: List[List[str]]) -> List[str]:
        """
        合并多个追踪标记列表，去重并保持顺序
        Args:
            marks_list: 多个追踪标记列表
        Returns:
            合并后的追踪标记列表
        """
        seen = set()
        merged = []
        
        for marks in marks_list:
            for mark in marks:
                if mark not in seen:
                    seen.add(mark)
                    merged.append(mark)
                    
        logger.info(f"合并追踪标记: 输入 {sum(len(m) for m in marks_list)} 个, 输出 {len(merged)} 个")
        
        return merged
    
    def get_document_by_mark(self, mark_str: str) -> Optional[TrackedDocument]:
        """
        根据追踪标记获取文档
        Args:
            mark_str: 追踪标记字符串
        Returns:
            追踪文档对象
        """
        return self.document_registry.get(mark_str)
    
    def maintain_tracking_through_pipeline(self, 
                                          original_marks: List[str],
                                          processed_content: str) -> Dict[str, Any]:
        """
        在处理流程中维护追踪信息
        Args:
            original_marks: 原始追踪标记
            processed_content: 处理后的内容
        Returns:
            包含处理内容和追踪信息的字典
        """
        return {
            'content': processed_content,
            'tracking_marks': original_marks,
            'source_info': self.extract_final_sources(original_marks)
        }
    
    def format_output_with_tracking(self, 
                                   question: str,
                                   answer: str,
                                   tracking_marks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        格式化带追踪信息的输出
        Args:
            question: 用户问题
            answer: 生成的答案
            tracking_marks: 追踪标记列表
        Returns:
            统一格式的JSON输出
        """
        if tracking_marks is None:
            tracking_marks = self.active_tracking_marks
            
        source_info = self.extract_final_sources(tracking_marks)
        
        output = {
            "question": question,
            "answer": answer,
            "source_file": source_info["source_file"],
            "page_number": source_info["page_number"]
        }
        
        return output
    
    def clear_active_tracking(self):
        """清空活动追踪标记"""
        self.active_tracking_marks.clear()
        logger.info("已清空活动追踪标记")
        
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """
        获取追踪统计信息
        Returns:
            统计信息字典
        """
        stats = {
            "total_tracked_documents": len(self.document_registry),
            "active_tracking_marks": len(self.active_tracking_marks),
            "tracking_history_size": len(self.tracking_history),
            "unique_source_files": len(set(m.source_file for m in self.tracking_history)),
            "unique_pages": len(set((m.source_file, m.page_number) for m in self.tracking_history))
        }
        
        return stats
    
    def _extract_source_from_content(self, doc: Any) -> str:
        """
        从文档内容中提取来源文件名
        Args:
            doc: 文档对象
        Returns:
            来源文件名
        """
        content = self._get_content(doc)
        
        # 尝试匹配常见的文件名模式
        patterns = [
            r"来源[:：]\s*([^\s,，]+\.pdf)",
            r"文档[:：]\s*([^\s,，]+\.pdf)",
            r"Source:\s*([^\s,]+\.pdf)",
            r"([^\s]+\.pdf)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
                
        return "unknown.pdf"
    
    def _extract_page_from_content(self, doc: Any) -> int:
        """
        从文档内容中提取页码
        Args:
            doc: 文档对象
        Returns:
            页码
        """
        content = self._get_content(doc)
        
        # 尝试匹配页码模式
        patterns = [
            r"页码[:：]\s*(\d+)",
            r"第\s*(\d+)\s*页",
            r"Page[:：]\s*(\d+)",
            r"p\.?\s*(\d+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return int(match.group(1))
                
        return 1
    
    def _get_content(self, doc: Any) -> str:
        """
        从文档对象中提取内容
        Args:
            doc: 文档对象
        Returns:
            文档内容字符串
        """
        if hasattr(doc, 'content'):
            return doc.content
        elif hasattr(doc, 'text'):
            return doc.text
        elif isinstance(doc, str):
            return doc
        elif isinstance(doc, dict):
            return doc.get('content', doc.get('text', str(doc)))
        else:
            return str(doc)
    
    def _get_content_snippet(self, doc: Any, max_length: int = 100) -> str:
        """
        获取文档内容片段
        Args:
            doc: 文档对象
            max_length: 最大长度
        Returns:
            内容片段
        """
        content = self._get_content(doc)
        if len(content) > max_length:
            return content[:max_length] + "..."
        return content
    
    def export_tracking_history(self, filepath: str):
        """
        导出追踪历史到文件
        Args:
            filepath: 输出文件路径
        """
        history_data = []
        for mark in self.tracking_history:
            history_data.append({
                "source_file": mark.source_file,
                "page_number": mark.page_number,
                "doc_id": mark.doc_id,
                "content_snippet": mark.content_snippet,
                "relevance_score": mark.relevance_score,
                "mark_string": mark.to_string()
            })
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"导出追踪历史到 {filepath}, 共 {len(history_data)} 条记录")