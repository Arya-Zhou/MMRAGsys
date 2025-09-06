"""
MinIO数据加载器
处理MinIO解析后的JSON和图像数据
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MinIODataLoader:
    """MinIO数据加载器"""
    
    def __init__(self, base_path: str = ""):
        """
        初始化数据加载器
        Args:
            base_path: 数据基础路径
        """
        self.base_path = Path(base_path) if base_path else Path("")
        
    def load_document(self, json_file_path: str, images_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        加载单个文档的数据
        Args:
            json_file_path: JSON文件路径
            images_dir: 图像目录路径（可选，默认为JSON文件同目录下的images）
        Returns:
            标准化的文档列表
        """
        json_path = self.base_path / json_file_path
        
        # 确定图像目录
        if images_dir:
            images_path = self.base_path / images_dir
        else:
            images_path = json_path.parent / "images"
            
        logger.info(f"加载文档: {json_path}")
        logger.info(f"图像目录: {images_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            documents = []
            source_file = json_path.stem  # 使用文件名作为source_file
            
            for i, item in enumerate(raw_data):
                doc = self._process_item(item, i, source_file, images_path)
                if doc:
                    documents.append(doc)
                    
            logger.info(f"成功加载 {len(documents)} 个文档项")
            return documents
            
        except Exception as e:
            logger.error(f"加载文档失败: {e}")
            return []
            
    def _process_item(self, 
                     item: Dict[str, Any], 
                     item_id: int,
                     source_file: str,
                     images_path: Path) -> Optional[Dict[str, Any]]:
        """
        处理单个数据项
        Args:
            item: 原始数据项
            item_id: 项目ID
            source_file: 源文件名
            images_path: 图像路径
        Returns:
            标准化的文档项
        """
        item_type = item.get("type", "")
        page_idx = item.get("page_idx", 0)
        
        if item_type == "text":
            return self._process_text_item(item, item_id, source_file, page_idx)
        elif item_type == "table":
            return self._process_table_item(item, item_id, source_file, page_idx, images_path)
        elif item_type == "image":
            return self._process_image_item(item, item_id, source_file, page_idx, images_path)
        else:
            logger.warning(f"未知的数据类型: {item_type}")
            return None
            
    def _process_text_item(self, 
                          item: Dict[str, Any], 
                          item_id: int,
                          source_file: str,
                          page_idx: int) -> Dict[str, Any]:
        """
        处理文本项
        Args:
            item: 文本项数据
            item_id: 项目ID
            source_file: 源文件名
            page_idx: 页码
        Returns:
            标准化文档项
        """
        text_content = item.get("text", "").strip()
        text_level = item.get("text_level", 0)
        
        # 跳过空文本
        if not text_content:
            return None
            
        # 构建内容，包含层级信息
        if text_level > 0:
            content = f"[标题{text_level}] {text_content}"
        else:
            content = text_content
            
        return {
            "id": f"{source_file}_text_{item_id}",
            "content": content,
            "data_type": "text",
            "source_file": f"{source_file}.pdf",
            "page_number": page_idx + 1,  # 转换为1开始的页码
            "metadata": {
                "type": "text",
                "text_level": text_level,
                "original_text": text_content
            }
        }
        
    def _process_table_item(self, 
                           item: Dict[str, Any], 
                           item_id: int,
                           source_file: str,
                           page_idx: int,
                           images_path: Path) -> Dict[str, Any]:
        """
        处理表格项
        Args:
            item: 表格项数据
            item_id: 项目ID
            source_file: 源文件名
            page_idx: 页码
            images_path: 图像路径
        Returns:
            标准化文档项
        """
        table_body = item.get("table_body", "")
        table_caption = item.get("table_caption", [])
        table_footnote = item.get("table_footnote", [])
        img_path = item.get("img_path", "")
        
        # 构建表格内容
        content_parts = []
        
        # 添加标题
        if table_caption:
            content_parts.append(f"表格标题: {'; '.join(table_caption)}")
            
        # 添加HTML表格内容（简化处理）
        if table_body:
            # 简单提取表格文本内容
            import re
            # 移除HTML标签，保留文本
            table_text = re.sub(r'<[^>]+>', ' ', table_body)
            table_text = re.sub(r'\s+', ' ', table_text).strip()
            content_parts.append(f"表格内容: {table_text}")
            
        # 添加脚注
        if table_footnote:
            content_parts.append(f"表格脚注: {'; '.join(table_footnote)}")
            
        content = "\n".join(content_parts)
        
        # 图像路径处理
        full_img_path = None
        if img_path:
            full_img_path = str(images_path / img_path.replace("images/", ""))
            
        return {
            "id": f"{source_file}_table_{item_id}",
            "content": content,
            "data_type": "table", 
            "source_file": f"{source_file}.pdf",
            "page_number": page_idx + 1,
            "metadata": {
                "type": "table",
                "table_body": table_body,
                "table_caption": table_caption,
                "table_footnote": table_footnote,
                "img_path": full_img_path,
                "original_img_path": img_path
            }
        }
        
    def _process_image_item(self, 
                           item: Dict[str, Any], 
                           item_id: int,
                           source_file: str,
                           page_idx: int,
                           images_path: Path) -> Dict[str, Any]:
        """
        处理图像项
        Args:
            item: 图像项数据
            item_id: 项目ID
            source_file: 源文件名
            page_idx: 页码
            images_path: 图像路径
        Returns:
            标准化文档项
        """
        image_caption = item.get("image_caption", [])
        image_footnote = item.get("image_footnote", [])
        img_path = item.get("img_path", "")
        
        # 构建图像内容描述
        content_parts = []
        
        # 添加标题
        if image_caption:
            content_parts.append(f"图像标题: {'; '.join(image_caption)}")
            
        # 添加脚注
        if image_footnote:
            content_parts.append(f"图像脚注: {'; '.join(image_footnote)}")
            
        # 如果没有描述信息，提供基本描述
        if not content_parts:
            content_parts.append("图像内容（需要视觉分析）")
            
        content = "\n".join(content_parts)
        
        # 图像路径处理
        full_img_path = None
        if img_path:
            full_img_path = str(images_path / img_path.replace("images/", ""))
            
        return {
            "id": f"{source_file}_image_{item_id}",
            "content": content,
            "data_type": "image",
            "source_file": f"{source_file}.pdf",
            "page_number": page_idx + 1,
            "metadata": {
                "type": "image", 
                "image_caption": image_caption,
                "image_footnote": image_footnote,
                "img_path": full_img_path,
                "original_img_path": img_path
            }
        }
        
    def load_multiple_documents(self, 
                               json_files: List[str],
                               images_dirs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        批量加载多个文档
        Args:
            json_files: JSON文件路径列表
            images_dirs: 图像目录路径列表（可选）
        Returns:
            所有文档的标准化数据
        """
        all_documents = []
        
        if images_dirs is None:
            images_dirs = [None] * len(json_files)
            
        for json_file, images_dir in zip(json_files, images_dirs):
            documents = self.load_document(json_file, images_dir)
            all_documents.extend(documents)
            
        logger.info(f"批量加载完成，总计 {len(all_documents)} 个文档项")
        return all_documents
        
    def get_document_statistics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取文档统计信息
        Args:
            documents: 文档列表
        Returns:
            统计信息
        """
        stats = {
            "total_documents": len(documents),
            "by_type": {},
            "by_file": {},
            "page_range": {}
        }
        
        for doc in documents:
            # 按类型统计
            data_type = doc.get("data_type", "unknown")
            stats["by_type"][data_type] = stats["by_type"].get(data_type, 0) + 1
            
            # 按文件统计
            source_file = doc.get("source_file", "unknown")
            stats["by_file"][source_file] = stats["by_file"].get(source_file, 0) + 1
            
            # 页码范围统计
            page_num = doc.get("page_number", 0)
            if source_file not in stats["page_range"]:
                stats["page_range"][source_file] = {"min": page_num, "max": page_num}
            else:
                stats["page_range"][source_file]["min"] = min(
                    stats["page_range"][source_file]["min"], page_num
                )
                stats["page_range"][source_file]["max"] = max(
                    stats["page_range"][source_file]["max"], page_num
                )
                
        return stats


# 便捷函数
def load_from_mineru(json_file_path: str, 
                     images_dir: Optional[str] = None,
                     base_path: str = "") -> List[Dict[str, Any]]:
    """
    便捷函数：从MinIO数据加载文档
    Args:
        json_file_path: JSON文件路径
        images_dir: 图像目录路径
        base_path: 基础路径
    Returns:
        标准化文档列表
    """
    loader = MinIODataLoader(base_path)
    return loader.load_document(json_file_path, images_dir)