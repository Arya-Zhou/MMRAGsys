"""
批量数据索引管理器
用于管理大规模PDF解析数据的导入、索引和查询
"""

import os
import json
import time
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFIndexManager:
    """PDF文档索引管理器"""
    
    def __init__(self, index_dir: str = "indexes"):
        """
        初始化索引管理器
        Args:
            index_dir: 索引存储目录
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # 索引文件路径
        self.master_index_file = self.index_dir / "master_index.json"
        self.document_map_file = self.index_dir / "document_map.json"
        
        # 加载或初始化索引
        self.master_index = self._load_master_index()
        self.document_map = self._load_document_map()
        
    def _load_master_index(self) -> Dict[str, Any]:
        """加载主索引"""
        if self.master_index_file.exists():
            with open(self.master_index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "total_pdfs": 0,
            "total_documents": 0,
            "pdfs": {},
            "last_update": None
        }
        
    def _load_document_map(self) -> Dict[str, Any]:
        """加载文档映射"""
        if self.document_map_file.exists():
            with open(self.document_map_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
        
    def register_pdf(self, 
                     pdf_name: str,
                     json_path: str,
                     images_dir: str,
                     metadata: Optional[Dict] = None) -> str:
        """
        注册PDF文档
        Args:
            pdf_name: PDF文件名
            json_path: JSON文件路径
            images_dir: 图像目录
            metadata: 元数据
        Returns:
            PDF ID
        """
        # 生成PDF ID
        pdf_id = hashlib.md5(pdf_name.encode()).hexdigest()[:12]
        
        # 更新主索引
        self.master_index["pdfs"][pdf_id] = {
            "name": pdf_name,
            "json_path": json_path,
            "images_dir": images_dir,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
            "document_count": 0,
            "status": "registered"
        }
        
        self.master_index["total_pdfs"] += 1
        self._save_master_index()
        
        logger.info(f"注册PDF: {pdf_name} (ID: {pdf_id})")
        return pdf_id
        
    def scan_directory(self, base_dir: str, pattern: str = "*_content_list.json") -> List[Dict]:
        """
        扫描目录查找所有PDF解析文件
        Args:
            base_dir: 基础目录
            pattern: 文件匹配模式
        Returns:
            找到的PDF文件列表
        """
        base_path = Path(base_dir)
        pdf_files = []
        
        logger.info(f"扫描目录: {base_path}")
        
        # 递归查找所有匹配的JSON文件
        json_files = list(base_path.rglob(pattern))
        
        for json_file in json_files:
            # 提取PDF名称
            pdf_name = json_file.stem.replace("_content_list", "")
            
            # 查找对应的images目录
            images_dir = json_file.parent / "images"
            if not images_dir.exists():
                images_dir = json_file.parent / "auto" / "images"
                
            pdf_info = {
                "pdf_name": pdf_name,
                "json_path": str(json_file),
                "images_dir": str(images_dir) if images_dir.exists() else None,
                "file_size": json_file.stat().st_size
            }
            
            pdf_files.append(pdf_info)
            
        logger.info(f"找到 {len(pdf_files)} 个PDF解析文件")
        return pdf_files
        
    def batch_register(self, pdf_files: List[Dict]) -> List[str]:
        """
        批量注册PDF文件
        Args:
            pdf_files: PDF文件信息列表
        Returns:
            PDF ID列表
        """
        pdf_ids = []
        
        for pdf_info in pdf_files:
            pdf_id = self.register_pdf(
                pdf_name=pdf_info["pdf_name"],
                json_path=pdf_info["json_path"],
                images_dir=pdf_info.get("images_dir", ""),
                metadata={"file_size": pdf_info.get("file_size", 0)}
            )
            pdf_ids.append(pdf_id)
            
        logger.info(f"批量注册完成: {len(pdf_ids)} 个PDF")
        return pdf_ids
        
    def update_document_count(self, pdf_id: str, count: int):
        """
        更新文档计数
        Args:
            pdf_id: PDF ID
            count: 文档数量
        """
        if pdf_id in self.master_index["pdfs"]:
            self.master_index["pdfs"][pdf_id]["document_count"] = count
            self.master_index["total_documents"] += count
            self._save_master_index()
            
    def update_status(self, pdf_id: str, status: str):
        """
        更新PDF状态
        Args:
            pdf_id: PDF ID
            status: 状态 (registered/loading/loaded/error)
        """
        if pdf_id in self.master_index["pdfs"]:
            self.master_index["pdfs"][pdf_id]["status"] = status
            self.master_index["pdfs"][pdf_id]["last_update"] = datetime.now().isoformat()
            self._save_master_index()
            
    def get_pdf_info(self, pdf_id: str) -> Optional[Dict]:
        """
        获取PDF信息
        Args:
            pdf_id: PDF ID
        Returns:
            PDF信息
        """
        return self.master_index["pdfs"].get(pdf_id)
        
    def get_unloaded_pdfs(self) -> List[str]:
        """
        获取未加载的PDF列表
        Returns:
            PDF ID列表
        """
        unloaded = []
        for pdf_id, info in self.master_index["pdfs"].items():
            if info["status"] != "loaded":
                unloaded.append(pdf_id)
        return unloaded
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        Returns:
            统计信息
        """
        status_counts = {}
        for pdf_info in self.master_index["pdfs"].values():
            status = pdf_info["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            
        return {
            "total_pdfs": self.master_index["total_pdfs"],
            "total_documents": self.master_index["total_documents"],
            "status_distribution": status_counts,
            "last_update": self.master_index.get("last_update")
        }
        
    def _save_master_index(self):
        """保存主索引"""
        self.master_index["last_update"] = datetime.now().isoformat()
        with open(self.master_index_file, 'w', encoding='utf-8') as f:
            json.dump(self.master_index, f, ensure_ascii=False, indent=2)
            
    def save_document_map(self, pdf_id: str, doc_ids: List[str]):
        """
        保存文档映射
        Args:
            pdf_id: PDF ID
            doc_ids: 文档ID列表
        """
        self.document_map[pdf_id] = doc_ids
        with open(self.document_map_file, 'w', encoding='utf-8') as f:
            json.dump(self.document_map, f, ensure_ascii=False, indent=2)
            
    def export_index_report(self, output_file: str = "index_report.md"):
        """
        导出索引报告
        Args:
            output_file: 输出文件名
        """
        stats = self.get_statistics()
        
        report = []
        report.append("# PDF索引报告\n")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("\n## 总体统计\n")
        report.append(f"- 总PDF数: {stats['total_pdfs']}")
        report.append(f"- 总文档数: {stats['total_documents']}")
        report.append(f"- 平均每PDF文档数: {stats['total_documents'] / max(stats['total_pdfs'], 1):.1f}")
        
        report.append("\n## 状态分布\n")
        for status, count in stats['status_distribution'].items():
            report.append(f"- {status}: {count}")
            
        report.append("\n## PDF列表\n")
        report.append("| PDF名称 | 状态 | 文档数 | 注册时间 |")
        report.append("|---------|------|--------|----------|")
        
        for pdf_id, info in self.master_index["pdfs"].items():
            name = info['name'][:30] + "..." if len(info['name']) > 30 else info['name']
            status = info['status']
            doc_count = info['document_count']
            reg_time = info.get('registered_at', 'N/A')[:10]
            report.append(f"| {name} | {status} | {doc_count} | {reg_time} |")
            
        # 写入文件
        output_path = self.index_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
            
        logger.info(f"索引报告已导出到: {output_path}")


class BatchLoader:
    """批量数据加载器"""
    
    def __init__(self, rag_system, index_manager: PDFIndexManager):
        """
        初始化批量加载器
        Args:
            rag_system: RAG系统实例
            index_manager: 索引管理器
        """
        self.rag_system = rag_system
        self.index_manager = index_manager
        
    def load_all_pdfs(self, batch_size: int = 10, save_interval: int = 5):
        """
        加载所有PDF文档
        Args:
            batch_size: 批次大小
            save_interval: 保存间隔
        """
        unloaded = self.index_manager.get_unloaded_pdfs()
        
        if not unloaded:
            logger.info("所有PDF已加载")
            return
            
        logger.info(f"开始加载 {len(unloaded)} 个PDF")
        
        loaded_count = 0
        error_count = 0
        
        for i, pdf_id in enumerate(unloaded):
            pdf_info = self.index_manager.get_pdf_info(pdf_id)
            
            if not pdf_info:
                continue
                
            logger.info(f"加载 [{i+1}/{len(unloaded)}]: {pdf_info['name']}")
            
            try:
                # 更新状态为加载中
                self.index_manager.update_status(pdf_id, "loading")
                
                # 加载文档
                count = self.rag_system.add_documents_from_mineru(
                    json_file_path=pdf_info['json_path'],
                    images_dir=pdf_info.get('images_dir')
                )
                
                # 更新文档计数和状态
                self.index_manager.update_document_count(pdf_id, count)
                self.index_manager.update_status(pdf_id, "loaded")
                
                loaded_count += 1
                logger.info(f"成功加载 {count} 个文档项")
                
            except Exception as e:
                logger.error(f"加载失败: {e}")
                self.index_manager.update_status(pdf_id, "error")
                error_count += 1
                
            # 定期保存进度
            if (i + 1) % save_interval == 0:
                logger.info(f"保存进度: 已加载 {loaded_count} 个PDF")
                # 这里可以添加检查点保存逻辑
                
        logger.info(f"批量加载完成: 成功 {loaded_count}, 失败 {error_count}")
        
        # 导出报告
        self.index_manager.export_index_report()
        
    def resume_loading(self):
        """恢复加载（从上次中断处继续）"""
        stats = self.index_manager.get_statistics()
        
        if 'loading' in stats['status_distribution']:
            loading_count = stats['status_distribution']['loading']
            logger.info(f"发现 {loading_count} 个正在加载的PDF，重置状态")
            
            # 重置loading状态为registered
            for pdf_id, info in self.index_manager.master_index["pdfs"].items():
                if info["status"] == "loading":
                    self.index_manager.update_status(pdf_id, "registered")
                    
        # 继续加载
        self.load_all_pdfs()


# 便捷函数
def setup_batch_loading(base_dir: str, rag_system) -> BatchLoader:
    """
    设置批量加载
    Args:
        base_dir: 数据目录
        rag_system: RAG系统实例
    Returns:
        批量加载器
    """
    # 创建索引管理器
    index_manager = PDFIndexManager()
    
    # 扫描目录
    pdf_files = index_manager.scan_directory(base_dir)
    
    # 批量注册
    if pdf_files:
        index_manager.batch_register(pdf_files)
        
    # 创建批量加载器
    batch_loader = BatchLoader(rag_system, index_manager)
    
    return batch_loader