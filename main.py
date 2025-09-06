"""
RAG系统主入口
集成所有组件的完整RAG系统
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# 导入向量化模型
try:
    from sentence_transformers import SentenceTransformer
    ENCODER_AVAILABLE = True
except ImportError:
    ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers未安装，将使用随机向量")

# 导入核心组件
from core.hybrid_store import HybridVectorStore, Document
from core.document_tracker import DocumentTracker
from core.reranker import Reranker
from core.cache import CacheManager

# 导入CoT组件
from cot.reasoner import CoTReasoner
from cot.complexity_detector import ComplexityDetector
from cot.template_manager import TemplateManager
from cot.quality_monitor import QualityMonitor
from cot.exception_handler import ExceptionHandler, ExceptionLevel

# 导入队列组件
from queue.manager import QueueManager, DataType

# 导入监控组件
from monitor.metrics import MetricsCollector
from monitor.degradation import DegradationManager, SystemMetrics

# 导入配置
from config.settings import settings

# 导入数据加载器
from data.mineru_loader import MinIODataLoader

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.get("system.log_level", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGSystem:
    """完整的RAG系统"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化RAG系统
        Args:
            config: 自定义配置
        """
        logger.info("正在初始化RAG系统...")
        
        # 更新配置
        if config:
            for key, value in config.items():
                settings.set(key, value)
                
        # 初始化向量编码器
        self._init_encoder()
                
        # 初始化组件
        self._init_components()
        
        # 初始化数据加载器
        self.mineru_loader = MinIODataLoader()
        
        logger.info("RAG系统初始化完成")
        
    def _init_encoder(self):
        """初始化向量编码器"""
        if ENCODER_AVAILABLE:
            try:
                self.encoder = SentenceTransformer('BAAI/bge-base-zh-v1.5')
                logger.info("成功加载向量编码器: BAAI/bge-base-zh-v1.5")
            except Exception as e:
                logger.warning(f"向量编码器加载失败: {e}, 将使用随机向量")
                self.encoder = None
        else:
            self.encoder = None
            
    def encode_text(self, text: str) -> np.ndarray:
        """
        对文本进行向量编码
        Args:
            text: 文本内容
        Returns:
            向量表示
        """
        if self.encoder is not None:
            return self.encoder.encode(text)
        else:
            # 使用随机向量作为fallback
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(768)
        
    def _init_components(self):
        """初始化所有组件"""
        
        # 核心组件
        self.hybrid_store = HybridVectorStore(
            dense_weight=settings.get("hybrid_retrieval.dense_weight"),
            sparse_weight=settings.get("hybrid_retrieval.sparse_weight"),
            embedding_dim=settings.get("hybrid_retrieval.embedding_dim")
        )
        
        self.document_tracker = DocumentTracker()
        
        self.reranker = Reranker(
            model_name=settings.get("reranker.model_name"),
            batch_size=settings.get("reranker.batch_size"),
            use_fp16=settings.get("reranker.use_fp16")
        )
        
        self.cache_manager = CacheManager()
        
        # CoT组件
        self.complexity_detector = ComplexityDetector()
        
        self.cot_reasoner = CoTReasoner(
            min_steps=settings.get("cot.min_steps"),
            max_steps=settings.get("cot.max_steps"),
            quality_threshold=settings.get("cot.quality_threshold"),
            max_retries=settings.get("cot.max_retries")
        )
        
        self.template_manager = TemplateManager()
        
        self.quality_monitor = QualityMonitor(
            quality_threshold=settings.get("quality.quality_threshold")
        )
        
        self.exception_handler = ExceptionHandler()
        
        # 队列管理
        self.queue_manager = QueueManager(
            queue_config=settings.get("queue")
        )
        
        # 监控组件
        self.metrics_collector = MetricsCollector()
        self.degradation_manager = DegradationManager()
        
        # 注册处理器
        self._register_processors()
        
        # 启动队列处理
        self.queue_manager.start()
        
    def _register_processors(self):
        """注册队列处理器"""
        
        def text_processor(batch_data: List[Tuple[str, Any]]) -> List[Any]:
            """文本批处理器"""
            results = []
            for query, data in batch_data:
                try:
                    result = self._process_single_query(query, data, DataType.TEXT)
                    results.append(result)
                except Exception as e:
                    logger.error(f"文本处理失败: {e}")
                    results.append(None)
            return results
            
        def image_processor(batch_data: List[Tuple[str, Any]]) -> List[Any]:
            """图像批处理器"""
            results = []
            for query, data in batch_data:
                try:
                    result = self._process_single_query(query, data, DataType.IMAGE)
                    results.append(result)
                except Exception as e:
                    logger.error(f"图像处理失败: {e}")
                    results.append(None)
            return results
            
        def table_processor(batch_data: List[Tuple[str, Any]]) -> List[Any]:
            """表格批处理器"""
            results = []
            for query, data in batch_data:
                try:
                    result = self._process_single_query(query, data, DataType.TABLE)
                    results.append(result)
                except Exception as e:
                    logger.error(f"表格处理失败: {e}")
                    results.append(None)
            return results
            
        self.queue_manager.register_processor(DataType.TEXT, text_processor)
        self.queue_manager.register_processor(DataType.IMAGE, image_processor)
        self.queue_manager.register_processor(DataType.TABLE, table_processor)
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        添加文档到系统
        Args:
            documents: 文档列表
        """
        doc_objects = []
        for doc_data in documents:
            # 如果没有提供embedding，自动生成
            embedding = doc_data.get("embedding")
            if embedding is None:
                content = doc_data.get("content", "")
                embedding = self.encode_text(content)
                
            doc = Document(
                id=doc_data.get("id", ""),
                content=doc_data.get("content", ""),
                embedding=embedding,
                metadata=doc_data.get("metadata", {}),
                source_file=doc_data.get("source_file"),
                page_number=doc_data.get("page_number")
            )
            doc_objects.append(doc)
            
        self.hybrid_store.add_documents(doc_objects)
        logger.info(f"添加了 {len(documents)} 个文档")
        
    def add_documents_from_mineru(self, 
                                 json_file_path: str,
                                 images_dir: Optional[str] = None,
                                 base_path: str = "") -> int:
        """
        从MinIO解析结果添加文档
        Args:
            json_file_path: JSON文件路径
            images_dir: 图像目录路径
            base_path: 基础路径
        Returns:
            添加的文档数量
        """
        logger.info(f"从MinIO数据加载文档: {json_file_path}")
        
        # 设置数据加载器的基础路径
        if base_path:
            self.mineru_loader = MinIODataLoader(base_path)
        
        # 加载文档
        documents = self.mineru_loader.load_document(json_file_path, images_dir)
        
        if not documents:
            logger.warning("未加载到任何文档")
            return 0
            
        # 转换为系统格式并添加向量
        doc_objects = []
        for doc_data in documents:
            # 自动生成向量
            embedding = self.encode_text(doc_data["content"])
            
            doc = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                embedding=embedding,
                metadata=doc_data.get("metadata", {}),
                source_file=doc_data.get("source_file"),
                page_number=doc_data.get("page_number")
            )
            doc_objects.append(doc)
            
        # 添加到系统
        self.hybrid_store.add_documents(doc_objects)
        
        # 获取统计信息
        stats = self.mineru_loader.get_document_statistics(documents)
        logger.info(f"MinIO数据加载完成: {stats}")
        
        return len(documents)
        
    def batch_load_mineru_documents(self,
                                   json_files: List[str],
                                   images_dirs: Optional[List[str]] = None,
                                   base_path: str = "") -> int:
        """
        批量加载MinIO解析的文档
        Args:
            json_files: JSON文件路径列表
            images_dirs: 图像目录列表
            base_path: 基础路径
        Returns:
            总共添加的文档数量
        """
        logger.info(f"批量加载MinIO文档: {len(json_files)} 个文件")
        
        if base_path:
            self.mineru_loader = MinIODataLoader(base_path)
            
        total_added = 0
        
        for i, json_file in enumerate(json_files):
            images_dir = images_dirs[i] if images_dirs and i < len(images_dirs) else None
            count = self.add_documents_from_mineru(
                json_file_path=json_file,
                images_dir=images_dir,
                base_path=""  # 已经设置了base_path
            )
            total_added += count
            logger.info(f"文件 {i+1}/{len(json_files)} 处理完成，添加 {count} 个文档项")
            
        logger.info(f"批量加载完成，总计添加 {total_added} 个文档项")
        return total_added
        
    def query(self, 
             query: str,
             query_embedding: Optional[np.ndarray] = None,
             data_type: str = "text",
             top_k: int = 10) -> Dict[str, Any]:
        """
        查询接口
        Args:
            query: 查询文本
            query_embedding: 查询向量
            data_type: 数据类型
            top_k: 返回结果数
        Returns:
            查询结果
        """
        start_time = time.time()
        
        try:
            # 如果没有提供查询向量，自动生成
            if query_embedding is None:
                query_embedding = self.encode_text(query)
                
            # 检查缓存
            cached_result = self.cache_manager.get(query, query_embedding)
            if cached_result:
                self.metrics_collector.collect_metric("cache_hit", 1)
                return cached_result
            else:
                self.metrics_collector.collect_metric("cache_hit", 0)
                
            # 混合检索
            retrieved_docs = self.hybrid_store.search(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k * 2,  # 获取更多候选用于重排序
                use_hybrid=True
            )
            
            # 创建文档追踪
            tracking_marks, docs_with_tracking = self.document_tracker.create_tracking_marks(retrieved_docs)
            
            # 重排序
            current_config = self.degradation_manager.get_current_config()
            rerank_limit = current_config["rerank_candidates"].get(data_type, 25)
            reranked_docs = self.reranker.rerank(
                query=query,
                candidates=retrieved_docs[:rerank_limit],
                top_k=top_k,
                data_type=data_type
            )
            
            # 更新文档追踪（重排序后）
            reranked_doc_objects = [doc for doc, score in reranked_docs]
            tracking_marks, docs_with_tracking = self.document_tracker.create_tracking_marks(reranked_doc_objects)
            
            # 复杂度检测
            complexity = self.complexity_detector.detect_complexity(query)
            question_type = self.complexity_detector.classify_question_type(query)
            
            # 决定是否使用CoT
            use_cot = (complexity == "complex" and 
                      self.degradation_manager.cot_enabled and
                      current_config.get("cot_enabled", True))
                      
            if use_cot:
                # CoT推理
                answer, final_tracking = self._process_with_cot(
                    query, docs_with_tracking, question_type, data_type
                )
            else:
                # 直接生成答案
                answer = self._generate_simple_answer(query, docs_with_tracking)
                final_tracking = tracking_marks
                
            # 格式化输出
            result = self.document_tracker.format_output_with_tracking(
                question=query,
                answer=answer,
                tracking_marks=final_tracking
            )
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 收集指标
            self._collect_metrics(processing_time, use_cot, question_type)
            
            # 更新缓存
            self.cache_manager.put(query, result, query_embedding)
            
            # 检查系统状态
            self._check_system_status(processing_time)
            
            return result
            
        except Exception as e:
            # 异常处理
            exc_info = self.exception_handler.handle_by_level(
                exception=e,
                context={"query": query, "data_type": data_type}
            )
            
            # 返回降级结果
            return {
                "question": query,
                "answer": f"处理失败: {str(e)} (NoUseCoT)",
                "source_file": "",
                "page_number": []
            }
            
    def _process_single_query(self, 
                            query: str,
                            data: Any,
                            data_type: DataType) -> Dict[str, Any]:
        """
        处理单个查询（队列内部使用）
        Args:
            query: 查询
            data: 数据
            data_type: 数据类型
        Returns:
            处理结果
        """
        # 这里简化处理，实际应该包含完整的处理流程
        return self.query(
            query=query,
            query_embedding=data.get("embedding") if isinstance(data, dict) else None,
            data_type=data_type.value
        )
        
    def _process_with_cot(self,
                         query: str,
                         docs_with_tracking: Dict[str, Any],
                         question_type: str,
                         data_type: str) -> Tuple[str, List[str]]:
        """
        使用CoT处理查询
        Args:
            query: 查询
            docs_with_tracking: 带追踪的文档
            question_type: 问题类型
            data_type: 数据类型
        Returns:
            (答案, 追踪标记)
        """
        try:
            # 执行CoT推理
            answer, tracking_marks = self.cot_reasoner.process_with_cot(
                query=query,
                docs_with_tracking=docs_with_tracking,
                question_type=question_type,
                data_type=data_type
            )
            
            # 质量评估
            quality_report = self.quality_monitor.evaluate_quality(
                reasoning_chain=None,  # 这里应该传入实际的推理链
                answer=answer,
                query=query,
                question_type=question_type
            )
            
            if not quality_report.passed:
                logger.warning(f"CoT质量未达标: {quality_report.failure_reasons}")
                # 可以选择重试或降级
                
            return answer, tracking_marks
            
        except Exception as e:
            logger.error(f"CoT处理失败: {e}")
            # 降级到简单答案
            answer = self._generate_simple_answer(query, docs_with_tracking)
            return answer + " (NoUseCoT)", docs_with_tracking.get("tracking", [])
            
    def _generate_simple_answer(self, 
                               query: str,
                               docs_with_tracking: Dict[str, Any]) -> str:
        """
        生成简单答案（不使用CoT）
        Args:
            query: 查询
            docs_with_tracking: 带追踪的文档
        Returns:
            答案
        """
        documents = docs_with_tracking.get("content", [])
        
        if not documents:
            return "未找到相关信息。"
            
        # 简单地返回第一个相关文档的内容
        if isinstance(documents[0], str):
            content = documents[0][:500]
        else:
            content = str(documents[0])[:500]
            
        return f"根据相关文档：{content}..."
        
    def _collect_metrics(self, 
                        processing_time: float,
                        use_cot: bool,
                        question_type: str):
        """
        收集性能指标
        Args:
            processing_time: 处理时间
            use_cot: 是否使用CoT
            question_type: 问题类型
        """
        self.metrics_collector.collect_metrics(
            operation_type="query",
            metrics_data={
                "latency": processing_time * 1000,  # 转换为毫秒
                "cot_used": 1 if use_cot else 0
            }
        )
        
    def _check_system_status(self, processing_time: float):
        """
        检查系统状态并应用降级
        Args:
            processing_time: 处理时间
        """
        # 获取系统指标
        try:
            import psutil
            cpu_usage = psutil.cpu_percent(interval=0)
            memory_usage = psutil.virtual_memory().percent
        except:
            cpu_usage = 0
            memory_usage = 0
            
        metrics = SystemMetrics(
            latency=processing_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage
        )
        
        # 评估并应用降级
        suggested_level = self.degradation_manager.evaluate_system_status(metrics)
        if suggested_level != self.degradation_manager.current_level:
            self.degradation_manager.apply_degradation(suggested_level)
            
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估系统性能
        Args:
            test_data: 测试数据
        Returns:
            评估结果
        """
        results = []
        
        for test_case in test_data:
            query = test_case["question"]
            ground_truth = test_case.get("ground_truth", {})
            
            # 执行查询
            result = self.query(query)
            
            # 评估结果
            evaluation = self._evaluate_single_result(result, ground_truth)
            
            results.append({
                "query": query,
                "result": result,
                "evaluation": evaluation
            })
            
        # 汇总统计
        stats = self._calculate_evaluation_stats(results)
        
        return {
            "results": results,
            "statistics": stats
        }
        
    def _evaluate_single_result(self,
                               result: Dict[str, Any],
                               ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        评估单个结果
        Args:
            result: 查询结果
            ground_truth: 真实答案
        Returns:
            评估分数
        """
        scores = {}
        
        # 页面匹配度
        if "pages" in ground_truth:
            predicted_pages = set(result.get("page_number", []))
            truth_pages = set(ground_truth["pages"])
            scores["page_match"] = 0.25 if predicted_pages == truth_pages else 0.0
            
        # 文件匹配度
        if "source_files" in ground_truth:
            predicted_files = set(result.get("source_file", "").split(","))
            truth_files = set(ground_truth["source_files"])
            scores["file_match"] = 0.25 if predicted_files == truth_files else 0.0
            
        # 答案相似度
        if "answer" in ground_truth:
            scores["content_similarity"] = self._calculate_text_similarity(
                result.get("answer", ""),
                ground_truth["answer"]
            ) * 0.5
            
        # 总分
        scores["total"] = sum(scores.values())
        
        return scores
        
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度
        Args:
            text1: 文本1
            text2: 文本2
        Returns:
            相似度分数
        """
        # 简化的Jaccard相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
        
    def _calculate_evaluation_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """
        计算评估统计
        Args:
            results: 评估结果列表
        Returns:
            统计信息
        """
        all_scores = [r["evaluation"].get("total", 0) for r in results]
        
        return {
            "total_cases": len(results),
            "average_score": np.mean(all_scores) if all_scores else 0,
            "pass_rate": sum(1 for s in all_scores if s >= 0.6) / len(all_scores) if all_scores else 0,
            "score_distribution": {
                "min": np.min(all_scores) if all_scores else 0,
                "max": np.max(all_scores) if all_scores else 0,
                "std": np.std(all_scores) if all_scores else 0
            }
        }
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        Returns:
            统计信息
        """
        return {
            "queue": self.queue_manager.get_statistics(),
            "degradation": self.degradation_manager.get_statistics(),
            "cache": self.cache_manager.get_all_stats(),
            "metrics": self.metrics_collector.get_dashboard_data(),
            "exceptions": self.exception_handler.get_exception_statistics(),
            "quality": self.quality_monitor.get_statistics()
        }
        
    def shutdown(self):
        """关闭系统"""
        logger.info("正在关闭RAG系统...")
        
        # 停止队列处理
        self.queue_manager.stop()
        
        # 导出统计信息
        stats = self.get_statistics()
        with open("rag_system_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
            
        logger.info("RAG系统已关闭")


def main():
    """主函数"""
    # 创建RAG系统实例
    rag_system = RAGSystem()
    
    # 演示1：使用示例数据（原有功能）
    print("=== 演示1: 手动添加文档 ===")
    sample_documents = [
        {
            "id": "doc1",
            "content": "北京是中华人民共和国的首都，是中国的政治、文化中心。",
            "source_file": "china.pdf",
            "page_number": 1
        },
        {
            "id": "doc2",
            "content": "人工智能是计算机科学的一个分支，致力于创建智能机器。",
            "source_file": "ai.pdf", 
            "page_number": 5
        }
    ]
    
    rag_system.add_documents(sample_documents)
    
    # 查询测试
    result = rag_system.query("北京是哪个国家的首都？")
    print("查询结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 演示2：加载MinIO数据（如果存在）
    print("\n=== 演示2: MinIO数据加载 ===")
    
    # 检查是否有示例数据
    import os
    data_example_path = "../data_example"
    json_file = "艾力斯-公司深度报告商业化成绩显著产品矩阵持续拓宽-25070718页_content_list.json"
    
    if os.path.exists(os.path.join(data_example_path, json_file)):
        print("发现MinIO示例数据，正在加载...")
        
        # 加载MinIO数据
        count = rag_system.add_documents_from_mineru(
            json_file_path=json_file,
            base_path=data_example_path
        )
        
        print(f"成功加载 {count} 个MinIO文档项")
        
        # 测试查询MinIO数据
        mineru_queries = [
            "艾力斯公司的主要业务是什么？",
            "伏美替尼的特点有哪些？",
            "公司的投资建议是什么？"
        ]
        
        for query in mineru_queries:
            print(f"\n查询: {query}")
            result = rag_system.query(query)
            print(f"答案: {result['answer'][:100]}...")
            print(f"来源: {result['source_file']}, 页码: {result['page_number']}")
            
    else:
        print("未找到MinIO示例数据，跳过演示2")
        print(f"请确保数据位于: {os.path.abspath(data_example_path)}")
    
    # 获取统计信息
    print("\n=== 系统统计信息 ===")
    stats = rag_system.get_statistics()
    
    # 显示关键统计
    if 'queue' in stats and 'processing_stats' in stats['queue']:
        print("队列处理统计:")
        for data_type, stat in stats['queue']['processing_stats'].items():
            if stat['processed'] > 0:
                print(f"  {data_type}: 处理 {stat['processed']} 个，平均时间 {stat['avg_time']:.3f}s")
    
    if 'cache' in stats:
        for cache_type, cache_stat in stats['cache'].items():
            if isinstance(cache_stat, dict) and 'hit_rate' in cache_stat:
                print(f"{cache_type} 缓存命中率: {cache_stat['hit_rate']:.2%}")
    
    # 关闭系统
    rag_system.shutdown()


if __name__ == "__main__":
    main()