"""
队列管理器
管理多队列调度和资源分配
"""

import time
import threading
from queue import Queue, PriorityQueue, Empty
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataType(Enum):
    """数据类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class QueueRequest:
    """队列请求"""
    request_id: str
    data_type: DataType
    query: str
    data: Any
    priority: int = 5  # 1-10, 1最高优先级
    timestamp: float = field(default_factory=time.time)
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """用于优先队列比较"""
        return self.priority < other.priority


@dataclass
class QueueResponse:
    """队列响应"""
    request_id: str
    result: Any
    processing_time: float
    success: bool = True
    error: Optional[str] = None


class QueueManager:
    """队列管理器"""
    
    def __init__(self, queue_config: Optional[Dict] = None):
        """
        初始化队列管理器
        Args:
            queue_config: 队列配置
        """
        # 默认配置
        default_config = {
            "text": {"resource_allocation": 0.4, "max_size": 1000, "batch_size": 20},
            "image": {"resource_allocation": 0.4, "max_size": 1000, "batch_size": 10},
            "table": {"resource_allocation": 0.2, "max_size": 1000, "batch_size": 10}
        }
        
        self.config = queue_config or default_config
        
        # 创建队列
        self.queues = {
            DataType.TEXT: PriorityQueue(maxsize=self.config["text"]["max_size"]),
            DataType.IMAGE: PriorityQueue(maxsize=self.config["image"]["max_size"]),
            DataType.TABLE: PriorityQueue(maxsize=self.config["table"]["max_size"])
        }
        
        # 结果存储
        self.results = {}
        self.results_lock = threading.Lock()
        
        # 处理器注册
        self.processors = {}
        
        # 统计信息
        self.stats = {
            DataType.TEXT: {"processed": 0, "failed": 0, "total_time": 0},
            DataType.IMAGE: {"processed": 0, "failed": 0, "total_time": 0},
            DataType.TABLE: {"processed": 0, "failed": 0, "total_time": 0}
        }
        
        # 工作线程
        self.workers = {}
        self.running = False
        
        # 请求去重
        self.request_cache = {}
        self.cache_lock = threading.Lock()
        
        logger.info("初始化队列管理器")
        
    def register_processor(self, data_type: DataType, processor: Callable):
        """
        注册处理器
        Args:
            data_type: 数据类型
            processor: 处理函数
        """
        self.processors[data_type] = processor
        logger.info(f"注册{data_type.value}类型处理器")
        
    def submit_request(self, 
                      query: str,
                      data: Any,
                      data_type: DataType = DataType.TEXT,
                      priority: int = 5,
                      callback: Optional[Callable] = None) -> str:
        """
        提交请求到队列
        Args:
            query: 查询
            data: 数据
            data_type: 数据类型
            priority: 优先级
            callback: 回调函数
        Returns:
            请求ID
        """
        # 请求去重检查
        cache_key = self._generate_cache_key(query, data_type)
        
        with self.cache_lock:
            if cache_key in self.request_cache:
                cached_result = self.request_cache[cache_key]
                if time.time() - cached_result["timestamp"] < 60:  # 60秒内的重复请求
                    logger.info(f"命中请求缓存: {cache_key}")
                    if callback:
                        callback(cached_result["result"])
                    return cached_result["request_id"]
                    
        # 创建请求
        request_id = str(uuid.uuid4())
        request = QueueRequest(
            request_id=request_id,
            data_type=data_type,
            query=query,
            data=data,
            priority=priority,
            callback=callback
        )
        
        # 添加到队列
        try:
            queue = self.queues[data_type]
            queue.put_nowait(request)
            logger.info(f"提交请求到{data_type.value}队列: {request_id}")
            
        except Exception as e:
            logger.error(f"队列已满: {e}")
            if callback:
                callback(QueueResponse(
                    request_id=request_id,
                    result=None,
                    processing_time=0,
                    success=False,
                    error="Queue full"
                ))
                
        return request_id
        
    def start(self):
        """启动队列处理"""
        if self.running:
            logger.warning("队列管理器已在运行")
            return
            
        self.running = True
        
        # 为每种数据类型启动工作线程
        for data_type in DataType:
            worker_count = self._get_worker_count(data_type)
            self.workers[data_type] = []
            
            for i in range(worker_count):
                worker = threading.Thread(
                    target=self._worker_loop,
                    args=(data_type,),
                    daemon=True,
                    name=f"{data_type.value}_worker_{i}"
                )
                worker.start()
                self.workers[data_type].append(worker)
                
        logger.info("队列管理器已启动")
        
    def stop(self):
        """停止队列处理"""
        self.running = False
        logger.info("正在停止队列管理器...")
        
        # 等待所有工作线程结束
        for workers in self.workers.values():
            for worker in workers:
                worker.join(timeout=5)
                
        logger.info("队列管理器已停止")
        
    def _worker_loop(self, data_type: DataType):
        """
        工作线程循环
        Args:
            data_type: 数据类型
        """
        queue = self.queues[data_type]
        batch_size = self.config[data_type.value]["batch_size"]
        batch = []
        last_process_time = time.time()
        
        while self.running:
            try:
                # 收集批次
                timeout = 0.5 if not batch else 0.1
                
                try:
                    request = queue.get(timeout=timeout)
                    batch.append(request)
                except Empty:
                    pass
                    
                # 处理批次的条件
                should_process = (
                    len(batch) >= batch_size or  # 批次已满
                    (batch and time.time() - last_process_time > 0.5) or  # 超时
                    (batch and queue.empty())  # 队列为空
                )
                
                if should_process and batch:
                    self._process_batch(batch, data_type)
                    batch = []
                    last_process_time = time.time()
                    
            except Exception as e:
                logger.error(f"{data_type.value}工作线程异常: {e}")
                
    def _process_batch(self, batch: List[QueueRequest], data_type: DataType):
        """
        处理请求批次
        Args:
            batch: 请求批次
            data_type: 数据类型
        """
        processor = self.processors.get(data_type)
        if not processor:
            logger.warning(f"未找到{data_type.value}类型处理器")
            return
            
        # 请求去重
        unique_batch = self._deduplicate_batch(batch)
        
        # 结果广播准备
        broadcast_map = self._prepare_broadcast_map(batch, unique_batch)
        
        # 批量处理
        start_time = time.time()
        
        try:
            # 提取数据进行批处理
            batch_data = [(req.query, req.data) for req in unique_batch]
            results = processor(batch_data)
            
            # 处理结果
            for i, request in enumerate(unique_batch):
                processing_time = time.time() - start_time
                
                response = QueueResponse(
                    request_id=request.request_id,
                    result=results[i] if i < len(results) else None,
                    processing_time=processing_time,
                    success=True
                )
                
                # 广播结果
                self._broadcast_result(response, broadcast_map.get(request.request_id, []))
                
                # 缓存结果
                self._cache_result(request, response)
                
                # 更新统计
                self._update_stats(data_type, processing_time, True)
                
        except Exception as e:
            logger.error(f"批处理失败: {e}")
            
            # 处理失败情况
            for request in batch:
                response = QueueResponse(
                    request_id=request.request_id,
                    result=None,
                    processing_time=0,
                    success=False,
                    error=str(e)
                )
                
                if request.callback:
                    request.callback(response)
                    
                self._update_stats(data_type, 0, False)
                
    def _deduplicate_batch(self, batch: List[QueueRequest]) -> List[QueueRequest]:
        """
        批次去重
        Args:
            batch: 原始批次
        Returns:
            去重后的批次
        """
        seen = set()
        unique = []
        
        for request in batch:
            cache_key = self._generate_cache_key(request.query, request.data_type)
            if cache_key not in seen:
                seen.add(cache_key)
                unique.append(request)
                
        if len(unique) < len(batch):
            logger.info(f"批次去重: {len(batch)} -> {len(unique)}")
            
        return unique
        
    def _prepare_broadcast_map(self, 
                               original_batch: List[QueueRequest],
                               unique_batch: List[QueueRequest]) -> Dict[str, List[QueueRequest]]:
        """
        准备结果广播映射
        Args:
            original_batch: 原始批次
            unique_batch: 去重批次
        Returns:
            广播映射
        """
        broadcast_map = {}
        
        # 建立查询到唯一请求的映射
        query_to_unique = {}
        for req in unique_batch:
            cache_key = self._generate_cache_key(req.query, req.data_type)
            query_to_unique[cache_key] = req.request_id
            broadcast_map[req.request_id] = []
            
        # 映射所有请求到唯一请求
        for req in original_batch:
            cache_key = self._generate_cache_key(req.query, req.data_type)
            unique_id = query_to_unique.get(cache_key)
            if unique_id and unique_id != req.request_id:
                broadcast_map[unique_id].append(req)
                
        return broadcast_map
        
    def _broadcast_result(self, response: QueueResponse, duplicate_requests: List[QueueRequest]):
        """
        广播结果到重复请求
        Args:
            response: 响应结果
            duplicate_requests: 重复请求列表
        """
        # 处理原始请求
        with self.results_lock:
            self.results[response.request_id] = response
            
        # 广播到重复请求
        for req in duplicate_requests:
            dup_response = QueueResponse(
                request_id=req.request_id,
                result=response.result,
                processing_time=response.processing_time,
                success=response.success,
                error=response.error
            )
            
            with self.results_lock:
                self.results[req.request_id] = dup_response
                
            if req.callback:
                req.callback(dup_response)
                
        if duplicate_requests:
            logger.info(f"广播结果到{len(duplicate_requests)}个重复请求")
            
    def _cache_result(self, request: QueueRequest, response: QueueResponse):
        """
        缓存结果
        Args:
            request: 请求
            response: 响应
        """
        cache_key = self._generate_cache_key(request.query, request.data_type)
        
        with self.cache_lock:
            self.request_cache[cache_key] = {
                "request_id": request.request_id,
                "result": response,
                "timestamp": time.time()
            }
            
            # 清理过期缓存
            self._cleanup_cache()
            
    def _cleanup_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []
        
        for key, value in self.request_cache.items():
            if current_time - value["timestamp"] > 300:  # 5分钟过期
                expired_keys.append(key)
                
        for key in expired_keys:
            del self.request_cache[key]
            
        if expired_keys:
            logger.info(f"清理了{len(expired_keys)}个过期缓存")
            
    def _generate_cache_key(self, query: str, data_type: DataType) -> str:
        """
        生成缓存键
        Args:
            query: 查询
            data_type: 数据类型
        Returns:
            缓存键
        """
        import hashlib
        content = f"{data_type.value}:{query}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def _get_worker_count(self, data_type: DataType) -> int:
        """
        获取工作线程数
        Args:
            data_type: 数据类型
        Returns:
            工作线程数
        """
        resource_allocation = self.config[data_type.value]["resource_allocation"]
        # 基于资源分配计算线程数
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        return max(1, int(cpu_count * resource_allocation))
        
    def _update_stats(self, data_type: DataType, processing_time: float, success: bool):
        """
        更新统计信息
        Args:
            data_type: 数据类型
            processing_time: 处理时间
            success: 是否成功
        """
        stats = self.stats[data_type]
        
        if success:
            stats["processed"] += 1
            stats["total_time"] += processing_time
        else:
            stats["failed"] += 1
            
    def get_result(self, request_id: str, timeout: float = 10) -> Optional[QueueResponse]:
        """
        获取处理结果
        Args:
            request_id: 请求ID
            timeout: 超时时间
        Returns:
            响应结果
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.results_lock:
                if request_id in self.results:
                    return self.results.pop(request_id)
                    
            time.sleep(0.1)
            
        return None
        
    def optimize_batch_processing(self, requests: List[Tuple[str, Any, DataType]]) -> List[str]:
        """
        批量处理优化
        Args:
            requests: [(查询, 数据, 数据类型)]列表
        Returns:
            请求ID列表
        """
        # 按数据类型分组
        grouped = {}
        for query, data, data_type in requests:
            if data_type not in grouped:
                grouped[data_type] = []
            grouped[data_type].append((query, data))
            
        request_ids = []
        
        # 批量提交
        for data_type, items in grouped.items():
            # 简单问题优先
            sorted_items = sorted(items, key=lambda x: len(x[0]))
            
            for query, data in sorted_items:
                request_id = self.submit_request(query, data, data_type, priority=5)
                request_ids.append(request_id)
                
        logger.info(f"批量提交{len(request_ids)}个请求")
        
        return request_ids
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        Returns:
            统计信息字典
        """
        queue_sizes = {
            data_type.value: self.queues[data_type].qsize()
            for data_type in DataType
        }
        
        return {
            "queue_sizes": queue_sizes,
            "processing_stats": {
                data_type.value: {
                    "processed": self.stats[data_type]["processed"],
                    "failed": self.stats[data_type]["failed"],
                    "avg_time": (self.stats[data_type]["total_time"] / 
                               max(self.stats[data_type]["processed"], 1))
                }
                for data_type in DataType
            },
            "cache_size": len(self.request_cache),
            "resource_allocation": {
                data_type.value: self.config[data_type.value]["resource_allocation"]
                for data_type.value in ["text", "image", "table"]
            }
        }