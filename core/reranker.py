"""
重排序模块
使用BAAI/bge-reranker-v2-m3模型对检索结果进行重排序
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import logging
import time
from queue import Queue
from threading import Thread, Lock
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入transformers库
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers库未安装，将使用模拟重排序功能")


@dataclass
class RerankRequest:
    """重排序请求"""
    query: str
    candidates: List[str]
    data_type: str = 'text'
    request_id: str = ""
    callback: Optional[Any] = None


@dataclass 
class RerankResult:
    """重排序结果"""
    scores: List[float]
    indices: List[int]
    request_id: str = ""


class Reranker:
    """重排序模块"""
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-reranker-v2-m3",
                 device: str = None,
                 batch_size: int = 8,
                 use_fp16: bool = False):
        """
        初始化重排序器
        Args:
            model_name: 模型名称
            device: 计算设备
            batch_size: 批处理大小
            use_fp16: 是否使用FP16精度
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # 候选数量配置
        self.candidate_limits = {
            'text': 25,
            'image': 10,
            'table': 25
        }
        
        # 性能监控
        self.cpu_threshold = 60
        self.latency_threshold = 5
        self.quantization_mode = 'fp16' if use_fp16 else 'fp32'
        self.last_mode_switch = time.time()
        self.min_switch_interval = 300  # 5分钟
        
        # 批处理队列
        self.request_queue = Queue(maxsize=100)
        self.batch_wait_time = 0.1  # 100ms
        
        # 加载模型
        self._load_model()
        
        # 启动批处理线程
        self.batch_thread = Thread(target=self._batch_processor, daemon=True)
        self.batch_thread.start()
        
        logger.info(f"初始化重排序器: device={self.device}, batch_size={batch_size}")
        
    def _load_model(self):
        """加载重排序模型"""
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                
                if self.use_fp16 and self.device.type == 'cuda':
                    self.model = self.model.half()
                    
                self.model.eval()
                logger.info(f"成功加载模型: {self.model_name}")
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                self.model = None
                self.tokenizer = None
        else:
            logger.warning("使用模拟重排序功能")
            self.model = None
            self.tokenizer = None
            
    def rerank(self, 
              query: str,
              candidates: List[Any],
              top_k: int = 10,
              data_type: str = 'text') -> List[Tuple[Any, float]]:
        """
        对候选文档进行重排序
        Args:
            query: 查询文本
            candidates: 候选文档列表
            top_k: 返回前k个结果
            data_type: 数据类型(text/image/table)
        Returns:
            [(文档, 分数)]列表
        """
        if not candidates:
            return []
            
        # 根据数据类型限制候选数量
        limit = self.candidate_limits.get(data_type, 25)
        candidates = candidates[:limit]
        
        # 提取候选文本
        candidate_texts = []
        for candidate in candidates:
            if isinstance(candidate, str):
                candidate_texts.append(candidate)
            elif hasattr(candidate, 'content'):
                candidate_texts.append(candidate.content)
            elif hasattr(candidate, 'text'):
                candidate_texts.append(candidate.text)
            else:
                candidate_texts.append(str(candidate))
                
        # 执行重排序
        if self.model is not None:
            scores = self._compute_scores(query, candidate_texts)
        else:
            # 模拟重排序（使用简单的文本相似度）
            scores = self._simulate_rerank(query, candidate_texts)
            
        # 排序并返回top_k结果
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        
        for idx in ranked_indices:
            if idx < len(candidates):
                results.append((candidates[idx], float(scores[idx])))
                
        logger.info(f"重排序完成: 输入{len(candidates)}个候选，输出{len(results)}个结果")
        
        return results
    
    def _compute_scores(self, query: str, candidates: List[str]) -> np.ndarray:
        """
        计算重排序分数
        Args:
            query: 查询文本
            candidates: 候选文本列表
        Returns:
            分数数组
        """
        scores = []
        
        # 检查是否需要切换量化模式
        self._check_quantization_mode()
        
        # 分批处理
        for i in range(0, len(candidates), self.batch_size):
            batch_candidates = candidates[i:i+self.batch_size]
            batch_scores = self._process_batch(query, batch_candidates)
            scores.extend(batch_scores)
            
        return np.array(scores)
    
    def _process_batch(self, query: str, candidates: List[str]) -> List[float]:
        """
        处理一批候选文档
        Args:
            query: 查询文本
            candidates: 候选文本批次
        Returns:
            分数列表
        """
        # 构建输入对
        pairs = [[query, candidate] for candidate in candidates]
        
        # 编码
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # 前向传播
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1)
            
            # 转换为概率
            if scores.dim() > 0:
                scores = torch.sigmoid(scores)
            else:
                scores = torch.sigmoid(scores.unsqueeze(0))
                
        return scores.cpu().numpy().tolist()
    
    def _simulate_rerank(self, query: str, candidates: List[str]) -> np.ndarray:
        """
        模拟重排序（当模型不可用时）
        Args:
            query: 查询文本  
            candidates: 候选文本列表
        Returns:
            模拟分数数组
        """
        scores = []
        query_words = set(query.lower().split())
        
        for candidate in candidates:
            candidate_words = set(candidate.lower().split())
            # 简单的词重叠度计算
            overlap = len(query_words & candidate_words)
            score = overlap / (len(query_words) + 1)
            scores.append(score)
            
        return np.array(scores)
    
    def _check_quantization_mode(self):
        """检查并切换量化模式"""
        if not self.model or not torch.cuda.is_available():
            return
            
        current_time = time.time()
        if current_time - self.last_mode_switch < self.min_switch_interval:
            return
            
        # 获取系统状态（这里简化处理）
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
        except:
            cpu_percent = 0
            
        # 决定是否切换模式
        should_use_int8 = cpu_percent >= self.cpu_threshold
        
        if should_use_int8 and self.quantization_mode == 'fp16':
            logger.info("切换到INT8量化模式")
            self.quantization_mode = 'int8'
            self._apply_int8_quantization()
            self.last_mode_switch = current_time
            
        elif not should_use_int8 and self.quantization_mode == 'int8':
            logger.info("切换回FP16模式")
            self.quantization_mode = 'fp16'
            self._apply_fp16_mode()
            self.last_mode_switch = current_time
            
    def _apply_int8_quantization(self):
        """应用INT8量化"""
        if self.model and hasattr(torch, 'quantization'):
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            except Exception as e:
                logger.warning(f"INT8量化失败: {e}")
                
    def _apply_fp16_mode(self):
        """应用FP16模式"""
        if self.model and self.device.type == 'cuda':
            try:
                self.model = self.model.half()
            except Exception as e:
                logger.warning(f"FP16转换失败: {e}")
                
    def _batch_processor(self):
        """批处理线程"""
        while True:
            batch = []
            start_time = time.time()
            
            # 收集批次
            while len(batch) < self.batch_size:
                timeout = self.batch_wait_time - (time.time() - start_time)
                if timeout <= 0:
                    break
                    
                try:
                    request = self.request_queue.get(timeout=timeout)
                    batch.append(request)
                except:
                    break
                    
            if batch:
                self._process_batch_requests(batch)
                
            time.sleep(0.01)
            
    def _process_batch_requests(self, batch: List[RerankRequest]):
        """
        处理批量重排序请求
        Args:
            batch: 请求批次
        """
        for request in batch:
            try:
                results = self.rerank(
                    request.query,
                    request.candidates,
                    data_type=request.data_type
                )
                
                if request.callback:
                    request.callback(results)
                    
            except Exception as e:
                logger.error(f"处理重排序请求失败: {e}")
                
    def rerank_async(self,
                    query: str,
                    candidates: List[Any],
                    data_type: str = 'text',
                    callback: Any = None) -> str:
        """
        异步重排序
        Args:
            query: 查询文本
            candidates: 候选文档
            data_type: 数据类型
            callback: 回调函数
        Returns:
            请求ID
        """
        import uuid
        request_id = str(uuid.uuid4())
        
        request = RerankRequest(
            query=query,
            candidates=candidates,
            data_type=data_type,
            request_id=request_id,
            callback=callback
        )
        
        try:
            self.request_queue.put_nowait(request)
            logger.info(f"添加异步重排序请求: {request_id}")
        except:
            logger.warning("请求队列已满，执行同步重排序")
            results = self.rerank(query, candidates, data_type=data_type)
            if callback:
                callback(results)
                
        return request_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取重排序统计信息
        Returns:
            统计信息字典
        """
        stats = {
            "model_name": self.model_name,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "quantization_mode": self.quantization_mode,
            "queue_size": self.request_queue.qsize(),
            "candidate_limits": self.candidate_limits,
            "model_loaded": self.model is not None
        }
        
        return stats