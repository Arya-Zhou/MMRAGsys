"""
缓存系统
实现两级缓存架构：L1精确缓存 + L2语义缓存
"""

import hashlib
import time
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np
import logging
from threading import Lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: float = 86400  # 默认24小时
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return time.time() - self.timestamp > self.ttl
    
    def update_access(self):
        """更新访问信息"""
        self.access_count += 1
        self.timestamp = time.time()


class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, capacity: int = 1000):
        """
        初始化LRU缓存
        Args:
            capacity: 缓存容量
        """
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = Lock()
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        Args:
            key: 缓存键
        Returns:
            缓存值或None
        """
        with self.lock:
            if key in self.cache:
                # 移动到末尾（最近使用）
                entry = self.cache.pop(key)
                
                # 检查是否过期
                if entry.is_expired():
                    self.misses += 1
                    return None
                    
                entry.update_access()
                self.cache[key] = entry
                self.hits += 1
                return entry.value
            else:
                self.misses += 1
                return None
                
    def put(self, key: str, value: Any, ttl: float = 86400, metadata: Dict = None):
        """
        设置缓存值
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
            metadata: 元数据
        """
        with self.lock:
            # 如果键已存在，先删除
            if key in self.cache:
                del self.cache[key]
                
            # 检查容量
            elif len(self.cache) >= self.capacity:
                # 删除最老的项（第一个）
                self.cache.popitem(last=False)
                
            # 添加新项
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                metadata=metadata or {}
            )
            self.cache[key] = entry
            
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            
            return {
                "size": len(self.cache),
                "capacity": self.capacity,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate
            }


class SemanticCache:
    """语义缓存实现"""
    
    def __init__(self, 
                 capacity: int = 5000,
                 similarity_threshold: float = 0.95):
        """
        初始化语义缓存
        Args:
            capacity: 缓存容量
            similarity_threshold: 相似度阈值
        """
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.cache = OrderedDict()
        self.embeddings = []
        self.lock = Lock()
        self.hits = 0
        self.misses = 0
        
    def get(self, query: str, query_embedding: np.ndarray) -> Optional[Any]:
        """
        基于语义相似度获取缓存
        Args:
            query: 查询文本
            query_embedding: 查询向量
        Returns:
            缓存值或None
        """
        with self.lock:
            if not self.embeddings:
                self.misses += 1
                return None
                
            # 计算相似度
            similarities = self._compute_similarities(query_embedding)
            
            # 找到最相似的缓存项
            max_idx = np.argmax(similarities)
            max_similarity = similarities[max_idx]
            
            if max_similarity >= self.similarity_threshold:
                # 获取对应的缓存键
                cache_keys = list(self.cache.keys())
                if max_idx < len(cache_keys):
                    key = cache_keys[max_idx]
                    entry = self.cache[key]
                    
                    # 检查是否过期
                    if entry.is_expired():
                        # 删除过期项
                        del self.cache[key]
                        del self.embeddings[max_idx]
                        self.misses += 1
                        return None
                        
                    entry.update_access()
                    self.hits += 1
                    
                    logger.info(f"语义缓存命中，相似度: {max_similarity:.4f}")
                    return entry.value
                    
            self.misses += 1
            return None
            
    def put(self, 
           query: str,
           query_embedding: np.ndarray,
           value: Any,
           ttl: float = 86400,
           metadata: Dict = None):
        """
        添加语义缓存
        Args:
            query: 查询文本
            query_embedding: 查询向量
            value: 缓存值
            ttl: 过期时间
            metadata: 元数据
        """
        with self.lock:
            # 生成缓存键
            key = self._generate_key(query)
            
            # 检查是否已存在
            if key in self.cache:
                # 更新现有项
                idx = list(self.cache.keys()).index(key)
                self.embeddings[idx] = query_embedding
                self.cache[key].value = value
                self.cache[key].timestamp = time.time()
                return
                
            # 检查容量
            if len(self.cache) >= self.capacity:
                # 删除最老的项
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.embeddings[0]
                
            # 添加新项
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                metadata=metadata or {}
            )
            self.cache[key] = entry
            self.embeddings.append(query_embedding)
            
    def _compute_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        计算查询向量与缓存向量的相似度
        Args:
            query_embedding: 查询向量
        Returns:
            相似度数组
        """
        if not self.embeddings:
            return np.array([])
            
        # 归一化查询向量
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # 计算余弦相似度
        similarities = []
        for embedding in self.embeddings:
            embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
            similarity = np.dot(query_norm, embedding_norm)
            similarities.append(similarity)
            
        return np.array(similarities)
    
    def _generate_key(self, query: str) -> str:
        """生成缓存键"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.embeddings.clear()
            self.hits = 0
            self.misses = 0
            
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            
            return {
                "size": len(self.cache),
                "capacity": self.capacity,
                "similarity_threshold": self.similarity_threshold,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate
            }


class CoTCache:
    """CoT推理专用缓存"""
    
    def __init__(self,
                 capacity: int = 5000,
                 similarity_threshold: float = 0.9,
                 ttl: float = 86400):
        """
        初始化CoT缓存
        Args:
            capacity: 缓存容量
            similarity_threshold: 相似度阈值
            ttl: 过期时间
        """
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.cache = OrderedDict()
        self.lock = Lock()
        
    def get_cache_key(self, query: str, question_type: str) -> str:
        """
        生成CoT缓存键
        Args:
            query: 查询文本
            question_type: 问题类型
        Returns:
            缓存键
        """
        combined = f"{query}|{question_type}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, query: str, question_type: str) -> Optional[Dict[str, Any]]:
        """
        获取CoT推理结果
        Args:
            query: 查询文本
            question_type: 问题类型
        Returns:
            推理结果或None
        """
        key = self.get_cache_key(query, question_type)
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                if entry.is_expired():
                    del self.cache[key]
                    return None
                    
                entry.update_access()
                
                # 移动到末尾
                self.cache.move_to_end(key)
                
                logger.info(f"CoT缓存命中: {question_type}")
                return entry.value
                
            return None
            
    def put(self, 
           query: str,
           question_type: str,
           reasoning_chain: List[str],
           answer: str,
           quality_score: float,
           metadata: Dict = None):
        """
        缓存CoT推理结果
        Args:
            query: 查询文本
            question_type: 问题类型
            reasoning_chain: 推理链
            answer: 答案
            quality_score: 质量分数
            metadata: 元数据
        """
        key = self.get_cache_key(query, question_type)
        
        value = {
            "reasoning_chain": reasoning_chain,
            "answer": answer,
            "quality_score": quality_score,
            "question_type": question_type,
            "timestamp": time.time()
        }
        
        with self.lock:
            # 检查容量
            if key not in self.cache and len(self.cache) >= self.capacity:
                # 删除最老的项
                self.cache.popitem(last=False)
                
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=self.ttl,
                metadata=metadata or {}
            )
            
            self.cache[key] = entry
            
    def clear_by_type(self, question_type: str):
        """
        清除特定类型的缓存
        Args:
            question_type: 问题类型
        """
        with self.lock:
            keys_to_delete = []
            for key, entry in self.cache.items():
                if entry.value.get("question_type") == question_type:
                    keys_to_delete.append(key)
                    
            for key in keys_to_delete:
                del self.cache[key]
                
            logger.info(f"清除了 {len(keys_to_delete)} 个 {question_type} 类型的缓存")
            
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.lock:
            type_counts = {}
            total_quality = 0
            
            for entry in self.cache.values():
                qtype = entry.value.get("question_type", "unknown")
                type_counts[qtype] = type_counts.get(qtype, 0) + 1
                total_quality += entry.value.get("quality_score", 0)
                
            avg_quality = total_quality / len(self.cache) if self.cache else 0
            
            return {
                "size": len(self.cache),
                "capacity": self.capacity,
                "type_distribution": type_counts,
                "avg_quality_score": avg_quality,
                "ttl": self.ttl
            }


class CacheManager:
    """统一缓存管理器"""
    
    def __init__(self):
        """初始化缓存管理器"""
        # L1精确缓存
        self.l1_cache = LRUCache(capacity=1000)
        
        # L2语义缓存
        self.l2_cache = SemanticCache(capacity=5000)
        
        # CoT专用缓存
        self.cot_cache = CoTCache(capacity=5000)
        
        logger.info("初始化缓存管理器")
        
    def get(self, 
           query: str,
           query_embedding: Optional[np.ndarray] = None,
           use_semantic: bool = True) -> Optional[Any]:
        """
        获取缓存值
        Args:
            query: 查询文本
            query_embedding: 查询向量
            use_semantic: 是否使用语义缓存
        Returns:
            缓存值或None
        """
        # 首先尝试L1精确缓存
        key = hashlib.md5(query.encode()).hexdigest()
        result = self.l1_cache.get(key)
        
        if result is not None:
            logger.info("L1缓存命中")
            return result
            
        # 尝试L2语义缓存
        if use_semantic and query_embedding is not None:
            result = self.l2_cache.get(query, query_embedding)
            if result is not None:
                logger.info("L2语义缓存命中")
                # 同时更新L1缓存
                self.l1_cache.put(key, result)
                return result
                
        return None
        
    def put(self,
           query: str,
           value: Any,
           query_embedding: Optional[np.ndarray] = None,
           ttl: float = 86400):
        """
        设置缓存值
        Args:
            query: 查询文本
            value: 缓存值
            query_embedding: 查询向量
            ttl: 过期时间
        """
        # 更新L1缓存
        key = hashlib.md5(query.encode()).hexdigest()
        self.l1_cache.put(key, value, ttl)
        
        # 更新L2缓存
        if query_embedding is not None:
            self.l2_cache.put(query, query_embedding, value, ttl)
            
        logger.info("缓存已更新")
        
    def invalidate_related(self, keywords: List[str]):
        """
        使相关缓存失效
        Args:
            keywords: 关键词列表
        """
        # 这里简化处理，实际可以更精细
        invalidated = 0
        
        with self.l1_cache.lock:
            keys_to_delete = []
            for key, entry in self.l1_cache.cache.items():
                # 检查是否包含关键词
                if isinstance(entry.value, str):
                    for keyword in keywords:
                        if keyword in entry.value:
                            keys_to_delete.append(key)
                            break
                            
            for key in keys_to_delete:
                del self.l1_cache.cache[key]
                invalidated += 1
                
        logger.info(f"失效了 {invalidated} 个相关缓存")
        
    def clear_all(self):
        """清空所有缓存"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.cot_cache.clear()
        logger.info("所有缓存已清空")
        
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有缓存统计"""
        return {
            "l1_cache": self.l1_cache.get_stats(),
            "l2_cache": self.l2_cache.get_stats(),
            "cot_cache": self.cot_cache.get_stats()
        }
        
    def warm_up(self, hot_queries: List[Tuple[str, Any]]):
        """
        缓存预热
        Args:
            hot_queries: [(查询, 结果)]列表
        """
        for query, result in hot_queries:
            key = hashlib.md5(query.encode()).hexdigest()
            self.l1_cache.put(key, result)
            
        logger.info(f"预热了 {len(hot_queries)} 个热点查询")