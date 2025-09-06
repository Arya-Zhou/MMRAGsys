"""
降级机制
实现系统降级策略
"""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """降级级别"""
    NORMAL = 0          # 正常
    LEVEL_1 = 1         # 一级降级
    LEVEL_2 = 2         # 二级降级
    COT_DISABLED = 3    # CoT关闭


@dataclass
class SystemMetrics:
    """系统指标"""
    latency: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_count: int = 0
    consecutive_errors: int = 0
    timestamp: float = field(default_factory=time.time)


class DegradationManager:
    """降级管理器"""
    
    def __init__(self):
        """初始化降级管理器"""
        
        # 降级状态
        self.current_level = DegradationLevel.NORMAL
        self.cot_enabled = True
        
        # 降级触发条件
        self.degradation_thresholds = {
            DegradationLevel.LEVEL_1: {
                "latency": 3.0,           # 延迟>3秒
                "cpu_usage": 80,          # CPU>80%
                "consecutive_errors": 3   # 连续3次错误
            },
            DegradationLevel.LEVEL_2: {
                "latency": 6.0,           # 延迟>6秒
                "cpu_usage": 90,          # CPU>90%
                "consecutive_errors": 5   # 连续5次错误
            }
        }
        
        # 降级配置
        self.degradation_configs = {
            DegradationLevel.NORMAL: {
                "rerank_candidates": {"text": 25, "image": 10, "table": 25},
                "cache_enabled": True,
                "cot_enabled": True,
                "batch_size": {"text": 20, "image": 10, "table": 10}
            },
            DegradationLevel.LEVEL_1: {
                "rerank_candidates": {"text": 15, "image": 8, "table": 15},
                "cache_enabled": True,
                "cot_enabled": True,
                "batch_size": {"text": 10, "image": 5, "table": 5}
            },
            DegradationLevel.LEVEL_2: {
                "rerank_candidates": {"text": 10, "image": 5, "table": 10},
                "cache_enabled": True,
                "cot_enabled": False,  # 禁用CoT
                "batch_size": {"text": 5, "image": 3, "table": 3}
            }
        }
        
        # 恢复条件
        self.recovery_thresholds = {
            "latency": 2.0,
            "cpu_usage": 60,
            "error_rate": 0.01
        }
        
        # 监控数据
        self.metrics_history = []
        self.lock = threading.Lock()
        
        # 降级历史
        self.degradation_history = []
        
        logger.info("初始化降级管理器")
        
    def evaluate_system_status(self, metrics: SystemMetrics) -> DegradationLevel:
        """
        评估系统状态并决定降级级别
        Args:
            metrics: 系统指标
        Returns:
            建议的降级级别
        """
        # 记录指标
        with self.lock:
            self.metrics_history.append(metrics)
            # 只保留最近100个数据点
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
                
        # 评估是否需要降级
        suggested_level = DegradationLevel.NORMAL
        
        # 检查二级降级条件
        level2_thresholds = self.degradation_thresholds[DegradationLevel.LEVEL_2]
        if (metrics.latency > level2_thresholds["latency"] or
            metrics.cpu_usage > level2_thresholds["cpu_usage"] or
            metrics.consecutive_errors >= level2_thresholds["consecutive_errors"]):
            suggested_level = DegradationLevel.LEVEL_2
            
        # 检查一级降级条件
        elif self.current_level == DegradationLevel.NORMAL:
            level1_thresholds = self.degradation_thresholds[DegradationLevel.LEVEL_1]
            if (metrics.latency > level1_thresholds["latency"] or
                metrics.cpu_usage > level1_thresholds["cpu_usage"] or
                metrics.consecutive_errors >= level1_thresholds["consecutive_errors"]):
                suggested_level = DegradationLevel.LEVEL_1
                
        # 检查恢复条件
        elif self.current_level != DegradationLevel.NORMAL:
            if self._check_recovery_conditions():
                suggested_level = max(DegradationLevel.NORMAL, 
                                     DegradationLevel(self.current_level.value - 1))
                
        return suggested_level
        
    def apply_degradation(self, level: DegradationLevel) -> Dict[str, Any]:
        """
        应用降级策略
        Args:
            level: 降级级别
        Returns:
            降级后的配置
        """
        if level == self.current_level:
            return self.get_current_config()
            
        old_level = self.current_level
        self.current_level = level
        
        # 记录降级历史
        self.degradation_history.append({
            "timestamp": time.time(),
            "from_level": old_level.name,
            "to_level": level.name,
            "reason": self._get_degradation_reason()
        })
        
        config = self.degradation_configs[level]
        
        # 特殊处理CoT状态
        self.cot_enabled = config.get("cot_enabled", True)
        
        if old_level == DegradationLevel.NORMAL and level != DegradationLevel.NORMAL:
            logger.warning(f"系统降级: {old_level.name} -> {level.name}")
        elif old_level != DegradationLevel.NORMAL and level == DegradationLevel.NORMAL:
            logger.info(f"系统恢复: {old_level.name} -> {level.name}")
        else:
            logger.info(f"降级调整: {old_level.name} -> {level.name}")
            
        return config
        
    def _check_recovery_conditions(self) -> bool:
        """
        检查恢复条件
        Returns:
            是否满足恢复条件
        """
        if len(self.metrics_history) < 10:
            return False
            
        # 检查最近10个指标
        recent_metrics = self.metrics_history[-10:]
        
        avg_latency = sum(m.latency for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        error_count = sum(m.error_count for m in recent_metrics)
        
        return (avg_latency < self.recovery_thresholds["latency"] and
                avg_cpu < self.recovery_thresholds["cpu_usage"] and
                error_count == 0)
                
    def _get_degradation_reason(self) -> str:
        """
        获取降级原因
        Returns:
            降级原因描述
        """
        if not self.metrics_history:
            return "未知原因"
            
        latest = self.metrics_history[-1]
        reasons = []
        
        if latest.latency > 3:
            reasons.append(f"高延迟({latest.latency:.2f}s)")
        if latest.cpu_usage > 80:
            reasons.append(f"高CPU({latest.cpu_usage:.1f}%)")
        if latest.consecutive_errors > 0:
            reasons.append(f"连续错误({latest.consecutive_errors}次)")
            
        return ", ".join(reasons) if reasons else "系统压力"
        
    def get_current_config(self) -> Dict[str, Any]:
        """
        获取当前配置
        Returns:
            当前降级配置
        """
        return self.degradation_configs[self.current_level]
        
    def force_degradation(self, level: DegradationLevel):
        """
        强制降级
        Args:
            level: 降级级别
        """
        logger.warning(f"强制降级到 {level.name}")
        self.apply_degradation(level)
        
    def disable_cot(self):
        """禁用CoT"""
        self.cot_enabled = False
        logger.warning("CoT已禁用")
        
    def enable_cot(self):
        """启用CoT"""
        if self.current_level != DegradationLevel.LEVEL_2:
            self.cot_enabled = True
            logger.info("CoT已启用")
        else:
            logger.warning("当前降级级别不允许启用CoT")
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        Returns:
            统计信息字典
        """
        stats = {
            "current_level": self.current_level.name,
            "cot_enabled": self.cot_enabled,
            "current_config": self.get_current_config(),
            "degradation_count": len(self.degradation_history),
            "recent_degradations": self.degradation_history[-5:] if self.degradation_history else []
        }
        
        if self.metrics_history:
            recent = self.metrics_history[-10:]
            stats["recent_metrics"] = {
                "avg_latency": sum(m.latency for m in recent) / len(recent),
                "avg_cpu": sum(m.cpu_usage for m in recent) / len(recent),
                "total_errors": sum(m.error_count for m in recent)
            }
            
        return stats