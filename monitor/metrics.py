"""
监控指标采集器
采集和管理系统监控指标
"""

import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """指标数据点"""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    """告警规则"""
    metric_name: str
    threshold: float
    operator: str  # ">", "<", ">=", "<=", "=="
    window_size: int = 60  # 秒
    description: str = ""


class MetricsCollector:
    """监控指标采集器"""
    
    def __init__(self):
        """初始化指标采集器"""
        
        # 指标存储
        self.metrics = {}
        self.metrics_lock = threading.Lock()
        
        # 指标优先级
        self.metric_priorities = {
            "answer_relevance": 0,      # P0级
            "retrieval_precision": 0,    # P0级
            "p95_latency": 1,           # P1级
            "error_rate": 2,            # P2级
            "cache_hit_rate": 2,        # P2级
            "cot_failure_rate": 1,      # P1级
            "cpu_usage": 2,             # P2级
            "memory_usage": 2,          # P2级
            "queue_size": 2             # P2级
        }
        
        # 告警规则
        self.alert_rules = [
            AlertRule("answer_relevance", 0.7, "<", 300, "答案相关性低于阈值"),
            AlertRule("p95_latency", 5000, ">", 60, "P95延迟超过5秒"),
            AlertRule("error_rate", 0.1, ">", 300, "错误率超过10%"),
            AlertRule("cache_hit_rate", 0.3, "<", 600, "缓存命中率低于30%"),
            AlertRule("cot_failure_rate", 0.1, ">", 300, "CoT失败率超过10%"),
            AlertRule("cpu_usage", 80, ">", 60, "CPU使用率超过80%"),
            AlertRule("memory_usage", 90, ">", 60, "内存使用率超过90%")
        ]
        
        # 告警历史
        self.alerts = deque(maxlen=1000)
        
        # 存储后端（预留接口）
        self.storage_backend = None
        
        logger.info("初始化监控指标采集器")
        
    def collect_metric(self, 
                      name: str,
                      value: float,
                      labels: Optional[Dict[str, str]] = None):
        """
        采集单个指标
        Args:
            name: 指标名称
            value: 指标值
            labels: 标签
        """
        metric_point = MetricPoint(
            name=name,
            value=value,
            labels=labels or {}
        )
        
        with self.metrics_lock:
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=10000)
            self.metrics[name].append(metric_point)
            
        # 检查告警
        self._check_alerts(name, value)
        
        # 发送到存储后端
        if self.storage_backend:
            self.storage_backend.store(metric_point)
            
    def collect_metrics(self,
                       operation_type: str,
                       metrics_data: Dict[str, float]):
        """
        批量采集指标
        Args:
            operation_type: 操作类型
            metrics_data: 指标数据字典
        """
        for name, value in metrics_data.items():
            self.collect_metric(
                name=name,
                value=value,
                labels={"operation": operation_type}
            )
            
    def get_metric_stats(self,
                        name: str,
                        window: int = 300) -> Dict[str, float]:
        """
        获取指标统计
        Args:
            name: 指标名称
            window: 时间窗口（秒）
        Returns:
            统计信息
        """
        with self.metrics_lock:
            if name not in self.metrics:
                return {}
                
            current_time = time.time()
            recent_points = [
                p for p in self.metrics[name]
                if current_time - p.timestamp <= window
            ]
            
            if not recent_points:
                return {}
                
            values = [p.value for p in recent_points]
            
            return {
                "count": len(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "p50": np.percentile(values, 50),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99)
            }
            
    def get_retrieval_metrics(self) -> Dict[str, float]:
        """
        获取检索评估指标
        Returns:
            检索指标字典
        """
        metrics = {}
        
        # 基础检索指标
        for metric_name in ["recall", "precision", "f1_score", "hit_at_5", 
                           "mrr", "ndcg_at_10", "map"]:
            stats = self.get_metric_stats(metric_name)
            if stats:
                metrics[metric_name] = stats.get("mean", 0)
                
        return metrics
        
    def get_quality_metrics(self) -> Dict[str, float]:
        """
        获取质量评估指标
        Returns:
            质量指标字典
        """
        metrics = {}
        
        # 输出质量指标
        for metric_name in ["page_match_score", "file_match_score", 
                           "content_similarity_score", "total_quality_score"]:
            stats = self.get_metric_stats(metric_name)
            if stats:
                metrics[metric_name] = stats.get("mean", 0)
                
        return metrics
        
    def _check_alerts(self, metric_name: str, value: float):
        """
        检查告警规则
        Args:
            metric_name: 指标名称
            value: 指标值
        """
        for rule in self.alert_rules:
            if rule.metric_name != metric_name:
                continue
                
            # 获取时间窗口内的值
            stats = self.get_metric_stats(metric_name, rule.window_size)
            if not stats:
                continue
                
            check_value = stats.get("mean", value)
            
            # 检查条件
            triggered = False
            if rule.operator == ">":
                triggered = check_value > rule.threshold
            elif rule.operator == "<":
                triggered = check_value < rule.threshold
            elif rule.operator == ">=":
                triggered = check_value >= rule.threshold
            elif rule.operator == "<=":
                triggered = check_value <= rule.threshold
            elif rule.operator == "==":
                triggered = check_value == rule.threshold
                
            if triggered:
                self._trigger_alert(rule, check_value)
                
    def _trigger_alert(self, rule: AlertRule, value: float):
        """
        触发告警
        Args:
            rule: 告警规则
            value: 当前值
        """
        alert = {
            "timestamp": time.time(),
            "metric": rule.metric_name,
            "value": value,
            "threshold": rule.threshold,
            "operator": rule.operator,
            "description": rule.description,
            "priority": self.metric_priorities.get(rule.metric_name, 2)
        }
        
        self.alerts.append(alert)
        
        # 根据优先级记录日志
        if alert["priority"] == 0:
            logger.critical(f"P0告警: {rule.description} (当前值: {value:.2f})")
        elif alert["priority"] == 1:
            logger.error(f"P1告警: {rule.description} (当前值: {value:.2f})")
        else:
            logger.warning(f"P2告警: {rule.description} (当前值: {value:.2f})")
            
    def register_storage(self, storage_backend: Any):
        """
        注册存储后端
        Args:
            storage_backend: 存储后端实例
        """
        self.storage_backend = storage_backend
        logger.info("注册监控存储后端")
        
    def export_metrics(self, filepath: str):
        """
        导出指标数据
        Args:
            filepath: 输出文件路径
        """
        export_data = {
            "timestamp": time.time(),
            "metrics": {},
            "alerts": list(self.alerts)
        }
        
        with self.metrics_lock:
            for name, points in self.metrics.items():
                recent_points = list(points)[-100:]  # 最近100个点
                export_data["metrics"][name] = [
                    {
                        "value": p.value,
                        "timestamp": p.timestamp,
                        "labels": p.labels
                    }
                    for p in recent_points
                ]
                
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"导出指标数据到 {filepath}")
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        获取仪表盘数据
        Returns:
            仪表盘数据字典
        """
        return {
            "retrieval_metrics": self.get_retrieval_metrics(),
            "quality_metrics": self.get_quality_metrics(),
            "system_metrics": {
                "p95_latency": self.get_metric_stats("p95_latency", 60).get("p95", 0),
                "error_rate": self.get_metric_stats("error_rate", 300).get("mean", 0),
                "cache_hit_rate": self.get_metric_stats("cache_hit_rate", 600).get("mean", 0),
                "cot_failure_rate": self.get_metric_stats("cot_failure_rate", 300).get("mean", 0)
            },
            "recent_alerts": list(self.alerts)[-10:],
            "metric_priorities": self.metric_priorities
        }