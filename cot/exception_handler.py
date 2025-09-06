"""
异常处理器
实现4级异常分类处理系统
"""

import logging
import traceback
from typing import Any, Dict, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass, field
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExceptionLevel(Enum):
    """异常级别枚举"""
    A_SYSTEM = "A"      # 系统级异常
    B_LOGIC = "B"       # 逻辑级异常
    C_QUALITY = "C"     # 质量级异常
    D_PERFORMANCE = "D" # 性能级异常


@dataclass
class ExceptionInfo:
    """异常信息"""
    level: ExceptionLevel
    exception_type: str
    message: str
    timestamp: float = field(default_factory=time.time)
    traceback: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    handled: bool = False


class ExceptionHandler:
    """异常处理器"""
    
    def __init__(self):
        """初始化异常处理器"""
        
        # 异常分类映射
        self.exception_classification = {
            # A级：系统级异常
            MemoryError: ExceptionLevel.A_SYSTEM,
            TimeoutError: ExceptionLevel.A_SYSTEM,
            RuntimeError: ExceptionLevel.A_SYSTEM,
            SystemError: ExceptionLevel.A_SYSTEM,
            OSError: ExceptionLevel.A_SYSTEM,
            
            # B级：逻辑级异常
            ValueError: ExceptionLevel.B_LOGIC,
            TypeError: ExceptionLevel.B_LOGIC,
            KeyError: ExceptionLevel.B_LOGIC,
            IndexError: ExceptionLevel.B_LOGIC,
            AttributeError: ExceptionLevel.B_LOGIC,
            
            # C级：质量级异常
            AssertionError: ExceptionLevel.C_QUALITY,
            
            # D级：性能级异常
            Warning: ExceptionLevel.D_PERFORMANCE
        }
        
        # 自定义异常分类
        self.custom_classifications = {
            "LogicContradiction": ExceptionLevel.B_LOGIC,
            "InfiniteLoop": ExceptionLevel.B_LOGIC,
            "PremiseMissing": ExceptionLevel.B_LOGIC,
            "InvalidReasoning": ExceptionLevel.B_LOGIC,
            
            "LowConfidence": ExceptionLevel.C_QUALITY,
            "IncompleteReasoning": ExceptionLevel.C_QUALITY,
            "IrrelevantSteps": ExceptionLevel.C_QUALITY,
            "HallucinationDetected": ExceptionLevel.C_QUALITY,
            
            "SlowResponse": ExceptionLevel.D_PERFORMANCE,
            "HighResourceUsage": ExceptionLevel.D_PERFORMANCE,
            "CacheInefficiency": ExceptionLevel.D_PERFORMANCE,
            "QueueBacklog": ExceptionLevel.D_PERFORMANCE
        }
        
        # 处理策略
        self.handling_strategies = {
            ExceptionLevel.A_SYSTEM: self._handle_system_exception,
            ExceptionLevel.B_LOGIC: self._handle_logic_exception,
            ExceptionLevel.C_QUALITY: self._handle_quality_exception,
            ExceptionLevel.D_PERFORMANCE: self._handle_performance_exception
        }
        
        # 异常历史
        self.exception_history = []
        
        # 回调函数注册
        self.callbacks = {
            ExceptionLevel.A_SYSTEM: [],
            ExceptionLevel.B_LOGIC: [],
            ExceptionLevel.C_QUALITY: [],
            ExceptionLevel.D_PERFORMANCE: []
        }
        
        logger.info("初始化异常处理器")
        
    def classify_exception(self, exception: Exception) -> ExceptionLevel:
        """
        异常分级分类
        Args:
            exception: 异常对象
        Returns:
            异常级别
        """
        # 检查内置异常类型
        for exc_type, level in self.exception_classification.items():
            if isinstance(exception, exc_type):
                return level
                
        # 检查自定义异常（通过异常消息）
        exc_message = str(exception)
        for custom_name, level in self.custom_classifications.items():
            if custom_name.lower() in exc_message.lower():
                return level
                
        # 默认归类为B级逻辑异常
        return ExceptionLevel.B_LOGIC
        
    def handle_by_level(self, exception: Exception, level: Optional[ExceptionLevel] = None, 
                       context: Optional[Dict[str, Any]] = None) -> ExceptionInfo:
        """
        按级别处理异常
        Args:
            exception: 异常对象
            level: 异常级别（可选，会自动分类）
            context: 上下文信息
        Returns:
            异常信息
        """
        # 自动分类
        if level is None:
            level = self.classify_exception(exception)
            
        # 创建异常信息
        exc_info = ExceptionInfo(
            level=level,
            exception_type=type(exception).__name__,
            message=str(exception),
            traceback=traceback.format_exc(),
            context=context or {}
        )
        
        # 记录异常
        self.exception_history.append(exc_info)
        
        # 根据级别处理
        handler = self.handling_strategies.get(level)
        if handler:
            handler(exc_info)
            
        # 触发回调
        self._trigger_callbacks(level, exc_info)
        
        logger.info(f"处理{level.value}级异常: {exc_info.exception_type}")
        
        return exc_info
        
    def _handle_system_exception(self, exc_info: ExceptionInfo):
        """
        处理A级系统异常（立即处理）
        Args:
            exc_info: 异常信息
        """
        logger.critical(f"系统级异常: {exc_info.message}")
        
        # 立即采取行动
        if "memory" in exc_info.message.lower():
            # 内存问题：触发内存清理
            self._emergency_memory_cleanup()
            
        elif "timeout" in exc_info.message.lower():
            # 超时问题：中断长时运行任务
            self._abort_long_running_tasks()
            
        elif "model" in exc_info.message.lower():
            # 模型问题：尝试重新加载
            self._reload_models()
            
        # 标记为已处理
        exc_info.handled = True
        
    def _handle_logic_exception(self, exc_info: ExceptionInfo):
        """
        处理B级逻辑异常（需要重试）
        Args:
            exc_info: 异常信息
        """
        logger.error(f"逻辑级异常: {exc_info.message}")
        
        # 判断是否可重试
        if exc_info.retry_count < 3:
            exc_info.retry_count += 1
            logger.info(f"准备重试 ({exc_info.retry_count}/3)")
            # 这里应该触发重试机制
            
        else:
            logger.warning("重试次数已达上限，降级处理")
            # 降级到C级处理
            self._handle_quality_exception(exc_info)
            
    def _handle_quality_exception(self, exc_info: ExceptionInfo):
        """
        处理C级质量异常（可降级处理）
        Args:
            exc_info: 异常信息
        """
        logger.warning(f"质量级异常: {exc_info.message}")
        
        # 降级策略
        if "confidence" in exc_info.message.lower():
            logger.info("置信度低，使用备用方案")
            
        elif "incomplete" in exc_info.message.lower():
            logger.info("推理不完整，返回部分结果")
            
        elif "hallucination" in exc_info.message.lower():
            logger.info("检测到幻觉，过滤可疑内容")
            
        # 可以继续执行，但标记质量问题
        exc_info.context["quality_degraded"] = True
        exc_info.handled = True
        
    def _handle_performance_exception(self, exc_info: ExceptionInfo):
        """
        处理D级性能异常（监控告警）
        Args:
            exc_info: 异常信息
        """
        logger.info(f"性能级异常: {exc_info.message}")
        
        # 记录性能指标
        if "slow" in exc_info.message.lower():
            self._log_performance_metric("response_time", exc_info.context.get("duration", 0))
            
        elif "resource" in exc_info.message.lower():
            self._log_performance_metric("resource_usage", exc_info.context.get("usage", 0))
            
        elif "cache" in exc_info.message.lower():
            self._log_performance_metric("cache_efficiency", exc_info.context.get("hit_rate", 0))
            
        # 不影响执行，仅记录
        exc_info.handled = True
        
    def register_callback(self, level: ExceptionLevel, callback: Callable):
        """
        注册异常回调函数
        Args:
            level: 异常级别
            callback: 回调函数
        """
        if level in self.callbacks:
            self.callbacks[level].append(callback)
            logger.info(f"注册{level.value}级异常回调")
            
    def _trigger_callbacks(self, level: ExceptionLevel, exc_info: ExceptionInfo):
        """
        触发回调函数
        Args:
            level: 异常级别
            exc_info: 异常信息
        """
        for callback in self.callbacks.get(level, []):
            try:
                callback(exc_info)
            except Exception as e:
                logger.error(f"回调执行失败: {e}")
                
    def _emergency_memory_cleanup(self):
        """紧急内存清理"""
        import gc
        gc.collect()
        logger.info("执行紧急内存清理")
        
    def _abort_long_running_tasks(self):
        """中止长时运行任务"""
        # 这里应该实现具体的任务中止逻辑
        logger.info("中止长时运行任务")
        
    def _reload_models(self):
        """重新加载模型"""
        # 这里应该实现模型重新加载逻辑
        logger.info("尝试重新加载模型")
        
    def _log_performance_metric(self, metric_name: str, value: Any):
        """
        记录性能指标
        Args:
            metric_name: 指标名称
            value: 指标值
        """
        logger.info(f"性能指标 - {metric_name}: {value}")
        
    def create_custom_exception(self, name: str, level: ExceptionLevel, message: str) -> Exception:
        """
        创建自定义异常
        Args:
            name: 异常名称
            level: 异常级别
            message: 异常消息
        Returns:
            异常对象
        """
        # 注册自定义异常
        self.custom_classifications[name] = level
        
        # 创建异常
        class CustomException(Exception):
            pass
            
        CustomException.__name__ = name
        return CustomException(message)
        
    def get_exception_statistics(self) -> Dict[str, Any]:
        """
        获取异常统计信息
        Returns:
            统计信息字典
        """
        stats = {
            "total": len(self.exception_history),
            "by_level": {},
            "by_type": {},
            "handled": 0,
            "retried": 0
        }
        
        for exc_info in self.exception_history:
            # 按级别统计
            level = exc_info.level.value
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
            
            # 按类型统计
            exc_type = exc_info.exception_type
            stats["by_type"][exc_type] = stats["by_type"].get(exc_type, 0) + 1
            
            # 统计处理情况
            if exc_info.handled:
                stats["handled"] += 1
            if exc_info.retry_count > 0:
                stats["retried"] += 1
                
        return stats
        
    def clear_history(self):
        """清空异常历史"""
        self.exception_history.clear()
        logger.info("已清空异常历史")
        
    def export_exception_log(self, filepath: str):
        """
        导出异常日志
        Args:
            filepath: 输出文件路径
        """
        import json
        
        log_data = []
        for exc_info in self.exception_history:
            log_data.append({
                "level": exc_info.level.value,
                "type": exc_info.exception_type,
                "message": exc_info.message,
                "timestamp": exc_info.timestamp,
                "retry_count": exc_info.retry_count,
                "handled": exc_info.handled,
                "context": exc_info.context
            })
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "statistics": self.get_exception_statistics(),
                "exceptions": log_data
            }, f, ensure_ascii=False, indent=2)
            
        logger.info(f"导出异常日志到 {filepath}")