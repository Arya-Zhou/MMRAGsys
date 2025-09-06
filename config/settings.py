"""
系统配置文件
"""

import os
from typing import Dict, Any

# 基础配置
BASE_CONFIG = {
    "system": {
        "name": "RAG Improvement System",
        "version": "3.0",
        "debug": False,
        "log_level": "INFO"
    },
    
    # 混合检索配置
    "hybrid_retrieval": {
        "dense_weight": 0.7,
        "sparse_weight": 0.3,
        "rrf_k": 60,
        "embedding_dim": 768,
        "default_top_k": 10
    },
    
    # BM25配置
    "bm25": {
        "k1": 1.5,
        "b": 0.75
    },
    
    # 重排序配置
    "reranker": {
        "model_name": "BAAI/bge-reranker-v2-m3",
        "batch_size": 8,
        "use_fp16": True,
        "candidate_limits": {
            "text": 25,
            "image": 10,
            "table": 25
        },
        "performance": {
            "cpu_threshold": 60,
            "latency_threshold": 5,
            "min_switch_interval": 300
        }
    },
    
    # 缓存配置
    "cache": {
        "l1_capacity": 1000,
        "l2_capacity": 5000,
        "l2_similarity_threshold": 0.95,
        "cot_capacity": 5000,
        "cot_similarity_threshold": 0.9,
        "default_ttl": 86400
    },
    
    # CoT推理配置
    "cot": {
        "min_steps": 3,
        "max_steps": 8,
        "quality_threshold": 0.6,
        "max_retries": 3,
        "timeout_config": {
            3: 10,
            5: 20,
            8: 35
        },
        "temperature_schedule": [0.7, 0.5, 0.3, 0.9]
    },
    
    # 复杂度检测配置
    "complexity": {
        "complexity_weights": {
            # 强复杂度指示词
            "分析": 0.9,
            "对比": 0.85,
            "评估": 0.9,
            "推理": 0.95,
            
            # 中等复杂度指示词
            "解释": 0.6,
            "描述": 0.5,
            "列出": 0.4,
            
            # 简单指示词
            "是什么": 0.2,
            "定义": 0.1,
            "查找": 0.1
        }
    },
    
    # 质量监控配置
    "quality": {
        "quality_threshold": 0.6,
        "quality_weights": {
            "completeness": 0.3,
            "consistency": 0.4,
            "relevance": 0.3
        },
        "type_benchmarks": {
            "事实查询类": {"completeness": 0.7, "consistency": 0.8, "relevance": 0.9},
            "分析推理类": {"completeness": 0.8, "consistency": 0.7, "relevance": 0.7},
            "对比评估类": {"completeness": 0.85, "consistency": 0.75, "relevance": 0.7},
            "因果解释类": {"completeness": 0.75, "consistency": 0.8, "relevance": 0.75},
            "流程规划类": {"completeness": 0.9, "consistency": 0.7, "relevance": 0.7},
            "综合论述类": {"completeness": 0.85, "consistency": 0.75, "relevance": 0.8}
        }
    },
    
    # 队列管理配置
    "queue": {
        "text": {
            "resource_allocation": 0.4,
            "max_size": 1000,
            "batch_size": 20,
            "batch_wait_time": 0.1
        },
        "image": {
            "resource_allocation": 0.4,
            "max_size": 1000,
            "batch_size": 10,
            "batch_wait_time": 0.1
        },
        "table": {
            "resource_allocation": 0.2,
            "max_size": 1000,
            "batch_size": 10,
            "batch_wait_time": 0.1
        }
    },
    
    # 监控配置
    "monitoring": {
        "metrics_retention": 10000,
        "alert_thresholds": {
            "answer_relevance": 0.7,
            "p95_latency": 5000,
            "error_rate": 0.1,
            "cache_hit_rate": 0.3,
            "cot_failure_rate": 0.1,
            "cpu_usage": 80,
            "memory_usage": 90
        },
        "metric_priorities": {
            "answer_relevance": 0,
            "retrieval_precision": 0,
            "p95_latency": 1,
            "error_rate": 2,
            "cache_hit_rate": 2
        }
    },
    
    # 降级配置
    "degradation": {
        "level1_thresholds": {
            "latency": 3.0,
            "cpu_usage": 80,
            "consecutive_errors": 3
        },
        "level2_thresholds": {
            "latency": 6.0,
            "cpu_usage": 90,
            "consecutive_errors": 5
        },
        "recovery_thresholds": {
            "latency": 2.0,
            "cpu_usage": 60,
            "error_rate": 0.01
        }
    },
    
    # 评估配置
    "evaluation": {
        "retrieval_thresholds": {
            "recall": 0.75,
            "precision": 0.55,
            "f1_score": 0.65,
            "hit_at_5": 0.85,
            "mrr": 0.70,
            "ndcg_at_10": 0.65,
            "map": 0.60
        },
        "quality_thresholds": {
            "total_score": 0.6,
            "page_match": 0.15,
            "file_match": 0.15,
            "content_similarity": 0.25
        }
    },
    
    # 日志配置
    "logging": {
        "log_dir": "logs",
        "log_file": "rag_system.log",
        "max_file_size": 104857600,  # 100MB
        "backup_count": 7,
        "cot_log_file": "cot_reasoning.log"
    }
}


class Settings:
    """配置管理类"""
    
    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        """
        初始化配置
        Args:
            custom_config: 自定义配置
        """
        self.config = BASE_CONFIG.copy()
        
        # 合并自定义配置
        if custom_config:
            self._merge_config(custom_config)
            
        # 从环境变量加载配置
        self._load_env_config()
        
    def _merge_config(self, custom_config: Dict[str, Any]):
        """
        合并自定义配置
        Args:
            custom_config: 自定义配置
        """
        def deep_merge(base: Dict, custom: Dict):
            for key, value in custom.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
                    
        deep_merge(self.config, custom_config)
        
    def _load_env_config(self):
        """从环境变量加载配置"""
        # 示例：RAG_DEBUG=true
        if os.getenv("RAG_DEBUG", "").lower() == "true":
            self.config["system"]["debug"] = True
            
        # 示例：RAG_LOG_LEVEL=DEBUG
        if os.getenv("RAG_LOG_LEVEL"):
            self.config["system"]["log_level"] = os.getenv("RAG_LOG_LEVEL")
            
        # 示例：RAG_DENSE_WEIGHT=0.8
        if os.getenv("RAG_DENSE_WEIGHT"):
            try:
                self.config["hybrid_retrieval"]["dense_weight"] = float(os.getenv("RAG_DENSE_WEIGHT"))
            except ValueError:
                pass
                
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值
        Args:
            key_path: 配置路径，如 "hybrid_retrieval.dense_weight"
            default: 默认值
        Returns:
            配置值
        """
        keys = key_path.split(".")
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
        
    def set(self, key_path: str, value: Any):
        """
        设置配置值
        Args:
            key_path: 配置路径
            value: 配置值
        """
        keys = key_path.split(".")
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
            
        config[keys[-1]] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        Returns:
            配置字典
        """
        return self.config.copy()
        
    def save(self, filepath: str):
        """
        保存配置到文件
        Args:
            filepath: 文件路径
        """
        import json
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
            
    @classmethod
    def load(cls, filepath: str) -> 'Settings':
        """
        从文件加载配置
        Args:
            filepath: 文件路径
        Returns:
            配置实例
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        return cls(custom_config=config)


# 全局配置实例
settings = Settings()