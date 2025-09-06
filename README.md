# RAG改进系统 v3.0

这是一个完整的RAG（Retrieval-Augmented Generation）系统实现，包含混合检索、重排序、CoT推理链等高级功能。

## 系统架构

```
gaijin_short_v3/
├── core/                    # 核心组件
│   ├── hybrid_store.py     # 混合向量存储
│   ├── document_tracker.py # 文档追踪系统
│   ├── reranker.py         # 重排序模块
│   └── cache.py            # 缓存系统
├── cot/                     # CoT推理组件
│   ├── reasoner.py         # CoT推理引擎
│   ├── complexity_detector.py # 复杂度检测器
│   ├── template_manager.py # 模板管理器
│   ├── quality_monitor.py  # 质量监控器
│   └── exception_handler.py # 异常处理器
├── queue/                   # 队列管理
│   └── manager.py          # 队列管理器
├── monitor/                 # 监控组件
│   ├── metrics.py          # 指标采集器
│   └── degradation.py      # 降级机制
├── config/                  # 配置
│   └── settings.py         # 系统配置
├── main.py                  # 主入口
└── requirements.txt         # 依赖包
```

## 主要特性

### 1. 混合检索
- 稠密检索（向量相似度）权重：0.7
- 稀疏检索（BM25）权重：0.3
- RRF倒数排名融合，K=60

### 2. 重排序
- 模型：BAAI/bge-reranker-v2-m3
- 差异化处理：Text(25个)、Image(10个)、Table(25个)
- 动态量化：FP16/INT8自动切换

### 3. CoT推理链
- 自动复杂度检测
- 动态步骤调整（3-8步）
- 质量评估系统
- 差异化模板（Text/Image/Table）

### 4. 文档追踪
- 全流程追踪标记
- 格式："XX.pdf-p1"
- 统一JSON输出

### 5. 缓存系统
- L1精确缓存（1000条）
- L2语义缓存（5000条）
- CoT专用缓存（5000条）

### 6. 降级机制
- 两级系统降级
- CoT独立降级
- 自动恢复机制

### 7. 监控体系
- P0级：答案相关性、检索精度
- P1级：P95延迟
- P2级：错误率、缓存命中率

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 基础使用
```python
from main import RAGSystem

# 创建系统实例
rag = RAGSystem()

# 添加文档
documents = [
    {
        "id": "doc1",
        "content": "文档内容",
        "source_file": "example.pdf",
        "page_number": 1,
        "embedding": embedding_vector
    }
]
rag.add_documents(documents)

# 查询
result = rag.query("你的问题", query_embedding=embedding_vector)
print(result)
```

### 输出格式
```json
{
    "question": "用户问题",
    "answer": "系统答案",
    "source_file": "XX.pdf,YY.pdf",
    "page_number": [1, 3, 7]
}
```

## 配置说明

主要配置项在 `config/settings.py` 中：

- `hybrid_retrieval`: 混合检索配置
- `reranker`: 重排序配置
- `cache`: 缓存配置
- `cot`: CoT推理配置
- `queue`: 队列管理配置
- `monitoring`: 监控配置
- `degradation`: 降级配置

## 性能指标

目标性能：
- 检索精度提升 >30%
- CoT推理质量 >0.8
- P95延迟 <5秒
- 系统可用性 >99.9%
- 缓存命中率 >30%
- CoT失败率 <10%

## 开发计划

- [x] Phase 1: 基础混合检索+文档追踪
- [x] Phase 2: 重排序集成
- [x] Phase 3: CoT推理链
- [x] Phase 4: 系统集成优化

## 许可证

MIT License