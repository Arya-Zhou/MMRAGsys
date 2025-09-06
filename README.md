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

### 1. MinIO数据处理（新增）
- 自动解析MinIO输出的JSON文件
- 支持text、table、image三种数据类型
- 自动处理图像路径和元数据
- 保持页码和文件来源追踪

### 2. 混合检索
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

# 方式1: 手动添加文档
documents = [
    {
        "id": "doc1",
        "content": "文档内容",
        "source_file": "example.pdf",
        "page_number": 1
    }
]
rag.add_documents(documents)

# 方式2: 从MinIO数据加载（新增）
count = rag.add_documents_from_mineru(
    json_file_path="document_content_list.json",
    base_path="data_directory"
)

# 查询
result = rag.query("你的问题")
print(result)
```

### MinIO数据使用（新增）
```python
# 单个文件加载
rag.add_documents_from_mineru(
    json_file_path="report_content_list.json",
    images_dir="images",  # 可选，默认为同目录下的images文件夹
    base_path="/path/to/data"
)

# 批量加载多个文件
json_files = ["file1_content_list.json", "file2_content_list.json"]
total_count = rag.batch_load_mineru_documents(
    json_files=json_files,
    base_path="/path/to/data"
)

# 使用MinIO数据工具
# 分析数据结构
python mineru_tool.py analyze document_content_list.json --base-path ../data_example

# 批量处理
python mineru_tool.py batch /path/to/mineru/data --output analysis.json

# 测试RAG系统
python mineru_tool.py test document_content_list.json --base-path ../data_example
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

## 运行系统

```bash
# 基础演示模式
python run_rag.py --mode demo

# 交互式模式
python run_rag.py --mode interactive

# MinIO数据演示（新增）
python run_rag.py --mode mineru

# 直接运行主程序（包含MinIO演示）
python main.py

# 使用MinIO工具分析数据
python mineru_tool.py analyze ../data_example/艾力斯-公司深度报告商业化成绩显著产品矩阵持续拓宽-25070718页_content_list.json --base-path ../data_example
```

## MinIO数据格式

系统支持的MinIO JSON数据格式：
```json
[
    {
        "type": "text",
        "text": "文本内容",
        "text_level": 1,
        "page_idx": 0
    },
    {
        "type": "table", 
        "table_body": "<table>...</table>",
        "table_caption": ["表格标题"],
        "table_footnote": ["表格脚注"],
        "img_path": "images/table_image.jpg",
        "page_idx": 0
    },
    {
        "type": "image",
        "image_caption": ["图像标题"],
        "image_footnote": ["图像脚注"],
        "img_path": "images/chart.jpg",
        "page_idx": 1
    }
]
```

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