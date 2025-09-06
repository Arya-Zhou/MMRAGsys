import sys
import os
sys.path.append('gaijin_short_v3')

import numpy as np
from main import RAGSystem
import json

# 使用模拟向量
def mock_encode(text):
    """模拟向量编码"""
    return np.random.randn(768)

# 创建系统
rag = RAGSystem()

# 准备测试数据
test_docs = [
    {
        "id": "test1",
        "content": "这是测试文档内容",
        "source_file": "test.pdf",
        "page_number": 1,
        "embedding": mock_encode("测试")
    }
]

rag.add_documents(test_docs)

# 测试查询
result = rag.query(
    query="测试查询",
    query_embedding=mock_encode("测试查询")
)

print("系统运行成功！")
print(json.dumps(result, ensure_ascii=False, indent=2))