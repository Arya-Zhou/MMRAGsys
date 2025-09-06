import numpy as np
from sentence_transformers import SentenceTransformer
from main import RAGSystem
import json

# 初始化向量模型
encoder = SentenceTransformer('BAAI/bge-base-zh-v1.5')

# 创建RAG系统
rag_system = RAGSystem()

# 准备示例文档
documents = [
    {
        "id": "doc1",
        "content": "北京是中华人民共和国的首都，是中国的政治、文化中心。",
        "source_file": "china.pdf",
        "page_number": 1
    },
    {
        "id": "doc2",
        "content": "上海是中国最大的经济中心和金融中心。",
        "source_file": "china.pdf",
        "page_number": 2
    }
]

# 为文档生成向量
for doc in documents:
    doc["embedding"] = encoder.encode(doc["content"])

# 添加文档
rag_system.add_documents(documents)

# 测试查询
query = "中国的首都是哪里？"
query_embedding = encoder.encode(query)

result = rag_system.query(
    query=query,
    query_embedding=query_embedding,
    data_type="text"
)

print(json.dumps(result, ensure_ascii=False, indent=2))