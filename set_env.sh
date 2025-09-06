#!/bin/bash
# 环境配置脚本 - 用于设置RAG系统环境变量

# 基础配置
export RAG_LOG_LEVEL=INFO
export RAG_DEBUG=false  
export RAG_DENSE_WEIGHT=0.7

# 可选高级配置
# export RAG_QUALITY_THRESHOLD=0.6
# export RAG_MODEL_NAME=BAAI/bge-reranker-v2-m3
# export RAG_BATCH_SIZE=8

echo "✅ RAG系统环境变量已设置："
echo "  - 日志级别: $RAG_LOG_LEVEL"
echo "  - 调试模式: $RAG_DEBUG" 
echo "  - 稠密检索权重: $RAG_DENSE_WEIGHT"

# Windows用户使用 set_env.bat