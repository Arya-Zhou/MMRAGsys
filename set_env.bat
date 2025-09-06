@echo off
REM Windows环境配置脚本 - 用于设置RAG系统环境变量

REM 基础配置
set RAG_LOG_LEVEL=INFO
set RAG_DEBUG=false
set RAG_DENSE_WEIGHT=0.7

REM 可选高级配置
REM set RAG_QUALITY_THRESHOLD=0.6
REM set RAG_MODEL_NAME=BAAI/bge-reranker-v2-m3
REM set RAG_BATCH_SIZE=8

echo ✅ RAG系统环境变量已设置：
echo   - 日志级别: %RAG_LOG_LEVEL%
echo   - 调试模式: %RAG_DEBUG%
echo   - 稠密检索权重: %RAG_DENSE_WEIGHT%

echo.
echo 现在您可以运行: python run_rag.py