#!/usr/bin/env python3
"""
RAG系统启动脚本
使用示例: python run_rag.py
"""

import os
import sys

# 设置环境变量（可以在这里配置，也可以在终端中export）
os.environ.setdefault("RAG_LOG_LEVEL", "INFO")
os.environ.setdefault("RAG_DEBUG", "false")
os.environ.setdefault("RAG_DENSE_WEIGHT", "0.7")

# 添加项目路径到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入主程序
from main import RAGSystem
import json


def demo_basic_usage():
    """基础使用演示"""
    print("=== RAG系统基础使用演示 ===")
    
    # 创建系统
    rag = RAGSystem()
    
    # 添加示例文档
    documents = [
        {
            "id": "doc1", 
            "content": "北京是中华人民共和国的首都，位于华北平原北部。",
            "source_file": "china_geography.pdf",
            "page_number": 1
        },
        {
            "id": "doc2",
            "content": "上海是中国最大的城市，是重要的经济和金融中心。",
            "source_file": "china_geography.pdf", 
            "page_number": 2
        },
        {
            "id": "doc3",
            "content": "人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。",
            "source_file": "ai_basics.pdf",
            "page_number": 1
        },
        {
            "id": "doc4",
            "content": "机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习模式。",
            "source_file": "ai_basics.pdf",
            "page_number": 3
        }
    ]
    
    print(f"正在添加 {len(documents)} 个文档...")
    rag.add_documents(documents)
    print("文档添加完成！")
    
    # 测试查询
    test_queries = [
        "中国的首都是哪里？",
        "什么是人工智能？", 
        "上海在中国的作用是什么？",
        "机器学习与人工智能的关系是什么？"
    ]
    
    print("\n=== 开始测试查询 ===")
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 查询 {i}: {query}")
        
        try:
            result = rag.query(query)
            
            print("✅ 查询结果:")
            print(f"  答案: {result['answer'][:100]}...")
            print(f"  来源: {result['source_file']}")
            print(f"  页码: {result['page_number']}")
            
        except Exception as e:
            print(f"❌ 查询失败: {e}")
    
    # 显示系统统计
    print("\n=== 系统统计信息 ===")
    stats = rag.get_statistics()
    
    # 显示关键统计信息
    if 'queue' in stats:
        print(f"队列处理统计: {stats['queue']['processing_stats']}")
    
    if 'cache' in stats:
        print(f"缓存命中率: L1={stats['cache'].get('l1_cache', {}).get('hit_rate', 0):.2%}")
    
    # 关闭系统
    print("\n正在关闭系统...")
    rag.shutdown()
    print("系统已关闭")


def interactive_mode():
    """交互式模式"""
    print("=== RAG系统交互式模式 ===")
    print("输入 'quit' 或 'exit' 退出")
    
    rag = RAGSystem()
    
    # 预加载一些示例文档
    default_docs = [
        {"id": "demo1", "content": "Python是一种高级编程语言，广泛用于数据科学和人工智能。", "source_file": "programming.pdf", "page_number": 1},
        {"id": "demo2", "content": "深度学习是机器学习的一个分支，使用多层神经网络。", "source_file": "ai_advanced.pdf", "page_number": 5},
        {"id": "demo3", "content": "自然语言处理（NLP）是AI的一个重要应用领域。", "source_file": "ai_advanced.pdf", "page_number": 12}
    ]
    
    rag.add_documents(default_docs)
    print("已预加载示例文档")
    
    while True:
        try:
            query = input("\n🤖 请输入您的问题: ").strip()
            
            if query.lower() in ['quit', 'exit', '退出']:
                break
                
            if not query:
                print("请输入有效问题")
                continue
                
            print("🔍 正在搜索...")
            result = rag.query(query)
            
            print(f"\n📋 答案: {result['answer']}")
            print(f"📚 来源: {result['source_file']}")
            print(f"📄 页码: {result['page_number']}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 处理出错: {e}")
    
    rag.shutdown()
    print("\n再见！")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG系统启动器')
    parser.add_argument('--mode', choices=['demo', 'interactive'], default='demo',
                       help='运行模式: demo(演示模式) 或 interactive(交互模式)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志级别
    os.environ["RAG_LOG_LEVEL"] = args.log_level
    
    print("🚀 启动RAG系统...")
    print(f"📊 日志级别: {args.log_level}")
    print(f"🎯 运行模式: {args.mode}")
    
    if args.mode == 'demo':
        demo_basic_usage()
    elif args.mode == 'interactive':
        interactive_mode()


if __name__ == "__main__":
    main()