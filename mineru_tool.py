#!/usr/bin/env python3
"""
MinIO数据处理工具
专门用于处理MinIO解析后的JSON和图像数据
"""

import os
import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.mineru_loader import MinIODataLoader
from main import RAGSystem


def analyze_mineru_data(json_file_path: str, base_path: str = ""):
    """
    分析MinIO数据结构
    Args:
        json_file_path: JSON文件路径
        base_path: 基础路径
    """
    print(f"=== 分析MinIO数据: {json_file_path} ===")
    
    loader = MinIODataLoader(base_path)
    documents = loader.load_document(json_file_path)
    
    if not documents:
        print("❌ 未能加载任何数据")
        return
        
    # 获取统计信息
    stats = loader.get_document_statistics(documents)
    
    print(f"📊 数据统计:")
    print(f"  总文档项: {stats['total_documents']}")
    print(f"  按类型分布: {stats['by_type']}")
    print(f"  按文件分布: {stats['by_file']}")
    print(f"  页码范围: {stats['page_range']}")
    
    # 显示示例内容
    print(f"\n📝 示例内容:")
    
    for doc_type in ['text', 'table', 'image']:
        type_docs = [d for d in documents if d.get('data_type') == doc_type]
        if type_docs:
            doc = type_docs[0]
            print(f"\n{doc_type.upper()} 示例:")
            print(f"  ID: {doc['id']}")
            print(f"  内容: {doc['content'][:100]}...")
            print(f"  页码: {doc['page_number']}")
            if doc.get('metadata', {}).get('img_path'):
                print(f"  图像路径: {doc['metadata']['img_path']}")


def batch_process_mineru_files(data_dir: str, output_file: str = "mineru_analysis.json"):
    """
    批量处理MinIO文件
    Args:
        data_dir: 数据目录
        output_file: 输出文件
    """
    print(f"=== 批量处理MinIO文件: {data_dir} ===")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
        
    # 查找所有JSON文件
    json_files = list(data_path.glob("*_content_list.json"))
    
    if not json_files:
        print("❌ 未找到MinIO JSON文件")
        return
        
    print(f"📁 发现 {len(json_files)} 个JSON文件")
    
    loader = MinIODataLoader(str(data_path))
    all_results = []
    
    for json_file in json_files:
        print(f"处理文件: {json_file.name}")
        
        # 加载文档
        documents = loader.load_document(json_file.name)
        stats = loader.get_document_statistics(documents)
        
        result = {
            "file": json_file.name,
            "documents_count": len(documents),
            "statistics": stats
        }
        
        all_results.append(result)
        print(f"  ✅ 处理完成: {len(documents)} 个文档项")
        
    # 保存分析结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    print(f"📄 分析结果已保存到: {output_file}")


def test_rag_with_mineru(json_file_path: str, base_path: str = ""):
    """
    测试RAG系统与MinIO数据
    Args:
        json_file_path: JSON文件路径
        base_path: 基础路径
    """
    print(f"=== 测试RAG系统与MinIO数据 ===")
    
    # 创建RAG系统
    rag = RAGSystem()
    
    # 加载MinIO数据
    print("🔄 加载MinIO数据...")
    count = rag.add_documents_from_mineru(json_file_path, base_path=base_path)
    print(f"✅ 成功加载 {count} 个文档项")
    
    # 预定义测试查询
    test_queries = [
        "文档的主要内容是什么？",
        "有哪些重要的数据信息？",
        "文档中提到了哪些关键人物或机构？",
        "有什么投资建议或风险提示？",
        "文档包含哪些图表或表格？"
    ]
    
    print(f"\n🔍 开始测试查询...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n查询 {i}: {query}")
        
        try:
            result = rag.query(query)
            
            print(f"✅ 答案: {result['answer'][:150]}...")
            print(f"📚 来源: {result['source_file']}")
            print(f"📄 页码: {result['page_number']}")
            
        except Exception as e:
            print(f"❌ 查询失败: {e}")
    
    # 关闭系统
    rag.shutdown()
    print("\n✅ 测试完成")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MinIO数据处理工具')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 分析命令
    analyze_parser = subparsers.add_parser('analyze', help='分析MinIO数据结构')
    analyze_parser.add_argument('json_file', help='JSON文件路径')
    analyze_parser.add_argument('--base-path', default='', help='基础路径')
    
    # 批处理命令
    batch_parser = subparsers.add_parser('batch', help='批量处理MinIO文件')
    batch_parser.add_argument('data_dir', help='数据目录')
    batch_parser.add_argument('--output', default='mineru_analysis.json', help='输出文件')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='测试RAG系统')
    test_parser.add_argument('json_file', help='JSON文件路径')
    test_parser.add_argument('--base-path', default='', help='基础路径')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        analyze_mineru_data(args.json_file, args.base_path)
        
    elif args.command == 'batch':
        batch_process_mineru_files(args.data_dir, args.output)
        
    elif args.command == 'test':
        test_rag_with_mineru(args.json_file, args.base_path)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()