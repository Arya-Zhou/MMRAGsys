#!/usr/bin/env python3
"""
批量处理132个PDF解析文件
用于大规模数据导入和管理
"""

import os
import sys
import time
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import RAGSystem
from data.batch_manager import PDFIndexManager, BatchLoader, setup_batch_loading


def scan_and_register(base_dir: str):
    """
    扫描并注册所有PDF文件
    Args:
        base_dir: 数据基础目录
    """
    print(f"=== 扫描PDF解析文件: {base_dir} ===")
    
    # 创建索引管理器
    index_manager = PDFIndexManager()
    
    # 扫描目录
    pdf_files = index_manager.scan_directory(base_dir)
    
    if not pdf_files:
        print("❌ 未找到PDF解析文件")
        return
        
    print(f"✅ 找到 {len(pdf_files)} 个PDF文件")
    
    # 显示前5个文件
    print("\n📋 示例文件:")
    for i, pdf in enumerate(pdf_files[:5]):
        print(f"  {i+1}. {pdf['pdf_name']}")
        print(f"     JSON: {pdf['json_path']}")
        print(f"     大小: {pdf['file_size'] / 1024:.1f} KB")
        
    if len(pdf_files) > 5:
        print(f"  ... 还有 {len(pdf_files) - 5} 个文件")
        
    # 询问是否注册
    response = input("\n是否注册所有文件? (y/n): ")
    if response.lower() == 'y':
        pdf_ids = index_manager.batch_register(pdf_files)
        print(f"✅ 成功注册 {len(pdf_ids)} 个PDF文件")
        
        # 导出报告
        index_manager.export_index_report()
        print("📊 索引报告已生成: indexes/index_report.md")
    else:
        print("取消注册")


def batch_load(base_dir: str, batch_size: int = 10, resume: bool = False):
    """
    批量加载PDF文档到RAG系统
    Args:
        base_dir: 数据基础目录
        batch_size: 批次大小
        resume: 是否恢复加载
    """
    print(f"=== 批量加载PDF文档 ===")
    print(f"📁 数据目录: {base_dir}")
    print(f"📦 批次大小: {batch_size}")
    print(f"♻️ 恢复模式: {resume}")
    
    # 创建RAG系统
    print("\n🚀 初始化RAG系统...")
    rag_system = RAGSystem()
    
    # 设置批量加载
    batch_loader = setup_batch_loading(base_dir, rag_system)
    
    # 显示当前状态
    stats = batch_loader.index_manager.get_statistics()
    print(f"\n📊 当前状态:")
    print(f"  总PDF数: {stats['total_pdfs']}")
    print(f"  总文档数: {stats['total_documents']}")
    print(f"  状态分布: {stats['status_distribution']}")
    
    # 开始加载
    start_time = time.time()
    
    if resume:
        print("\n♻️ 恢复加载...")
        batch_loader.resume_loading()
    else:
        print("\n🔄 开始批量加载...")
        batch_loader.load_all_pdfs(batch_size=batch_size)
        
    elapsed_time = time.time() - start_time
    
    # 显示最终统计
    final_stats = batch_loader.index_manager.get_statistics()
    print(f"\n✅ 加载完成!")
    print(f"  耗时: {elapsed_time:.1f} 秒")
    print(f"  总文档数: {final_stats['total_documents']}")
    print(f"  状态分布: {final_stats['status_distribution']}")
    
    # 关闭系统
    rag_system.shutdown()


def check_status():
    """检查索引状态"""
    print("=== 检查索引状态 ===")
    
    index_manager = PDFIndexManager()
    stats = index_manager.get_statistics()
    
    print(f"\n📊 索引统计:")
    print(f"  总PDF数: {stats['total_pdfs']}")
    print(f"  总文档数: {stats['total_documents']}")
    print(f"  最后更新: {stats['last_update']}")
    
    print(f"\n📈 状态分布:")
    for status, count in stats['status_distribution'].items():
        percentage = count / max(stats['total_pdfs'], 1) * 100
        print(f"  {status}: {count} ({percentage:.1f}%)")
        
    # 显示未加载的PDF
    unloaded = index_manager.get_unloaded_pdfs()
    if unloaded:
        print(f"\n⚠️ 未加载的PDF: {len(unloaded)} 个")
        for pdf_id in unloaded[:5]:
            pdf_info = index_manager.get_pdf_info(pdf_id)
            if pdf_info:
                print(f"  - {pdf_info['name']} [{pdf_info['status']}]")
        if len(unloaded) > 5:
            print(f"  ... 还有 {len(unloaded) - 5} 个")
    else:
        print("\n✅ 所有PDF已加载")


def test_query(query: str):
    """
    测试查询已加载的数据
    Args:
        query: 查询文本
    """
    print(f"=== 测试查询: {query} ===")
    
    # 检查是否有已加载的数据
    index_manager = PDFIndexManager()
    stats = index_manager.get_statistics()
    
    if stats['total_documents'] == 0:
        print("❌ 没有已加载的文档，请先运行批量加载")
        return
        
    print(f"📊 当前已加载 {stats['total_documents']} 个文档")
    
    # 创建RAG系统
    print("\n🚀 初始化RAG系统...")
    rag_system = RAGSystem()
    
    # 重新加载已有数据（这里需要实现持久化存储）
    print("⚠️ 注意: 当前版本需要重新加载数据，建议实现持久化存储")
    
    # 执行查询
    print(f"\n🔍 执行查询: {query}")
    try:
        result = rag_system.query(query)
        
        print("\n✅ 查询结果:")
        print(f"答案: {result['answer'][:200]}...")
        print(f"来源: {result['source_file']}")
        print(f"页码: {result['page_number']}")
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        
    # 关闭系统
    rag_system.shutdown()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量处理PDF解析文件')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 扫描命令
    scan_parser = subparsers.add_parser('scan', help='扫描并注册PDF文件')
    scan_parser.add_argument('base_dir', help='数据基础目录')
    
    # 加载命令
    load_parser = subparsers.add_parser('load', help='批量加载PDF文档')
    load_parser.add_argument('base_dir', help='数据基础目录')
    load_parser.add_argument('--batch-size', type=int, default=10, help='批次大小')
    load_parser.add_argument('--resume', action='store_true', help='恢复加载')
    
    # 状态命令
    status_parser = subparsers.add_parser('status', help='检查索引状态')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='测试查询')
    test_parser.add_argument('query', help='查询文本')
    
    args = parser.parse_args()
    
    if args.command == 'scan':
        scan_and_register(args.base_dir)
        
    elif args.command == 'load':
        batch_load(args.base_dir, args.batch_size, args.resume)
        
    elif args.command == 'status':
        check_status()
        
    elif args.command == 'test':
        test_query(args.query)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()