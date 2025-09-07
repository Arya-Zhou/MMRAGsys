"""
RAG系统监控仪表板
提供实时性能监控和系统状态可视化
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardServer:
    """监控仪表板服务器"""
    
    def __init__(self, rag_system, port: int = 8080):
        """
        初始化仪表板服务器
        Args:
            rag_system: RAG系统实例
            port: 服务端口
        """
        self.rag_system = rag_system
        self.port = port
        self.metrics_history = []
        self.query_history = []
        self.max_history_size = 1000
        
    def collect_metrics(self) -> Dict[str, Any]:
        """收集当前指标"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": self._get_system_metrics(),
            "performance": self._get_performance_metrics(),
            "cache": self._get_cache_metrics(),
            "queue": self._get_queue_metrics()
        }
        
        # 添加到历史记录
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
            
        return metrics
        
    def _get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        try:
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        except:
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "disk_usage": 0
            }
            
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        # 从RAG系统获取性能统计
        stats = self.rag_system.get_statistics()
        
        return {
            "avg_query_time": self._calculate_avg_query_time(),
            "queries_per_minute": self._calculate_qpm(),
            "success_rate": self._calculate_success_rate()
        }
        
    def _get_cache_metrics(self) -> Dict[str, Any]:
        """获取缓存指标"""
        stats = self.rag_system.get_statistics()
        cache_stats = stats.get('cache', {})
        
        metrics = {}
        for cache_type, cache_data in cache_stats.items():
            if isinstance(cache_data, dict):
                metrics[cache_type] = {
                    "hit_rate": cache_data.get('hit_rate', 0),
                    "size": cache_data.get('size', 0),
                    "capacity": cache_data.get('capacity', 0)
                }
                
        return metrics
        
    def _get_queue_metrics(self) -> Dict[str, Any]:
        """获取队列指标"""
        stats = self.rag_system.get_statistics()
        queue_stats = stats.get('queue', {})
        
        return {
            "queue_sizes": queue_stats.get('queue_sizes', {}),
            "processing_stats": queue_stats.get('processing_stats', {})
        }
        
    def _calculate_avg_query_time(self) -> float:
        """计算平均查询时间"""
        if not self.query_history:
            return 0
            
        recent_queries = self.query_history[-100:]
        times = [q['processing_time'] for q in recent_queries if 'processing_time' in q]
        
        return sum(times) / len(times) if times else 0
        
    def _calculate_qpm(self) -> float:
        """计算每分钟查询数"""
        if not self.query_history:
            return 0
            
        one_minute_ago = time.time() - 60
        recent_queries = [q for q in self.query_history 
                         if q.get('timestamp', 0) > one_minute_ago]
        
        return len(recent_queries)
        
    def _calculate_success_rate(self) -> float:
        """计算成功率"""
        if not self.query_history:
            return 1.0
            
        recent = self.query_history[-100:]
        success = sum(1 for q in recent if q.get('success', False))
        
        return success / len(recent) if recent else 1.0
        
    def record_query(self, query: str, result: Any, processing_time: float, success: bool):
        """
        记录查询
        Args:
            query: 查询文本
            result: 查询结果
            processing_time: 处理时间
            success: 是否成功
        """
        record = {
            "timestamp": time.time(),
            "query": query[:100],  # 限制长度
            "processing_time": processing_time,
            "success": success
        }
        
        if success and result:
            record["source_file"] = result.get('source_file', '')
            record["page_numbers"] = result.get('page_number', [])
            
        self.query_history.append(record)
        
        if len(self.query_history) > self.max_history_size:
            self.query_history = self.query_history[-self.max_history_size:]
            
    def generate_html_dashboard(self) -> str:
        """生成HTML仪表板"""
        metrics = self.collect_metrics()
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG系统监控仪表板</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric-title {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }
        .metric-unit {
            font-size: 16px;
            color: #999;
            margin-left: 5px;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }
        .status-good { background: #4CAF50; }
        .status-warning { background: #FFC107; }
        .status-error { background: #F44336; }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s ease;
        }
        .table-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        th {
            background: #f5f5f5;
            font-weight: 600;
        }
        .refresh-info {
            text-align: right;
            color: #666;
            font-size: 12px;
            margin-top: 20px;
        }
    </style>
    <script>
        function refreshPage() {
            location.reload();
        }
        // 自动刷新（每5秒）
        setTimeout(refreshPage, 5000);
    </script>
</head>
<body>
    <div class="header">
        <h1>🚀 RAG系统监控仪表板</h1>
        <p>实时监控系统性能和运行状态</p>
        <p>更新时间: """ + metrics['timestamp'] + """</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-title">CPU使用率</div>
            <div class="metric-value">""" + f"{metrics['system']['cpu_percent']:.1f}" + """<span class="metric-unit">%</span></div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: """ + str(metrics['system']['cpu_percent']) + """%"></div>
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">内存使用率</div>
            <div class="metric-value">""" + f"{metrics['system']['memory_percent']:.1f}" + """<span class="metric-unit">%</span></div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: """ + str(metrics['system']['memory_percent']) + """%"></div>
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">平均查询时间</div>
            <div class="metric-value">""" + f"{metrics['performance']['avg_query_time']:.2f}" + """<span class="metric-unit">秒</span></div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">查询速率</div>
            <div class="metric-value">""" + f"{metrics['performance']['queries_per_minute']:.0f}" + """<span class="metric-unit">QPM</span></div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">成功率</div>
            <div class="metric-value">""" + f"{metrics['performance']['success_rate']*100:.1f}" + """<span class="metric-unit">%</span></div>
        </div>
    </div>
    
    <div class="chart-container">
        <h3>缓存状态</h3>
        <table>
            <tr>
                <th>缓存类型</th>
                <th>命中率</th>
                <th>使用量</th>
                <th>容量</th>
            </tr>
"""
        
        for cache_type, cache_data in metrics['cache'].items():
            html += f"""
            <tr>
                <td>{cache_type}</td>
                <td>{cache_data.get('hit_rate', 0)*100:.1f}%</td>
                <td>{cache_data.get('size', 0)}</td>
                <td>{cache_data.get('capacity', 0)}</td>
            </tr>
            """
            
        html += """
        </table>
    </div>
    
    <div class="table-container">
        <h3>最近查询</h3>
        <table>
            <tr>
                <th>时间</th>
                <th>查询</th>
                <th>耗时</th>
                <th>状态</th>
            </tr>
"""
        
        # 显示最近10个查询
        for query in self.query_history[-10:][::-1]:
            timestamp = datetime.fromtimestamp(query['timestamp']).strftime('%H:%M:%S')
            status_class = 'status-good' if query['success'] else 'status-error'
            status_text = '成功' if query['success'] else '失败'
            
            html += f"""
            <tr>
                <td>{timestamp}</td>
                <td>{query['query']}</td>
                <td>{query['processing_time']:.2f}s</td>
                <td><span class="status-indicator {status_class}"></span>{status_text}</td>
            </tr>
            """
            
        html += """
        </table>
    </div>
    
    <div class="refresh-info">
        页面每5秒自动刷新 | <a href="javascript:refreshPage()">立即刷新</a>
    </div>
</body>
</html>
"""
        
        return html
        
    def save_dashboard(self, output_file: str = "dashboard.html"):
        """
        保存仪表板到文件
        Args:
            output_file: 输出文件名
        """
        html = self.generate_html_dashboard()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
            
        logger.info(f"仪表板已保存到: {output_file}")
        
    def export_metrics(self, output_file: str = "metrics_export.json"):
        """
        导出指标数据
        Args:
            output_file: 输出文件名
        """
        export_data = {
            "export_time": datetime.now().isoformat(),
            "metrics_history": self.metrics_history[-100:],  # 最近100条
            "query_history": self.query_history[-100:],      # 最近100条
            "current_stats": self.rag_system.get_statistics()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"指标已导出到: {output_file}")