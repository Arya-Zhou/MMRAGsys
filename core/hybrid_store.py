"""
混合向量存储模块
实现稠密检索（向量）+ 稀疏检索（BM25）的混合检索机制
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import jieba
from collections import Counter
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """文档数据结构"""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    source_file: Optional[str] = None
    page_number: Optional[int] = None


class BM25:
    """BM25稀疏检索算法实现"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化BM25算法
        Args:
            k1: 词频饱和参数
            b: 文档长度归一化参数
        """
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = {}
        self.idf_scores = {}
        self.doc_count = 0
        
    def fit(self, documents: List[str]):
        """
        训练BM25模型
        Args:
            documents: 文档列表
        """
        self.documents = documents
        self.doc_count = len(documents)
        
        # 计算文档长度和词频
        tokenized_docs = []
        for doc in documents:
            tokens = list(jieba.cut(doc))
            tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))
            
            # 统计文档频率
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                
        # 计算平均文档长度
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # 计算IDF分数
        for token, freq in self.doc_freqs.items():
            self.idf_scores[token] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
            
    def score(self, query: str, doc_idx: int) -> float:
        """
        计算查询与文档的BM25分数
        Args:
            query: 查询文本
            doc_idx: 文档索引
        Returns:
            BM25分数
        """
        query_tokens = list(jieba.cut(query))
        doc_tokens = list(jieba.cut(self.documents[doc_idx]))
        doc_length = self.doc_lengths[doc_idx]
        
        # 计算词频
        doc_token_counts = Counter(doc_tokens)
        
        score = 0.0
        for token in query_tokens:
            if token not in self.idf_scores:
                continue
                
            tf = doc_token_counts.get(token, 0)
            idf = self.idf_scores[token]
            
            # BM25公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * numerator / denominator
            
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        搜索最相关的文档
        Args:
            query: 查询文本
            top_k: 返回前k个结果
        Returns:
            [(文档索引, 分数)]列表
        """
        scores = []
        for idx in range(self.doc_count):
            score = self.score(query, idx)
            scores.append((idx, score))
            
        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridVectorStore:
    """混合向量存储，支持稠密和稀疏检索"""
    
    def __init__(self, 
                 dense_weight: float = 0.7,
                 sparse_weight: float = 0.3,
                 embedding_dim: int = 768):
        """
        初始化混合向量存储
        Args:
            dense_weight: 稠密检索权重
            sparse_weight: 稀疏检索权重
            embedding_dim: 向量维度
        """
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.embedding_dim = embedding_dim
        
        # 文档存储
        self.documents: List[Document] = []
        self.embeddings: List[np.ndarray] = []
        self.contents: List[str] = []
        
        # BM25检索器
        self.bm25 = BM25()
        
        # RRF融合参数
        self.rrf_k = 60
        
        logger.info(f"初始化混合向量存储: dense_weight={dense_weight}, sparse_weight={sparse_weight}")
        
    def add_documents(self, documents: List[Document]):
        """
        添加文档到存储
        Args:
            documents: 文档列表
        """
        for doc in documents:
            self.documents.append(doc)
            if doc.embedding is not None:
                self.embeddings.append(doc.embedding)
            self.contents.append(doc.content)
            
        # 更新BM25索引
        self.bm25.fit(self.contents)
        logger.info(f"添加了 {len(documents)} 个文档到存储")
        
    def dense_search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        稠密向量检索
        Args:
            query_embedding: 查询向量
            top_k: 返回前k个结果
        Returns:
            [(文档索引, 相似度分数)]列表
        """
        if not self.embeddings:
            return []
            
        # 计算余弦相似度
        embeddings_array = np.array(self.embeddings)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        
        similarities = np.dot(embeddings_norm, query_norm)
        
        # 获取top_k个最相似的文档
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(idx, similarities[idx]) for idx in top_indices]
        
        return results
    
    def sparse_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        稀疏检索（BM25）
        Args:
            query: 查询文本
            top_k: 返回前k个结果
        Returns:
            [(文档索引, BM25分数)]列表
        """
        return self.bm25.search(query, top_k)
    
    def rrf_fusion(self, 
                   dense_results: List[Tuple[int, float]], 
                   sparse_results: List[Tuple[int, float]],
                   top_k: int = 10) -> List[Tuple[int, float]]:
        """
        使用倒数排名融合(RRF)合并检索结果
        Args:
            dense_results: 稠密检索结果
            sparse_results: 稀疏检索结果
            top_k: 返回前k个结果
        Returns:
            融合后的结果列表
        """
        # 计算RRF分数
        rrf_scores = {}
        
        # 处理稠密检索结果
        for rank, (doc_idx, score) in enumerate(dense_results):
            rrf_score = self.dense_weight / (self.rrf_k + rank + 1)
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + rrf_score
            
        # 处理稀疏检索结果
        for rank, (doc_idx, score) in enumerate(sparse_results):
            rrf_score = self.sparse_weight / (self.rrf_k + rank + 1)
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + rrf_score
            
        # 按RRF分数排序
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results[:top_k]
    
    def search(self, 
              query: str,
              query_embedding: Optional[np.ndarray] = None,
              top_k: int = 10,
              use_hybrid: bool = True,
              **kwargs) -> List[Document]:
        """
        混合搜索接口，兼容SimpleVectorStore接口
        Args:
            query: 查询文本
            query_embedding: 查询向量（可选）
            top_k: 返回前k个结果
            use_hybrid: 是否使用混合检索
            **kwargs: 其他参数
        Returns:
            相关文档列表
        """
        if not self.documents:
            logger.warning("文档存储为空")
            return []
            
        results = []
        
        if use_hybrid and query_embedding is not None:
            # 执行混合检索
            dense_results = self.dense_search(query_embedding, top_k=top_k*2)
            sparse_results = self.sparse_search(query, top_k=top_k*2)
            
            # RRF融合
            fused_results = self.rrf_fusion(dense_results, sparse_results, top_k=top_k)
            
            # 获取文档
            for doc_idx, score in fused_results:
                if doc_idx < len(self.documents):
                    doc = self.documents[doc_idx]
                    # 添加检索分数到元数据
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['retrieval_score'] = score
                    results.append(doc)
                    
        elif query_embedding is not None:
            # 仅使用稠密检索
            dense_results = self.dense_search(query_embedding, top_k=top_k)
            for doc_idx, score in dense_results:
                if doc_idx < len(self.documents):
                    doc = self.documents[doc_idx]
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['retrieval_score'] = score
                    results.append(doc)
                    
        else:
            # 仅使用稀疏检索
            sparse_results = self.sparse_search(query, top_k=top_k)
            for doc_idx, score in sparse_results:
                if doc_idx < len(self.documents):
                    doc = self.documents[doc_idx]
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['retrieval_score'] = score
                    results.append(doc)
                    
        logger.info(f"检索到 {len(results)} 个相关文档")
        return results
    
    def set_weights(self, dense_weight: float, sparse_weight: float):
        """
        动态调整检索权重
        Args:
            dense_weight: 稠密检索权重
            sparse_weight: 稀疏检索权重
        """
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        logger.info(f"更新检索权重: dense={dense_weight}, sparse={sparse_weight}")
        
    def clear(self):
        """清空存储"""
        self.documents.clear()
        self.embeddings.clear()
        self.contents.clear()
        self.bm25 = BM25()
        logger.info("已清空混合向量存储")