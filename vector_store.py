"""
向量存储和检索引擎模块

这个模块负责：
1. 将文档块转换为向量嵌入
2. 存储向量到向量数据库
3. 执行相似度检索

【向量检索流程图】
文本块 -> 嵌入模型 -> 向量(embedding) -> 存入向量数据库
                                              ↓
用户查询 -> 嵌入模型 -> 查询向量 -> 相似度搜索 -> 返回相关文档块

【为什么用向量检索？】
1. 语义理解：能找到语义相关而非仅仅关键词匹配的内容
2. 跨语言：不同语言的相似内容可以匹配
3. 容错性：即使用户查询表述不精确也能找到相关内容
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import VectorStore, MetadataFilters, MetadataFilter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


class VectorStoreManager:
    """
    向量存储管理器
    
    负责向量数据库的创建、管理和检索
    
    【向量数据库的作用】
    1. 高效存储：压缩存储高维向量
    2. 快速检索：使用近似最近邻(ANN)算法快速查找相似向量
    3. 持久化：数据可以保存到磁盘，下次启动无需重新处理
    
    【ChromaDB特点】
    - 轻量级，适合本地开发
    - 支持元数据过滤
    - 自动处理向量索引
    """
    
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag_documents",
        embed_model: Optional[OpenAIEmbedding] = None
    ):
        """
        初始化向量存储管理器
        
        参数:
            persist_dir: 向量数据库持久化目录
            collection_name: 集合名称（类似数据库表名）
            embed_model: 嵌入模型，用于将文本转换为向量
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # 初始化嵌入模型
        # 【核心组件】嵌入模型将文本转换为高维向量
        # OpenAI的text-embedding-3-small模型生成1536维向量
        self.embed_model = embed_model or OpenAIEmbedding(
            model="text-embedding-3-small"
        )
        
        # 创建持久化目录
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化ChromaDB客户端
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 获取或创建集合
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # 创建LlamaIndex的向量存储包装器
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        
        # 创建存储上下文
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # 向量索引（懒加载）
        self._index: Optional[VectorStoreIndex] = None
    
    @property
    def index(self) -> VectorStoreIndex:
        """
        获取向量索引（懒加载模式）
        
        【VectorStoreIndex的作用】
        1. 管理文档节点的向量化和存储
        2. 提供检索接口
        3. 协调嵌入模型和向量存储
        """
        if self._index is None:
            self._index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model
            )
        return self._index
    
    def add_nodes(self, nodes: List[TextNode]) -> None:
        """
        将节点添加到向量存储
        
        【处理流程】
        1. 遍历每个节点
        2. 调用嵌入模型将文本转换为向量
        3. 将向量和元数据存入ChromaDB
        
        参数:
            nodes: 文本节点列表
        """
        if not nodes:
            return
        
        # 使用索引添加节点，会自动进行向量化
        # 【关键操作】这里是真正将文档存入向量库的地方
        for node in nodes:
            self.index.insert_node(node)
        
        print(f"✓ 已添加 {len(nodes)} 个节点到向量存储")
    
    def get_retriever(
        self,
        similarity_top_k: int = 5
    ) -> VectorIndexRetriever:
        """
        获取检索器
        
        【检索器的作用】
        检索器负责执行向量相似度搜索，返回最相关的文档块
        
        参数:
            similarity_top_k: 返回最相似的K个结果
            
        返回:
            向量检索器实例
        """
        return VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k
        )
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[NodeWithScore]:
        """
        执行检索
        
        【检索过程详解】
        1. 将用户查询转换为向量（使用相同的嵌入模型）
        2. 在向量空间中查找与查询向量最近的K个文档向量
        3. 返回对应的文档块及其相似度分数
        
        【相似度计算】
        使用余弦相似度：cos(θ) = A·B / (|A| * |B|)
        值范围[-1, 1]，越接近1表示越相似
        
        参数:
            query: 用户查询文本
            top_k: 返回结果数量
            
        返回:
            带有相似度分数的节点列表
        """
        retriever = self.get_retriever(similarity_top_k=top_k)
        
        # 【核心检索调用】执行向量相似度搜索
        nodes_with_scores = retriever.retrieve(query)
        
        return nodes_with_scores
    
    def retrieve_with_filters(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 5
    ) -> List[NodeWithScore]:
        """
        带元数据过滤的检索
        
        【元数据过滤的应用场景】
        1. 按文档来源过滤：只搜索特定文档
        2. 按时间过滤：只搜索最近的文档
        3. 按类型过滤：只搜索特定类型的文档
        
        参数:
            query: 查询文本
            filters: 过滤条件，如 {"source": "document.pdf"}
            top_k: 返回结果数量
            
        返回:
            过滤后的检索结果
        """
        # 构建元数据过滤器
        metadata_filters = MetadataFilters(
            filters=[
                MetadataFilter(key=k, value=v)
                for k, v in filters.items()
            ]
        )
        
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
            filters=metadata_filters
        )
        
        return retriever.retrieve(query)
    
    def delete_collection(self) -> None:
        """删除当前集合（清空所有数据）"""
        self.chroma_client.delete_collection(self.collection_name)
        self._index = None
        print(f"✓ 已删除集合: {self.collection_name}")
    
    def get_collection_count(self) -> int:
        """获取集合中的文档数量"""
        return self.collection.count()


class RetrieverEngine:
    """
    检索引擎 - 封装高级检索功能
    
    【检索策略】
    1. 基础向量检索：纯语义相似度
    2. 混合检索：结合关键词和语义
    3. 重排序：对初步结果进行二次排序
    """
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        """
        初始化检索引擎
        
        参数:
            vector_store_manager: 向量存储管理器
        """
        self.vsm = vector_store_manager
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        执行搜索并返回格式化结果
        
        参数:
            query: 查询文本
            top_k: 返回结果数量
            min_score: 最低相似度阈值
            
        返回:
            格式化的搜索结果列表
        """
        # 执行检索
        nodes_with_scores = self.vsm.retrieve(query, top_k=top_k)
        
        # 格式化结果
        results = []
        for node_with_score in nodes_with_scores:
            if node_with_score.score >= min_score:
                results.append({
                    "text": node_with_score.node.text,
                    "score": node_with_score.score,
                    "metadata": node_with_score.node.metadata,
                    "node_id": node_with_score.node.node_id
                })
        
        return results
    
    def search_with_context(
        self,
        query: str,
        top_k: int = 3
    ) -> str:
        """
        搜索并返回组装好的上下文文本
        
        【用途】
        直接为LLM提供检索到的上下文
        
        参数:
            query: 查询文本
            top_k: 返回结果数量
            
        返回:
            拼接后的上下文文本
        """
        results = self.search(query, top_k=top_k)
        
        if not results:
            return "未找到相关内容"
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[文档片段 {i}] (相关度: {result['score']:.3f})\n"
                f"来源: {result['metadata'].get('file_name', '未知')}\n"
                f"{result['text']}"
            )
        
        return "\n\n---\n\n".join(context_parts)


def demo_vector_store():
    """
    演示向量存储和检索功能
    """
    # 创建一些示例节点
    sample_nodes = [
        TextNode(
            text="Python是一种高级编程语言，以其简洁的语法和强大的功能著称。",
            metadata={"topic": "python", "source": "demo"}
        ),
        TextNode(
            text="机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。",
            metadata={"topic": "ml", "source": "demo"}
        ),
        TextNode(
            text="深度学习使用多层神经网络来处理复杂的模式识别任务。",
            metadata={"topic": "dl", "source": "demo"}
        )
    ]
    
    # 初始化向量存储
    vsm = VectorStoreManager(
        persist_dir="./demo_chroma",
        collection_name="demo_collection"
    )
    
    # 添加节点
    vsm.add_nodes(sample_nodes)
    
    # 创建检索引擎
    engine = RetrieverEngine(vsm)
    
    # 执行检索
    query = "什么是深度学习？"
    print(f"\n查询: {query}\n")
    
    results = engine.search(query, top_k=2)
    
    print("检索结果:")
    for i, result in enumerate(results, 1):
        print(f"\n--- 结果 {i} ---")
        print(f"相关度: {result['score']:.3f}")
        print(f"内容: {result['text']}")
    
    # 清理
    vsm.delete_collection()


if __name__ == "__main__":
    demo_vector_store()
