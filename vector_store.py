"""
向量存储和检索引擎模块 - RAG系统的第二阶段：向量化存储

这个模块负责：
1. 将文档块转换为向量嵌入
2. 存储向量到向量数据库
3. 执行相似度检索

================================================================================
                    【第二阶段：向量化存储阶段】完整流程图
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                  第二阶段：向量化存储 (Vectorization & Storage)              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【输入】List[TextNode] - 来自第一阶段的文本节点                            │
│                                                                             │
│  【Step 1】OpenAIEmbedding 向量化                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  文本块(text) → 嵌入模型 → 向量(embedding)                          │   │
│  │                                                                     │   │
│  │  嵌入模型的工作原理：                                                │   │
│  │  ┌────────────────────────────────────────────────────────────┐    │   │
│  │  │  输入: "机器学习是人工智能的一个分支"                        │    │   │
│  │  │                      ↓                                      │    │   │
│  │  │  OpenAI Embedding API (text-embedding-3-small)              │    │   │
│  │  │                      ↓                                      │    │   │
│  │  │  输出: [0.023, -0.156, 0.089, ..., 0.034]  # 1536维向量    │    │   │
│  │  └────────────────────────────────────────────────────────────┘    │   │
│  │                                                                     │   │
│  │  向量的特性：                                                        │   │
│  │  • 语义相似的文本 → 向量在空间中距离近                              │   │
│  │  • 语义不同的文本 → 向量在空间中距离远                              │   │
│  │  • 向量维度：1536 (text-embedding-3-small)                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                            ↓                                                │
│  【Step 2】VectorStoreIndex 构建索引                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  VectorStoreIndex 是 LlamaIndex 的核心索引结构                      │   │
│  │                                                                     │   │
│  │  职责：                                                              │   │
│  │  1. 协调嵌入模型和向量存储                                          │   │
│  │  2. 管理文档节点的添加和删除                                        │   │
│  │  3. 提供统一的检索接口                                              │   │
│  │                                                                     │   │
│  │  内部流程：                                                          │   │
│  │  TextNode → embed_model.get_text_embedding() → vector              │   │
│  │          → vector_store.add() → 存入数据库                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                            ↓                                                │
│  【Step 3】ChromaVectorStore 持久化存储                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ChromaDB 存储：                                                     │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │  id      │ embedding (1536维) │ metadata      │ document    │  │   │
│  │  ├─────────┼─────────────────────┼───────────────┼─────────────┤  │   │
│  │  │  doc_1  │ [0.02, -0.15, ...]  │ {file, page}  │ "文本内容"  │  │   │
│  │  │  doc_2  │ [0.08, 0.23, ...]   │ {file, page}  │ "文本内容"  │  │   │
│  │  │  ...    │ ...                 │ ...           │ ...         │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  │                                                                     │   │
│  │  【HNSW索引】(Hierarchical Navigable Small World)                   │   │
│  │  • 一种高效的近似最近邻搜索算法                                      │   │
│  │  • 检索复杂度：O(log n)，支持百万级向量                              │   │
│  │  • 使用余弦相似度作为距离度量                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                            ↓                                                │
│  【输出】持久化的向量数据库，可供第三阶段检索使用                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
                           【检索流程图】
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                           向量检索流程                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  用户查询: "什么是机器学习？"                                                │
│       ↓                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Step 1: 查询向量化                                                  │   │
│  │  "什么是机器学习？" → OpenAIEmbedding → [0.12, -0.08, ...]          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       ↓                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Step 2: 相似度搜索 (HNSW算法)                                       │   │
│  │                                                                     │   │
│  │  在向量空间中查找与查询向量最近的K个向量：                           │   │
│  │                                                                     │   │
│  │  余弦相似度 = (A · B) / (|A| × |B|)                                 │   │
│  │  值范围: [-1, 1]，越接近1越相似                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       ↓                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Step 3: 返回结果                                                    │   │
│  │  NodeWithScore {                                                     │   │
│  │    node: TextNode (原始文档块)                                       │   │
│  │    score: 0.89 (相似度分数)                                          │   │
│  │  }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

【为什么用向量检索？】
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  1. 语义理解：能找到语义相关而非仅仅关键词匹配的内容                         │
│     - 查询"汽车"可以匹配到"轿车"、"机动车"等相关内容                        │
│                                                                             │
│  2. 跨语言：不同语言的相似内容可以匹配                                       │
│     - 英文的"machine learning"可以匹配中文的"机器学习"                       │
│                                                                             │
│  3. 容错性：即使用户查询表述不精确也能找到相关内容                           │
│     - 拼写错误、同义词替换都能正常工作                                       │
│                                                                             │
│  4. 高效性：HNSW算法支持大规模数据的快速检索                                 │
│     - 百万级向量也能在毫秒级返回结果                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings

# ============================================================
# LlamaIndex 核心导入 - 第二阶段向量化存储所需的核心组件
# ============================================================

# ★★★【核心导入】VectorStoreIndex - LlamaIndex的核心索引类★★★
# VectorStoreIndex 是整个向量检索系统的心脏
# 职责：
#   1. 协调嵌入模型和向量存储
#   2. 管理文档节点的添加、删除、更新
#   3. 提供统一的检索接口
#   4. 处理文档的自动分块和向量化
from llama_index.core import VectorStoreIndex, StorageContext

# ★★★【核心导入】TextNode, NodeWithScore - 数据模型★★★
# TextNode: LlamaIndex中最基本的数据单元
#   - 包含文本内容、元数据、嵌入向量等
#   - 是文档分块后的表示形式
# NodeWithScore: 带有相似度分数的节点包装类
#   - 检索结果的标准格式
#   - 包含 node(TextNode) + score(float)
from llama_index.core.schema import TextNode, NodeWithScore

# ★★★【核心导入】VectorIndexRetriever - 向量检索器★★★
# VectorIndexRetriever: 执行向量相似度搜索的核心类
# 工作流程：
#   1. 将查询文本转换为向量
#   2. 在向量空间中搜索最近的K个向量
#   3. 返回对应的文档节点
from llama_index.core.retrievers import VectorIndexRetriever

# ★★★【核心导入】向量存储类型定义★★★
# VectorStore: 向量存储的抽象基类，定义了统一的接口
# MetadataFilters: 元数据过滤器集合，支持多个过滤条件的组合
# MetadataFilter: 单个元数据过滤条件，如 {"source": "doc.pdf"}
from llama_index.core.vector_stores.types import VectorStore, MetadataFilters, MetadataFilter

# ★★★【核心导入】OpenAIEmbedding - 嵌入模型★★★
# OpenAIEmbedding: 将文本转换为高维向量的模型
# 默认使用 text-embedding-3-small 模型
#   - 输出维度: 1536
#   - 适用场景: 文档检索、语义搜索、聚类
#   - 价格: $0.02/1M tokens (非常经济)
from llama_index.embeddings.openai import OpenAIEmbedding

# ★★★【核心导入】ChromaVectorStore - 向量数据库适配器★★★
# ChromaVectorStore: ChromaDB的LlamaIndex适配器
# ChromaDB特点:
#   - 开源、轻量级向量数据库
#   - 支持本地持久化存储
#   - 内置HNSW索引算法
#   - 支持元数据过滤
from llama_index.vector_stores.chroma import ChromaVectorStore


class VectorStoreManager:
    """
    向量存储管理器 - 第二阶段的核心类
    
    负责向量数据库的创建、管理和检索
    
    【向量数据库的作用】
    1. 高效存储：压缩存储高维向量（1536维）
    2. 快速检索：使用近似最近邻(ANN)算法快速查找相似向量
    3. 持久化：数据可以保存到磁盘，下次启动无需重新处理
    
    【ChromaDB特点】
    - 轻量级，适合本地开发和中小规模应用
    - 支持元数据过滤（按文件名、时间等条件筛选）
    - 自动处理向量索引（HNSW算法）
    - 数据持久化到磁盘
    
    【与其他向量数据库的比较】
    ┌─────────────┬──────────────┬──────────────┬────────────────┐
    │  数据库     │  适用场景     │  性能        │  部署复杂度    │
    ├─────────────┼──────────────┼──────────────┼────────────────┤
    │  ChromaDB   │ 本地/小规模   │ 中等         │ 低（嵌入式）   │
    │  Pinecone   │ 云端/大规模   │ 高           │ 低（SaaS）     │
    │  Milvus     │ 大规模生产    │ 很高         │ 高（需集群）   │
    │  Weaviate   │ 混合检索      │ 高           │ 中等           │
    │  pgvector   │ PostgreSQL   │ 中等         │ 低（扩展）     │
    └─────────────┴──────────────┴──────────────┴────────────────┘
    """
    
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag_documents",
        embed_model: Optional[OpenAIEmbedding] = None
    ):
        """
        初始化向量存储管理器
        
        【初始化流程】
        ┌─────────────────────────────────────────────────────────────┐
        │  1. 创建嵌入模型（用于文本向量化）                           │
        │  2. 创建持久化目录                                          │
        │  3. 初始化ChromaDB客户端                                    │
        │  4. 获取或创建集合（collection）                            │
        │  5. 创建LlamaIndex的向量存储包装器                          │
        │  6. 创建存储上下文                                          │
        └─────────────────────────────────────────────────────────────┘
        
        参数:
            persist_dir: 向量数据库持久化目录，数据会保存到这个目录
            collection_name: 集合名称（类似数据库表名），不同类型的数据可以用不同集合
            embed_model: 嵌入模型实例，用于将文本转换为向量
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # ★★★【核心组件】初始化嵌入模型★★★
        # 嵌入模型将文本转换为高维向量，这是向量检索的基础
        # OpenAI的text-embedding-3-small模型：
        #   - 生成1536维向量
        #   - 性价比高（$0.02/1M tokens）
        #   - 适合大多数RAG场景
        self.embed_model = embed_model or OpenAIEmbedding(
            model="text-embedding-3-small"
        )
        
        # 创建持久化目录
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # ★★★【核心组件】初始化ChromaDB客户端★★★
        # PersistentClient 将数据持久化到磁盘
        # anonymized_telemetry=False 关闭匿名遥测
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # ★★★【核心组件】获取或创建集合★★★
        # collection 类似于数据库中的表
        # metadata={"hnsw:space": "cosine"} 指定使用余弦相似度作为距离度量
        # HNSW (Hierarchical Navigable Small World) 是一种高效的近似最近邻算法
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # ★★★【核心组件】创建LlamaIndex的向量存储包装器★★★
        # ChromaVectorStore 是 LlamaIndex 和 ChromaDB 之间的适配器
        # 它将 LlamaIndex 的操作转换为 ChromaDB 的 API 调用
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        
        # ★★★【核心组件】创建存储上下文★★★
        # StorageContext 管理所有存储组件：
        #   - vector_store: 向量存储
        #   - doc_store: 文档存储
        #   - index_store: 索引存储
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # 向量索引（懒加载，第一次访问时才创建）
        self._index: Optional[VectorStoreIndex] = None
    
    @property
    def index(self) -> VectorStoreIndex:
        """
        获取向量索引（懒加载模式）
        
        【VectorStoreIndex的作用】
        VectorStoreIndex 是 LlamaIndex 中最常用的索引类型
        
        1. 管理文档节点的向量化和存储
        2. 提供统一的检索接口
        3. 协调嵌入模型和向量存储
        
        【懒加载的好处】
        - 只有真正需要检索时才创建索引
        - 节省初始化时间
        - 减少内存占用
        """
        if self._index is None:
            # ★★★【核心调用】从向量存储创建索引★★★
            # 这个方法会：
            # 1. 读取已有的向量数据
            # 2. 创建索引结构
            # 3. 关联嵌入模型（用于查询向量化）
            self._index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model
            )
        return self._index
    
    def add_nodes(self, nodes: List[TextNode]) -> None:
        """
        将节点添加到向量存储
        
        【完整处理流程】
        ┌─────────────────────────────────────────────────────────────┐
        │  for each node in nodes:                                    │
        │    ┌─────────────────────────────────────────────────────┐ │
        │    │  1. 提取文本内容: text = node.text                   │ │
        │    │  2. 调用嵌入模型: embedding = embed_model(text)     │ │
        │    │  3. 存入ChromaDB:                                    │ │
        │    │     - id: node.id_                                   │ │
        │    │     - embedding: [0.02, -0.15, ...]                  │ │
        │    │     - metadata: node.metadata                        │ │
        │    │     - document: node.text                            │ │
        │    └─────────────────────────────────────────────────────┘ │
        └─────────────────────────────────────────────────────────────┘
        
        参数:
            nodes: 来自第一阶段的文本节点列表
        """
        if not nodes:
            return
        
        # ★★★【核心调用】将节点添加到向量索引★★★
        # insert_node() 方法会：
        # 1. 调用嵌入模型将文本转换为向量
        # 2. 将向量、元数据、原文存入ChromaDB
        # 3. 更新索引结构
        # 这是【第二阶段最关键的步骤】
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
        检索器是第三阶段Agent检索的核心组件，负责：
        1. 接收用户查询
        2. 执行向量相似度搜索
        3. 返回最相关的文档块
        
        【similarity_top_k 的选择】
        - 太小（1-2）：可能遗漏相关信息
        - 太大（10+）：可能引入噪音，增加LLM处理成本
        - 推荐：3-5，根据实际效果调整
        
        参数:
            similarity_top_k: 返回最相似的K个结果
            
        返回:
            VectorIndexRetriever 向量检索器实例
        """
        # ★★★【核心调用】创建向量检索器★★★
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
        执行检索 - 这是第二阶段输出的主要接口
        
        【检索过程详解】
        ┌─────────────────────────────────────────────────────────────┐
        │  Step 1: 查询向量化                                         │
        │  ────────────────────                                       │
        │  query = "什么是机器学习？"                                  │
        │       ↓                                                     │
        │  embed_model.get_text_embedding(query)                      │
        │       ↓                                                     │
        │  query_vector = [0.12, -0.08, 0.23, ...]  # 1536维向量      │
        │                                                             │
        │  Step 2: 相似度搜索                                         │
        │  ────────────────────                                       │
        │  在向量空间中查找与query_vector最近的K个向量                 │
        │  使用余弦相似度：cos(θ) = A·B / (|A| × |B|)                 │
        │  值范围: [-1, 1]，越接近1表示越相似                          │
        │                                                             │
        │  Step 3: 返回结果                                           │
        │  ────────────────────                                       │
        │  返回带有相似度分数的节点列表，按分数降序排列                │
        └─────────────────────────────────────────────────────────────┘
        
        参数:
            query: 用户查询文本
            top_k: 返回结果数量
            
        返回:
            带有相似度分数的节点列表 NodeWithScore
        """
        retriever = self.get_retriever(similarity_top_k=top_k)
        
        # ★★★【核心调用】执行向量相似度搜索★★★
        # 这是【第二阶段向第三阶段提供数据的关键方法】
        # 返回的 NodeWithScore 包含：
        #   - node: TextNode（原始文档块）
        #   - score: float（相似度分数，范围通常在0-1）
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
        ┌─────────────────────────────────────────────────────────────┐
        │  场景1: 按文档来源过滤                                       │
        │  filters = {"file_name": "技术文档.pdf"}                    │
        │  → 只在技术文档中搜索                                        │
        │                                                             │
        │  场景2: 按时间范围过滤                                       │
        │  filters = {"year": "2024"}                                │
        │  → 只搜索2024年的文档                                        │
        │                                                             │
        │  场景3: 按文档类型过滤                                       │
        │  filters = {"doc_type": "api"}                             │
        │  → 只搜索API相关文档                                         │
        └─────────────────────────────────────────────────────────────┘
        
        【过滤原理】
        1. 先根据metadata过滤符合条件的文档
        2. 再在过滤后的文档中执行向量相似度搜索
        3. 这样可以减少搜索范围，提高精确度
        
        参数:
            query: 查询文本
            filters: 过滤条件字典，如 {"source": "document.pdf"}
            top_k: 返回结果数量
            
        返回:
            过滤后的检索结果
        """
        # ★★★【核心调用】构建元数据过滤器★★★
        # MetadataFilters 封装多个过滤条件
        # MetadataFilter 表示单个过滤条件 (key, value)
        metadata_filters = MetadataFilters(
            filters=[
                MetadataFilter(key=k, value=v)
                for k, v in filters.items()
            ]
        )
        
        # ★★★【核心调用】创建带过滤器的检索器★★★
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
            filters=metadata_filters
        )
        
        return retriever.retrieve(query)
    
    def delete_collection(self) -> None:
        """
        删除当前集合（清空所有数据）
        
        【使用场景】
        - 需要重新导入所有文档时
        - 切换到完全不同的知识库时
        - 测试清理时
        """
        self.chroma_client.delete_collection(self.collection_name)
        self._index = None
        print(f"✓ 已删除集合: {self.collection_name}")
    
    def get_collection_count(self) -> int:
        """
        获取集合中的文档数量
        
        返回:
            当前集合中存储的向量数量
        """
        return self.collection.count()


class RetrieverEngine:
    """
    检索引擎 - 封装高级检索功能，是第二阶段与第三阶段的桥梁
    
    【检索策略】
    ┌─────────────────────────────────────────────────────────────┐
    │  1. 基础向量检索（当前实现）                                 │
    │     - 纯语义相似度匹配                                       │
    │     - 使用余弦相似度计算                                     │
    │                                                             │
    │  2. 混合检索（可扩展）                                       │
    │     - 结合关键词(BM25)和语义向量                             │
    │     - 提高精确匹配的召回率                                   │
    │                                                             │
    │  3. 重排序（可扩展）                                         │
    │     - 对初步结果进行二次排序                                 │
    │     - 使用更精确的模型重新计算相似度                         │
    └─────────────────────────────────────────────────────────────┘
    
    【在RAG系统中的位置】
    第二阶段输出 → RetrieverEngine → 第三阶段输入
    """
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        """
        初始化检索引擎
        
        参数:
            vector_store_manager: 向量存储管理器（第二阶段的核心类）
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
        
        【搜索流程】
        ┌─────────────────────────────────────────────────────────────┐
        │  1. 调用向量存储管理器执行检索                              │
        │  2. 过滤掉相似度低于阈值的结果                              │
        │  3. 格式化为字典列表返回                                    │
        └─────────────────────────────────────────────────────────────┘
        
        参数:
            query: 查询文本
            top_k: 返回结果数量
            min_score: 最低相似度阈值（0-1），低于此值的结果会被过滤
            
        返回:
            格式化的搜索结果列表，每个结果包含：
            - text: 文档内容
            - score: 相似度分数
            - metadata: 元数据
            - node_id: 节点ID
        """
        # ★★★【核心调用】执行向量检索★★★
        nodes_with_scores = self.vsm.retrieve(query, top_k=top_k)
        
        # 格式化结果，过滤低分结果
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
        直接为第三阶段的LLM提供检索到的上下文
        返回的文本可以直接插入到Prompt中
        
        【输出格式示例】
        ┌─────────────────────────────────────────────────────────────┐
        │  [文档片段 1] (相关度: 0.892)                               │
        │  来源: 技术文档.pdf                                         │
        │  机器学习是人工智能的一个分支...                            │
        │                                                             │
        │  ---                                                        │
        │                                                             │
        │  [文档片段 2] (相关度: 0.756)                               │
        │  来源: 产品手册.pdf                                         │
        │  深度学习是机器学习的子领域...                              │
        └─────────────────────────────────────────────────────────────┘
        
        参数:
            query: 查询文本
            top_k: 返回结果数量
            
        返回:
            拼接后的上下文文本，可直接用于LLM的Prompt
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
    演示向量存储和检索功能 - 第二阶段完整流程演示
    
    【演示流程】
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                                                             │
    │  Step 1: 创建示例文本节点                                                    │
    │  ─────────────────────────────────────────────────────────────────────────  │
    │  创建3个包含不同主题的TextNode                                               │
    │  - Python介绍                                                               │
    │  - 机器学习介绍                                                              │
    │  - 深度学习介绍                                                              │
    │                                                                             │
    │  Step 2: 初始化向量存储管理器                                                │
    │  ─────────────────────────────────────────────────────────────────────────  │
    │  VectorStoreManager 会：                                                    │
    │  - 创建持久化目录                                                            │
    │  - 初始化ChromaDB                                                           │
    │  - 准备嵌入模型                                                              │
    │                                                                             │
    │  Step 3: 添加节点到向量存储                                                  │
    │  ─────────────────────────────────────────────────────────────────────────  │
    │  add_nodes() 会：                                                           │
    │  - 调用OpenAI Embedding API向量化每个节点                                    │
    │  - 将向量和元数据存入ChromaDB                                               │
    │                                                                             │
    │  Step 4: 创建检索引擎并执行检索                                              │
    │  ─────────────────────────────────────────────────────────────────────────  │
    │  RetrieverEngine.search() 会：                                              │
    │  - 将查询向量化                                                              │
    │  - 在向量空间中搜索最相似的文档                                              │
    │  - 返回带有相似度分数的结果                                                  │
    │                                                                             │
    │  Step 5: 清理演示数据                                                        │
    │  ─────────────────────────────────────────────────────────────────────────  │
    │  删除演示用的集合                                                            │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    【运行结果示例】
    查询: 什么是深度学习？
    
    检索结果:
    --- 结果 1 ---
    相关度: 0.892
    内容: 深度学习使用多层神经网络来处理复杂的模式识别任务。
    
    --- 结果 2 ---
    相关度: 0.756
    内容: 机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。
    """
    # ★★★【演示Step 1】创建示例文本节点★★★
    # TextNode 是 LlamaIndex 中最基本的数据单元
    # 每个节点包含：
    #   - text: 文本内容
    #   - metadata: 元数据（用于过滤和溯源）
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
    
    # ★★★【演示Step 2】初始化向量存储管理器★★★
    # persist_dir: 数据持久化目录
    # collection_name: 集合名称（类似数据库表名）
    vsm = VectorStoreManager(
        persist_dir="./demo_chroma",
        collection_name="demo_collection"
    )
    
    # ★★★【演示Step 3】添加节点到向量存储★★★
    # 这一步会调用OpenAI Embedding API将文本转换为向量
    vsm.add_nodes(sample_nodes)
    
    # ★★★【演示Step 4】创建检索引擎并执行检索★★★
    # RetrieverEngine 是第二阶段与第三阶段的桥梁
    engine = RetrieverEngine(vsm)
    
    # 执行语义搜索
    # 即使查询词"深度学习"与文档中的词不完全匹配
    # 向量检索也能找到语义相关的内容
    query = "什么是深度学习？"
    print(f"\n查询: {query}\n")
    
    results = engine.search(query, top_k=2)
    
    print("检索结果:")
    for i, result in enumerate(results, 1):
        print(f"\n--- 结果 {i} ---")
        print(f"相关度: {result['score']:.3f}")
        print(f"内容: {result['text']}")
    
    # ★★★【演示Step 5】清理演示数据★★★
    vsm.delete_collection()


if __name__ == "__main__":
    demo_vector_store()
