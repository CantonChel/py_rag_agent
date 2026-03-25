"""
增强版向量存储模块

支持动态向量维度、完整的关系映射和图片处理

【数据结构详解】

┌─────────────────────────────────────────────────────────────────────┐
│                     知识库完整数据结构                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【原始文档 Document】                                               │
│  ├── doc_id: 唯一标识                                               │
│  ├── file_path: 文件路径                                            │
│  ├── file_type: 文件类型(pdf/docx/html...)                         │
│  ├── total_pages: 总页数                                            │
│  └── created_at: 创建时间                                           │
│                                                                     │
│  【文本块 TextChunk】                                                │
│  ├── chunk_id: 唯一标识                                             │
│  ├── text: 文本内容                                                 │
│  ├── embedding: [向量] ← 维度由embedding模型决定                    │
│  ├── metadata:                                                      │
│  │   ├── doc_id: 所属文档ID                                         │
│  │   ├── chunk_index: 在文档中的序号                                │
│  │   ├── page_number: 来源页码                                      │
│  │   ├── start_char/end_char: 字符位置                              │
│  │   └── section_title: 所属章节                                    │
│  └── relationships:                                                 │
│      ├── prev_chunk: 前一块ID                                       │
│      ├── next_chunk: 后一块ID                                       │
│      ├── parent_chunk: 父块ID(父子分块时)                           │
│      └── related_images: [关联图片ID列表]                           │
│                                                                     │
│  【图片节点 ImageNode】                                              │
│  ├── image_id: 唯一标识                                             │
│  ├── image_data: base64或二进制数据                                 │
│  ├── image_path: 存储路径(如果保存到文件)                           │
│  ├── image_embedding: [多模态向量] ← 可选，用于图片检索             │
│  ├── metadata:                                                      │
│  │   ├── doc_id: 所属文档                                           │
│  │   ├── page_number: 所在页码                                      │
│  │   └── caption: 图片说明                                          │
│  └── relationships:                                                 │
│      └── related_chunks: [关联的文本块ID列表]                       │
│                                                                     │
│  【关系映射表】                                                      │
│  chunk_to_image: {chunk_id -> [image_ids]}                         │
│  image_to_chunk: {image_id -> [chunk_ids]}                         │
│  doc_to_chunks: {doc_id -> [chunk_ids]}                            │
│  doc_to_images: {doc_id -> [image_ids]}                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
"""

import os
import json
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime

import chromadb
from chromadb.config import Settings

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, ImageNode
from llama_index.core.vector_stores.types import VectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass
class EmbeddingModelConfig:
    """
    嵌入模型配置
    
    【动态维度说明】
    不同的embedding模型输出不同维度的向量：
    - OpenAI text-embedding-3-small: 1536维
    - OpenAI text-embedding-3-large: 3072维
    - BGE-large-zh: 1024维
    - BGE-M3: 1024维
    
    系统会自动检测模型输出的维度，无需手动指定
    """
    model_name: str
    api_type: str = "openai"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # 已知模型的维度映射
    KNOWN_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "bge-large-zh": 1024,
        "bge-large-en": 1024,
        "bge-m3": 1024,
        "bge-small-zh": 512,
        "text2vec-large-chinese": 1024,
    }
    
    def get_dimension(self) -> int:
        """
        获取模型的向量维度
        
        如果模型不在已知列表中，会通过实际调用来获取
        """
        if self.model_name in self.KNOWN_DIMENSIONS:
            return self.KNOWN_DIMENSIONS[self.model_name]
        # 未知模型，返回None，需要动态检测
        return None


@dataclass
class ChunkNode:
    """
    文本块节点数据结构
    
    封装了文本块的所有信息，包括内容、向量和关系
    """
    chunk_id: str
    text: str
    doc_id: str
    chunk_index: int
    embedding: Optional[List[float]] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    related_image_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_text_node(self) -> TextNode:
        """转换为LlamaIndex的TextNode"""
        node = TextNode(
            id_=self.chunk_id,
            text=self.text,
            metadata={
                "doc_id": self.doc_id,
                "chunk_index": self.chunk_index,
                "page_number": self.page_number,
                "section_title": self.section_title,
                "start_char": self.start_char,
                "end_char": self.end_char,
                "prev_chunk_id": self.prev_chunk_id,
                "next_chunk_id": self.next_chunk_id,
                "parent_chunk_id": self.parent_chunk_id,
                "child_chunk_ids": self.child_chunk_ids,
                "related_image_ids": self.related_image_ids,
                **self.metadata
            }
        )
        
        # 设置关系
        if self.prev_chunk_id:
            node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=self.prev_chunk_id
            )
        if self.next_chunk_id:
            node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=self.next_chunk_id
            )
        if self.parent_chunk_id:
            node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                node_id=self.parent_chunk_id
            )
        
        return node


@dataclass
class ImageNodeData:
    """
    图片节点数据结构
    """
    image_id: str
    doc_id: str
    image_data: Optional[bytes] = None
    image_path: Optional[str] = None
    image_embedding: Optional[List[float]] = None
    page_number: Optional[int] = None
    caption: Optional[str] = None
    related_chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_image_node(self) -> ImageNode:
        """转换为LlamaIndex的ImageNode"""
        return ImageNode(
            id_=self.image_id,
            image=self.image_path or "",
            text=self.caption or "",
            metadata={
                "doc_id": self.doc_id,
                "page_number": self.page_number,
                "caption": self.caption,
                "related_chunk_ids": self.related_chunk_ids,
                **self.metadata
            }
        )


@dataclass
class RelationshipMapping:
    """
    关系映射表
    
    【映射表的作用】
    1. 快速查找：O(1)时间查找相关节点
    2. 双向导航：可以从chunk找image，也可以反向查找
    3. 文档聚合：快速获取某个文档的所有chunks
    """
    # chunk到其他实体
    chunk_to_doc: Dict[str, str] = field(default_factory=dict)
    chunk_to_prev: Dict[str, str] = field(default_factory=dict)
    chunk_to_next: Dict[str, str] = field(default_factory=dict)
    chunk_to_parent: Dict[str, str] = field(default_factory=dict)
    chunk_to_children: Dict[str, List[str]] = field(default_factory=dict)
    chunk_to_images: Dict[str, List[str]] = field(default_factory=dict)
    
    # image到其他实体
    image_to_doc: Dict[str, str] = field(default_factory=dict)
    image_to_chunks: Dict[str, List[str]] = field(default_factory=dict)
    
    # doc到其他实体
    doc_to_chunks: Dict[str, List[str]] = field(default_factory=dict)
    doc_to_images: Dict[str, List[str]] = field(default_factory=dict)
    
    def add_chunk(self, chunk: ChunkNode):
        """添加chunk的关系映射"""
        cid = chunk.chunk_id
        
        self.chunk_to_doc[cid] = chunk.doc_id
        
        if chunk.prev_chunk_id:
            self.chunk_to_prev[cid] = chunk.prev_chunk_id
        if chunk.next_chunk_id:
            self.chunk_to_next[cid] = chunk.next_chunk_id
        if chunk.parent_chunk_id:
            self.chunk_to_parent[cid] = chunk.parent_chunk_id
        if chunk.child_chunk_ids:
            self.chunk_to_children[cid] = chunk.child_chunk_ids
        if chunk.related_image_ids:
            self.chunk_to_images[cid] = chunk.related_image_ids
        
        # 更新doc到chunk的映射
        if chunk.doc_id not in self.doc_to_chunks:
            self.doc_to_chunks[chunk.doc_id] = []
        self.doc_to_chunks[chunk.doc_id].append(cid)
    
    def add_image(self, image: ImageNodeData):
        """添加image的关系映射"""
        iid = image.image_id
        
        self.image_to_doc[iid] = image.doc_id
        
        if image.related_chunk_ids:
            self.image_to_chunks[iid] = image.related_chunk_ids
        
        # 更新doc到image的映射
        if image.doc_id not in self.doc_to_images:
            self.doc_to_images[image.doc_id] = []
        self.doc_to_images[image.doc_id].append(iid)
    
    def get_adjacent_chunks(self, chunk_id: str, window: int = 1) -> List[str]:
        """
        获取相邻的chunk（滑动窗口）
        
        参数:
            chunk_id: 当前chunk ID
            window: 窗口大小（前后各取几个）
            
        返回:
            相邻chunk ID列表
        """
        result = []
        
        # 向前查找
        current = chunk_id
        for _ in range(window):
            if current in self.chunk_to_prev:
                prev_id = self.chunk_to_prev[current]
                result.insert(0, prev_id)
                current = prev_id
            else:
                break
        
        # 添加自己
        result.append(chunk_id)
        
        # 向后查找
        current = chunk_id
        for _ in range(window):
            if current in self.chunk_to_next:
                next_id = self.chunk_to_next[current]
                result.append(next_id)
                current = next_id
            else:
                break
        
        return result
    
    def get_context_chunks(self, chunk_id: str, include_siblings: bool = True, include_parent: bool = True) -> List[str]:
        """
        获取上下文相关的chunks
        
        参数:
            chunk_id: 当前chunk ID
            include_siblings: 是否包含兄弟节点（同一父节点的其他子节点）
            include_parent: 是否包含父节点
            
        返回:
            上下文chunk ID列表
        """
        result = [chunk_id]
        
        # 包含父节点
        if include_parent and chunk_id in self.chunk_to_parent:
            result.append(self.chunk_to_parent[chunk_id])
        
        # 包含兄弟节点
        if include_siblings and chunk_id in self.chunk_to_parent:
            parent_id = self.chunk_to_parent[chunk_id]
            if parent_id in self.chunk_to_children:
                siblings = [c for c in self.chunk_to_children[parent_id] if c != chunk_id]
                result.extend(siblings)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "chunk_to_doc": self.chunk_to_doc,
            "chunk_to_prev": self.chunk_to_prev,
            "chunk_to_next": self.chunk_to_next,
            "chunk_to_parent": self.chunk_to_parent,
            "chunk_to_children": self.chunk_to_children,
            "chunk_to_images": self.chunk_to_images,
            "image_to_doc": self.image_to_doc,
            "image_to_chunks": self.image_to_chunks,
            "doc_to_chunks": self.doc_to_chunks,
            "doc_to_images": self.doc_to_images,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipMapping":
        """从字典反序列化"""
        return cls(**data)


class EnhancedVectorStore:
    """
    增强版向量存储
    
    【主要改进】
    1. 动态检测embedding维度
    2. 完整的关系映射
    3. 图片存储支持
    4. 持久化关系数据
    """
    
    def __init__(
        self,
        persist_dir: str = "./vector_store",
        collection_name: str = "knowledge_base",
        embed_model_config: Optional[EmbeddingModelConfig] = None,
        embed_model: Optional[Any] = None
    ):
        """
        初始化向量存储
        
        参数:
            persist_dir: 持久化目录
            collection_name: 集合名称
            embed_model_config: 嵌入模型配置
            embed_model: 嵌入模型实例（直接传入）
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化嵌入模型
        if embed_model:
            self.embed_model = embed_model
        elif embed_model_config:
            self.embed_model = self._create_embed_model(embed_model_config)
        else:
            # 默认使用OpenAI的small模型
            self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        # 【动态检测向量维度】
        # 通过实际调用获取模型输出维度
        self.embedding_dimension = self._detect_embedding_dimension()
        print(f"✓ 检测到嵌入模型维度: {self.embedding_dimension}")
        
        # 初始化ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 创建文本集合
        self.text_collection = self.chroma_client.get_or_create_collection(
            name=f"{collection_name}_texts",
            metadata={"hnsw:space": "cosine", "dimension": self.embedding_dimension}
        )
        
        # 创建图片集合（可选）
        self.image_collection = self.chroma_client.get_or_create_collection(
            name=f"{collection_name}_images",
            metadata={"hnsw:space": "cosine"}
        )
        
        # 初始化关系映射
        self.relationships = self._load_relationships()
        
        # LlamaIndex组件
        self.vector_store = ChromaVectorStore(chroma_collection=self.text_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self._index: Optional[VectorStoreIndex] = None
    
    def _create_embed_model(self, config: EmbeddingModelConfig):
        """根据配置创建嵌入模型"""
        if config.api_type == "openai":
            return OpenAIEmbedding(
                model=config.model_name,
                api_key=config.api_key,
                api_base=config.api_base
            )
        # 可以添加其他模型支持
        raise ValueError(f"不支持的API类型: {config.api_type}")
    
    def _detect_embedding_dimension(self) -> int:
        """
        动态检测嵌入模型的输出维度
        
        【检测方法】
        1. 先检查已知模型列表
        2. 未知模型则发送测试文本获取实际维度
        """
        # 尝试从模型名称获取
        if hasattr(self.embed_model, 'model_name'):
            model_name = self.embed_model.model_name
            if model_name in EmbeddingModelConfig.KNOWN_DIMENSIONS:
                return EmbeddingModelConfig.KNOWN_DIMENSIONS[model_name]
        
        # 通过实际调用获取维度
        test_text = "test"
        try:
            embedding = self.embed_model.get_text_embedding(test_text)
            return len(embedding)
        except Exception as e:
            print(f"警告: 无法检测嵌入维度，使用默认值1536: {e}")
            return 1536
    
    def _load_relationships(self) -> RelationshipMapping:
        """加载关系映射"""
        mapping_file = Path(self.persist_dir) / "relationships.json"
        if mapping_file.exists():
            with open(mapping_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return RelationshipMapping.from_dict(data)
        return RelationshipMapping()
    
    def _save_relationships(self):
        """保存关系映射"""
        mapping_file = Path(self.persist_dir) / "relationships.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.relationships.to_dict(), f, ensure_ascii=False, indent=2)
    
    @property
    def index(self) -> VectorStoreIndex:
        """获取向量索引（懒加载）"""
        if self._index is None:
            self._index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model
            )
        return self._index
    
    def add_chunks(self, chunks: List[ChunkNode]) -> None:
        """
        添加文本块到向量存储
        
        【处理流程】
        1. 为每个chunk生成embedding
        2. 存储到ChromaDB
        3. 更新关系映射
        4. 持久化关系数据
        
        参数:
            chunks: 文本块列表
        """
        if not chunks:
            return
        
        # 转换为TextNode并添加
        nodes = [chunk.to_text_node() for chunk in chunks]
        
        # 批量添加到索引
        for node in nodes:
            self.index.insert_node(node)
        
        # 更新关系映射
        for chunk in chunks:
            self.relationships.add_chunk(chunk)
        
        # 保存关系
        self._save_relationships()
        
        print(f"✓ 添加了 {len(chunks)} 个文本块")
    
    def add_images(self, images: List[ImageNodeData]) -> None:
        """
        添加图片到存储
        
        【图片处理选项】
        1. 只存储元数据（image_path）
        2. 存储base64编码
        3. 可选：生成图片embedding用于多模态检索
        
        参数:
            images: 图片节点列表
        """
        if not images:
            return
        
        for image in images:
            # 如果有图片embedding，存储到向量库
            if image.image_embedding:
                self.image_collection.add(
                    ids=[image.image_id],
                    embeddings=[image.image_embedding],
                    metadatas=[{
                        "doc_id": image.doc_id,
                        "page_number": image.page_number,
                        "caption": image.caption or "",
                        "image_path": image.image_path or "",
                        "related_chunks": json.dumps(image.related_chunk_ids)
                    }]
                )
            
            # 更新关系映射
            self.relationships.add_image(image)
        
        self._save_relationships()
        print(f"✓ 添加了 {len(images)} 个图片")
    
    def link_chunk_to_image(self, chunk_id: str, image_id: str):
        """
        建立chunk和图片的关联
        
        参数:
            chunk_id: 文本块ID
            image_id: 图片ID
        """
        # 更新chunk到image的映射
        if chunk_id not in self.relationships.chunk_to_images:
            self.relationships.chunk_to_images[chunk_id] = []
        if image_id not in self.relationships.chunk_to_images[chunk_id]:
            self.relationships.chunk_to_images[chunk_id].append(image_id)
        
        # 更新image到chunk的映射
        if image_id not in self.relationships.image_to_chunks:
            self.relationships.image_to_chunks[image_id] = []
        if chunk_id not in self.relationships.image_to_chunks[image_id]:
            self.relationships.image_to_chunks[image_id].append(chunk_id)
        
        self._save_relationships()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_context: bool = True,
        context_window: int = 1
    ) -> List[Dict[str, Any]]:
        """
        执行检索
        
        【检索流程】
        1. 查询转向量
        2. 向量相似度搜索
        3. 可选：获取上下文chunks
        4. 可选：获取关联图片
        
        参数:
            query: 查询文本
            top_k: 返回结果数
            include_context: 是否包含上下文
            context_window: 上下文窗口大小
            
        返回:
            检索结果列表
        """
        # 获取查询embedding
        query_embedding = self.embed_model.get_text_embedding(query)
        
        # 执行向量搜索
        results = self.text_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results['ids'][0])):
            chunk_id = results['ids'][0][i]
            text = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            result = {
                "chunk_id": chunk_id,
                "text": text,
                "score": 1 - distance,
                "metadata": metadata,
            }
            
            # 获取上下文chunks
            if include_context:
                context_ids = self.relationships.get_adjacent_chunks(
                    chunk_id, window=context_window
                )
                result["context_chunk_ids"] = context_ids
            
            # 获取关联图片
            image_ids = self.relationships.chunk_to_images.get(chunk_id, [])
            result["related_images"] = image_ids
            
            formatted_results.append(result)
        
        return formatted_results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取chunk"""
        results = self.text_collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )
        
        if results['ids']:
            return {
                "chunk_id": chunk_id,
                "text": results['documents'][0],
                "metadata": results['metadatas'][0]
            }
        return None
    
    def get_document_chunks(self, doc_id: str) -> List[str]:
        """获取文档的所有chunks"""
        return self.relationships.doc_to_chunks.get(doc_id, [])
    
    def get_document_images(self, doc_id: str) -> List[str]:
        """获取文档的所有图片"""
        return self.relationships.doc_to_images.get(doc_id, [])
    
    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        return {
            "text_count": self.text_collection.count(),
            "image_count": self.image_collection.count(),
            "document_count": len(self.relationships.doc_to_chunks),
            "embedding_dimension": self.embedding_dimension,
            "persist_dir": self.persist_dir,
            "collection_name": self.collection_name
        }
    
    def clear(self):
        """清空所有数据"""
        self.chroma_client.delete_collection(f"{self.collection_name}_texts")
        self.chroma_client.delete_collection(f"{self.collection_name}_images")
        
        # 删除关系文件
        mapping_file = Path(self.persist_dir) / "relationships.json"
        if mapping_file.exists():
            mapping_file.unlink()
        
        # 重新初始化
        self.text_collection = self.chroma_client.get_or_create_collection(
            name=f"{self.collection_name}_texts",
            metadata={"hnsw:space": "cosine", "dimension": self.embedding_dimension}
        )
        self.image_collection = self.chroma_client.get_or_create_collection(
            name=f"{self.collection_name}_images",
            metadata={"hnsw:space": "cosine"}
        )
        self.relationships = RelationshipMapping()
        self._index = None
        
        print("✓ 已清空所有数据")


def demo_enhanced_store():
    """演示增强版向量存储"""
    # 创建存储
    store = EnhancedVectorStore(
        persist_dir="./demo_store",
        collection_name="demo"
    )
    
    # 创建示例chunks
    chunks = [
        ChunkNode(
            chunk_id="doc1_chunk_0",
            text="人工智能是计算机科学的一个分支。",
            doc_id="doc1",
            chunk_index=0,
            next_chunk_id="doc1_chunk_1"
        ),
        ChunkNode(
            chunk_id="doc1_chunk_1",
            text="机器学习是AI的核心技术之一。",
            doc_id="doc1",
            chunk_index=1,
            prev_chunk_id="doc1_chunk_0",
            next_chunk_id="doc1_chunk_2"
        ),
        ChunkNode(
            chunk_id="doc1_chunk_2",
            text="深度学习使用多层神经网络。",
            doc_id="doc1",
            chunk_index=2,
            prev_chunk_id="doc1_chunk_1"
        ),
    ]
    
    # 添加chunks
    store.add_chunks(chunks)
    
    # 创建示例图片
    images = [
        ImageNodeData(
            image_id="doc1_img_0",
            doc_id="doc1",
            caption="神经网络架构图",
            related_chunk_ids=["doc1_chunk_2"]
        )
    ]
    
    # 添加图片
    store.add_images(images)
    
    # 建立关联
    store.link_chunk_to_image("doc1_chunk_2", "doc1_img_0")
    
    # 测试检索
    print("\n【检索测试】")
    results = store.retrieve("什么是机器学习？", top_k=2)
    for r in results:
        print(f"\nChunk: {r['chunk_id']}")
        print(f"Score: {r['score']:.3f}")
        print(f"Text: {r['text']}")
        print(f"Context: {r.get('context_chunk_ids', [])}")
        print(f"Images: {r.get('related_images', [])}")
    
    # 打印统计
    print("\n【存储统计】")
    print(json.dumps(store.get_stats(), indent=2))
    
    # 清理
    store.clear()


if __name__ == "__main__":
    demo_enhanced_store()
