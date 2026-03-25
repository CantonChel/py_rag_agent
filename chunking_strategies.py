"""
分块策略模块

提供多种文档分块策略，用于将长文档切分成适合检索的小块

【分块策略对比图】

┌─────────────────────────────────────────────────────────────────────┐
│                         分块策略选择指南                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【1. 固定大小分块 (FixedSizeChunker)】                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ [块1: 512 tokens] [块2: 512 tokens] [块3: 512 tokens]       │   │
│  │      ↑重叠↑            ↑重叠↑                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  优点: 简单可控、向量均匀                                         │
│  缺点: 可能切断语义、忽略文档结构                                 │
│  适用: 格式统一的文档、快速原型                                   │
│                                                                     │
│  【2. 句子分块 (SentenceChunker)】                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ [句子1+句子2]  [句子3+句子4]  [句子5]                        │   │
│  │   (完整句子)    (完整句子)    (完整句子)                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  优点: 保持句子完整性、语义连贯                                   │
│  缺点: 块大小不均、可能超过token限制                              │
│  适用: 叙述性文本、文章                                           │
│                                                                     │
│  【3. 语义分块 (SemanticChunker)】                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ [语义段1]      [语义段2]        [语义段3]                    │   │
│  │  ↑相似度高      ↑相似度下降点    ↑新主题                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  优点: 语义完整、主题清晰                                         │
│  缺点: 计算成本高、需要embedding模型                              │
│  适用: 高质量检索场景                                             │
│                                                                     │
│  【4. 递归分块 (RecursiveChunker)】                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 章节1 ─┬─ 段落1 ─┬─ 句子1                                    │   │
│  │        │         └─ 句子2                                    │   │
│  │        └─ 段落2 ─── 句子3                                    │   │
│  │ 章节2 ─── 段落3 ...                                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  优点: 保持层级结构、上下文完整                                   │
│  缺点: 需要文档有清晰结构                                         │
│  适用: 技术文档、手册                                             │
│                                                                     │
│  【5. 父子分块 (ParentChildChunker)】                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 父块(大): [============== 1024 tokens ==============]        │   │
│  │              ↓ 包含                                           │   │
│  │ 子块(小): [===256===] [===256===] [===256===] [===256===]   │   │
│  │            ↑                                                  │   │
│  │         检索时用子块，返回时用父块                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  优点: 检索精确+上下文完整                                        │
│  缺点: 存储成本高、实现复杂                                       │
│  适用: 需要完整上下文的高质量RAG                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

【分块参数说明】
- chunk_size: 块的最大大小（通常以token或字符计）
- chunk_overlap: 相邻块之间的重叠大小（保证上下文连贯）
- separator: 分隔符（用于切分的边界字符）
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitter,
    TokenTextSplitter,
    HierarchicalNodeParser,
)
from llama_index.core import Document


class ChunkStrategy(Enum):
    """分块策略枚举"""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    PARENT_CHILD = "parent_child"


@dataclass
class ChunkMetadata:
    """
    块的元数据信息
    
    【元数据的作用】
    1. 追溯来源：知道这个块来自哪个文档的哪个位置
    2. 建立关系：记录与其他块的关系
    3. 辅助检索：用于过滤和排序
    """
    doc_id: str
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    related_images: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "prev_chunk_id": self.prev_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "parent_chunk_id": self.parent_chunk_id,
            "child_chunk_ids": self.child_chunk_ids,
            "related_images": self.related_images,
        }


@dataclass
class ChunkResult:
    """
    分块结果
    
    包含所有分块及其元数据和关系映射
    """
    nodes: List[TextNode]
    metadata: List[ChunkMetadata]
    doc_id: str
    total_chunks: int
    # 关系映射
    chunk_to_images: Dict[str, List[str]] = field(default_factory=dict)
    image_to_chunks: Dict[str, List[str]] = field(default_factory=dict)


class BaseChunker(ABC):
    """
    分块器基类
    
    所有分块策略都需要继承此类
    """
    
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkResult:
        """
        执行分块
        
        参数:
            text: 待分块的文本
            metadata: 文档级元数据
            
        返回:
            分块结果
        """
        pass
    
    def _generate_node_id(self, doc_id: str, chunk_index: int) -> str:
        """生成节点ID"""
        return f"{doc_id}_chunk_{chunk_index}"


class FixedSizeChunker(BaseChunker):
    """
    固定大小分块器
    
    【工作原理】
    1. 将文本按固定token数切分
    2. 相邻块之间有重叠，保证上下文连贯
    3. 使用tiktoken精确计算token数
    
    【适用场景】
    - 格式统一的文档
    - 对块大小有严格要求
    - 快速原型开发
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = " "
    ):
        """
        初始化固定大小分块器
        
        参数:
            chunk_size: 块大小（token数）
            chunk_overlap: 重叠大小（token数）
            separator: 分隔符
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        
        # 使用LlamaIndex的TokenTextSplitter
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator
        )
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkResult:
        """
        执行固定大小分块
        
        参数:
            text: 待分块文本
            metadata: 文档元数据
            
        返回:
            分块结果
        """
        metadata = metadata or {}
        doc_id = metadata.get("doc_id", "unknown")
        
        # 创建临时文档
        doc = Document(text=text, metadata=metadata)
        
        # 分块
        nodes = self.splitter.get_nodes_from_documents([doc])
        
        # 构建元数据和关系
        chunk_metadata_list = []
        for i, node in enumerate(nodes):
            node.node_id = self._generate_node_id(doc_id, i)
            
            chunk_meta = ChunkMetadata(
                doc_id=doc_id,
                chunk_index=i,
                total_chunks=len(nodes),
                start_char=0,
                end_char=len(node.text),
                page_number=metadata.get("page_number"),
                section_title=metadata.get("section_title"),
                prev_chunk_id=self._generate_node_id(doc_id, i - 1) if i > 0 else None,
                next_chunk_id=self._generate_node_id(doc_id, i + 1) if i < len(nodes) - 1 else None,
            )
            chunk_metadata_list.append(chunk_meta)
            
            # 更新节点的元数据
            node.metadata.update(chunk_meta.to_dict())
            
            # 设置节点关系
            if i > 0:
                node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                    node_id=self._generate_node_id(doc_id, i - 1)
                )
            if i < len(nodes) - 1:
                node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                    node_id=self._generate_node_id(doc_id, i + 1)
                )
        
        return ChunkResult(
            nodes=nodes,
            metadata=chunk_metadata_list,
            doc_id=doc_id,
            total_chunks=len(nodes)
        )


class SentenceChunker(BaseChunker):
    """
    句子分块器
    
    【工作原理】
    1. 使用NLP技术识别句子边界
    2. 按句子组合成块，直到达到大小限制
    3. 保证不会在句子中间切分
    
    【句子边界识别】
    - 中文：。！？；等
    - 英文：. ! ? ; 等
    - 特殊处理：小数点、缩写等
    
    【适用场景】
    - 叙述性文本
    - 需要保持句子完整性
    - 问答对
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        paragraph_separator: str = "\n\n"
    ):
        """
        初始化句子分块器
        
        参数:
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            paragraph_separator: 段落分隔符
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.paragraph_separator = paragraph_separator
        
        # 使用LlamaIndex的SentenceSplitter
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator=paragraph_separator,
            # 二次切分的正则（按句子边界）
            secondary_chunking_regex="[^，。！？；,.;!?]+[，。！？；,.;!?]?"
        )
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkResult:
        """
        执行句子分块
        """
        metadata = metadata or {}
        doc_id = metadata.get("doc_id", "unknown")
        
        doc = Document(text=text, metadata=metadata)
        nodes = self.splitter.get_nodes_from_documents([doc])
        
        chunk_metadata_list = []
        for i, node in enumerate(nodes):
            node.node_id = self._generate_node_id(doc_id, i)
            
            chunk_meta = ChunkMetadata(
                doc_id=doc_id,
                chunk_index=i,
                total_chunks=len(nodes),
                start_char=0,
                end_char=len(node.text),
                prev_chunk_id=self._generate_node_id(doc_id, i - 1) if i > 0 else None,
                next_chunk_id=self._generate_node_id(doc_id, i + 1) if i < len(nodes) - 1 else None,
            )
            chunk_metadata_list.append(chunk_meta)
            node.metadata.update(chunk_meta.to_dict())
        
        return ChunkResult(
            nodes=nodes,
            metadata=chunk_metadata_list,
            doc_id=doc_id,
            total_chunks=len(nodes)
        )


class SemanticChunker(BaseChunker):
    """
    语义分块器
    
    【工作原理】
    1. 将文本按句子分割
    2. 对每个句子计算embedding
    3. 计算相邻句子的相似度
    4. 在相似度显著下降的位置切分
    
    【相似度断点检测】
    - 计算相邻句子的余弦相似度
    - 当相似度低于某个百分位时，认为是主题转换点
    - 在这些点进行切分
    
    【适用场景】
    - 高质量检索
    - 多主题文档
    - 需要语义完整性
    """
    
    def __init__(
        self,
        embed_model,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95
    ):
        """
        初始化语义分块器
        
        参数:
            embed_model: 嵌入模型实例
            buffer_size: 计算相似度时的句子缓冲数
            breakpoint_percentile_threshold: 断点百分位阈值
        """
        self.embed_model = embed_model
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        
        self.splitter = SemanticSplitter(
            embed_model=embed_model,
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold
        )
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkResult:
        """
        执行语义分块
        """
        metadata = metadata or {}
        doc_id = metadata.get("doc_id", "unknown")
        
        doc = Document(text=text, metadata=metadata)
        nodes = self.splitter.get_nodes_from_documents([doc])
        
        chunk_metadata_list = []
        for i, node in enumerate(nodes):
            node.node_id = self._generate_node_id(doc_id, i)
            
            chunk_meta = ChunkMetadata(
                doc_id=doc_id,
                chunk_index=i,
                total_chunks=len(nodes),
                start_char=0,
                end_char=len(node.text),
                prev_chunk_id=self._generate_node_id(doc_id, i - 1) if i > 0 else None,
                next_chunk_id=self._generate_node_id(doc_id, i + 1) if i < len(nodes) - 1 else None,
            )
            chunk_metadata_list.append(chunk_meta)
            node.metadata.update(chunk_meta.to_dict())
        
        return ChunkResult(
            nodes=nodes,
            metadata=chunk_metadata_list,
            doc_id=doc_id,
            total_chunks=len(nodes)
        )


class ParentChildChunker(BaseChunker):
    """
    父子分块器
    
    【工作原理】
    1. 创建大块（父块）和小块（子块）
    2. 小块用于精确检索
    3. 检索到小块后，返回对应的父块作为上下文
    
    【优势】
    - 检索精确：小块更容易匹配查询
    - 上下文完整：返回大块提供完整上下文
    - 灵活性：可以调整父子块大小比例
    
    【适用场景】
    - 需要完整上下文的高质量RAG
    - 文档主题跨度大
    - 用户查询通常很短
    """
    
    def __init__(
        self,
        parent_chunk_size: int = 1024,
        parent_chunk_overlap: int = 100,
        child_chunk_size: int = 256,
        child_chunk_overlap: int = 50
    ):
        """
        初始化父子分块器
        
        参数:
            parent_chunk_size: 父块大小
            parent_chunk_overlap: 父块重叠
            child_chunk_size: 子块大小
            child_chunk_overlap: 子块重叠
        """
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        
        # 使用LlamaIndex的层级解析器
        self.hierarchical_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[parent_chunk_size, child_chunk_size],
            chunk_overlaps=[parent_chunk_overlap, child_chunk_overlap]
        )
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkResult:
        """
        执行父子分块
        """
        metadata = metadata or {}
        doc_id = metadata.get("doc_id", "unknown")
        
        doc = Document(text=text, metadata=metadata)
        
        # 获取层级节点
        all_nodes = self.hierarchical_parser.get_nodes_from_documents([doc])
        
        # 分离父节点和子节点
        parent_nodes = [n for n in all_nodes if len(n.text) > self.child_chunk_size]
        child_nodes = [n for n in all_nodes if len(n.text) <= self.child_chunk_size]
        
        # 建立父子关系
        chunk_metadata_list = []
        child_index = 0
        
        for parent_idx, parent_node in enumerate(parent_nodes):
            parent_node.node_id = self._generate_node_id(doc_id, parent_idx)
            parent_node.metadata["is_parent"] = True
            
            # 找到属于这个父节点的子节点
            child_ids = []
            for child in child_nodes:
                if child.relationships.get(NodeRelationship.PARENT):
                    if child.relationships[NodeRelationship.PARENT].node_id == parent_node.node_id:
                        child.node_id = f"{parent_node.node_id}_child_{len(child_ids)}"
                        child.metadata["parent_id"] = parent_node.node_id
                        child.metadata["is_child"] = True
                        child_ids.append(child.node_id)
            
            chunk_meta = ChunkMetadata(
                doc_id=doc_id,
                chunk_index=parent_idx,
                total_chunks=len(parent_nodes),
                start_char=0,
                end_char=len(parent_node.text),
                child_chunk_ids=child_ids,
            )
            chunk_metadata_list.append(chunk_meta)
            parent_node.metadata.update(chunk_meta.to_dict())
        
        # 合并所有节点（用于索引）
        all_result_nodes = parent_nodes + child_nodes
        
        return ChunkResult(
            nodes=all_result_nodes,
            metadata=chunk_metadata_list,
            doc_id=doc_id,
            total_chunks=len(parent_nodes)
        )


class ChunkerFactory:
    """
    分块器工厂
    
    根据策略名称创建对应的分块器
    """
    
    @staticmethod
    def create(
        strategy: ChunkStrategy,
        embed_model=None,
        **kwargs
    ) -> BaseChunker:
        """
        创建分块器
        
        参数:
            strategy: 分块策略
            embed_model: 嵌入模型（语义分块需要）
            **kwargs: 其他参数
            
        返回:
            分块器实例
        """
        if strategy == ChunkStrategy.FIXED_SIZE:
            return FixedSizeChunker(**kwargs)
        elif strategy == ChunkStrategy.SENTENCE:
            return SentenceChunker(**kwargs)
        elif strategy == ChunkStrategy.SEMANTIC:
            if embed_model is None:
                raise ValueError("语义分块需要提供嵌入模型")
            return SemanticChunker(embed_model=embed_model, **kwargs)
        elif strategy == ChunkStrategy.PARENT_CHILD:
            return ParentChildChunker(**kwargs)
        else:
            raise ValueError(f"未知的分块策略: {strategy}")


def demo_chunking():
    """演示不同分块策略"""
    sample_text = """
# 人工智能概述

人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。

## 机器学习

机器学习是AI的核心技术之一。它使计算机能够从数据中学习，而无需显式编程。

### 监督学习
监督学习使用标记数据进行训练。模型学习输入和输出之间的映射关系。

### 无监督学习
无监督学习使用未标记数据。模型尝试发现数据中的隐藏结构。

## 深度学习

深度学习是机器学习的一个子领域，使用多层神经网络来处理复杂任务。
    """
    
    print("="*60)
    print("【固定大小分块】")
    print("="*60)
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    result = chunker.chunk(sample_text, {"doc_id": "test_doc"})
    for i, node in enumerate(result.nodes):
        print(f"\n--- 块 {i+1} ---")
        print(node.text[:100] + "...")
    
    print("\n" + "="*60)
    print("【句子分块】")
    print("="*60)
    chunker = SentenceChunker(chunk_size=150, chunk_overlap=30)
    result = chunker.chunk(sample_text, {"doc_id": "test_doc"})
    for i, node in enumerate(result.nodes):
        print(f"\n--- 块 {i+1} ---")
        print(node.text[:100] + "...")
    
    print("\n" + "="*60)
    print("【父子分块】")
    print("="*60)
    chunker = ParentChildChunker(
        parent_chunk_size=200,
        child_chunk_size=50
    )
    result = chunker.chunk(sample_text, {"doc_id": "test_doc"})
    print(f"父块数: {result.total_chunks}")
    print(f"总节点数: {len(result.nodes)}")


if __name__ == "__main__":
    demo_chunking()
