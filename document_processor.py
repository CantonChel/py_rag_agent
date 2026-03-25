"""
文档处理模块
    
这个模块负责处理原始文档，包括：
1. 数据清洗：去除无用字符、标准化文本
2. 文档分块：将长文档切分成适合检索的小块
    
【文档处理流程图】
原始文档 -> 加载器读取 -> 文本清洗 -> 分块处理 -> 生成节点(Node) -> 存入向量库
"""

import re
from typing import List, Optional
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitter, NodeParser
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.embeddings.openai import OpenAIEmbedding


class DocumentCleaner:
    """
    文档清洗器
        
    负责对原始文本进行预处理，去除噪音数据，提高检索质量
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        清洗文本内容
        
        清洗步骤：
        1. 去除多余的空白字符（连续空格、制表符等）
        2. 去除特殊控制字符
        3. 标准化换行符
        4. 去除页码等无意义内容
            
        参数:
            text: 原始文本
            
        返回:
            清洗后的文本
        """
        if not text:
            return ""
            
        # 去除控制字符（保留换行和制表符）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
            
        # 将多个连续空格替换为单个空格
        text = re.sub(r'[ \t]+', ' ', text)
            
        # 将多个连续换行替换为两个换行（保留段落结构）
        text = re.sub(r'\n{3,}', '\n\n', text)
            
        # 去除行首行尾的空白
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
            
        # 去除可能的页码标记（如 "第 1 页" 或 "- 1 -" 等）
        text = re.sub(r'[-–—]?\s*\d+\s*[-–—]?\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^第\s*\d+\s*页', '', text, flags=re.MULTILINE)
            
        return text.strip()
    
    @staticmethod
    def remove_headers_footers(text: str) -> str:
        """
        去除页眉页脚
        
        参数:
            text: 原始文本
            
        返回:
            去除页眉页脚后的文本
        """
        lines = text.split('\n')
        cleaned_lines = []
            
        for line in lines:
            line = line.strip()
            # 跳过可能是页眉页脚的短行（可根据实际文档格式调整）
            if len(line) < 5 and re.match(r'^[\d\s\-–—]+$', line):
                continue
            cleaned_lines.append(line)
            
        return '\n'.join(cleaned_lines)


class DocumentChunker:
    """
    文档分块器
        
    将长文档切分成合适大小的块，这是RAG系统的核心步骤之一
        
    【为什么需要分块？】
    1. LLM有token限制，无法处理超长文档
    2. 向量检索的精度：小块更容易精确匹配用户查询
    3. 上下文相关性：避免检索到过多无关内容
        
    【分块策略】
    1. 固定大小分块：简单但可能切断语义
    2. 语义分块：基于句子/段落边界，保持语义完整
    3. 递归分块：按层级结构逐级切分
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_semantic_splitting: bool = False,
        embed_model: Optional[OpenAIEmbedding] = None
    ):
        """
        初始化分块器
            
        参数:
            chunk_size: 每个块的最大token数
            chunk_overlap: 相邻块之间的重叠token数（保证上下文连贯性）
            use_semantic_splitting: 是否使用语义分块
            embed_model: 嵌入模型（语义分块需要）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_splitting = use_semantic_splitting
            
        if use_semantic_splitting:
            if embed_model is None:
                raise ValueError("语义分块需要提供嵌入模型")
            self.splitter = SemanticSplitter(
                embed_model=embed_model,
                buffer_size=1,
                breakpoint_percentile_threshold=95
            )
        else:
            # 使用句子分割器，按句子边界切分
            self.splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                paragraph_separator="\n\n",
                secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?"
            )
    
    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        """
        将文档列表分块成节点列表
            
        【分块后的节点结构】
        每个TextNode包含：
        - text: 文本内容
        - metadata: 元数据（来源、页码等）
        - relationships: 与其他节点的关系（前后节点）
        - embedding: 向量嵌入（后续生成）
            
        参数:
            documents: 文档列表
            
        返回:
            节点列表
        """
        nodes = self.splitter.get_nodes_from_documents(documents)
            
        # 为每个节点添加额外的元数据
        for i, node in enumerate(nodes):
            node.metadata["chunk_index"] = i
            node.metadata["total_chunks"] = len(nodes)
                
        return nodes
    
    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[TextNode]:
        """
        将单个文本分块
        
        参数:
            text: 文本内容
            metadata: 额外的元数据
            
        返回:
            节点列表
        """
        document = Document(text=text, metadata=metadata or {})
        return self.chunk_documents([document])


class DocumentProcessor:
    """
    文档处理器 - 整合清洗和分块流程
        
    【完整处理流程】
    1. 加载文档（支持PDF、TXT、MD等格式）
    2. 数据清洗（去除噪音、标准化格式）
    3. 文档分块（切分成合适大小的块）
    4. 生成节点（带有元数据和关系信息）
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_semantic_splitting: bool = False,
        embed_model: Optional[OpenAIEmbedding] = None
    ):
        """
        初始化文档处理器
            
        参数:
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            use_semantic_splitting: 是否使用语义分块
            embed_model: 嵌入模型
        """
        self.cleaner = DocumentCleaner()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_semantic_splitting=use_semantic_splitting,
            embed_model=embed_model
        )
    
    def load_documents(self, file_path: str) -> List[Document]:
        """
        从文件或目录加载文档
        
        【支持的文件格式】
        - PDF (.pdf)
        - 文本文件 (.txt)
        - Markdown (.md)
        - CSV (.csv)
        - JSON (.json)
        - Word文档 (.docx) - 需要额外依赖
            
        参数:
            file_path: 文件路径或目录路径
            
        返回:
            文档列表
        """
        path = Path(file_path)
            
        if path.is_file():
            # 单个文件
            documents = SimpleDirectoryReader(
                input_files=[str(path)]
            ).load_data()
        else:
            # 目录
            documents = SimpleDirectoryReader(
                str(path),
                recursive=True,
                required_exts=[".pdf", ".txt", ".md", ".csv", ".json"]
            ).load_data()
            
        return documents
    
    def process_documents(
        self,
        documents: List[Document],
        clean_text: bool = True
    ) -> List[TextNode]:
        """
        处理文档：清洗 + 分块
        
        【处理步骤详解】
        1. 遍历每个文档
        2. 对文档文本进行清洗
        3. 更新文档内容
        4. 调用分块器进行分块
            
        参数:
            documents: 原始文档列表
            clean_text: 是否进行文本清洗
            
        返回:
            处理后的节点列表
        """
        if clean_text:
            cleaned_documents = []
            for doc in documents:
                # 清洗文本
                cleaned_text = self.cleaner.clean_text(doc.text)
                cleaned_text = self.cleaner.remove_headers_footers(cleaned_text)
                    
                # 创建新的文档对象，保留原有元数据
                cleaned_doc = Document(
                    text=cleaned_text,
                    metadata=doc.metadata.copy(),
                    doc_id=doc.doc_id
                )
                cleaned_documents.append(cleaned_doc)
                
            documents = cleaned_documents
            
        # 分块处理
        nodes = self.chunker.chunk_documents(documents)
            
        return nodes
    
    def process_file(
        self,
        file_path: str,
        clean_text: bool = True
    ) -> List[TextNode]:
        """
        完整处理流程：加载文件 -> 清洗 -> 分块
        
        这是外部调用的主要入口
        
        参数:
            file_path: 文件路径
            clean_text: 是否清洗文本
            
        返回:
            处理后的节点列表
        """
        # 步骤1: 加载文档
        documents = self.load_documents(file_path)
        print(f"✓ 加载了 {len(documents)} 个文档")
            
        # 步骤2: 处理文档（清洗 + 分块）
        nodes = self.process_documents(documents, clean_text=clean_text)
        print(f"✓ 生成了 {len(nodes)} 个节点")
            
        # 打印分块统计信息
        self._print_chunk_stats(nodes)
            
        return nodes
    
    def _print_chunk_stats(self, nodes: List[TextNode]) -> None:
        """打印分块统计信息"""
        if not nodes:
            return
            
        lengths = [len(node.text) for node in nodes]
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)
            
        print(f"\n【分块统计】")
        print(f"  - 总块数: {len(nodes)}")
        print(f"  - 平均长度: {avg_length:.0f} 字符")
        print(f"  - 最短块: {min_length} 字符")
        print(f"  - 最长块: {max_length} 字符")


def demo_processing():
    """
    演示文档处理流程
    """
    # 示例文本
    sample_text = """
    这是一篇关于人工智能的技术文档。
        
    第一章：机器学习基础
        
    机器学习是人工智能的一个分支，它使计算机能够从数据中学习，
    而不需要显式编程。机器学习算法可以分为三类：
    1. 监督学习
    2. 无监督学习  
    3. 强化学习
        
    深度学习是机器学习的一个子领域，使用多层神经网络来处理
    复杂的模式识别任务。
    """
    
    # 创建处理器
    processor = DocumentProcessor(
        chunk_size=100,
        chunk_overlap=20
    )
    
    # 处理文本
    nodes = processor.chunk_text(sample_text, metadata={"source": "demo"})
    
    print("\n【分块结果】")
    for i, node in enumerate(nodes):
        print(f"\n--- 块 {i+1} ---")
        print(f"内容: {node.text[:100]}...")
        print(f"元数据: {node.metadata}")


if __name__ == "__main__":
    demo_processing()
