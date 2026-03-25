"""
文档处理模块 - RAG系统的第一阶段：文档摄入
    
这个模块是整个RAG系统的入口点，负责处理原始文档，包括：
1. 数据清洗：去除无用字符、标准化文本格式
2. 文档分块：将长文档切分成适合向量检索的小块
    
【文档摄入阶段 - 完整流程图】

┌─────────────────────────────────────────────────────────────────────┐
│                     第一阶段：文档摄入 (Document Ingestion)          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【输入】原始文档文件                                                │
│  └── 支持格式：PDF, TXT, MD, CSV, JSON, DOCX                       │
│                                                                     │
│  【Step 1】SimpleDirectoryReader 加载文档                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  文件路径 → 读取文件内容 → 创建 Document 对象                 │   │
│  │                                                              │   │
│  │  Document对象包含：                                          │   │
│  │  - text: 文档的原始文本内容                                   │   │
│  │  - metadata: 元数据（文件名、页码、创建时间等）                │   │
│  │  - doc_id: 文档的唯一标识符                                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                            ↓                                        │
│  【Step 2】DocumentCleaner 数据清洗                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  清洗操作：                                                   │   │
│  │  1. 去除控制字符（\x00-\x1f等不可见字符）                     │   │
│  │  2. 标准化空白字符（多个空格→单个空格）                        │   │
│  │  3. 标准化换行符（多个换行→最多两个）                          │   │
│  │  4. 去除页码标记（"第1页"、"- 1 -"等）                        │   │
│  │  5. 去除页眉页脚                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                            ↓                                        │
│  【Step 3】DocumentChunker 文档分块                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  分块策略：                                                   │   │
│  │                                                              │   │
│  │  【句子分块 - SentenceSplitter】（默认）                      │   │
│  │  ┌────────────────────────────────────────────────────┐     │   │
│  │  │ [句子1 + 句子2] [句子3 + 句子4] [句子5]             │     │   │
│  │  │      ↑重叠↑           ↑重叠↑                        │     │   │
│  │  │  (chunk_overlap 保证上下文连贯性)                   │     │   │
│  │  └────────────────────────────────────────────────────┘     │   │
│  │                                                              │   │
│  │  【语义分块 - SemanticSplitter】（高级）                     │   │
│  │  ┌────────────────────────────────────────────────────┐     │   │
│  │  │ [语义段1]    [语义段2]      [语义段3]               │     │   │
│  │  │  主题A        主题B           主题C                 │     │   │
│  │  │  (需要embedding模型计算语义相似度)                  │     │   │
│  │  └────────────────────────────────────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                            ↓                                        │
│  【Step 4】生成 TextNode 节点                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  每个TextNode（文本节点）包含：                               │   │
│  │                                                              │   │
│  │  1. id_: 节点的唯一标识符                                     │   │
│  │  2. text: 分块后的文本内容                                    │   │
│  │  3. metadata: 丰富的元数据                                    │   │
│  │     ├── file_name: 来源文件名                                 │   │
│  │     ├── page_number: 页码                                     │   │
│  │     ├── chunk_index: 块在文档中的序号                         │   │
│  │     └── total_chunks: 文档总块数                              │   │
│  │  4. relationships: 节点之间的关系                             │   │
│  │     ├── PREVIOUS: 前一个节点的ID                              │   │
│  │     ├── NEXT: 后一个节点的ID                                  │   │
│  │     └── PARENT: 父节点ID（父子分块时）                        │   │
│  │  5. embedding: 向量嵌入（在第二阶段生成）                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                            ↓                                        │
│  【输出】List[TextNode] - 处理好的文本节点列表                      │
│  └── 这些节点将传入第二阶段进行向量化存储                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

【为什么需要文档分块？】
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│  1. LLM的上下文窗口限制                                           │
│     - GPT-4: 8K/32K tokens                                       │
│     - 即使窗口很大，传入整本文档也会：                             │
│       * 增加API调用成本                                           │
│       * 增加响应延迟                                              │
│       * 包含大量无关信息                                          │
│                                                                   │
│  2. 向量检索的精度                                                │
│     - 向量相似度比较的是语义                                       │
│     - 小块（512 tokens）更容易精确匹配用户查询                     │
│     - 大块（整文档）会稀释语义，降低检索精度                       │
│                                                                   │
│  3. 上下文相关性                                                  │
│     - 用户提问通常只涉及文档的一小部分                             │
│     - 分块后只检索相关部分，避免噪音干扰                           │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
"""

import re
from typing import List, Optional
from pathlib import Path

# ============================================================
# LlamaIndex 核心导入 - 第一阶段文档摄入所需的核心组件
# ============================================================

# Document: LlamaIndex的文档对象，包含文本内容和元数据
# SimpleDirectoryReader: 简单目录读取器，用于从文件/目录加载文档
from llama_index.core import Document, SimpleDirectoryReader

# SentenceSplitter: 句子分割器，按句子边界进行分块（保持语义完整性）
# SemanticSplitter: 语义分割器，基于embedding相似度进行语义分块
# NodeParser: 节点解析器的基类
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitter, NodeParser

# TextNode: 文本节点，是LlamaIndex中最基本的数据单元
# NodeRelationship: 节点关系枚举（PREVIOUS, NEXT, PARENT, CHILD等）
# RelatedNodeInfo: 关联节点的信息
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

# OpenAIEmbedding: OpenAI的嵌入模型，用于将文本转换为向量
# （语义分块时需要用它计算文本块的语义相似度）
from llama_index.embeddings.openai import OpenAIEmbedding


class DocumentCleaner:
    """
    文档清洗器 - 数据预处理的第一步
        
    【职责】对原始文本进行清洗和标准化，提高后续检索的质量
        
    【为什么需要清洗？】
    ┌─────────────────────────────────────────────────────────────┐
    │  原始文档常见问题：                                           │
    │  1. PDF提取的文本包含乱码和控制字符                           │
    │  2. 复制粘贴的内容有多余的空白和换行                          │
    │  3. 页眉页脚、页码等噪音干扰检索                              │
    │  4. 不同来源的文档格式不统一                                  │
    │                                                             │
    │  清洗后的好处：                                               │
    │  1. 提高embedding质量（噪音少→语义更准确）                    │
    │  2. 减少存储空间（去除无用字符）                              │
    │  3. 提升检索精度（减少噪音匹配）                              │
    └─────────────────────────────────────────────────────────────┘
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        【核心方法】清洗文本内容
        
        执行一系列文本标准化操作，去除噪音数据
        
        清洗步骤（按顺序执行）：
        ┌────────────────────────────────────────────────────────┐
        │ Step 1: 去除控制字符                                    │
        │         删除 ASCII 0-31 范围内的不可见控制字符          │
        │         保留换行符(\n)和制表符(\t)                      │
        │                                                        │
        │ Step 2: 标准化空白字符                                  │
        │         多个连续空格 → 单个空格                         │
        │         多个连续制表符 → 单个制表符                     │
        │                                                        │
        │ Step 3: 标准化换行符                                    │
        │         3个以上连续换行 → 2个换行                       │
        │         （保留段落结构，但避免过多空行）                 │
        │                                                        │
        │ Step 4: 去除行首行尾空白                                │
        │         对每一行执行 strip() 操作                       │
        │                                                        │
        │ Step 5: 去除页码标记                                    │
        │         匹配模式："- 1 -", "第 1 页", "1" 等           │
        │         这些标记对检索没有意义                          │
        └────────────────────────────────────────────────────────┘
            
        参数:
            text: 原始文本内容
            
        返回:
            清洗后的文本内容
            
        示例:
            输入: "  Hello   World  \n\n\n\n  第 1 页  "
            输出: "Hello World"
        """
        if not text:
            return ""
            
        # 【Step 1】去除控制字符
        # 正则表达式说明：
        # \x00-\x08: ASCII 0-8 (NUL, SOH, STX, ETX, EOT, ENQ, ACK, BEL, BS)
        # \x0b: 垂直制表符
        # \x0c: 换页符
        # \x0e-\x1f: ASCII 14-31 (各种控制字符)
        # \x7f-\x9f: DEL 和扩展控制字符
        # 这些字符对文本内容没有意义，且可能干扰后续处理
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
            
        # 【Step 2】标准化水平空白字符
        # [ \t]+ 匹配一个或多个空格或制表符
        # 替换为单个空格，保证单词之间只有一个空格
        text = re.sub(r'[ \t]+', ' ', text)
            
        # 【Step 3】标准化垂直空白（换行）
        # \n{3,} 匹配3个或更多连续换行
        # 替换为2个换行，保留段落分隔但去除过多空行
        text = re.sub(r'\n{3,}', '\n\n', text)
            
        # 【Step 4】去除每行的首尾空白
        # 先按换行分割，对每行strip()，再重新组合
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
            
        # 【Step 5】去除页码标记
        # 正则1: [-–—]?\s*\d+\s*[-–—]?\s*$
        #   匹配行尾的页码，如 "- 1 -", "– 2 –", "3" 等
        #   [-–—]? 可选的横线（支持不同类型的破折号）
        #   \s* 零个或多个空白
        #   \d+ 一个或多个数字
        #   $ 行尾锚点
        #   flags=re.MULTILINE 使 ^ 和 $ 匹配每行的开头和结尾
        text = re.sub(r'[-–—]?\s*\d+\s*[-–—]?\s*$', '', text, flags=re.MULTILINE)
        
        # 正则2: ^第\s*\d+\s*页
        #   匹配中文页码格式，如 "第 1 页", "第2页" 等
        text = re.sub(r'^第\s*\d+\s*页', '', text, flags=re.MULTILINE)
            
        return text.strip()
    
    @staticmethod
    def remove_headers_footers(text: str) -> str:
        """
        去除页眉页脚
        
        【页眉页脚的特征】
        - 通常是很短的行（少于5个字符）
        - 只包含数字、横线等简单字符
        - 在每一页重复出现
        
        【处理策略】
        遍历每一行，如果符合页眉页脚特征则跳过
        
        注意：这是一个简单的启发式方法，对于复杂的页眉页脚
        可能需要更复杂的逻辑或机器学习方法
        
        参数:
            text: 原始文本
            
        返回:
            去除页眉页脚后的文本
        """
        lines = text.split('\n')
        cleaned_lines = []
            
        for line in lines:
            line = line.strip()
            
            # 判断是否可能是页眉页脚：
            # 1. 长度小于5个字符（短行）
            # 2. 只包含数字、空白和横线
            # ^[\d\s\-–—]+$ 的含义：
            #   ^ 行首
            #   [\d\s\-–—] 数字、空白、各种横线
            #   + 一个或多个
            #   $ 行尾
            if len(line) < 5 and re.match(r'^[\d\s\-–—]+$', line):
                continue  # 跳过这一行
            
            cleaned_lines.append(line)
            
        return '\n'.join(cleaned_lines)


class DocumentChunker:
    """
    文档分块器 - RAG系统的核心组件
        
    【职责】将长文档切分成合适大小的块（chunks）
        
    【分块的重要性】
    ┌─────────────────────────────────────────────────────────────┐
    │  分块是RAG系统中最关键的设计决策之一                         │
    │                                                             │
    │  分块太大：                                                  │
    │  ✗ 检索精度下降（块内包含过多无关信息）                      │
    │  ✗ LLM处理成本增加                                          │
    │  ✗ 响应速度变慢                                             │
    │                                                             │
    │  分块太小：                                                  │
    │  ✗ 上下文不完整（答案可能跨越多个块）                        │
    │  ✗ 需要检索更多块才能找到完整答案                            │
    │  ✗ 增加向量存储开销                                          │
    │                                                             │
    │  推荐配置：                                                  │
    │  • chunk_size: 256-512 tokens（约500-1000中文字符）         │
    │  • chunk_overlap: 10-20% 的 chunk_size                      │
    └─────────────────────────────────────────────────────────────┘
        
    【支持的两种分块策略】
        
    1. 句子分块（SentenceSplitter）- 默认策略
       ┌──────────────────────────────────────────────────────┐
       │  原理：按句子边界进行切分，保持句子完整性               │
       │                                                       │
       │  优点：                                                │
       │  + 保持语义完整性（不会在句子中间切断）                 │
       │  + 不需要额外的embedding调用                           │
       │  + 速度快，成本低                                      │
       │                                                       │
       │  缺点：                                                │
       │  - 块大小不均匀                                        │
       │  - 可能忽略段落间的语义边界                             │
       │                                                       │
       │  适用场景：                                            │
       │  • 叙述性文本（文章、博客）                             │
       │  • 格式统一的文档                                      │
       │  • 成本敏感的应用                                      │
       └──────────────────────────────────────────────────────┘
        
    2. 语义分块（SemanticSplitter）- 高级策略
       ┌──────────────────────────────────────────────────────┐
       │  原理：计算相邻句子的embedding相似度，                  │
       │        在相似度"断崖"处进行切分                        │
       │                                                       │
       │  工作流程：                                            │
       │  句子1 ─→ embedding ─┐                                │
       │                      ├─→ 计算余弦相似度                │
       │  句子2 ─→ embedding ─┘    ↓                           │
       │                       相似度 > 阈值?                   │
       │                      ↓         ↓                      │
       │                   合并到     开始新块                   │
       │                   当前块                               │
       │                                                       │
       │  优点：                                                │
       │  + 语义完整性最好                                      │
       │  + 自动识别主题边界                                    │
       │  + 块内内容高度相关                                    │
       │                                                       │
       │  缺点：                                                │
       │  - 需要调用embedding API（成本增加）                   │
       │  - 处理速度较慢                                        │
       │  - 块大小不可控                                        │
       │                                                       │
       │  适用场景：                                            │
       │  • 高质量检索场景                                      │
       │  • 主题变化明显的文档                                  │
       │  • 对检索质量要求极高                                  │
       └──────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_semantic_splitting: bool = False,
        embed_model: Optional[OpenAIEmbedding] = None
    ):
        """
        初始化文档分块器
        
        【参数详解】
            
        参数:
            chunk_size: int = 512
                每个块的最大token数量
                - 1 token ≈ 0.75 个英文单词
                - 1 token ≈ 1.5 个中文字符
                - 512 tokens ≈ 768 个中文字符
                - 推荐范围：256-1024
                
            chunk_overlap: int = 50
                相邻块之间的重叠token数
                - 重叠是为了保证跨块信息的上下文连贯性
                - 例如：块1的最后50 tokens 和 块2的前50 tokens 相同
                - 推荐范围：chunk_size 的 10-20%
                
                【重叠的作用示意图】
                ┌──────────────────────────────────────┐
                │ 块1: [A B C D E F G H I J]           │
                │                          ↑ 重叠部分  │
                │ 块2:              [F G H I J K L M]  │
                │                          ↑          │
                └──────────────────────────────────────┘
                如果用户问的问题答案在 F-H 附近，
                无论检索到块1还是块2都能获得完整上下文
                
            use_semantic_splitting: bool = False
                是否使用语义分块
                - False: 使用句子分块（推荐，性价比高）
                - True: 使用语义分块（高质量，高成本）
                
            embed_model: Optional[OpenAIEmbedding] = None
                嵌入模型实例（仅语义分块需要）
                - 用于计算文本的语义向量
                - 默认为 None，使用句子分块时不需要
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_splitting = use_semantic_splitting
            
        if use_semantic_splitting:
            # 【语义分块模式】
            if embed_model is None:
                raise ValueError("语义分块需要提供嵌入模型（embed_model参数）")
            
            # ★★★【核心调用】创建LlamaIndex的语义分割器★★★
            # SemanticSplitter 参数说明：
            # - embed_model: 嵌入模型，用于将文本转换为向量
            # - buffer_size: 缓冲区大小，决定比较多少个相邻句子
            #   buffer_size=1 表示比较当前句子和下一个句子
            # - breakpoint_percentile_threshold: 断点百分位阈值
            #   相似度低于此百分位的点会被认为是主题边界
            #   95 表示只有最不相似的5%的点会被认为是断点
            self.splitter = SemanticSplitter(
                embed_model=embed_model,
                buffer_size=1,
                breakpoint_percentile_threshold=95
            )
        else:
            # 【句子分块模式】（默认）
            
            # ★★★【核心调用】创建LlamaIndex的句子分割器★★★
            # SentenceSplitter 是 LlamaIndex 提供的智能分块器
            # 它会尽量在句子边界处切分，而不是在单词中间
            self.splitter = SentenceSplitter(
                # chunk_size: 每个块的最大token数
                chunk_size=chunk_size,
                
                # chunk_overlap: 相邻块之间的重叠token数
                # 这确保了跨块边界的上下文不会丢失
                chunk_overlap=chunk_overlap,
                
                # paragraph_separator: 段落分隔符
                # 分块器会优先在段落边界处切分
                paragraph_separator="\n\n",
                
                # secondary_chunking_regex: 二级分块正则表达式
                # 当需要在段落内部切分时，使用此正则确定切分点
                # [^,.;。？！]+ 匹配非标点字符
                # [,.;。？！]? 可选的句末标点
                # 效果：尽量在句号、问号、感叹号处切分
                secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?"
            )
    
    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        """
        【核心方法】将Document列表分块成TextNode列表
        
        这是从Document到Node的关键转换步骤
        
        【分块过程详解】
        ┌─────────────────────────────────────────────────────────────┐
        │  输入: List[Document]                                       │
        │  └── 每个Document包含完整的文档文本                          │
        │                                                             │
        │  处理: splitter.get_nodes_from_documents()                  │
        │  ┌───────────────────────────────────────────────────────┐ │
        │  │  1. 遍历每个Document                                   │ │
        │  │  2. 根据分块策略切分文本                                │ │
        │  │  3. 为每个块创建TextNode                               │ │
        │  │  4. 设置节点之间的关系（PREVIOUS, NEXT）               │ │
        │  │  5. 继承Document的元数据                               │ │
        │  └───────────────────────────────────────────────────────┘ │
        │                                                             │
        │  输出: List[TextNode]                                       │
        │  └── 每个TextNode是一个可检索的文本块                        │
        └─────────────────────────────────────────────────────────────┘
        
        【TextNode的完整结构】
        ┌─────────────────────────────────────────────────────────────┐
        │  TextNode {                                                  │
        │    id_: "doc_abc_chunk_0",        # 唯一标识                │
        │    text: "分块后的文本内容...",    # 文本内容                │
        │                                                              │
        │    metadata: {                     # 元数据字典              │
        │      "file_name": "report.pdf",   # 来源文件                │
        │      "page_number": 1,            # 页码                    │
        │      "chunk_index": 0,            # 块序号（本方法添加）     │
        │      "total_chunks": 10,          # 总块数（本方法添加）     │
        │      ...                          # 其他元数据               │
        │    },                                                        │
        │                                                              │
        │    relationships: {                # 节点关系                │
        │      PREVIOUS: RelatedNodeInfo(node_id="prev_id"),          │
        │      NEXT: RelatedNodeInfo(node_id="next_id"),              │
        │    },                                                        │
        │                                                              │
        │    embedding: None,               # 向量（第二阶段生成）     │
        │  }                                                          │
        └─────────────────────────────────────────────────────────────┘
            
        参数:
            documents: LlamaIndex Document对象列表
            
        返回:
            TextNode列表，每个节点代表一个可检索的文本块
        """
        # ★★★【核心调用】LlamaIndex的分块方法★★★
        # get_nodes_from_documents() 是 NodeParser 的核心方法
        # 它会：
        # 1. 遍历每个文档
        # 2. 根据配置的分块策略切分文本
        # 3. 为每个块创建 TextNode 对象
        # 4. 自动设置节点之间的 PREVIOUS/NEXT 关系
        # 5. 继承原文档的元数据
        nodes = self.splitter.get_nodes_from_documents(documents)
            
        # 为每个节点添加额外的元数据
        # 这些元数据在后续的检索和展示中很有用
        for i, node in enumerate(nodes):
            # chunk_index: 当前块在文档中的序号（从0开始）
            node.metadata["chunk_index"] = i
            # total_chunks: 文档被分成了多少块
            # 这有助于LLM理解上下文的完整性
            node.metadata["total_chunks"] = len(nodes)
                
        return nodes
    
    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[TextNode]:
        """
        【便捷方法】将单个文本字符串直接分块
        
        这是一个便捷方法，封装了 text → Document → TextNode 的转换过程
        
        使用场景：
        - 直接处理内存中的文本（不需要从文件加载）
        - 快速测试分块效果
        - 处理用户输入的文本
        
        参数:
            text: 要分块的文本内容
            metadata: 额外的元数据（会附加到每个节点）
            
        返回:
            TextNode列表
            
        示例:
            chunker = DocumentChunker(chunk_size=200)
            nodes = chunker.chunk_text(
                "这是一段长文本...",
                metadata={"source": "user_input", "timestamp": "2024-01-01"}
            )
        """
        # 步骤1: 将文本包装成LlamaIndex的Document对象
        # Document是LlamaIndex中表示文档的基本数据结构
        document = Document(text=text, metadata=metadata or {})
        
        # 步骤2: 调用chunk_documents进行分块
        return self.chunk_documents([document])


class DocumentProcessor:
    """
    文档处理器 - 整合文档摄入的完整流程
        
    【职责】整合 DocumentCleaner 和 DocumentChunker，提供一站式文档处理
        
    【完整处理流程】
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  process_file() 完整流程：                                   │
    │                                                             │
    │  文件路径                                                   │
    │     ↓                                                       │
    │  load_documents() ─→ 加载文件内容                           │
    │     ↓                                                       │
    │  process_documents() ─→ 清洗 + 分块                         │
    │     │                                                       │
    │     ├── DocumentCleaner.clean_text()                        │
    │     ├── DocumentCleaner.remove_headers_footers()            │
    │     └── DocumentChunker.chunk_documents()                   │
    │     ↓                                                       │
    │  List[TextNode] ─→ 输出，准备进入第二阶段                    │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
        
    【使用示例】
    ```python
    # 创建处理器
    processor = DocumentProcessor(
        chunk_size=512,
        chunk_overlap=50
    )
    
    # 处理单个文件
    nodes = processor.process_file("./documents/report.pdf")
    
    # 处理整个目录
    nodes = processor.process_file("./documents/")
    
    # 处理文本字符串
    nodes = processor.process_text("长文本内容...")
    ```
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
            chunk_size: 分块大小（token数）
            chunk_overlap: 分块重叠（token数）
            use_semantic_splitting: 是否使用语义分块
            embed_model: 嵌入模型（语义分块需要）
        """
        # 创建文档清洗器实例
        self.cleaner = DocumentCleaner()
        
        # 创建文档分块器实例
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_semantic_splitting=use_semantic_splitting,
            embed_model=embed_model
        )
    
    def load_documents(self, file_path: str) -> List[Document]:
        """
        【加载方法】从文件或目录加载文档
        
        使用LlamaIndex的SimpleDirectoryReader加载文档
        
        【支持的文件格式】
        ┌────────────────────────────────────────────────────────┐
        │  格式    │ 扩展名      │ 说明                          │
        ├──────────┼─────────────┼───────────────────────────────┤
        │  PDF     │ .pdf        │ 需要PyPDF2或pdfplumber        │
        │  文本    │ .txt        │ 纯文本文件                    │
        │  Markdown│ .md         │ Markdown文档                  │
        │  CSV     │ .csv        │ 表格数据                      │
        │  JSON    │ .json       │ JSON数据                      │
        │  Word    │ .docx       │ 需要python-docx               │
        │  HTML    │ .html       │ 需要BeautifulSoup             │
        │  EPUB    │ .epub       │ 电子书格式                    │
        └────────────────────────────────────────────────────────┘
        
        【SimpleDirectoryReader的工作原理】
        1. 扫描指定路径下的文件
        2. 根据文件扩展名选择合适的解析器
        3. 读取文件内容并提取文本
        4. 创建Document对象，包含：
           - text: 提取的文本内容
           - metadata: 文件名、路径等信息
           - doc_id: 自动生成的唯一ID
            
        参数:
            file_path: 文件路径或目录路径
            
        返回:
            Document对象列表
            
        示例:
            # 加载单个文件
            docs = processor.load_documents("./report.pdf")
            
            # 加载整个目录
            docs = processor.load_documents("./documents/")
        """
        path = Path(file_path)
            
        if path.is_file():
            # 【情况1】加载单个文件
            
            # ★★★【核心调用】使用SimpleDirectoryReader加载单个文件★★★
            # SimpleDirectoryReader 是 LlamaIndex 提供的通用文档加载器
            # input_files 参数接受文件路径列表
            documents = SimpleDirectoryReader(
                input_files=[str(path)]
            ).load_data()
            # load_data() 方法执行实际的文件读取和文本提取
            
        else:
            # 【情况2】加载整个目录
            
            # ★★★【核心调用】使用SimpleDirectoryReader加载目录★★★
            documents = SimpleDirectoryReader(
                str(path),                    # 目录路径
                recursive=True,               # 递归扫描子目录
                required_exts=[".pdf", ".txt", ".md", ".csv", ".json"]  # 只加载这些扩展名的文件
            ).load_data()
            
        return documents
    
    def process_documents(
        self,
        documents: List[Document],
        clean_text: bool = True
    ) -> List[TextNode]:
        """
        【核心方法】处理文档列表：清洗 + 分块
        
        这是文档处理的核心逻辑，将原始Document转换为可检索的TextNode
        
        【处理流程图】
        ┌─────────────────────────────────────────────────────────────┐
        │  输入: List[Document]                                       │
        │                                                             │
        │  for each document:                                         │
        │    ┌─────────────────────────────────────────────────────┐ │
        │    │ 1. 提取原始文本                                       │ │
        │    │    doc.text                                          │ │
        │    │                                                      │ │
        │    │ 2. 清洗文本                                           │ │
        │    │    cleaner.clean_text(text)                          │ │
        │    │    cleaner.remove_headers_footers(text)              │ │
        │    │                                                      │ │
        │    │ 3. 创建清洗后的Document                               │ │
        │    │    保留原有的metadata和doc_id                         │ │
        │    └─────────────────────────────────────────────────────┘ │
        │                                                             │
        │  批量分块:                                                  │
        │    chunker.chunk_documents(cleaned_documents)              │
        │                                                             │
        │  输出: List[TextNode]                                       │
        └─────────────────────────────────────────────────────────────┘
            
        参数:
            documents: 原始Document列表
            clean_text: 是否进行文本清洗（默认True）
            
        返回:
            处理后的TextNode列表
        """
        if clean_text:
            cleaned_documents = []
            
            for doc in documents:
                # 【Step 1】清洗文本内容
                # 去除控制字符、多余空白、页码等噪音
                cleaned_text = self.cleaner.clean_text(doc.text)
                
                # 【Step 2】去除页眉页脚
                cleaned_text = self.cleaner.remove_headers_footers(cleaned_text)
                    
                # 【Step 3】创建新的Document对象
                # 重要：保留原有的metadata和doc_id
                # 这样后续可以追溯每个块的来源
                cleaned_doc = Document(
                    text=cleaned_text,
                    metadata=doc.metadata.copy(),  # 复制元数据
                    doc_id=doc.doc_id              # 保留文档ID
                )
                cleaned_documents.append(cleaned_doc)
                
            documents = cleaned_documents
            
        # 【Step 4】执行分块
        # ★★★【核心调用】将清洗后的文档进行分块★★★
        # 这是最关键的一步，将长文档转换为可检索的文本块
        nodes = self.chunker.chunk_documents(documents)
            
        return nodes
    
    def process_file(
        self,
        file_path: str,
        clean_text: bool = True
    ) -> List[TextNode]:
        """
        【主入口方法】完整的文件处理流程
        
        这是对外暴露的主要接口，封装了完整的处理流程：
        加载 → 清洗 → 分块
        
        【调用流程】
        file_path → load_documents() → process_documents() → TextNode[]
        
        参数:
            file_path: 文件路径或目录路径
            clean_text: 是否清洗文本
            
        返回:
            处理后的TextNode列表，可直接传入向量存储
            
        使用示例:
            processor = DocumentProcessor(chunk_size=512)
            nodes = processor.process_file("./documents/report.pdf")
            
            # nodes 现在可以传入向量存储
            vector_store.add_nodes(nodes)
        """
        # 【Step 1】加载文档
        documents = self.load_documents(file_path)
        print(f"✓ 加载了 {len(documents)} 个文档")
            
        # 【Step 2】处理文档（清洗 + 分块）
        nodes = self.process_documents(documents, clean_text=clean_text)
        print(f"✓ 生成了 {len(nodes)} 个节点")
            
        # 【Step 3】打印统计信息
        self._print_chunk_stats(nodes)
            
        return nodes
    
    def _print_chunk_stats(self, nodes: List[TextNode]) -> None:
        """
        【辅助方法】打印分块统计信息
        
        用于调试和监控，了解分块的质量
        
        参数:
            nodes: TextNode列表
        """
        if not nodes:
            return
            
        # 计算每个块的文本长度
        lengths = [len(node.text) for node in nodes]
        
        # 计算统计指标
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)
            
        print(f"\n【分块统计】")
        print(f"  - 总块数: {len(nodes)}")
        print(f"  - 平均长度: {avg_length:.0f} 字符")
        print(f"  - 最短块: {min_length} 字符")
        print(f"  - 最长块: {max_length} 字符")
        
        # 【统计信息的意义】
        # - 如果最短块和最长块差距很大，说明分块不均匀
        # - 平均长度应该接近 chunk_size 对应的字符数
        # - 最短块如果太短（<50字符），可能需要调整分块参数


def demo_processing():
    """
    演示文档处理流程的完整示例
    
    展示如何使用DocumentProcessor处理文本
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
        chunk_size=100,      # 小块大小，用于演示
        chunk_overlap=20     # 20 tokens 的重叠
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
