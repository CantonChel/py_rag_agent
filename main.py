"""
RAG Agent 系统主程序入口

这个文件展示了完整的RAG系统使用流程

【完整系统架构图】

┌─────────────────────────────────────────────────────────────┐
│                     RAG 知识库 Agent 系统                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  文档处理    │ -> │  向量存储    │ -> │  检索引擎    │     │
│  │             │    │             │    │             │      │
│  │ - 加载文档   │    │ - 向量化    │    │ - 相似搜索   │     │
│  │ - 数据清洗   │    │ - 存储      │    │ - 结果排序   │     │
│  │ - 分块处理   │    │ - 索引      │    │ - 过滤      │      │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                           ↑                                 │
│                           │                                 │
│                    ┌──────┴──────┐                         │
│                    │  Agent工具   │                         │
│                    │             │                          │
│                    │ knowledge_  │                          │
│                    │ search      │                          │
│                    └──────┬──────┘                         │
│                           │                                 │
│                    ┌──────┴──────┐                         │
│                    │    Agent    │                         │
│                    │   (ReAct)   │                         │
│                    │             │                          │
│                    │ - 分析问题   │                         │
│                    │ - 决策调用   │                         │
│                    │ - 生成回答   │                         │
│                    └─────────────┘                         │
│                           ↑                                 │
│                           │                                 │
│                    ┌──────┴──────┐                         │
│                    │    用户      │                         │
│                    └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘

【使用流程】
1. 初始化系统：创建RAGSystem实例
2. 添加文档：ingest_documents() 方法
3. 对话查询：chat() 方法
"""

import os
import sys
from typing import Optional, List
from pathlib import Path

from dotenv import load_dotenv

from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_agent import RAGAgent, AgentWithHistory


class RAGSystem:
    """
    RAG系统 - 整合所有组件的统一接口
    
    【这是系统的主入口类】
    封装了所有功能，提供简单的API
    """
    
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "knowledge_base",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        verbose: bool = True
    ):
        """
        初始化RAG系统
        
        参数:
            persist_dir: 向量数据库存储目录
            collection_name: 集合名称
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小
            llm_model: 使用的LLM模型
            embedding_model: 使用的嵌入模型
            verbose: 是否显示详细日志
        """
        self.verbose = verbose
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        if verbose:
            print("正在初始化RAG系统...")
        
        # 初始化文档处理器
        # 【组件1】负责文档的加载、清洗和分块
        self.doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # 初始化向量存储
        # 【组件2】负责向量化和存储
        self.vector_store = VectorStoreManager(
            persist_dir=persist_dir,
            collection_name=collection_name
        )
        
        # 初始化Agent
        # 【组件3】负责智能检索和回答
        self.agent = RAGAgent(
            vector_store_manager=self.vector_store,
            llm=None,
            verbose=verbose
        )
        
        # 带历史的Agent封装
        self.agent_with_history = AgentWithHistory(self.agent)
        
        if verbose:
            print("✓ RAG系统初始化完成")
            print(f"  - 向量存储位置: {persist_dir}")
            print(f"  - 集合名称: {collection_name}")
            print(f"  - 当前文档数: {self.vector_store.get_collection_count()}")
    
    def ingest_documents(
        self,
        file_path: str,
        clean_text: bool = True
    ) -> int:
        """
        导入文档到知识库
        
        【完整的文档处理流程】
        
        步骤1: 加载文档
        ├── 支持PDF、TXT、MD等格式
        └── 递归读取目录中的文件
        
        步骤2: 数据清洗
        ├── 去除控制字符
        ├── 标准化空白
        ├── 去除页眉页脚
        └── 清理页码等噪音
        
        步骤3: 文档分块
        ├── 按句子边界切分
        ├── 保持语义完整性
        ├── 块之间有重叠
        └── 添加元数据
        
        步骤4: 向量化存储
        ├── 文本转向量(1536维)
        ├── 存入ChromaDB
        └── 建立向量索引
        
        参数:
            file_path: 文件或目录路径
            clean_text: 是否清洗文本
            
        返回:
            生成的节点数量
        """
        if self.verbose:
            print(f"\n开始导入文档: {file_path}")
        
        # 处理文档
        nodes = self.doc_processor.process_file(
            file_path,
            clean_text=clean_text
        )
        
        # 存入向量库
        self.vector_store.add_nodes(nodes)
        
        if self.verbose:
            print(f"\n✓ 文档导入完成")
            print(f"  - 总节点数: {len(nodes)}")
            print(f"  - 向量库总文档数: {self.vector_store.get_collection_count()}")
        
        return len(nodes)
    
    def chat(self, message: str) -> str:
        """
        与Agent对话
        
        【Agent处理流程】
        
        用户输入: "文档中提到了哪些机器学习算法？"
                ↓
        Agent分析: 这个问题涉及文档内容，需要检索
                ↓
        工具调用: knowledge_search(query="机器学习算法")
                ↓
        向量检索: 在向量库中搜索相似内容
                ↓
        返回结果: [相关文档片段1, 片段2, ...]
                ↓
        Agent推理: 基于检索结果整理答案
                ↓
        最终回答: "根据文档内容，提到的机器学习算法包括..."
        
        参数:
            message: 用户消息
            
        返回:
            Agent回复
        """
        return self.agent_with_history.chat(message)
    
    def chat_stream(self, message: str):
        """
        流式对话
        
        参数:
            message: 用户消息
            
        Yields:
            响应token
        """
        for token in self.agent.chat_stream(message):
            yield token
    
    def reset_conversation(self):
        """重置对话历史"""
        self.agent_with_history.clear_history()
        if self.verbose:
            print("✓ 对话历史已重置")
    
    def get_stats(self) -> dict:
        """
        获取系统统计信息
        
        返回:
            统计信息字典
        """
        return {
            "persist_dir": self.persist_dir,
            "collection_name": self.collection_name,
            "total_documents": self.vector_store.get_collection_count()
        }
    
    def clear_knowledge_base(self):
        """清空知识库"""
        self.vector_store.delete_collection()
        # 重新初始化
        self.vector_store = VectorStoreManager(
            persist_dir=self.persist_dir,
            collection_name=self.collection_name
        )
        self.agent = RAGAgent(
            vector_store_manager=self.vector_store,
            verbose=self.verbose
        )
        self.agent_with_history = AgentWithHistory(self.agent)
        
        if self.verbose:
            print("✓ 知识库已清空")


def interactive_mode(rag_system: RAGSystem):
    """
    交互式对话模式
    
    参数:
        rag_system: RAG系统实例
    """
    print("\n" + "="*60)
    print("RAG 知识库 Agent - 交互模式")
    print("="*60)
    print("命令:")
    print("  /stats  - 显示系统统计")
    print("  /reset  - 重置对话")
    print("  /clear  - 清空知识库")
    print("  /exit   - 退出程序")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input.startswith("/"):
                cmd = user_input.lower()
                if cmd == "/stats":
                    stats = rag_system.get_stats()
                    print(f"\n系统统计: {stats}")
                elif cmd == "/reset":
                    rag_system.reset_conversation()
                elif cmd == "/clear":
                    confirm = input("确定要清空知识库吗？(y/n): ")
                    if confirm.lower() == "y":
                        rag_system.clear_knowledge_base()
                elif cmd == "/exit":
                    print("\n再见！")
                    break
                else:
                    print(f"未知命令: {cmd}")
                continue
            
            # 对话
            print("\nAgent: ", end="")
            response = rag_system.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def main():
    """
    主函数 - 演示完整的使用流程
    """
    # 加载环境变量
    load_dotenv()
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("错误: 请设置OPENAI_API_KEY环境变量")
        print("可以创建.env文件: OPENAI_API_KEY=your-key-here")
        sys.exit(1)
    
    # 创建RAG系统
    rag = RAGSystem(
        persist_dir="./knowledge_base",
        collection_name="my_knowledge",
        chunk_size=512,
        chunk_overlap=50,
        verbose=True
    )
    
    # 检查是否有文档需要导入
    docs_dir = Path("./documents")
    if docs_dir.exists() and docs_dir.is_dir():
        print(f"\n发现文档目录: {docs_dir}")
        rag.ingest_documents(str(docs_dir))
    
    # 启动交互模式
    interactive_mode(rag)


if __name__ == "__main__":
    main()
