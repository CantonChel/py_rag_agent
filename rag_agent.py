"""
Agent检索工具模块

这个模块是整个RAG系统的核心，展示了如何将检索能力封装成Agent工具

【Agent工作流程图】

用户问题
    ↓
┌─────────────────────────────────────┐
│           Agent (LLM)               │
│                                     │
│  1. 分析问题：需要什么信息？        │
│  2. 决策：是否需要检索？            │
│  3. 调用：knowledge_search 工具     │
│  4. 推理：基于检索结果生成答案      │
└─────────────────────────────────────┘
                    ↓
         ┌─────────────────┐
         │  检索工具 (Tool) │
         │                 │
         │  1. 接收查询     │
         │  2. 向量检索     │
         │  3. 返回结果     │
         └─────────────────┘
                    ↓
           返回检索到的内容

【工具化 vs 工作流的区别】

工作流模式（Workflow）：
  用户问题 → 固定步骤：检索 → 生成答案
  特点：流程固定，每次都会先检索再回答

工具化模式（Tool-based Agent）：
  用户问题 → Agent自主决策 → 可能检索/可能不检索
  特点：Agent有自主权，根据问题决定是否需要检索
  
本实现采用【工具化模式】，Agent可以：
- 直接回答简单问题（不调用检索工具）
- 检索后回答需要知识库的问题
- 多次检索以获取更全面的信息
"""

from typing import List, Optional, Dict, Any, Type
from pydantic import BaseModel, Field

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool, ToolMetadata
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI

from vector_store import VectorStoreManager, RetrieverEngine


class KnowledgeSearchInput(BaseModel):
    """
    知识检索工具的输入参数模型
    
    使用Pydantic进行参数验证和文档生成
    """
    query: str = Field(
        ...,
        description="要在知识库中搜索的查询文本。应该是清晰、具体的问题或关键词。"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="返回的相关文档数量，范围1-10，默认5"
    )


class KnowledgeSearchTool(BaseTool):
    """
    知识检索工具 - Agent的核心工具
    
    【工具设计原则】
    1. 单一职责：只负责知识库检索
    2. 清晰的输入输出：让LLM理解如何使用
    3. 详细的描述：帮助Agent判断何时使用
    
    【Agent如何决定使用这个工具？】
    LLM会根据：
    1. 工具的name和description
    2. 用户问题的内容
    3. 上下文信息
    
    来判断是否需要调用这个工具
    """
    
    # 类属性定义
    retriever_engine: RetrieverEngine = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, retriever_engine: RetrieverEngine):
        """
        初始化知识检索工具
        
        参数:
            retriever_engine: 检索引擎实例
        """
        super().__init__(
            retriever_engine=retriever_engine,
            metadata=ToolMetadata(
                name="knowledge_search",
                description=(
                    "搜索知识库以获取相关信息。"
                    "当用户的问题涉及到知识库中的文档内容时使用此工具。"
                    "输入应该是一个清晰的搜索查询。"
                    "返回与查询最相关的文档片段。"
                    "【使用场景】用户询问具体信息、事实、概念解释等需要参考知识库的问题。"
                    "【不使用场景】闲聊、简单问候、通用知识问题（如'1+1等于几'）。"
                )
            )
        )
    
    def __call__(self, *args, **kwargs) -> Any:
        """使工具可调用"""
        return self.call(*args, **kwargs)
    
    def call(self, query: str, top_k: int = 5) -> str:
        """
        执行检索并返回结果
        
        【返回格式设计】
        返回格式化的文本，包含：
        1. 检索到的内容
        2. 相关度分数
        3. 来源信息
        
        这样Agent可以理解检索结果并基于此生成答案
        
        参数:
            query: 搜索查询
            top_k: 返回结果数量
            
        返回:
            格式化的检索结果文本
        """
        results = self.retriever_engine.search(query, top_k=top_k)
        
        if not results:
            return "在知识库中未找到相关信息。请尝试其他查询或直接回答用户问题。"
        
        output_parts = ["以下是知识库中找到的相关信息：\n"]
        
        for i, result in enumerate(results, 1):
            output_parts.append(
                f"【文档片段 {i}】(相关度: {result['score']:.2%})\n"
                f"来源: {result['metadata'].get('file_name', '未知来源')}\n"
                f"内容:\n{result['text']}\n"
            )
        
        return "\n".join(output_parts)


class RAGAgent:
    """
    RAG智能代理 - 整合所有组件的入口类
    
    【Agent工作原理】
    使用ReAct (Reasoning + Acting) 模式：
    1. Thought: 分析用户问题
    2. Action: 决定是否调用工具
    3. Observation: 观察工具返回结果
    4. Final Answer: 基于所有信息生成最终答案
    
    【完整交互流程示例】
    
    用户: "文档中提到的机器学习有哪些类型？"
    
    Agent思考: 用户询问文档中关于机器学习的内容，需要检索知识库
    
    Agent调用: knowledge_search(query="机器学习类型")
    
    工具返回: [检索到的相关文档片段...]
    
    Agent思考: 基于检索结果，我可以总结出三种机器学习类型...
    
    Agent回答: 根据文档内容，机器学习分为监督学习、无监督学习和强化学习三种类型...
    """
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm: Optional[LLM] = None,
        verbose: bool = True
    ):
        """
        初始化RAG Agent
        
        参数:
            vector_store_manager: 向量存储管理器
            llm: 大语言模型实例，默认使用GPT-4o-mini
            verbose: 是否打印详细日志
        """
        self.vsm = vector_store_manager
        self.verbose = verbose
        
        # 初始化LLM
        # 【核心组件】LLM是Agent的大脑，负责推理和决策
        self.llm = llm or OpenAI(
            model="gpt-4o-mini",
            temperature=0
        )
        
        # 创建检索引擎
        self.retriever_engine = RetrieverEngine(self.vsm)
        
        # 创建检索工具
        # 【关键步骤】将检索能力封装成工具供Agent调用
        self.search_tool = KnowledgeSearchTool(self.retriever_engine)
        
        # 创建ReAct Agent
        # 【核心】Agent使用ReAct模式，可以自主决定何时调用工具
        self.agent = ReActAgent.from_tools(
            tools=[self.search_tool],
            llm=self.llm,
            verbose=verbose,
            max_iterations=10,
            # Agent的系统提示词
            system_prompt=self._build_system_prompt()
        )
    
    def _build_system_prompt(self) -> str:
        """
        构建Agent的系统提示词
        
        【系统提示词的作用】
        1. 定义Agent的角色和能力
        2. 指导Agent如何使用工具
        3. 设定回答风格和限制
        
        返回:
            系统提示词文本
        """
        return """你是一个智能知识库助手，可以访问知识库进行信息检索。

【你的能力】
1. 搜索知识库获取相关信息
2. 基于检索结果回答用户问题
3. 如果知识库中没有相关信息，坦诚告知用户

【使用工具的原则】
1. 当用户问题涉及知识库中的具体内容时，使用knowledge_search工具检索
2. 如果是通用问题或简单对话，可以直接回答，不需要检索
3. 检索时使用清晰、具体的查询词
4. 可以多次检索以获取更全面的信息

【回答原则】
1. 基于检索到的内容回答，不要编造信息
2. 如果检索结果不足以回答问题，坦诚说明
3. 回答要清晰、有条理
4. 适当引用来源信息

请始终用中文回答用户问题。
"""
    
    def chat(self, message: str) -> str:
        """
        与Agent进行对话
        
        【这是外部调用的主要接口】
        
        参数:
            message: 用户消息
            
        返回:
            Agent的回复
        """
        # 【核心调用】Agent处理用户消息
        # Agent会自动：
        # 1. 分析问题
        # 2. 决定是否调用检索工具
        # 3. 基于结果生成回答
        response = self.agent.chat(message)
        
        return str(response)
    
    def chat_stream(self, message: str):
        """
        流式对话（逐步返回响应）
        
        参数:
            message: 用户消息
            
        Yields:
            响应的每个部分
        """
        response = self.agent.stream_chat(message)
        for token in response.response_gen:
            yield token
    
    def reset_memory(self):
        """重置Agent的对话记忆"""
        self.agent.reset()


class AgentWithHistory:
    """
    带对话历史的Agent封装
    
    支持多轮对话，保持上下文连贯性
    """
    
    def __init__(self, rag_agent: RAGAgent):
        """
        初始化
        
        参数:
            rag_agent: RAG Agent实例
        """
        self.agent = rag_agent
        self.conversation_history: List[Dict[str, str]] = []
    
    def chat(self, message: str) -> str:
        """
        带历史记录的对话
        
        参数:
            message: 用户消息
            
        返回:
            Agent回复
        """
        # 记录用户消息
        self.conversation_history.append({"role": "user", "content": message})
        
        # 获取Agent回复
        response = self.agent.chat(message)
        
        # 记录Agent回复
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.conversation_history
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        self.agent.reset_memory()


def demo_agent():
    """
    演示Agent的使用
    """
    from document_processor import DocumentProcessor
    
    # 准备示例文档
    sample_text = """
    Python编程语言指南
    
    Python是一种高级编程语言，由Guido van Rossum于1991年创建。
    Python的设计哲学强调代码的可读性和简洁性。
    
    主要特点：
    1. 简洁易学的语法
    2. 丰富的标准库
    3. 跨平台兼容性
    4. 支持多种编程范式
    
    应用领域：
    - Web开发
    - 数据科学
    - 人工智能
    - 自动化脚本
    """
    
    # 初始化组件
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
    nodes = processor.chunk_text(sample_text, metadata={"source": "demo"})
    
    vsm = VectorStoreManager(
        persist_dir="./demo_agent_chroma",
        collection_name="demo_agent"
    )
    vsm.add_nodes(nodes)
    
    # 创建Agent
    rag_agent = RAGAgent(vsm, verbose=True)
    
    # 测试对话
    print("\n" + "="*50)
    print("【测试1】需要检索的问题")
    print("="*50)
    response = rag_agent.chat("Python有哪些主要特点？")
    print(f"\n回答: {response}")
    
    print("\n" + "="*50)
    print("【测试2】不需要检索的问题")
    print("="*50)
    response = rag_agent.chat("你好，今天天气怎么样？")
    print(f"\n回答: {response}")
    
    # 清理
    vsm.delete_collection()


if __name__ == "__main__":
    demo_agent()
