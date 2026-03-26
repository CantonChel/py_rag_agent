"""
基于PgVector的RAG Agent模块

这个模块连接PgVectorStore和LLM，实现真正的RAG对话功能

【工作流程】
1. 用户提问
2. 对问题进行向量化
3. 在PgVector中检索相关文档块
4. 将检索结果作为上下文，调用LLM生成回答
5. 返回回答和来源信息
"""

from typing import Iterator, List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import httpx
from openai import OpenAI

from config import config
from pgvector_store import get_pgvector_store, ChunkRecord


@dataclass
class RAGResponse:
    """RAG响应数据结构"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    reasoning: Optional[str] = None


class EmbeddingService:
    """
    Embedding服务 - 将文本转换为向量
    
    使用配置的Embedding模型进行向量化
    """
    
    def __init__(self):
        self.api_key = config.embedding.api_key
        self.base_url = config.embedding.base_url
        self.model_name = config.embedding.model_name
        
        # 不继承系统代理环境变量，避免 SOCKS 代理缺少 socksio 时报错
        self._http_client = httpx.Client(trust_env=False, timeout=60.0)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=self._http_client
        )
    
    def embed_query(self, text: str) -> List[float]:
        """
        将文本转换为向量
        
        参数:
            text: 要向量化的文本
            
        返回:
            向量列表
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding


class RerankService:
    """
    重排序服务 - 对检索结果进行精细排序
    
    使用Rerank模型提高检索质量
    """
    
    def __init__(self):
        self.enabled = config.rerank.enabled
        self.api_key = config.rerank.api_key
        self.base_url = config.rerank.base_url
        self.model_name = config.rerank.model_name
        
        if self.enabled:
            # 不继承系统代理环境变量，避免 SOCKS 代理缺少 socksio 时报错
            self._http_client = httpx.Client(trust_env=False, timeout=60.0)
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=self._http_client
            )
    
    def rerank(
        self,
        query: str,
        chunks: List[Tuple[ChunkRecord, float]],
        top_k: int = 5
    ) -> List[Tuple[ChunkRecord, float]]:
        """
        对检索结果进行重排序
        
        参数:
            query: 用户查询
            chunks: 原始检索结果 (chunk, similarity)
            top_k: 返回数量
            
        返回:
            重排序后的结果
        """
        if not self.enabled or not chunks:
            return chunks[:top_k]
        
        try:
            documents = [chunk.text for chunk, _ in chunks]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个重排序助手。根据查询与文档的相关性进行评分。"},
                    {"role": "user", "content": f"查询: {query}\n\n请对以下文档按相关性排序:\n" + "\n---\n".join(documents[:10])}
                ],
                max_tokens=10
            )
            
            return chunks[:top_k]
            
        except Exception as e:
            print(f"Rerank失败: {e}")
            return chunks[:top_k]


class PgVectorRAGAgent:
    """
    基于PgVector的RAG Agent
    
    【这是核心类】
    整合向量检索和LLM生成，实现完整的RAG流程
    
    【使用方式】
    agent = PgVectorRAGAgent()
    response = agent.chat("这个文档主要讲了什么？")
    print(response.answer)
    """
    
    def __init__(self, verbose: bool = False):
        """
        初始化RAG Agent
        
        参数:
            verbose: 是否打印详细日志
        """
        self.verbose = verbose
        
        self.store = get_pgvector_store()
        self.embedding_service = EmbeddingService()
        self.rerank_service = RerankService()
        
        # 不继承系统代理环境变量，避免 SOCKS 代理缺少 socksio 时报错
        self._http_client = httpx.Client(trust_env=False, timeout=120.0)
        self.llm_client = OpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            http_client=self._http_client
        )
        self.llm_model = config.llm.model_name
        
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """
        构建系统提示词
        
        返回:
            系统提示词文本
        """
        return """你是一个智能知识库助手，可以访问知识库中的文档内容。

【你的能力】
1. 基于提供的知识库内容回答用户问题
2. 如果知识库中没有相关信息，坦诚告知用户
3. 回答要准确、有条理

【回答原则】
1. 只基于提供的上下文内容回答，不要编造信息
2. 如果上下文不足以回答问题，明确说明
3. 回答要清晰、简洁、有条理
4. 适当引用来源信息

请始终用中文回答用户问题。"""

    def _build_user_message(self, context: str, query: str) -> str:
        """构建用户提示词（包含上下文和问题）"""
        return f"""请基于以下知识库内容回答用户问题。

【知识库内容】
{context}

【用户问题】
{query}

请提供准确、有条理的回答："""

    def _extract_text(self, value: Any) -> str:
        """
        尽可能从不同结构中提取文本（兼容不同SDK返回结构）
        """
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, list):
            return "".join(self._extract_text(item) for item in value)

        if isinstance(value, dict):
            for key in ("text", "content", "reasoning_content", "reasoning", "output_text", "value"):
                if key in value:
                    text = self._extract_text(value.get(key))
                    if text:
                        return text
            return ""

        for attr in ("text", "content", "reasoning_content", "reasoning", "output_text", "value"):
            if hasattr(value, attr):
                text = self._extract_text(getattr(value, attr))
                if text:
                    return text

        return ""

    def _extract_reasoning_from_message(self, message: Any) -> str:
        """从非流式message中提取reasoning文本"""
        for attr in ("reasoning_content", "reasoning"):
            if hasattr(message, attr):
                text = self._extract_text(getattr(message, attr))
                if text:
                    return text

        extra = getattr(message, "model_extra", None)
        if isinstance(extra, dict):
            for key in ("reasoning_content", "reasoning"):
                text = self._extract_text(extra.get(key))
                if text:
                    return text
        return ""

    def _extract_reasoning_from_delta(self, delta: Any) -> str:
        """从流式delta中提取reasoning文本"""
        for attr in ("reasoning_content", "reasoning"):
            if hasattr(delta, attr):
                text = self._extract_text(getattr(delta, attr))
                if text:
                    return text

        extra = getattr(delta, "model_extra", None)
        if isinstance(extra, dict):
            for key in ("reasoning_content", "reasoning"):
                text = self._extract_text(extra.get(key))
                if text:
                    return text
        return ""

    def _stream_llm_response(self, user_message: str) -> Iterator[Dict[str, str]]:
        """调用LLM流式生成回答，返回answer/reasoning增量"""
        stream = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=2000,
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            answer_delta = self._extract_text(getattr(delta, "content", None))
            reasoning_delta = self._extract_reasoning_from_delta(delta)

            if answer_delta or reasoning_delta:
                yield {
                    "content": answer_delta,
                    "reasoning": reasoning_delta
                }

    def prepare_chat(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        use_rerank: bool = True
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        预处理对话请求，返回LLM输入消息和来源信息
        """
        chunks = self._retrieve(query, top_k, doc_ids, use_rerank)
        context, sources = self._build_context(chunks)
        user_message = self._build_user_message(context, query)
        return user_message, sources
    
    def _retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        use_rerank: bool = True
    ) -> List[Tuple[ChunkRecord, float]]:
        """
        检索相关文档块
        
        【核心检索流程】
        1. 对查询进行向量化
        2. 在PgVector中执行相似度搜索
        3. 可选：使用Rerank进行重排序
        
        参数:
            query: 用户查询
            top_k: 返回数量
            doc_ids: 限定文档ID
            use_rerank: 是否使用重排序
            
        返回:
            (ChunkRecord, similarity) 元组列表
        """
        query_embedding = self.embedding_service.embed_query(query)
        
        results = self.store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k * 2 if use_rerank else top_k,
            filter_doc_ids=doc_ids
        )
        
        if use_rerank and results:
            results = self.rerank_service.rerank(query, results, top_k)
        
        return results
    
    def _build_context(
        self,
        chunks: List[Tuple[ChunkRecord, float]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        构建LLM上下文和来源信息
        
        参数:
            chunks: 检索到的文档块
            
        返回:
            (上下文文本, 来源列表)
        """
        if not chunks:
            return "知识库中没有找到相关信息。", []
        
        context_parts = []
        sources = []
        
        for i, (chunk, similarity) in enumerate(chunks, 1):
            doc = self.store.get_document(chunk.doc_id)
            file_name = doc.file_name if doc else "未知来源"
            
            # 【重要】处理similarity可能为None的情况
            sim_display = f"{similarity:.2%}" if similarity is not None else "N/A"
            sim_value = round(similarity, 4) if similarity is not None else 0.0
            
            context_parts.append(
                f"【文档片段 {i}】(相关度: {sim_display})\n"
                f"来源: {file_name}\n"
                f"内容:\n{chunk.text}\n"
            )
            
            source_info = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "file_name": file_name,
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "similarity": sim_value,
                "page_number": chunk.page_number
            }
            sources.append(source_info)
        
        return "\n".join(context_parts), sources
    
    def chat(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        use_rerank: bool = True
    ) -> RAGResponse:
        """
        与知识库进行对话
        
        【这是主要的对外接口】
        
        参数:
            query: 用户问题
            top_k: 检索数量
            doc_ids: 限定文档ID列表
            use_rerank: 是否使用重排序
            
        返回:
            RAGResponse对象，包含answer、sources、query
        """
        if self.verbose:
            print(f"\n[用户问题] {query}")
        
        user_message, sources = self.prepare_chat(query, top_k, doc_ids, use_rerank)

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        message = response.choices[0].message
        answer = self._extract_text(getattr(message, "content", None))
        reasoning = self._extract_reasoning_from_message(message)

        if self.verbose:
            print(f"\n[回答]\n{answer}")
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=query,
            reasoning=reasoning or None
        )
    
    def chat_stream(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        use_rerank: bool = True
    ):
        """
        流式对话（逐步返回响应）
        
        参数:
            query: 用户问题
            top_k: 检索数量
            doc_ids: 限定文档ID列表
            use_rerank: 是否使用重排序
            
        Yields:
            响应的每个token
        """
        user_message, _ = self.prepare_chat(query, top_k, doc_ids, use_rerank)
        for delta in self._stream_llm_response(user_message):
            content = delta.get("content")
            if content:
                yield content

    def chat_stream_with_sources(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        use_rerank: bool = True
    ) -> Tuple[Iterator[Dict[str, str]], List[Dict[str, Any]]]:
        """
        流式对话（同时返回来源信息）
        """
        user_message, sources = self.prepare_chat(query, top_k, doc_ids, use_rerank)
        return self._stream_llm_response(user_message), sources


_pgvector_rag_agent = None


def get_pgvector_rag_agent() -> PgVectorRAGAgent:
    """
    获取PgVectorRAGAgent单例实例
    
    【使用方式】
    agent = get_pgvector_rag_agent()
    response = agent.chat("问题")
    """
    global _pgvector_rag_agent
    if _pgvector_rag_agent is None:
        _pgvector_rag_agent = PgVectorRAGAgent()
    return _pgvector_rag_agent
