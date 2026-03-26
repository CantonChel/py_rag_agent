"""
PostgreSQL + pgvector 向量存储模块

使用PostgreSQL作为关系型数据库，pgvector扩展存储和检索向量

【架构图】

┌─────────────────────────────────────────────────────────────────────┐
│                    PostgreSQL + pgvector 架构                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     PostgreSQL 数据库                          │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │                                                              │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │  │
│  │  │ documents   │  │   chunks    │  │   images    │          │  │
│  │  │ 表          │  │ 表+向量列   │  │ 表          │          │  │
│  │  │             │  │             │  │             │           │  │
│  │  │ - doc_id    │  │ - chunk_id  │  │ - image_id  │          │  │
│  │  │ - file_name │  │ - text      │  │ - path      │           │  │
│  │  │ - file_path │  │ - embedding │  │ - caption   │          │  │
│  │  │ - status    │  │ - metadata  │  │ - embedding │          │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘          │  │
│  │         │                │                │                  │  │
│  │         └────────────────┼────────────────┘                  │  │
│  │                          │                                   │  │
│  │               ┌──────────┴──────────┐                       │  │
│  │               │ chunk_image_relations│                       │  │
│  │               │ 关联表               │                       │  │
│  │               └─────────────────────┘                       │  │
│  │                                                              │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │ pgvector 扩展                                           │ │  │
│  │  │ - 向量存储 (vector类型)                                 │ │  │
│  │  │ - 余弦相似度搜索 (vector_cosine_ops)                    │ │  │
│  │  │ - IVFFlat索引加速检索                                   │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

【核心功能】
1. 文档管理：存储原始文档元数据
2. 向量存储：存储文本块的向量表示
3. 相似度搜索：使用pgvector进行向量检索
4. 关系维护：维护chunk之间的前后关系
"""

import uuid
import json
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from pgvector.psycopg2 import register_vector

from config import config


@dataclass
class DocumentRecord:
    """
    文档记录数据结构

    【对应数据库表】documents
    存储上传文档的元数据信息
    """
    doc_id: str
    file_name: str
    file_type: str
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    minio_bucket: Optional[str] = None
    minio_object_name: Optional[str] = None
    total_pages: int = 0
    total_chunks: int = 0
    status: str = "pending"  # pending, processing, completed, failed
    kb_id: str = "default"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class KnowledgeBaseRecord:
    """
    知识库记录数据结构

    【对应数据库表】knowledge_bases
    存储知识库的元数据信息
    """
    kb_id: str
    name: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class ChunkRecord:
    """
    文本块记录数据结构
    
    【对应数据库表】chunks
    存储文档分块后的文本和向量
    """
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    embedding: Optional[List[float]] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None


@dataclass
class ImageRecord:
    """
    图片记录数据结构
    
    【对应数据库表】images
    存储从文档中提取的图片信息
    """
    image_id: str
    doc_id: str
    image_path: Optional[str] = None
    minio_bucket: Optional[str] = None
    minio_object_name: Optional[str] = None
    caption: Optional[str] = None
    page_number: Optional[int] = None
    image_embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None


class PgVectorStore:
    """
    PostgreSQL + pgvector 向量存储管理器
    
    【这是向量存储的核心类】
    提供文档、文本块、图片的CRUD操作和向量检索功能
    
    【使用流程】
    1. 初始化连接：PgVectorStore()
    2. 添加文档：add_document()
    3. 添加文本块和向量：add_chunks()
    4. 相似度搜索：similarity_search()
    5. 关闭连接：close()
    """
    
    def __init__(self):
        """
        初始化PostgreSQL连接
        
        【连接参数来源】
        从config.py读取环境变量配置
        """
        self.conn = None
        self._connect()
    
    def _connect(self):
        """
        建立数据库连接
        
        【关键步骤】
        1. 使用psycopg2建立连接
        2. 注册pgvector类型
        3. 设置自动提交
        """
        try:
            pg_config = config.postgres
            self.conn = psycopg2.connect(
                host=pg_config.host,
                port=pg_config.port,
                user=pg_config.user,
                password=pg_config.password,
                database=pg_config.database
            )
            
            # 先设置自动提交模式，再注册pgvector扩展类型
            # 【重要】register_vector需要在非事务状态下执行
            self.conn.autocommit = True
            
            # 注册pgvector扩展类型
            # 【重要】这使得可以直接插入和查询vector类型
            register_vector(self.conn)
            print(f"✓ PostgreSQL连接成功: {pg_config.host}:{pg_config.port}/{pg_config.database}")
            
        except Exception as e:
            print(f"✗ PostgreSQL连接失败: {e}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            print("✓ PostgreSQL连接已关闭")
    
    # ============================================================
    # 文档管理
    # ============================================================
    
    def add_document(self, doc: DocumentRecord) -> str:
        """
        添加文档记录
        
        【参数】
            doc: DocumentRecord 文档记录对象
            
        【返回】
            文档ID
            
        【SQL执行】
            INSERT INTO documents (...)
        """
        sql = """
            INSERT INTO documents (
                doc_id, file_name, file_type, file_path, file_size,
                minio_bucket, minio_object_name, total_pages, total_chunks, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                file_name = EXCLUDED.file_name,
                file_type = EXCLUDED.file_type,
                file_path = EXCLUDED.file_path,
                file_size = EXCLUDED.file_size,
                minio_bucket = EXCLUDED.minio_bucket,
                minio_object_name = EXCLUDED.minio_object_name,
                total_pages = EXCLUDED.total_pages,
                total_chunks = EXCLUDED.total_chunks,
                status = EXCLUDED.status,
                updated_at = CURRENT_TIMESTAMP
            RETURNING doc_id
        """
        
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                doc.doc_id, doc.file_name, doc.file_type, doc.file_path,
                doc.file_size, doc.minio_bucket, doc.minio_object_name,
                doc.total_pages, doc.total_chunks, doc.status
            ))
            result = cur.fetchone()
            return result[0] if result else doc.doc_id
    
    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """
        获取文档记录
        
        【参数】
            doc_id: 文档ID
            
        【返回】
            DocumentRecord对象，不存在则返回None
        """
        sql = """
            SELECT doc_id, file_name, file_type, file_path, file_size,
                   minio_bucket, minio_object_name, total_pages, total_chunks,
                   status, created_at, updated_at
            FROM documents WHERE doc_id = %s
        """
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (doc_id,))
            row = cur.fetchone()
            if row:
                return DocumentRecord(**dict(row))
        return None
    
    def list_documents(
        self,
        status: Optional[str] = None,
        kb_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DocumentRecord]:
        """
        列出文档

        【参数】
            status: 按状态过滤（可选）
            kb_id: 按知识库ID过滤（可选）
            limit: 返回数量限制
            offset: 偏移量

        【返回】
            DocumentRecord列表
        """
        conditions = []
        params = []

        if status:
            conditions.append("status = %s")
            params.append(status)

        if kb_id:
            conditions.append("kb_id = %s")
            params.append(kb_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
            SELECT doc_id, file_name, file_type, file_path, file_size,
                   minio_bucket, minio_object_name, total_pages, total_chunks,
                   status, COALESCE(kb_id, 'default') as kb_id, created_at, updated_at
            FROM documents
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [DocumentRecord(**dict(row)) for row in rows]

    def update_document_status(self, doc_id: str, status: str, total_chunks: Optional[int] = None):
        """
        更新文档状态
        
        【参数】
            doc_id: 文档ID
            status: 新状态
            total_chunks: 总块数（可选）
        """
        if total_chunks is not None:
            sql = """
                UPDATE documents 
                SET status = %s, total_chunks = %s, updated_at = CURRENT_TIMESTAMP 
                WHERE doc_id = %s
            """
            params = (status, total_chunks, doc_id)
        else:
            sql = """
                UPDATE documents 
                SET status = %s, updated_at = CURRENT_TIMESTAMP 
                WHERE doc_id = %s
            """
            params = (status, doc_id)
        
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
    
    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档及其所有相关数据
        
        【级联删除】
        由于设置了ON DELETE CASCADE，删除文档时会自动删除：
        - 所有关联的chunks
        - 所有关联的images
        - 所有关联的relations
        """
        sql = "DELETE FROM documents WHERE doc_id = %s"
        
        with self.conn.cursor() as cur:
            cur.execute(sql, (doc_id,))
            return cur.rowcount > 0
    
    # ============================================================
    # 文本块管理
    # ============================================================
    
    def add_chunk(self, chunk: ChunkRecord) -> str:
        """
        添加单个文本块
        
        【参数】
            chunk: ChunkRecord 文本块记录
            
        【返回】
            chunk_id
        """
        sql = """
            INSERT INTO chunks (
                chunk_id, doc_id, chunk_index, text, embedding,
                page_number, section_title, start_char, end_char,
                prev_chunk_id, next_chunk_id, parent_chunk_id, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata
            RETURNING chunk_id
        """
        
        # 将embedding转换为PostgreSQL的vector格式
        embedding = chunk.embedding if chunk.embedding else None
        
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                chunk.chunk_id, chunk.doc_id, chunk.chunk_index, chunk.text,
                embedding, chunk.page_number, chunk.section_title,
                chunk.start_char, chunk.end_char,
                chunk.prev_chunk_id, chunk.next_chunk_id, chunk.parent_chunk_id,
                json.dumps(chunk.metadata)
            ))
            result = cur.fetchone()
            return result[0] if result else chunk.chunk_id
    
    def add_chunks_batch(self, chunks: List[ChunkRecord]) -> int:
        """
        批量添加文本块
        
        【参数】
            chunks: ChunkRecord列表
            
        【返回】
            成功插入的数量
            
        【性能优化】
        使用execute_values进行批量插入，比循环单条插入快10-100倍
        """
        if not chunks:
            return 0
        
        # 准备数据
        data = [
            (
                chunk.chunk_id, chunk.doc_id, chunk.chunk_index, chunk.text,
                chunk.embedding, chunk.page_number, chunk.section_title,
                chunk.start_char, chunk.end_char,
                chunk.prev_chunk_id, chunk.next_chunk_id, chunk.parent_chunk_id,
                json.dumps(chunk.metadata)
            )
            for chunk in chunks
        ]
        
        sql = """
            INSERT INTO chunks (
                chunk_id, doc_id, chunk_index, text, embedding,
                page_number, section_title, start_char, end_char,
                prev_chunk_id, next_chunk_id, parent_chunk_id, metadata
            ) VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding
        """
        
        with self.conn.cursor() as cur:
            execute_values(cur, sql, data)
            return len(chunks)
    
    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        """获取单个文本块"""
        sql = "SELECT * FROM chunks WHERE chunk_id = %s"
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (chunk_id,))
            row = cur.fetchone()
            if row:
                data = dict(row)
                # 转换embedding
                if data.get('embedding') is not None:
                    data['embedding'] = list(data['embedding'])
                return ChunkRecord(**data)
        return None
    
    def get_chunks_by_doc(self, doc_id: str) -> List[ChunkRecord]:
        """
        获取文档的所有文本块
        
        【按顺序返回】
        结果按chunk_index排序，确保文本块的顺序正确
        """
        sql = """
            SELECT * FROM chunks 
            WHERE doc_id = %s 
            ORDER BY chunk_index
        """
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (doc_id,))
            rows = cur.fetchall()
            results = []
            for row in rows:
                data = dict(row)
                if data.get('embedding') is not None:
                    data['embedding'] = list(data['embedding'])
                if 'id' in data:
                    del data['id']
                results.append(ChunkRecord(**data))
            return results
    
    def get_adjacent_chunks(self, chunk_id: str, window: int = 1) -> List[ChunkRecord]:
        """
        获取相邻的文本块（滑动窗口）
        
        【参数】
            chunk_id: 当前chunk ID
            window: 前后各取几个chunk
            
        【返回】
            包含当前chunk及其相邻chunk的列表
        """
        # 先获取当前chunk
        current = self.get_chunk(chunk_id)
        if not current:
            return []
        
        chunk_ids = []
        
        # 向前查找
        prev_id = current.prev_chunk_id
        for _ in range(window):
            if prev_id:
                chunk_ids.insert(0, prev_id)
                prev_chunk = self.get_chunk(prev_id)
                prev_id = prev_chunk.prev_chunk_id if prev_chunk else None
            else:
                break
        
        # 添加当前chunk
        chunk_ids.append(chunk_id)
        
        # 向后查找
        next_id = current.next_chunk_id
        for _ in range(window):
            if next_id:
                chunk_ids.append(next_id)
                next_chunk = self.get_chunk(next_id)
                next_id = next_chunk.next_chunk_id if next_chunk else None
            else:
                break
        
        # 批量获取
        if not chunk_ids:
            return [current]
        
        sql = """
            SELECT * FROM chunks 
            WHERE chunk_id = ANY(%s) 
            ORDER BY chunk_index
        """
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (chunk_ids,))
            rows = cur.fetchall()
            results = []
            for row in rows:
                data = dict(row)
                if data.get('embedding') is not None:
                    data['embedding'] = list(data['embedding'])
                results.append(ChunkRecord(**data))
            return results
    
    # ============================================================
    # 向量相似度搜索
    # ============================================================
    
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_doc_ids: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> List[Tuple[ChunkRecord, float]]:
        """
        向量相似度搜索
        
        【这是检索的核心方法】
        
        【参数】
            query_embedding: 查询向量
            top_k: 返回数量
            filter_doc_ids: 限定在哪些文档中搜索
            threshold: 相似度阈值（0-1），低于此值不返回
            
        【返回】
            (ChunkRecord, similarity) 元组列表
            similarity是余弦相似度，范围[-1, 1]，越大越相似
            
        【SQL说明】
            1 - (embedding <=> query) 计算余弦相似度
            <=> 是pgvector的余弦距离操作符
        """
        # 将Python列表转换为PostgreSQL vector格式的字符串
        # 【重要】pgvector需要显式转换为vector类型
        vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # 构建SQL - 使用显式类型转换
        if filter_doc_ids:
            sql = """
                SELECT 
                    *,
                    1 - (embedding <=> %s::vector) as similarity
                FROM chunks
                WHERE doc_id = ANY(%s)
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = (vector_str, filter_doc_ids, vector_str, top_k)
        else:
            sql = """
                SELECT 
                    *,
                    1 - (embedding <=> %s::vector) as similarity
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = (vector_str, vector_str, top_k)
        
        results = []
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            
            for row in rows:
                data = dict(row)
                similarity = data.pop('similarity', 0)
                
                # 【重要】删除数据库自增id字段，ChunkRecord不包含此字段
                if 'id' in data:
                    del data['id']
                
                # 应用阈值过滤
                if threshold is not None and similarity < threshold:
                    continue
                
                if data.get('embedding') is not None:
                    data['embedding'] = list(data['embedding'])
                
                chunk = ChunkRecord(**data)
                results.append((chunk, similarity))
        
        return results
    
    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.7
    ) -> List[Tuple[ChunkRecord, float]]:
        """
        混合搜索：向量相似度 + 关键词匹配
        
        【参数】
            query_embedding: 查询向量
            query_text: 查询文本
            top_k: 返回数量
            alpha: 向量搜索权重（0-1），1-alpha为关键词权重
            
        【返回】
            (ChunkRecord, score) 元组列表
        """
        sql = """
            SELECT 
                *,
                (
                    %s * (1 - (embedding <=> %s)) +
                    %s * ts_rank(
                        to_tsvector('simple', text),
                        plainto_tsquery('simple', %s)
                    )
                ) as score
            FROM chunks
            ORDER BY score DESC
            LIMIT %s
        """
        
        results = []
        vector_weight = alpha
        text_weight = 1 - alpha
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (
                vector_weight, query_embedding,
                text_weight, query_text,
                top_k
            ))
            rows = cur.fetchall()
            
            for row in rows:
                data = dict(row)
                score = data.pop('score', 0)
                
                if data.get('embedding') is not None:
                    data['embedding'] = list(data['embedding'])
                
                chunk = ChunkRecord(**data)
                results.append((chunk, score))
        
        return results
    
    # ============================================================
    # 图片管理
    # ============================================================
    
    def add_image(self, image: ImageRecord) -> str:
        """添加图片记录"""
        sql = """
            INSERT INTO images (
                image_id, doc_id, image_path, minio_bucket, minio_object_name,
                caption, page_number, image_embedding, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (image_id) DO UPDATE SET
                caption = EXCLUDED.caption,
                image_embedding = EXCLUDED.image_embedding
            RETURNING image_id
        """
        
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                image.image_id, image.doc_id, image.image_path,
                image.minio_bucket, image.minio_object_name,
                image.caption, image.page_number,
                image.image_embedding, json.dumps(image.metadata)
            ))
            result = cur.fetchone()
            return result[0] if result else image.image_id
    
    def get_images_by_doc(self, doc_id: str) -> List[ImageRecord]:
        """获取文档的所有图片"""
        sql = """
            SELECT image_id, doc_id, image_path, minio_bucket, minio_object_name,
                   caption, page_number, image_embedding, metadata, created_at
            FROM images WHERE doc_id = %s ORDER BY page_number
        """
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (doc_id,))
            rows = cur.fetchall()
            results = []
            for row in rows:
                data = dict(row)
                if data.get('image_embedding') is not None:
                    data['image_embedding'] = list(data['image_embedding'])
                results.append(ImageRecord(**data))
            return results
    
    def add_chunk_image_relation(self, chunk_id: str, image_id: str):
        """
        添加chunk与image的关联关系
        
        【关联表】
        chunk_image_relations 表记录哪些chunk包含哪些图片
        """
        sql = """
            INSERT INTO chunk_image_relations (chunk_id, image_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
        """
        
        with self.conn.cursor() as cur:
            cur.execute(sql, (chunk_id, image_id))
    
    def get_chunk_images(self, chunk_id: str) -> List[ImageRecord]:
        """
        获取与chunk关联的所有图片
        
        【通过关联表查询】
        """
        sql = """
            SELECT i.* FROM images i
            JOIN chunk_image_relations r ON i.image_id = r.image_id
            WHERE r.chunk_id = %s
        """
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (chunk_id,))
            rows = cur.fetchall()
            results = []
            for row in rows:
                data = dict(row)
                if data.get('image_embedding') is not None:
                    data['image_embedding'] = list(data['image_embedding'])
                results.append(ImageRecord(**data))
            return results
    
    # ============================================================
    # 统计和工具方法
    # ============================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        【返回】
        {
            'total_documents': 文档总数,
            'total_chunks': 文本块总数,
            'total_images': 图片总数,
            'documents_by_status': 按状态分类的文档数
        }
        """
        stats = {}
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 文档总数
            cur.execute("SELECT COUNT(*) as count FROM documents")
            stats['total_documents'] = cur.fetchone()['count']
            
            # 文本块总数
            cur.execute("SELECT COUNT(*) as count FROM chunks")
            stats['total_chunks'] = cur.fetchone()['count']
            
            # 图片总数
            cur.execute("SELECT COUNT(*) as count FROM images")
            stats['total_images'] = cur.fetchone()['count']
            
            # 按状态分类
            cur.execute("""
                SELECT status, COUNT(*) as count 
                FROM documents 
                GROUP BY status
            """)
            stats['documents_by_status'] = {row['status']: row['count'] for row in cur.fetchall()}
        
        return stats

    # ============================================================
    # 知识库管理
    # ============================================================

    def create_knowledge_base(self, kb_id: str, name: str, description: Optional[str] = None) -> str:
        """
        创建知识库

        【参数】
            kb_id: 知识库ID
            name: 知识库名称
            description: 描述（可选）

        【返回】
            kb_id
        """
        sql = """
            INSERT INTO knowledge_bases (kb_id, name, description)
            VALUES (%s, %s, %s)
            ON CONFLICT (kb_id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                updated_at = CURRENT_TIMESTAMP
            RETURNING kb_id
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (kb_id, name, description))
            return cur.fetchone()[0]

    def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBaseRecord]:
        """
        获取知识库

        【参数】
            kb_id: 知识库ID

        【返回】
            KnowledgeBaseRecord对象，不存在则返回None
        """
        sql = """
            SELECT kb_id, name, description, created_at, updated_at
            FROM knowledge_bases WHERE kb_id = %s
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (kb_id,))
            row = cur.fetchone()
            if row:
                return KnowledgeBaseRecord(**dict(row))
        return None

    def list_knowledge_bases(self) -> List[KnowledgeBaseRecord]:
        """
        列出所有知识库

        【返回】
            KnowledgeBaseRecord列表
        """
        sql = """
            SELECT kb_id, name, description, created_at, updated_at
            FROM knowledge_bases ORDER BY created_at DESC
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            return [KnowledgeBaseRecord(**dict(row)) for row in rows]

    def update_knowledge_base(self, kb_id: str, name: str = None, description: str = None) -> bool:
        """
        更新知识库

        【参数】
            kb_id: 知识库ID
            name: 新名称（可选）
            description: 新描述（可选）

        【返回】
            是否成功
        """
        updates = []
        params = []
        if name is not None:
            updates.append("name = %s")
            params.append(name)
        if description is not None:
            updates.append("description = %s")
            params.append(description)

        if not updates:
            return False

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(kb_id)

        sql = f"""
            UPDATE knowledge_bases SET {', '.join(updates)}
            WHERE kb_id = %s
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount > 0

    def delete_knowledge_base(self, kb_id: str) -> bool:
        """
        删除知识库（同时将文档的kb_id设为NULL）

        【参数】
            kb_id: 知识库ID

        【返回】
            是否成功
        """
        if kb_id == "default":
            return False  # 不能删除默认知识库

        with self.conn.cursor() as cur:
            # 先将文档的kb_id设为NULL
            cur.execute("UPDATE documents SET kb_id = NULL WHERE kb_id = %s", (kb_id,))
            # 删除知识库
            cur.execute("DELETE FROM knowledge_bases WHERE kb_id = %s", (kb_id,))
            return cur.rowcount > 0

    def clear_all(self):
        """
        清空所有数据
        
        【危险操作】仅用于测试环境
        """
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE chunk_image_relations CASCADE")
            cur.execute("TRUNCATE TABLE chunks CASCADE")
            cur.execute("TRUNCATE TABLE images CASCADE")
            cur.execute("TRUNCATE TABLE documents CASCADE")
        print("✓ 所有数据已清空")


# ============================================================
# 全局实例（单例模式）
# ============================================================

_pgvector_store = None

def get_pgvector_store() -> PgVectorStore:
    """
    获取PgVectorStore单例实例
    
    【使用方式】
    store = get_pgvector_store()
    store.similarity_search(...)
    """
    global _pgvector_store
    if _pgvector_store is None:
        _pgvector_store = PgVectorStore()
    return _pgvector_store
