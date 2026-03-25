"""
FastAPI后端接口

提供RAG系统的RESTful API接口

【API架构图】

┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI 接口架构                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      客户端请求                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    FastAPI 路由层                             │   │
│  │                                                             │   │
│  │  /api/documents     文档管理接口                             │   │
│  │  /api/chat          对话查询接口                             │   │
│  │  /api/search        向量搜索接口                             │   │
│  │  /api/health        健康检查接口                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│              ┌───────────────┼───────────────┐                     │
│              ▼               ▼               ▼                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │ MinIO存储   │  │ PgVector    │  │ RAG Agent   │                │
│  │ 原始文件    │  │ 向量存储    │  │ 检索生成    │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

【API端点说明】
POST   /api/documents/upload     上传文档
GET    /api/documents            获取文档列表
GET    /api/documents/{doc_id}   获取单个文档
DELETE /api/documents/{doc_id}   删除文档
POST   /api/chat                 对话查询（RAG）
POST   /api/search               向量相似度搜索
GET    /api/health               健康检查
GET    /api/stats                系统统计
"""

import os
import io
import uuid
import asyncio
import tempfile
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

from config import config
from storage.minio_storage import MinIOStorage
from pgvector_store import PgVectorStore, DocumentRecord, ChunkRecord, ImageRecord
from advanced_parser import DocumentParserFactory
from pgvector_rag_agent import get_pgvector_rag_agent


# ============================================================
# Pydantic 模型定义
# ============================================================

class UploadResponse(BaseModel):
    """上传响应模型"""
    doc_id: str
    file_name: str
    file_size: int
    message: str
    status: str = "processing"


class DocumentInfo(BaseModel):
    """文档信息模型"""
    doc_id: str
    file_name: str
    file_type: str
    file_size: Optional[int] = None
    total_pages: int = 0
    total_chunks: int = 0
    status: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    total: int
    documents: List[DocumentInfo]


class ChatRequest(BaseModel):
    """
    对话请求模型
    
    【参数说明】
    query: 用户问题
    doc_ids: 限定在哪些文档中搜索（可选）
    top_k: 返回的相关文档块数量
    use_rerank: 是否使用重排序
    """
    query: str = Field(..., description="用户问题", min_length=1)
    doc_ids: Optional[List[str]] = Field(None, description="限定搜索的文档ID列表")
    top_k: int = Field(5, description="返回的相关文档块数量", ge=1, le=20)
    use_rerank: bool = Field(True, description="是否使用重排序")
    conversation_id: Optional[str] = Field(None, description="会话ID，用于多轮对话")


class ChatResponse(BaseModel):
    """对话响应模型"""
    answer: str
    sources: List[Dict[str, Any]]
    conversation_id: str
    query: str


class SearchRequest(BaseModel):
    """
    搜索请求模型
    
    【用于纯向量搜索，不生成回答】
    """
    query: str = Field(..., description="搜索文本", min_length=1)
    top_k: int = Field(5, description="返回数量", ge=1, le=50)
    doc_ids: Optional[List[str]] = Field(None, description="限定搜索的文档ID列表")
    threshold: Optional[float] = Field(None, description="相似度阈值", ge=0, le=1)


class SearchResult(BaseModel):
    """搜索结果模型"""
    chunk_id: str
    doc_id: str
    text: str
    similarity: float
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    file_name: Optional[str] = None


class SearchResponse(BaseModel):
    """搜索响应模型"""
    query: str
    results: List[SearchResult]
    total: int


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    postgres: str
    minio: str
    timestamp: datetime


class StatsResponse(BaseModel):
    """统计信息响应"""
    total_documents: int
    total_chunks: int
    total_images: int
    documents_by_status: Dict[str, int]


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str
    detail: str
    timestamp: datetime


# ============================================================
# FastAPI 应用初始化
# ============================================================

app = FastAPI(
    title="RAG知识库Agent系统",
    description="""
    基于LlamaIndex的RAG知识库Agent系统API
    
    ## 功能特性
    - 📄 多格式文档支持（PDF、Word、HTML、Markdown）
    - 🔍 向量相似度搜索
    - 🤖 基于Agent的智能问答
    - 📦 MinIO文件存储
    - 🗄️ PostgreSQL + pgvector向量存储
    
    ## 使用流程
    1. 上传文档到系统
    2. 等待文档处理完成
    3. 使用/chat接口进行问答
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS（跨域资源共享）
# 【允许前端页面跨域访问API】
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 全局实例
# ============================================================

minio_storage = None
pgvector_store = None
rag_agent = None


def get_minio() -> MinIOStorage:
    """获取MinIO存储实例"""
    global minio_storage
    if minio_storage is None:
        minio_storage = MinIOStorage()
    return minio_storage


def get_pgvector() -> PgVectorStore:
    """获取PgVector存储实例"""
    global pgvector_store
    if pgvector_store is None:
        pgvector_store = PgVectorStore()
    return pgvector_store


# ============================================================
# 启动和关闭事件
# ============================================================

@app.on_event("startup")
async def startup_event():
    """
    应用启动时执行
    
    【初始化操作】
    1. 检查配置
    2. 初始化数据库连接
    3. 初始化MinIO连接
    """
    print("=" * 60)
    print("RAG知识库Agent系统启动中...")
    print("=" * 60)
    
    # 验证配置
    if not config.validate():
        print("❌ 配置验证失败，请检查.env文件")
        return
    
    # 打印配置信息
    config.print_config()
    
    # 初始化连接
    try:
        get_minio()
        get_pgvector()
        print("✓ 所有服务初始化完成")
    except Exception as e:
        print(f"❌ 服务初始化失败: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    global pgvector_store
    if pgvector_store:
        pgvector_store.close()
    print("RAG知识库Agent系统已关闭")


# ============================================================
# 文档管理接口
# ============================================================

@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    上传文档
    
    【处理流程】
    1. 验证文件类型和大小
    2. 上传到MinIO存储
    3. 创建文档记录
    4. 后台异步处理文档（解析、分块、向量化）
    
    【支持的文件类型】
    - PDF (.pdf)
    - Word (.docx, .doc)
    - HTML (.html, .htm)
    - Markdown (.md)
    - 纯文本 (.txt)
    """
    # 验证文件类型
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.app.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}。支持的类型: {config.app.allowed_extensions}"
        )
    
    # 验证文件大小
    content = await file.read()
    file_size = len(content)
    if file_size > config.app.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大: {file_size} bytes。最大允许: {config.app.max_file_size} bytes"
        )
    
    # 生成文档ID
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    
    try:
        # 上传到MinIO
        # 【重要】将bytes转换为BytesIO对象以支持seek操作
        minio = get_minio()
        upload_result = minio.upload_file(
            file_data=io.BytesIO(content),
            file_name=file.filename,
            content_type=file.content_type,
            prefix="documents",
            metadata={"doc_id": doc_id}
        )
        
        # 创建文档记录
        pgvector = get_pgvector()
        doc = DocumentRecord(
            doc_id=doc_id,
            file_name=file.filename,
            file_type=file_ext[1:],  # 去掉点号
            file_size=file_size,
            minio_bucket=upload_result.get("bucket"),
            minio_object_name=upload_result.get("object_name"),
            status="pending"
        )
        pgvector.add_document(doc)
        
        # 添加后台任务处理文档
        # 【核心功能】异步处理文档，不阻塞响应
        background_tasks.add_task(
            process_document_task,
            doc_id=doc_id,
            minio_object=upload_result.get("object_name")
        )
        
        return UploadResponse(
            doc_id=doc_id,
            file_name=file.filename,
            file_size=file_size,
            message="文档上传成功，正在后台处理",
            status="processing"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


async def process_document_task(doc_id: str, minio_object: str):
    """
    后台任务：处理文档
    
    【处理步骤】
    1. 从MinIO下载文件
    2. 解析文档内容（使用advanced_parser）
    3. 提取图片并上传到MinIO
    4. 分块处理
    5. 存储chunks和images到pgvector
    6. 更新文档状态
    """
    pgvector = get_pgvector()
    minio = get_minio()
    
    try:
        # 更新状态为处理中
        pgvector.update_document_status(doc_id, "processing")
        
        # 【核心步骤1】从MinIO下载文件
        file_data = minio.download_file(minio_object)
        if not file_data:
            raise Exception(f"无法从MinIO下载文件: {minio_object}")
        
        # 保存到临时文件（解析器需要文件路径）
        import tempfile
        file_ext = Path(minio_object).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name
        
        try:
            # 【核心步骤2】使用DocumentParserFactory解析文档
            parsed_doc = DocumentParserFactory.parse(tmp_path)
            print(f"✓ 文档解析完成: {len(parsed_doc.images)} 张图片, {len(parsed_doc.sections)} 个章节")
            
            # 【核心步骤3】处理并上传图片
            for img_info in parsed_doc.images:
                if img_info.image_data:
                    # 生成图片ID
                    img_id = f"img_{uuid.uuid4().hex[:12]}"
                    
                    # 上传图片到MinIO
                    img_ext = Path(img_info.source_path).suffix or ".png"
                    img_object_name = f"images/{doc_id}/{img_id}{img_ext}"
                    minio.client.put_object(
                        minio.bucket,
                        img_object_name,
                        io.BytesIO(img_info.image_data),
                        len(img_info.image_data),
                        content_type="image/png" if img_ext == ".png" else "image/jpeg"
                    )
                    
                    # 存储图片记录到数据库
                    img_record = ImageRecord(
                        image_id=img_id,
                        doc_id=doc_id,
                        page_number=img_info.page_number,
                        image_path=img_object_name,
                        minio_bucket=minio.bucket,
                        minio_object_name=img_object_name,
                        caption=img_info.caption
                    )
                    pgvector.add_image(img_record)
            
            # 【核心步骤4】文本分块（使用简化的分块逻辑）
            chunk_size = config.app.chunk_size
            chunk_overlap = config.app.chunk_overlap
            text = parsed_doc.markdown_content
            
            # 简单的滑动窗口分块
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]
                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text.strip(),
                        "chunk_index": len(chunks),
                        "metadata": {"doc_id": doc_id, "file_path": parsed_doc.file_path}
                    })
                start = end - chunk_overlap
            
            # 【核心步骤5】存储chunks到数据库
            for chunk_info in chunks:
                chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
                chunk_record = ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=chunk_info["text"],
                    chunk_index=chunk_info["chunk_index"],
                    metadata=chunk_info["metadata"]
                )
                pgvector.add_chunk(chunk_record)
            
            # 更新文档状态为完成
            pgvector.update_document_status(
                doc_id, 
                "completed", 
                total_chunks=len(chunks)
            )
            
            print(f"✓ 文档处理完成 [{doc_id}]: {len(chunks)} 个分块, {len(parsed_doc.images)} 张图片")
            
        finally:
            # 清理临时文件
            os.unlink(tmp_path)
        
    except Exception as e:
        import traceback
        print(f"✗ 文档处理失败 [{doc_id}]: {e}")
        traceback.print_exc()
        pgvector.update_document_status(doc_id, "failed")


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(
    status: Optional[str] = Query(None, description="按状态过滤"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
    offset: int = Query(0, ge=0, description="偏移量")
):
    """
    获取文档列表
    
    【参数】
    - status: 按状态过滤（pending, processing, completed, failed）
    - limit: 返回数量
    - offset: 分页偏移
    """
    pgvector = get_pgvector()
    docs = pgvector.list_documents(status=status, limit=limit, offset=offset)
    
    document_infos = [
        DocumentInfo(
            doc_id=doc.doc_id,
            file_name=doc.file_name,
            file_type=doc.file_type,
            file_size=doc.file_size,
            total_pages=doc.total_pages,
            total_chunks=doc.total_chunks,
            status=doc.status,
            created_at=doc.created_at,
            updated_at=doc.updated_at
        )
        for doc in docs
    ]
    
    return DocumentListResponse(
        total=len(document_infos),
        documents=document_infos
    )


class ChunkWithImages(BaseModel):
    chunk_id: str
    chunk_index: int
    text: str
    text_with_images: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    images: List[dict] = []


class DocumentWithContent(BaseModel):
    doc_id: str
    file_name: str
    status: str
    total_chunks: int
    total_images: int
    chunks: List[ChunkWithImages] = []


class DocumentContentResponse(BaseModel):
    total: int
    documents: List[DocumentWithContent]


@app.get("/api/documents/with-content", response_model=DocumentContentResponse)
async def get_documents_with_content():
    """
    获取所有文档及其分段内容，分段中的图片引用被替换为MinIO预签名URL
    
    图片引用格式：image://img_X_Y 其中X是页码，Y是该页的图片序号
    """
    pgvector = get_pgvector()
    minio = get_minio()
    
    docs = pgvector.list_documents(limit=1000)
    doc_contents = []
    
    for doc in docs:
        chunks = pgvector.get_chunks_by_doc(doc.doc_id)
        
        doc_images_sql = """
            SELECT image_id, minio_bucket, minio_object_name, page_number
            FROM images WHERE doc_id = %s
            ORDER BY page_number, image_id
        """
        
        with pgvector.conn.cursor() as cur:
            cur.execute(doc_images_sql, (doc.doc_id,))
            image_rows = cur.fetchall()
        
        page_image_map = {}
        for row in image_rows:
            image_id, minio_bucket, minio_object_name, page_number = row[0], row[1], row[2], row[3]
            if page_number not in page_image_map:
                page_image_map[page_number] = []
            
            url = None
            if minio_object_name:
                try:
                    from datetime import timedelta
                    url = minio.client.presigned_get_object(
                        minio_bucket or minio.bucket,
                        minio_object_name,
                        expires=timedelta(hours=1)
                    )
                except Exception as e:
                    print(f"生成预签名URL失败: {e}")
            
            page_image_map[page_number].append({
                'image_id': image_id,
                'minio_path': minio_object_name,
                'url': url
            })
        
        chunk_infos = []
        for chunk in chunks:
            import re
            text_with_images = chunk.text
            chunk_images = []
            
            image_refs = re.findall(r'!\[.*?\]\(image://(img_\d+_\d+)\)', chunk.text)
            
            for ref in image_refs:
                parts = ref.replace('img_', '').split('_')
                if len(parts) == 2:
                    page_num = int(parts[0])
                    img_idx = int(parts[1])
                    
                    if page_num in page_image_map and img_idx < len(page_image_map[page_num]):
                        img_info = page_image_map[page_num][img_idx]
                        chunk_images.append(img_info)
                        
                        if img_info['url']:
                            text_with_images = text_with_images.replace(
                                f'(image://{ref})',
                                f'({img_info["url"]})'
                            )
            
            chunk_infos.append(ChunkWithImages(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                text_with_images=text_with_images,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                images=chunk_images
            ))
        
        doc_contents.append(DocumentWithContent(
            doc_id=doc.doc_id,
            file_name=doc.file_name,
            status=doc.status,
            total_chunks=len(chunks),
            total_images=len(image_rows),
            chunks=chunk_infos
        ))
    
    return DocumentContentResponse(total=len(doc_contents), documents=doc_contents)


@app.get("/api/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str):
    """
    获取单个文档详情
    
    【参数】
    - doc_id: 文档ID
    """
    pgvector = get_pgvector()
    doc = pgvector.get_document(doc_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
    
    return DocumentInfo(
        doc_id=doc.doc_id,
        file_name=doc.file_name,
        file_type=doc.file_type,
        file_size=doc.file_size,
        total_pages=doc.total_pages,
        total_chunks=doc.total_chunks,
        status=doc.status,
        created_at=doc.created_at,
        updated_at=doc.updated_at
    )


class ImageInfo(BaseModel):
    """图片信息响应模型"""
    image_id: str
    doc_id: str
    page_number: Optional[int] = None
    image_path: Optional[str] = None
    caption: Optional[str] = None
    url: Optional[str] = None


class ImageListResponse(BaseModel):
    """图片列表响应模型"""
    total: int
    images: List[ImageInfo]


@app.get("/api/documents/{doc_id}/images", response_model=ImageListResponse)
async def get_document_images(doc_id: str):
    """
    获取文档的所有图片
    
    【参数】
    - doc_id: 文档ID
    
    【返回】
    - 图片列表，包含MinIO预签名URL
    """
    pgvector = get_pgvector()
    minio = get_minio()
    
    # 检查文档是否存在
    doc = pgvector.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
    
    # 获取图片记录
    images = pgvector.get_images_by_doc(doc_id)
    
    # 构建响应，包含预签名URL
    image_infos = []
    for img in images:
        # 生成预签名URL（有效期1小时）
        url = None
        if img.minio_object_name:
            try:
                from minio.commonconfig import CopySource
                from datetime import timedelta
                url = minio.client.presigned_get_object(
                    img.minio_bucket or minio.bucket,
                    img.minio_object_name,
                    expires=timedelta(hours=1)
                )
            except Exception as e:
                import traceback
                print(f"生成预签名URL失败: {e}")
                traceback.print_exc()
        
        image_infos.append(ImageInfo(
            image_id=img.image_id,
            doc_id=img.doc_id,
            page_number=img.page_number,
            image_path=img.image_path,
            caption=img.caption,
            url=url
        ))
    
    return ImageListResponse(
        total=len(image_infos),
        images=image_infos
    )


class ChunkInfo(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    text_preview: Optional[str] = None


class ChunkListResponse(BaseModel):
    total: int
    chunks: List[ChunkInfo]


@app.get("/api/documents/{doc_id}/chunks", response_model=ChunkListResponse)
async def get_document_chunks(doc_id: str):
    pgvector = get_pgvector()
    doc = pgvector.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
    
    chunks = pgvector.get_chunks_by_doc(doc_id)
    chunk_infos = []
    for chunk in chunks:
        text_preview = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
        chunk_infos.append(ChunkInfo(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            text_preview=text_preview
        ))
    
    return ChunkListResponse(
        total=len(chunk_infos),
        chunks=chunk_infos
    )


@app.get("/api/chunks", response_model=ChunkListResponse)
async def get_all_chunks(limit: int = 100, offset: int = 0):
    pgvector = get_pgvector()
    sql = """
        SELECT c.*, d.file_name 
        FROM chunks c 
        JOIN documents d ON c.doc_id = d.doc_id 
        ORDER BY d.file_name, c.chunk_index
        LIMIT %s OFFSET %s
    """
    
    with pgvector.conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks")
        total = cur.fetchone()[0]
        
        cur.execute(sql, (limit, offset))
        rows = cur.fetchall()
    
    chunk_infos = []
    for row in rows:
        chunk_id, doc_id, chunk_index, text = row[1], row[2], row[3], row[4]
        page_number = row[6]
        section_title = row[7]
        text_preview = text[:200] + "..." if len(text) > 200 else text
        chunk_infos.append(ChunkInfo(
            chunk_id=chunk_id,
            doc_id=doc_id,
            chunk_index=chunk_index,
            text=text,
            page_number=page_number,
            section_title=section_title,
            text_preview=text_preview
        ))
    
    return ChunkListResponse(total=total, chunks=chunk_infos)


@app.get("/api/images", response_model=ImageListResponse)
async def get_all_images(limit: int = 100, offset: int = 0):
    pgvector = get_pgvector()
    minio = get_minio()
    
    sql = """
        SELECT i.*, d.file_name 
        FROM images i 
        JOIN documents d ON i.doc_id = d.doc_id 
        ORDER BY d.file_name, i.page_number
        LIMIT %s OFFSET %s
    """
    
    with pgvector.conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM images")
        total = cur.fetchone()[0]
        
        cur.execute(sql, (limit, offset))
        rows = cur.fetchall()
    
    image_infos = []
    for row in rows:
        image_id = row[1]
        doc_id = row[2]
        image_path = row[3]
        minio_bucket = row[4]
        minio_object_name = row[5]
        caption = row[6]
        page_number = row[7]
        
        url = None
        if minio_object_name:
            try:
                from datetime import timedelta
                url = minio.client.presigned_get_object(
                    minio_bucket or minio.bucket,
                    minio_object_name,
                    expires=timedelta(hours=1)
                )
            except Exception as e:
                print(f"生成预签名URL失败: {e}")
        
        image_infos.append(ImageInfo(
            image_id=image_id,
            doc_id=doc_id,
            page_number=page_number,
            image_path=image_path,
            caption=caption,
            url=url
        ))
    
    return ImageListResponse(total=total, images=image_infos)


class DocumentUpdateRequest(BaseModel):
    """
    文档更新请求模型
    
    【用于更新文档元数据】
    目前支持：文件名重命名
    """
    file_name: Optional[str] = Field(None, description="新的文件名")


class DocumentUpdateResponse(BaseModel):
    """文档更新响应模型"""
    doc_id: str
    file_name: str
    message: str


@app.put("/api/documents/{doc_id}", response_model=DocumentUpdateResponse)
async def update_document(doc_id: str, request: DocumentUpdateRequest):
    """
    更新文档信息
    
    【功能说明】
    更新文档的元数据信息，目前支持重命名文件
    
    【参数】
    - doc_id: 文档ID
    - file_name: 新的文件名（可选）
    """
    pgvector = get_pgvector()
    
    doc = pgvector.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
    
    update_fields = []
    update_values = []
    
    if request.file_name is not None:
        update_fields.append("file_name = %s")
        update_values.append(request.file_name)
    
    if not update_fields:
        raise HTTPException(status_code=400, detail="没有提供要更新的字段")
    
    update_values.append(doc_id)
    sql = f"UPDATE documents SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP WHERE doc_id = %s"
    
    try:
        with pgvector.conn.cursor() as cur:
            cur.execute(sql, update_values)
        
        new_file_name = request.file_name if request.file_name else doc.file_name
        return DocumentUpdateResponse(
            doc_id=doc_id,
            file_name=new_file_name,
            message="文档更新成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    删除文档
    
    【级联删除】
    删除文档时会同时删除：
    - MinIO中的原始文件
    - MinIO中的所有关联图片
    - pgvector中的所有chunks
    - pgvector中的所有images
    - chunk_image_relations中的所有关联关系
    """
    pgvector = get_pgvector()
    minio = get_minio()
    
    doc = pgvector.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
    
    try:
        images_sql = """
            SELECT image_id, minio_bucket, minio_object_name 
            FROM images WHERE doc_id = %s
        """
        images_to_delete = []
        with pgvector.conn.cursor() as cur:
            cur.execute(images_sql, (doc_id,))
            for row in cur.fetchall():
                minio_bucket, minio_object_name = row[1], row[2]
                if minio_object_name:
                    images_to_delete.append((minio_bucket or minio.bucket, minio_object_name))
        
        if doc.minio_bucket and doc.minio_object_name:
            minio.delete_file(doc.minio_object_name, doc.minio_bucket)
            print(f"✓ 已删除原始文件: {doc.minio_object_name}")
        
        for bucket, object_name in images_to_delete:
            try:
                minio.delete_file(object_name, bucket)
                print(f"✓ 已删除图片: {object_name}")
            except Exception as e:
                print(f"⚠ 删除图片失败 {object_name}: {e}")
        
        pgvector.delete_document(doc_id)
        print(f"✓ 已删除数据库记录: {doc_id}")
        
        return {
            "message": f"文档已删除: {doc_id}", 
            "doc_id": doc_id,
            "deleted_images": len(images_to_delete)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


# ============================================================
# 对话和搜索接口
# ============================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG对话接口
    
    【这是核心功能接口】
    
    【处理流程】
    1. 接收用户问题
    2. 对问题进行向量化
    3. 在向量数据库中检索相关文档块
    4. 可选：使用Rerank对结果重排序
    5. 将检索结果作为上下文，调用LLM生成回答
    6. 返回回答和来源
    
    【请求示例】
    ```json
    {
        "query": "什么是RAG？",
        "top_k": 5,
        "use_rerank": true
    }
    ```
    """
    conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
    
    try:
        agent = get_pgvector_rag_agent()
        
        result = agent.chat(
            query=request.query,
            top_k=request.top_k,
            doc_ids=request.doc_ids,
            use_rerank=request.use_rerank
        )
        
        return ChatResponse(
            answer=result.answer,
            sources=result.sources,
            conversation_id=conversation_id,
            query=request.query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对话处理失败: {str(e)}")


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    向量相似度搜索
    
    【纯搜索接口，不生成回答】
    
    【处理流程】
    1. 对查询文本进行向量化
    2. 在pgvector中执行相似度搜索
    3. 返回最相关的文档块
    
    【请求示例】
    ```json
    {
        "query": "机器学习基础",
        "top_k": 10,
        "threshold": 0.7
    }
    ```
    """
    try:
        pgvector = get_pgvector()
        
        # 这里需要对查询进行向量化
        # 为了演示，我们使用模拟向量
        # 实际应该调用embedding模型
        
        # from llama_index.embeddings.openai import OpenAIEmbedding
        # embed_model = OpenAIEmbedding(...)
        # query_embedding = embed_model.get_query_embedding(request.query)
        
        # 模拟向量（实际应替换）
        query_embedding = [0.1] * 1536  # 假设维度为1536
        
        # 执行搜索
        results = pgvector.similarity_search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            filter_doc_ids=request.doc_ids,
            threshold=request.threshold
        )
        
        # 转换结果
        search_results = []
        for chunk, similarity in results:
            # 获取文档信息
            doc = pgvector.get_document(chunk.doc_id)
            file_name = doc.file_name if doc else None
            
            search_results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
                similarity=round(similarity, 4),
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                file_name=file_name
            ))
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total=len(search_results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


# ============================================================
# 系统状态接口
# ============================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查
    
    【检查各服务状态】
    - PostgreSQL连接
    - MinIO连接
    """
    postgres_status = "ok"
    minio_status = "ok"
    
    # 检查PostgreSQL
    try:
        pgvector = get_pgvector()
        # 执行简单查询测试连接
        pgvector.get_stats()
    except Exception:
        postgres_status = "error"
    
    # 检查MinIO
    try:
        minio = get_minio()
        # 检查bucket是否存在
        minio.client.bucket_exists(minio.bucket)
    except Exception:
        minio_status = "error"
    
    overall_status = "ok" if postgres_status == "ok" and minio_status == "ok" else "degraded"
    
    return HealthResponse(
        status=overall_status,
        postgres=postgres_status,
        minio=minio_status,
        timestamp=datetime.now()
    )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """
    获取系统统计信息
    
    【返回】
    - 文档总数
    - 文本块总数
    - 图片总数
    - 按状态分类的文档数
    """
    pgvector = get_pgvector()
    stats = pgvector.get_stats()
    
    return StatsResponse(
        total_documents=stats.get("total_documents", 0),
        total_chunks=stats.get("total_chunks", 0),
        total_images=stats.get("total_images", 0),
        documents_by_status=stats.get("documents_by_status", {})
    )


# ============================================================
# 错误处理
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("启动RAG知识库Agent系统API服务")
    print("=" * 60)
    print(f"API文档: http://localhost:{config.app.api_port}/docs")
    print(f"交互文档: http://localhost:{config.app.api_port}/redoc")
    print("=" * 60)
    
    uvicorn.run(
        "api:app",
        host=config.app.api_host,
        port=config.app.api_port,
        reload=True,
        log_level="info"
    )
