-- 初始化PostgreSQL数据库
-- 创建pgvector扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 创建文档表
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id VARCHAR(255) UNIQUE NOT NULL,
    file_name VARCHAR(500) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_path VARCHAR(1000),
    file_size BIGINT,
    minio_bucket VARCHAR(255),
    minio_object_name VARCHAR(500),
    total_pages INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建文本块表
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id VARCHAR(255) UNIQUE NOT NULL,
    doc_id VARCHAR(255) NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding vector(1536),
    page_number INTEGER,
    section_title VARCHAR(500),
    start_char INTEGER DEFAULT 0,
    end_char INTEGER DEFAULT 0,
    prev_chunk_id VARCHAR(255),
    next_chunk_id VARCHAR(255),
    parent_chunk_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建图片表
CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id VARCHAR(255) UNIQUE NOT NULL,
    doc_id VARCHAR(255) NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    image_path VARCHAR(1000),
    minio_bucket VARCHAR(255),
    minio_object_name VARCHAR(500),
    caption TEXT,
    page_number INTEGER,
    image_embedding vector(512),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建chunk与image关联表
CREATE TABLE IF NOT EXISTS chunk_image_relations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id VARCHAR(255) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    image_id VARCHAR(255) REFERENCES images(image_id) ON DELETE CASCADE,
    UNIQUE(chunk_id, image_id)
);

-- 创建向量索引（使用IVFFlat）
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- 创建普通索引
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS chunks_chunk_index_idx ON chunks(doc_id, chunk_index);
CREATE INDEX IF NOT EXISTS images_doc_id_idx ON images(doc_id);
CREATE INDEX IF NOT EXISTS documents_status_idx ON documents(status);

-- 创建更新时间触发器
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- 插入测试数据（可选）
-- INSERT INTO documents (doc_id, file_name, file_type, status) 
-- VALUES ('test_doc_001', 'test.pdf', 'pdf', 'completed');
