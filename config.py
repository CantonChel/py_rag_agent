"""
配置管理模块

统一管理所有环境变量和配置项

【架构图】

┌─────────────────────────────────────────────────────────────────────┐
│                        系统架构                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │  前端页面   │────▶│  FastAPI    │────▶│   LLM/嵌入  │           │
│  │  (Vue.js)   │     │   后端      │     │    模型     │           │
│  └─────────────┘     └──────┬──────┘     └─────────────┘           │
│                             │                                      │
│              ┌──────────────┼──────────────┐                       │
│              ▼              ▼              ▼                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │   MinIO     │  │ PostgreSQL  │  │   Redis     │                │
│  │  原始文件   │  │ + pgvector  │  │   缓存      │                │
│  │  对象存储   │  │  向量存储   │  │             │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
"""

import os
from typing import Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """大语言模型配置"""
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("LLM_BASE_URL"))
    model_name: str = field(default_factory=lambda: os.getenv("LLM_MODEL_NAME", "gpt-4-turbo"))


@dataclass
class EmbeddingConfig:
    """Embedding模型配置"""
    api_key: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", ""))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("EMBEDDING_BASE_URL"))
    model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"))
    
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
        """获取模型维度"""
        return self.KNOWN_DIMENSIONS.get(self.model_name, 1536)


@dataclass
class RerankConfig:
    """Rerank模型配置"""
    enabled: bool = field(default_factory=lambda: os.getenv("ENABLE_RERANK", "false").lower() == "true")
    api_key: str = field(default_factory=lambda: os.getenv("RERANK_API_KEY", ""))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("RERANK_BASE_URL"))
    model_name: str = field(default_factory=lambda: os.getenv("RERANK_MODEL_NAME", "rerank-multilingual-v3.0"))


@dataclass
class MinIOConfig:
    """
    MinIO对象存储配置
    
    【MinIO的作用】
    存储用户上传的原始文件（PDF、Word等）
    - 支持大文件存储
    - 支持文件版本管理
    - 提供预签名URL用于下载
    """
    endpoint: str = field(default_factory=lambda: os.getenv("MINIO_ENDPOINT", "localhost:9000"))
    access_key: str = field(default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
    secret_key: str = field(default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "minioadmin123"))
    bucket: str = field(default_factory=lambda: os.getenv("MINIO_BUCKET", "rag-documents"))
    secure: bool = field(default_factory=lambda: os.getenv("MINIO_SECURE", "false").lower() == "true")


@dataclass
class PostgresConfig:
    """
    PostgreSQL + pgvector配置
    
    【pgvector的作用】
    1. 存储文本块和元数据
    2. 存储向量并进行相似度搜索
    3. 维护chunk之间的关系
    
    【数据库URL格式】
    postgresql://user:password@host:port/database
    """
    host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "postgres123"))
    database: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "rag_db"))
    
    @property
    def url(self) -> str:
        """获取数据库连接URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_url(self) -> str:
        """获取异步数据库连接URL"""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    """Redis缓存配置"""
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD") or None)
    db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    
    @property
    def url(self) -> str:
        """获取Redis连接URL"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


@dataclass
class AppConfig:
    """应用配置"""
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    max_file_size: int = field(default_factory=lambda: int(os.getenv("MAX_FILE_SIZE", "104857600")))
    allowed_extensions: List[str] = field(default_factory=lambda: 
        os.getenv("ALLOWED_EXTENSIONS", ".pdf,.docx,.doc,.html,.md,.txt").split(","))


class Config:
    """
    全局配置类
    
    【访问方式】
    config.llm.api_key
    config.minio.endpoint
    config.postgres.url
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self):
        self.llm = LLMConfig()
        self.embedding = EmbeddingConfig()
        self.rerank = RerankConfig()
        self.minio = MinIOConfig()
        self.postgres = PostgresConfig()
        self.redis = RedisConfig()
        self.app = AppConfig()
    
    def validate(self) -> bool:
        """验证必要的配置"""
        errors = []
        
        if not self.llm.api_key:
            errors.append("LLM_API_KEY 未设置")
        
        if not self.embedding.api_key:
            errors.append("EMBEDDING_API_KEY 未设置")
        
        if errors:
            print("❌ 配置验证失败：")
            for error in errors:
                print(f"   - {error}")
            return False
        
        return True
    
    def print_config(self):
        """打印当前配置"""
        print("="*60)
        print("【当前配置】")
        print("="*60)
        print(f"LLM: {self.llm.model_name}")
        print(f"Embedding: {self.embedding.model_name} (维度: {self.embedding.get_dimension()})")
        print(f"Rerank: {'启用' if self.rerank.enabled else '禁用'}")
        print("-"*60)
        print(f"MinIO: {self.minio.endpoint}/{self.minio.bucket}")
        print(f"PostgreSQL: {self.postgres.host}:{self.postgres.port}/{self.postgres.database}")
        print(f"Redis: {self.redis.host}:{self.redis.port}")
        print("-"*60)
        print(f"API: http://{self.app.api_host}:{self.app.api_port}")
        print(f"分块: {self.app.chunk_size} tokens, 重叠 {self.app.chunk_overlap}")
        print("="*60)


config = Config()


if __name__ == "__main__":
    config.print_config()
