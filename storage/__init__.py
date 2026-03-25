"""
存储模块初始化
"""
from .minio_storage import minio_storage, MinIOStorage

__all__ = [
    "minio_storage",
    "MinIOStorage",
]
