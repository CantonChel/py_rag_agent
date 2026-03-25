"""
MinIO文件存储模块

用于存储和管理上传的原始文件

【MinIO存储架构】

┌─────────────────────────────────────────────────────────────────────┐
│                      MinIO存储结构                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Bucket: rag-documents                                              │
│  ├── documents/                      # 原始文档                     │
│  │   ├── doc_001.pdf                                               │
│  │   ├── doc_002.docx                                              │
│  │   └── ...                                                        │
│  ├── images/                         # 提取的图片                   │
│  │   ├── doc_001_page1_img1.png                                    │
│  │   └── ...                                                        │
│  └── temp/                           # 临时文件                     │
│      └── upload_xxx.tmp                                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

【文件处理流程】
1. 用户上传文件 → 临时存储到temp/
2. 验证文件类型和大小
3. 移动到documents/目录，生成唯一文件名
4. 返回文件路径用于后续处理
"""

import io
import uuid
import hashlib
from typing import Optional, List, BinaryIO, Dict, Any
from datetime import timedelta
from pathlib import Path

from minio import Minio
from minio.error import S3Error
from minio.deleteobjects import DeleteObject

from config import config


class MinIOStorage:
    """
    MinIO文件存储管理器
    
    【主要功能】
    1. 文件上传和下载
    2. 文件删除
    3. 预签名URL生成（用于直接下载）
    4. 存储桶管理
    """
    
    def __init__(self):
        """初始化MinIO客户端"""
        self.client = Minio(
            config.minio.endpoint,
            access_key=config.minio.access_key,
            secret_key=config.minio.secret_key,
            secure=config.minio.secure
        )
        self.bucket = config.minio.bucket
        self._ensure_bucket()
    
    def _ensure_bucket(self):
        """确保存储桶存在"""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                print(f"✓ 创建MinIO存储桶: {self.bucket}")
        except S3Error as e:
            print(f"✗ MinIO存储桶创建失败: {e}")
    
    def upload_file(
        self,
        file_data: BinaryIO,
        file_name: str,
        content_type: Optional[str] = None,
        prefix: str = "documents",
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        上传文件到MinIO
        
        【上传流程】
        1. 生成唯一对象名称
        2. 计算文件大小
        3. 上传到MinIO
        4. 返回文件信息
        
        参数:
            file_data: 文件数据流
            file_name: 原始文件名
            content_type: MIME类型
            prefix: 存储路径前缀
            metadata: 文件元数据
            
        返回:
            上传结果字典
        """
        # 生成唯一对象名称
        file_ext = Path(file_name).suffix
        unique_id = str(uuid.uuid4())
        object_name = f"{prefix}/{unique_id}{file_ext}"
        
        # 获取文件大小
        file_data.seek(0, 2)
        file_size = file_data.tell()
        file_data.seek(0)
        
        # 默认content_type
        if not content_type:
            content_type = self._get_content_type(file_ext)
        
        try:
            # 上传文件
            self.client.put_object(
                self.bucket,
                object_name,
                file_data,
                file_size,
                content_type=content_type,
                metadata=metadata
            )
            
            return {
                "success": True,
                "object_name": object_name,
                "bucket": self.bucket,
                "file_name": file_name,
                "file_size": file_size,
                "content_type": content_type,
                "message": "文件上传成功"
            }
        except S3Error as e:
            return {
                "success": False,
                "error": str(e),
                "message": "文件上传失败"
            }
    
    def upload_bytes(
        self,
        data: bytes,
        file_name: str,
        content_type: Optional[str] = None,
        prefix: str = "documents"
    ) -> Dict[str, Any]:
        """
        上传字节数据到MinIO
        
        参数:
            data: 字节数据
            file_name: 文件名
            content_type: MIME类型
            prefix: 存储路径前缀
            
        返回:
            上传结果
        """
        return self.upload_file(
            io.BytesIO(data),
            file_name,
            content_type,
            prefix
        )
    
    def download_file(self, object_name: str) -> Optional[bytes]:
        """
        从MinIO下载文件
        
        参数:
            object_name: 对象名称
            
        返回:
            文件字节数据，失败返回None
        """
        try:
            response = self.client.get_object(self.bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            print(f"✗ 文件下载失败: {e}")
            return None
    
    def get_file_stream(self, object_name: str):
        """
        获取文件流（用于大文件下载）
        
        参数:
            object_name: 对象名称
            
        返回:
            文件流对象
        """
        try:
            return self.client.get_object(self.bucket, object_name)
        except S3Error as e:
            print(f"✗ 获取文件流失败: {e}")
            return None
    
    def delete_file(self, object_name: str) -> bool:
        """
        删除文件
        
        参数:
            object_name: 对象名称
            
        返回:
            是否删除成功
        """
        try:
            self.client.remove_object(self.bucket, object_name)
            return True
        except S3Error as e:
            print(f"✗ 文件删除失败: {e}")
            return False
    
    def delete_files(self, object_names: List[str]) -> Dict[str, int]:
        """
        批量删除文件
        
        参数:
            object_names: 对象名称列表
            
        返回:
            删除统计
        """
        delete_objects = [DeleteObject(name) for name in object_names]
        
        try:
            errors = self.client.remove_objects(self.bucket, delete_objects)
            error_count = len(list(errors))
            return {
                "total": len(object_names),
                "deleted": len(object_names) - error_count,
                "errors": error_count
            }
        except S3Error as e:
            print(f"✗ 批量删除失败: {e}")
            return {"total": len(object_names), "deleted": 0, "errors": len(object_names)}
    
    def get_presigned_url(
        self,
        object_name: str,
        expires: timedelta = timedelta(hours=1)
    ) -> Optional[str]:
        """
        获取预签名下载URL
        
        【用途】
        生成一个临时URL，用户可以直接通过浏览器下载文件
        无需经过后端服务器
        
        参数:
            object_name: 对象名称
            expires: URL过期时间
            
        返回:
            预签名URL
        """
        try:
            url = self.client.presigned_get_object(
                self.bucket,
                object_name,
                expires=expires
            )
            return url
        except S3Error as e:
            print(f"✗ 生成预签名URL失败: {e}")
            return None
    
    def file_exists(self, object_name: str) -> bool:
        """
        检查文件是否存在
        
        参数:
            object_name: 对象名称
            
        返回:
            文件是否存在
        """
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error:
            return False
    
    def get_file_info(self, object_name: str) -> Optional[Dict[str, Any]]:
        """
        获取文件信息
        
        参数:
            object_name: 对象名称
            
        返回:
            文件信息字典
        """
        try:
            stat = self.client.stat_object(self.bucket, object_name)
            return {
                "object_name": object_name,
                "size": stat.size,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified,
                "etag": stat.etag,
                "metadata": stat.metadata
            }
        except S3Error as e:
            print(f"✗ 获取文件信息失败: {e}")
            return None
    
    def list_files(
        self,
        prefix: str = "",
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        列出文件
        
        参数:
            prefix: 路径前缀
            recursive: 是否递归列出
            
        返回:
            文件列表
        """
        files = []
        try:
            objects = self.client.list_objects(
                self.bucket,
                prefix=prefix,
                recursive=recursive
            )
            for obj in objects:
                files.append({
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "etag": obj.etag,
                    "is_dir": obj.is_dir
                })
        except S3Error as e:
            print(f"✗ 列出文件失败: {e}")
        
        return files
    
    def calculate_md5(self, file_data: BinaryIO) -> str:
        """
        计算文件MD5值
        
        参数:
            file_data: 文件数据流
            
        返回:
            MD5哈希值
        """
        file_data.seek(0)
        md5_hash = hashlib.md5()
        for chunk in iter(lambda: file_data.read(8192), b""):
            md5_hash.update(chunk)
        file_data.seek(0)
        return md5_hash.hexdigest()
    
    def _get_content_type(self, file_ext: str) -> str:
        """
        根据文件扩展名获取MIME类型
        
        参数:
            file_ext: 文件扩展名
            
        返回:
            MIME类型
        """
        content_types = {
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".ppt": "application/vnd.ms-powerpoint",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".txt": "text/plain",
            ".html": "text/html",
            ".htm": "text/html",
            ".md": "text/markdown",
            ".json": "application/json",
            ".xml": "application/xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
        }
        return content_types.get(file_ext.lower(), "application/octet-stream")


# 创建全局实例
minio_storage = MinIOStorage()


if __name__ == "__main__":
    # 测试MinIO连接
    print("测试MinIO连接...")
    files = minio_storage.list_files()
    print(f"存储桶中的文件数: {len(files)}")
    
    # 测试上传
    test_content = b"Hello, MinIO!"
    result = minio_storage.upload_bytes(
        test_content,
        "test.txt",
        prefix="temp"
    )
    print(f"上传结果: {result}")
    
    if result["success"]:
        # 测试下载
        data = minio_storage.download_file(result["object_name"])
        print(f"下载内容: {data}")
        
        # 测试预签名URL
        url = minio_storage.get_presigned_url(result["object_name"])
        print(f"预签名URL: {url[:50]}...")
        
        # 清理测试文件
        minio_storage.delete_file(result["object_name"])
        print("测试文件已删除")
