"""
接口测试脚本

用于测试RAG系统API接口的功能

【测试范围】
1. 健康检查接口
2. 文档上传接口
3. 文档列表接口
4. 向量搜索接口
5. 对话接口

【使用方法】
python test_api.py

【前置条件】
1. Docker容器已启动（PostgreSQL + pgvector + MinIO）
2. FastAPI服务已启动（python api.py）
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional

# API基础地址
API_BASE = "http://localhost:8000"

# 测试结果记录
test_results = []


def print_header(title: str):
    """打印测试标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(test_name: str, success: bool, message: str = "", data: Any = None):
    """打印测试结果"""
    status = "✅ 通过" if success else "❌ 失败"
    print(f"\n{status} - {test_name}")
    if message:
        print(f"   消息: {message}")
    if data:
        print(f"   数据: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}")
    
    test_results.append({
        "test": test_name,
        "success": success,
        "message": message
    })


def test_health():
    """
    测试健康检查接口
    
    【接口】GET /api/health
    【预期】返回各服务的健康状态
    """
    print_header("测试健康检查接口")
    
    try:
        response = requests.get(f"{API_BASE}/api/health", timeout=10)
        data = response.json()
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        success = (
            response.status_code == 200 and
            data.get("status") in ["ok", "degraded"]
        )
        
        print_result(
            "健康检查",
            success,
            f"系统状态: {data.get('status')}, PostgreSQL: {data.get('postgres')}, MinIO: {data.get('minio')}"
        )
        return success
        
    except Exception as e:
        print_result("健康检查", False, str(e))
        return False


def test_stats():
    """
    测试统计信息接口
    
    【接口】GET /api/stats
    【预期】返回文档、文本块、图片的统计数量
    """
    print_header("测试统计信息接口")
    
    try:
        response = requests.get(f"{API_BASE}/api/stats", timeout=10)
        data = response.json()
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        success = response.status_code == 200 and "total_documents" in data
        
        print_result(
            "统计信息",
            success,
            f"文档: {data.get('total_documents')}, 文本块: {data.get('total_chunks')}, 图片: {data.get('total_images')}"
        )
        return data
        
    except Exception as e:
        print_result("统计信息", False, str(e))
        return None


def test_upload_document(file_path: str = None):
    """
    测试文档上传接口
    
    【接口】POST /api/documents/upload
    【预期】上传文件成功，返回doc_id
    """
    print_header("测试文档上传接口")
    
    # 如果没有指定文件，创建一个测试文件
    if not file_path or not os.path.exists(file_path):
        test_content = """
# RAG知识库系统测试文档

这是一个用于测试RAG知识库系统的示例文档。

## 什么是RAG？

RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。
它首先从知识库中检索相关文档，然后将检索结果作为上下文输入给大语言模型，生成更准确的回答。

## RAG的优势

1. 减少幻觉：基于真实文档生成回答
2. 知识更新：可以随时更新知识库
3. 可追溯：回答有明确的来源

## 技术架构

RAG系统通常包含以下组件：
- 文档处理器：负责解析和分块
- 向量数据库：存储文档向量
- 检索引擎：进行相似度搜索
- 大语言模型：生成最终回答
        """
        
        # 创建临时测试文件
        test_file = "/tmp/test_rag_document.md"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)
        file_path = test_file
        print(f"创建测试文件: {file_path}")
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "text/markdown")}
            response = requests.post(
                f"{API_BASE}/api/documents/upload",
                files=files,
                timeout=30
            )
        
        data = response.json()
        print(f"响应状态码: {response.status_code}")
        print(f"响应数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        success = (
            response.status_code == 200 and
            "doc_id" in data
        )
        
        print_result(
            "文档上传",
            success,
            data.get("message", ""),
            {"doc_id": data.get("doc_id"), "file_name": data.get("file_name")}
        )
        
        return data.get("doc_id") if success else None
        
    except Exception as e:
        print_result("文档上传", False, str(e))
        return None


def test_list_documents():
    """
    测试文档列表接口
    
    【接口】GET /api/documents
    【预期】返回文档列表
    """
    print_header("测试文档列表接口")
    
    try:
        response = requests.get(
            f"{API_BASE}/api/documents",
            params={"limit": 20, "offset": 0},
            timeout=10
        )
        data = response.json()
        
        print(f"响应状态码: {response.status_code}")
        print(f"文档数量: {data.get('total', 0)}")
        
        # 打印文档列表
        for doc in data.get("documents", []):
            print(f"  - {doc.get('file_name')} ({doc.get('status')})")
        
        success = response.status_code == 200 and "documents" in data
        
        print_result(
            "文档列表",
            success,
            f"共 {data.get('total', 0)} 个文档"
        )
        
        return data.get("documents", [])
        
    except Exception as e:
        print_result("文档列表", False, str(e))
        return []


def test_get_document(doc_id: str):
    """
    测试获取单个文档接口
    
    【接口】GET /api/documents/{doc_id}
    【预期】返回文档详情
    """
    print_header("测试获取单个文档接口")
    
    if not doc_id:
        print_result("获取文档", False, "没有可用的文档ID")
        return None
    
    try:
        response = requests.get(
            f"{API_BASE}/api/documents/{doc_id}",
            timeout=10
        )
        data = response.json()
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        success = response.status_code == 200 and data.get("doc_id") == doc_id
        
        print_result(
            "获取文档",
            success,
            f"文档: {data.get('file_name')}, 状态: {data.get('status')}"
        )
        
        return data
        
    except Exception as e:
        print_result("获取文档", False, str(e))
        return None


def test_search(query: str = "什么是RAG"):
    """
    测试向量搜索接口
    
    【接口】POST /api/search
    【预期】返回相关文档块
    """
    print_header("测试向量搜索接口")
    
    try:
        response = requests.post(
            f"{API_BASE}/api/search",
            json={
                "query": query,
                "top_k": 5
            },
            timeout=30
        )
        data = response.json()
        
        print(f"响应状态码: {response.status_code}")
        print(f"查询: {data.get('query')}")
        print(f"结果数: {data.get('total', 0)}")
        
        # 打印搜索结果
        for i, result in enumerate(data.get("results", [])[:3], 1):
            print(f"\n  结果 {i}:")
            print(f"    相似度: {result.get('similarity', 0):.4f}")
            print(f"    文本: {result.get('text', '')[:100]}...")
        
        success = response.status_code == 200
        
        print_result(
            "向量搜索",
            success,
            f"查询 '{query}'，找到 {data.get('total', 0)} 个结果"
        )
        
        return data
        
    except Exception as e:
        print_result("向量搜索", False, str(e))
        return None


def test_chat(query: str = "请介绍一下RAG技术的优势"):
    """
    测试对话接口
    
    【接口】POST /api/chat
    【预期】返回回答和来源
    """
    print_header("测试对话接口")
    
    try:
        response = requests.post(
            f"{API_BASE}/api/chat",
            json={
                "query": query,
                "top_k": 5,
                "use_rerank": True
            },
            timeout=60
        )
        data = response.json()
        
        print(f"响应状态码: {response.status_code}")
        print(f"查询: {data.get('query')}")
        print(f"回答: {data.get('answer', '')[:200]}...")
        print(f"来源数: {len(data.get('sources', []))}")
        
        success = response.status_code == 200 and "answer" in data
        
        print_result(
            "对话接口",
            success,
            f"会话ID: {data.get('conversation_id')}"
        )
        
        return data
        
    except Exception as e:
        print_result("对话接口", False, str(e))
        return None


def test_delete_document(doc_id: str):
    """
    测试删除文档接口
    
    【接口】DELETE /api/documents/{doc_id}
    【预期】删除成功
    """
    print_header("测试删除文档接口")
    
    if not doc_id:
        print_result("删除文档", False, "没有可用的文档ID")
        return False
    
    try:
        response = requests.delete(
            f"{API_BASE}/api/documents/{doc_id}",
            timeout=10
        )
        data = response.json()
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        success = response.status_code == 200
        
        print_result(
            "删除文档",
            success,
            data.get("message", "")
        )
        
        return success
        
    except Exception as e:
        print_result("删除文档", False, str(e))
        return False


def run_all_tests():
    """
    运行所有测试
    
    【测试顺序】
    1. 健康检查
    2. 统计信息
    3. 文档上传
    4. 文档列表
    5. 获取文档详情
    6. 向量搜索
    7. 对话接口
    8. 删除文档
    """
    print("\n" + "=" * 60)
    print("  RAG知识库Agent系统 - API接口测试")
    print("=" * 60)
    print(f"API地址: {API_BASE}")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 健康检查
    if not test_health():
        print("\n❌ 服务不可用，请确保Docker容器和FastAPI服务已启动")
        print("\n启动命令:")
        print("  1. docker-compose up -d")
        print("  2. python api.py")
        return
    
    # 2. 统计信息
    test_stats()
    
    # 3. 文档上传
    doc_id = test_upload_document()
    
    # 4. 文档列表
    test_list_documents()
    
    # 5. 获取文档详情
    if doc_id:
        test_get_document(doc_id)
        
        # 等待文档处理完成
        print("\n等待文档处理...")
        time.sleep(3)
    
    # 6. 向量搜索
    test_search()
    
    # 7. 对话接口
    test_chat()
    
    # 8. 删除文档（清理测试数据）
    if doc_id:
        test_delete_document(doc_id)
    
    # 打印测试汇总
    print_summary()


def print_summary():
    """打印测试汇总"""
    print("\n" + "=" * 60)
    print("  测试汇总")
    print("=" * 60)
    
    total = len(test_results)
    passed = sum(1 for r in test_results if r["success"])
    failed = total - passed
    
    print(f"\n总测试数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"通过率: {passed/total*100:.1f}%" if total > 0 else "N/A")
    
    if failed > 0:
        print("\n失败的测试:")
        for r in test_results:
            if not r["success"]:
                print(f"  ❌ {r['test']}: {r['message']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 检查是否指定了自定义API地址
    if len(sys.argv) > 1:
        API_BASE = sys.argv[1]
    
    run_all_tests()
