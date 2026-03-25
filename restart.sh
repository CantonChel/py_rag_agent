#!/bin/bash

# ============================================================
# RAG项目重启脚本
# 顺序: 停止Docker -> 停止主进程 -> 启动Docker -> 启动主进程
# ============================================================

set -e

PROJECT_DIR="/Users/luolinhao/Documents/trae_projects/py_agentic_rag"
cd "$PROJECT_DIR"

echo "=========================================="
echo "RAG项目重启脚本"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================
# 第一步: 停止主进程 (API 8000, Frontend 3000)
# ============================================================
echo ""
echo -e "${YELLOW}[1/4] 停止主进程...${NC}"

# 停止API服务 (端口8000)
API_PID=$(lsof -ti :8000 2>/dev/null || true)
if [ -n "$API_PID" ]; then
    echo "  停止API服务 (PID: $API_PID)..."
    kill -9 $API_PID 2>/dev/null || true
    echo -e "  ${GREEN}✓ API服务已停止${NC}"
else
    echo "  API服务未运行"
fi

# 停止前端服务 (端口3000)
FRONTEND_PID=$(lsof -ti :3000 2>/dev/null || true)
if [ -n "$FRONTEND_PID" ]; then
    echo "  停止前端服务 (PID: $FRONTEND_PID)..."
    kill -9 $FRONTEND_PID 2>/dev/null || true
    echo -e "  ${GREEN}✓ 前端服务已停止${NC}"
else
    echo "  前端服务未运行"
fi

# ============================================================
# 第二步: 停止Docker容器
# ============================================================
echo ""
echo -e "${YELLOW}[2/4] 停止Docker容器...${NC}"
docker-compose down
echo -e "${GREEN}✓ Docker容器已停止${NC}"

# ============================================================
# 第三步: 启动Docker容器
# ============================================================
echo ""
echo -e "${YELLOW}[3/4] 启动Docker容器...${NC}"
docker-compose up -d

# 等待PostgreSQL就绪
echo "  等待PostgreSQL就绪..."
sleep 3
for i in {1..30}; do
    if docker exec rag_postgres pg_isready -U postgres > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓ PostgreSQL已就绪${NC}"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

# 等待MinIO就绪
echo "  等待MinIO就绪..."
for i in {1..30}; do
    if curl -s http://localhost:19000/minio/health/live > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓ MinIO已就绪${NC}"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

echo -e "${GREEN}✓ Docker容器已启动${NC}"

# ============================================================
# 第四步: 启动主进程
# ============================================================
echo ""
echo -e "${YELLOW}[4/4] 启动主进程...${NC}"

# 启动API服务
echo "  启动API服务..."
source .venv/bin/activate
nohup python api.py > logs/api.log 2>&1 &
API_PID=$!
echo "  API服务已启动 (PID: $API_PID)"

# 等待API服务就绪
echo "  等待API服务就绪..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓ API服务已就绪${NC}"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

# 启动前端服务
echo "  启动前端服务..."
cd frontend
nohup python -m http.server 3000 > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "  前端服务已启动 (PID: $FRONTEND_PID)"

# ============================================================
# 完成
# ============================================================
echo ""
echo "=========================================="
echo -e "${GREEN}✓ 重启完成!${NC}"
echo "=========================================="
echo ""
echo "服务地址:"
echo "  - 前端页面:   http://localhost:3000"
echo "  - API文档:    http://localhost:8000/docs"
echo "  - MinIO控制台: http://localhost:19001"
echo ""
echo "日志文件:"
echo "  - API日志:    logs/api.log"
echo "  - 前端日志:   logs/frontend.log"
echo ""
