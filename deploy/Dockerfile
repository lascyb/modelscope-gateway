# ModelScope API Gateway
# 构建命令: docker build -t modelscope-gateway -f deploy/Dockerfile .
# 运行命令: docker run -p 8000:8000 --env-file .env modelscope-gateway

FROM python:3.11-slim

LABEL maintainer="ModelScope Gateway"
LABEL description="ModelScope API 负载均衡网关"

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制核心模块
COPY core/ ./core/

# 复制服务入口和配置
COPY server.py .
COPY models_config.json .

# 创建数据目录
RUN mkdir -p /app/usage /app/limits

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/')" || exit 1

# 启动命令
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
