FROM python:3.12-slim

WORKDIR /app

# 系统依赖：tini 处理 PID 1 信号转发；curl 用于 HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY main.py .

# 非 root 用户 + 准备目录（usage_data + token 缓存挂载点）
RUN useradd -m -u 1000 -s /bin/bash app \
    && mkdir -p /app/usage_data /home/app/.config/databricks-claude-lb \
    && chown -R app:app /app /home/app/.config

USER app

EXPOSE 8000

# Pod 内自检：30s 间隔，3 次失败视为不健康（K8s probe 是真正的健康判定）
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/live || exit 1

# tini 接管 PID 1，确保 SIGTERM 正确转发，触发 uvicorn graceful shutdown
ENTRYPOINT ["/usr/bin/tini", "--"]

# uvicorn 已经在 main.py 里通过 if __name__ == "__main__" 启动；
# Docker 场景直接调 uvicorn 命令，避免 reload=True
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--timeout-keep-alive", "600", \
     "--timeout-graceful-shutdown", "30"]
