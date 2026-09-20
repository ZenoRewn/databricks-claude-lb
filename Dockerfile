ARG DEPENDENCY_STAGE=dependencies-online
FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9 AS base

WORKDIR /app

# 系统依赖：tini 处理 PID 1 信号转发；curl 用于 HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt requirements.lock ./

FROM base AS dependencies-online
RUN pip install --no-cache-dir -r requirements.lock

FROM base AS dependencies-offline
RUN --mount=type=bind,source=.wheelhouse,target=/wheelhouse,readonly \
    pip install --no-cache-dir --no-index --find-links=/wheelhouse -r requirements.lock

FROM ${DEPENDENCY_STAGE} AS runtime

# 应用代码
# main.py 在 2026-09-08 拆出了本地模块与静态资源，必须一并入镜像：
#   usage_store.py  —— 顶层 import（main.py:1888），无保护；缺失 → ModuleNotFoundError，启动即崩
#   otel_setup.py   —— lifespan 内 import + except Exception 兜底；缺失 tracing 静默失能
#   dashboard.html  —— import 时按 __file__ 同级路径读取，except OSError 降级空壳
COPY main.py effort_compat.py request_telemetry.py request_budget.py admission.py gateway_lifecycle.py upstream_body.py usage_store.py otel_setup.py response_semantics.py cleanup_observability.py release_probe.py dashboard.html ./

ARG SOURCE_REVISION=working-tree
RUN SOURCE_REVISION="$SOURCE_REVISION" python -c 'import os,json,pathlib,hashlib,importlib.metadata as m,subprocess; pathlib.Path("/app/build-info.json").write_text(json.dumps({"source_revision":os.environ["SOURCE_REVISION"],"files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in pathlib.Path("/app").glob("*.py")},"packages":{d.metadata["Name"]:d.version for d in m.distributions()},"os_packages":subprocess.check_output(["dpkg-query","-W"],text=True)},sort_keys=True))'

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
