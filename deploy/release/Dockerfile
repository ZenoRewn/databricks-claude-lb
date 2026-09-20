ARG DEPENDENCY_STAGE=dependencies-online
FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9 AS base
WORKDIR /app
COPY operations/release/requirements.lock /app/requirements.lock
FROM base AS dependencies-online
RUN pip install --no-cache-dir -r requirements.lock
FROM base AS dependencies-offline
RUN --mount=type=bind,source=.release-wheelhouse,target=/wheelhouse,readonly \
    pip install --no-cache-dir --no-index --find-links=/wheelhouse -r requirements.lock
FROM ${DEPENDENCY_STAGE} AS runtime
COPY operations/__init__.py /app/operations/__init__.py
COPY operations/release /app/operations/release
RUN useradd -m -u 1000 app
USER 1000
EXPOSE 9000
ENTRYPOINT ["python","-m","operations.release.controller"]
