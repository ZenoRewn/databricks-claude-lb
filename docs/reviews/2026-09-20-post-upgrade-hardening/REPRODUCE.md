# 本地验证重现入口

Author: Zeno Ren

以下命令只准备本地环境。隔离 MySQL 与 Kind 的创建、证书、镜像及故障入口见 [测试说明](../../../tests/release/README.md)。禁止替换为默认 Kubernetes context。

本轮的三节点定义保存在 [kind.yaml](kind.yaml)，Python 3.11/3.13 回归构建夹具保存在 [Dockerfile.test-matrix](Dockerfile.test-matrix)。夹具的 Python 标签不代表不可变版本；每轮都要记录实际镜像 ID。

```bash
python -m pip install -r requirements-test.lock
python -m pytest tests/ -q --tb=short

docker build --platform linux/amd64 \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t claude-lb:local-review .

docker build --platform linux/amd64 -f deploy/release/Dockerfile \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t lb-release-controller:local-review .
```

本轮使用经过 PyPI SHA-256 核对的离线 wheel 和 `DEPENDENCY_STAGE=dependencies-offline`。生产应用镜像内测试示例（测试文件与说明文档只读，应用源码不挂载）：

```bash
docker run --rm --network none --platform linux/amd64 --user root \
  -e PYTHONPATH=/app \
  -v "$PWD/tests:/app/tests:ro" \
  -v "$PWD/operations:/app/operations:ro" \
  -v "$PWD/docs:/app/docs:ro" \
  -v "$PWD/CLAUDE.md:/app/CLAUDE.md:ro" \
  -v "$PWD/.test-wheelhouse:/wheels:ro" \
  -v "$PWD/requirements-test.lock:/app/requirements-test.lock:ro" \
  --entrypoint sh claude-lb:local-review -c \
  'pip install --no-index --find-links=/wheels -r /app/requirements-test.lock && python -m pytest /app/tests -q --tb=short'
```

协调器镜像只挂载 `tests` 与离线 pytest 依赖，执行 `pytest /app/tests/release`；不得覆盖它的 `/app/operations`。应用镜像不包含发布器，因此应用回归中独立挂载 operations 工具不构成对发布器候选镜像的验证。

Kind 普通发布、各阶段中断和故障模式使用 `tests/release/run_kind_scenarios.py`。另执行 `tests/release/run_kind_contention.py --context ... --kubeconfig ... --output NEW_FILE.json` 检查十轮真实 Lease 竞争。每个场景使用新的 ID/输出文件，不覆盖失败材料。

版本、基础镜像、平台与实际回执必须跟随新一轮测试重新记录。不要把本报告的 local image ID 或已清理的 loopback registry digest 复用为可拉取的生产镜像。
