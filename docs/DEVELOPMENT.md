# 开发与验证

Author: Zeno Ren

Python 3.11+，推荐 3.12；前端为独立 HTML、CSS 和原生 JavaScript，没有 React/npm 构建链。Node.js 仅用于 UI 行为回归，Chart.js 使用固定版本及 SRI 校验。

## 模块边界

| 范围 | 主要文件 |
|---|---|
| HTTP 路由、provider 代理与负载均衡 | `main.py` |
| 准入、请求预算、生命周期 | `admission.py`、`request_budget.py`、`gateway_lifecycle.py` |
| 请求与发送观测、安全诊断、阶段时间 | `request_telemetry.py`、`safe_diagnostics.py`、`request_timing.py` |
| 协议结果、Chat 适配、上游错误体 | `response_semantics.py`、`chat_adapter.py`、`upstream_body.py` |
| 参数与渠道能力 | `effort_compat.py`、`model_capabilities.py` |
| 用量与 Copilot 估算 | `usage_store.py`、`copilot_pricing.py` |
| Dashboard 与构建身份 | `dashboard.html`、`build_metadata.py` |
| 报告、实验、构建与受控发布 | `operations/` |

## 回归

```bash
python -m pip install -r requirements-test.lock
python -m pytest tests/ -q --tb=short
node tests/test_copilot_pricing_ui.cjs
node tests/test_dashboard_ui.cjs
```

MySQL 的 opt-in 测试必须使用一次性隔离实例，不指向生产。发布工具的 Kind/中断/接管场景有独立环境与证据要求，见 [发布测试说明](../tests/release/README.md)。不为通过 CI 削弱协议、账本或退出码断言。

## UI 预览

```bash
python tests/dashboard_preview.py
```

访问终端打印的 loopback 地址。全部渠道、账本和版本均为合成数据，没有生产数据库或上游调用；模拟 DELETE 不删除任何数据。`?scenario=empty`、`error`、`unknown`、`modified` 用于未配置、请求失败和版本状态验收。

界面采用语义 CSS 变量、清晰的信息层级、键盘导航和 reduced-motion 处理，参考 BoardUI 设计规则并保留现有技术栈；不将其称为 BoardUI React 组件实现。核验深浅主题、表格溢出、四个视图、手动/暂停刷新、版本链接和失败提示。Chart.js 无法加载时仍显示表格。

版本 API `/version` 仅返回提交及文件匹配状态。前端链接只由合法 40 位 Git SHA 和固定仓库地址构造；未知构建不猜版本。管理凭据只用于用户明确确认的清理请求，不持久保存或发往 GitHub。

## 镜像与来源

```bash
python -m operations.build_identity \
  --build-tag claude-lb:local \
  --platform linux/amd64
```

工具把 Git SHA、运行文件清单及本地修改状态绑定到构建。未提交代码会标注修改状态；不能靠手写源码注解伪装一致。新增运行模块时同步 `build_metadata.RUNTIME_FILES`、发布器 `APP_FILES`、Docker COPY 和打包验证。

本地运行示例：

```bash
docker run --rm -p 127.0.0.1:8000:8000 \
  --env LB_API_KEY --env DATABRICKS_TOKEN_1 \
  -v "$PWD/config.yaml:/app/config.yaml:ro" \
  -v "$PWD/usage_data:/app/usage_data" \
  claude-lb:local
```

先创建可写的数据目录，并传入配置中所有实际引用的环境变量；不要挂入与另一个 writer 共用的 JSON 存储。Copilot 缓存使用单独只读凭据挂载，细节见 [配置指南](CONFIGURATION.md)。

`.wheelhouse`、`.release-wheelhouse` 和 `.test-wheelhouse` 是被忽略的离线构建/测试输入。已有构建确实使用过这些目录，不能仅因体积大就当作无用文件删除。普通 Python/pytest 缓存可以再生；私有发布目录包含备份和恢复材料，保持独立、限权且不进入 Git。

## 文档维护

README 只保留入口、快速开始和关键边界；操作细节归到对应指南，阶段证据归到 `docs/reviews/`。新增路由同步 [完整 API 表](../CLAUDE.md#api-端点)，已有测试会核对文档与真实路由及鉴权行为。

生产发布是独立验收层。已合并 GitHub、镜像构建通过、Pod Ready 和完整业务发布成功分别记录，参见 [AKS](AKS.md) 与 [发布流程](RELEASE_TOOL.md)。
