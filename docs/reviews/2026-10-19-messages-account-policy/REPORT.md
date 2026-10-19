# 2026-10-19 Messages 面候选判断绕过账户策略（生产实测）

Author: Zeno Ren

源码 `53615db`，线上此前 `1d00c58`。本次修的不是功能缺口，而是一条**已经在生产发生过**
的策略绕过：一个运维从未启用的 GHCP 账户服务了 `/v1/messages` 流量。

| | |
|---|---|
| 源码 | `53615db20abf5db70515e75e2180d8d468190a59`（PR #61） |
| 镜像 | `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:b8825556…` |
| source manifest | `561f391b55bd18d3ccc9c77fb676cf9825f36853531a816daacde904ea5e2c95`（24 个运行文件，清单未变） |
| 维护窗口 | **78.5 秒**，11 阶段一次通过（第二次提交；第一次在 preflight 403 中止，见下） |
| 业务探针 | 三协议均 `completed` + `marker_found`，`persistence_verified: true`，3 条用量事件 |
| 线上 | `53615db` / `matched`，Pod `claude-lb-58bd6bcc4d-b77hw`，1/1、0 重启 |

## 怎么发现的

为那条**从未在生产验证过**的 GHCP messages 跳取证：临时给 `claude-opus-5-5` 配
`providers: ['copilot']`，发一次 `/v1/messages`。请求成功，但事件是这样的：

```
provider: "databricks"   endpoint_alias: "zenoren-msft"   forwarded_model: "claude-opus-5.5"
```

一个 GHCP 账户名、一个 GHCP 拼法，却标着 databricks —— **事件自己自相矛盾**。而
`zenoren-msft` 在 config 里是 `enabled: false`，`/admin/model-routing/claude-opus-5-5`
同时报它 `account_never_enabled`。诊断与运行时打架，正是 CLAUDE.md 说过「比没有诊断
更糟」的那一种。

## 缺陷一：Messages 面绕过四层账户策略

三个 messages 池都是 `ClaudeProxy`，它经 `LoadBalancer.select_endpoint()` 选端点，而那里
唯一的兼容判断是 **config 声明的 `models` 列表**。Copilot 账户声明 `models: []`（= all），
于是该谓词恒真：

| 层 | 实现位置 | 修复前 |
|---|---|---|
| 账户启用态（含三态） | `_copilot_support_reason` | **被跳过** |
| `allowed_models` | 同上 | **被跳过** |
| per-model 账户选择 | 同上 | **被跳过** |
| 已发现 capability | 同上 | **被跳过** |

四层都住在 `CopilotEndpoint.supports()` 背后，而 `select_endpoint` 从不调它。CLAUDE.md
早就写了「筛选只能加在 `supports()`：那是所有候选判断的唯一收口」—— OpenAI 面守住了
这条，messages 面没有。**这不是遗漏一个条件，是整条收口没接上。**

修法是新的 `endpoint_permits_model(endpoint, model)`：有 `supports()` 就问它，没有的
（ADB/Foundry 的 `WorkspaceEndpoint`）保持原先的 `models` 列表行为**完全不变**。两处细节：

- **不传 `api_type`**。Copilot 的 `api_types` 只能取 `responses`/`chat`，传进去会把每个
  账户从它确实在服务的面上排除掉。`_messages_path_views` 既有的问法也是这样。
- 同一个谓词也用在「没有端点允许这个模型」的判定上，所以候选被策略清空时抛
  **404 `unsupported_model`**（在 `_messages_failover_allowed` 白名单里，交给下一条路径）
  而不是 `unavailable`。一组被停用的账户不是对这个请求的回答。

## 缺陷二：遥测归因（同一行代码）

`inference_call` 被写死传 `'databricks'`，所以 Foundry 与 GHCP-messages 的每一次 attempt
在诊断事件**和** `lb_upstream_send_started_total{provider=...}` 里都标成 databricks。

**用量账本不受影响** —— 那个记录点 2026-10-18 已改用 `self.provider`。所以这是同一条
不变量在**第二个记录点**失效，而失效的恰好是运维问「这条请求是哪条路径服务的」时读的
那一份。两处改用 `getattr(self,'provider','databricks')`，与同一函数里既有的
`observe_route_budget` 一致。

**硬取 `self.provider` 会出事**：测试里有用 `object.__new__` 手搭的 proxy，`AttributeError`
落在重试循环里会让整个文件挂死 —— 实测 `test_resilience_routing.py` 超过 75 秒不返回，
另有 8 条失败。这是我自己造的回归，由分文件计时定位后改回 `getattr`。

## 第一次提交在 preflight 403 中止

`last_failure: {"action": "preflight", "api_status": 403}`，`maintenance_started_at: null`
—— **从未进入维护窗口，生产未被触碰**，版本、Pod、generation 全程不变。

原因是 2026-10-18 的凭据迁移给 Deployment 加了 `envFrom: claude-lb-upstream-credentials`，
而控制器在目标命名空间的 Role 只列了 `claude-lb-copilot-auth` 与
`claude-lb-credential-key`。preflight 的 `_verify_references()` 逐个读取模板引用的每个对象
并比对摘要，漏一个就是 403。

**这一步仓库里本来就写着**（`docs/AKS.md`：「给 Deployment 新增任何 ConfigMap/Secret
引用，都要把该对象名加进 `Role/lb-release-controller-target` 的 `resourceNames`」，还记着
「2026-10-14 首轮发布就是这么中止的」），我在迁移那一轮**漏了**。同一处还写明
**`plan` 阶段不会暴露这个问题**，因为 plan 用的是运维自己的身份而不是控制器的
ServiceAccount —— 本次完全复现：`plan` 报 `production_changed: false` /
`server_dry_run_verified: true`，然后 preflight 403。

补上 `resourceNames` 并以控制器 SA 核验（`kubectl auth can-i … → yes`）后换新 release ID
重跑，一次通过。**代价要说清**：凭据迁移虽然没有影响生产服务，却让**此后任何一次发布**
都会在 preflight 失败，直到补上这一条。那一轮的回执已回填这一步。

## 线上实测（修复后）

同样强制 `providers: ['copilot']` 发一次 `/v1/messages`：

```
provider=copilot          endpoint=zenoren-icloud     forwarded=claude-opus-5.5
provider=databricks       endpoint=westus-adb-ws      forwarded=databricks-claude-opus-5
```

- 落在 **`zenoren-icloud`** —— 唯一 `supports()` 允许的账户（已启用 + 有 claude 能力），
  不再是从未启用的 `zenoren-msft`
- 事件归因 `provider=copilot` 与 `endpoint_alias` **不再互相矛盾**；ADB 的 attempt 仍报
  databricks
- 响应 `stop_reason: end_turn`、内容 `POLICY_OK`。验证后已 DELETE 规则恢复默认

## 顺带撤回两处自相矛盾的旧表述

`CLAUDE.md` 与 `docs/CONFIGURATION.md` 都还写着「`/v1/messages` 仍然只走 Databricks，
需要双向适配器」，而**同一份文件几行之后**就是 2026-10-17 实测推翻它的那条不变量。后者
还断言「Claude Code 这类走 `/v1/messages` 的客户端**用不到**这条通路」—— 恰好说反了。

新增机械守卫：该句只能作为**撤回说明里的引述**出现，不能作为当前主张（有效性已验证：
塞回一条断言即变红）。

## 验证

- `python3 -m pytest tests/ -q` → **1496 passed, 6 skipped, 1389 subtests**；前端四套全过
- 新增 `tests/test_messages_account_policy.py`（12 条）。**有效性已验证**：把两处改回原样，
  9 条红

## 边界

- **这条绕过存在多久无从断定。** `select_endpoint` 的谓词一直如此，而 messages 面接入
  Copilot 是 2026-10-17 的事；在那之前该路径上只有 ADB 的 `WorkspaceEndpoint`（没有
  `supports()`），所以**行为上**不受影响。期间是否有真实流量落到未启用账户，日志保留
  期内可查的部分没有，但不能就此断言没有发生过
- 修复只保证**候选判断**过收口。它不改会话亲和的哈希口径，也不追认任何已经发生的请求
- `api_types` 仍只能取 `responses`/`chat`，所以**无法表达「这个账户不服务 messages 面」**。
  目前所有已启用账户都被当作可服务该面，这是现状而不是认证
- Foundry 与 GHCP-messages 的历史遥测数据仍标着 databricks，**不回填**
