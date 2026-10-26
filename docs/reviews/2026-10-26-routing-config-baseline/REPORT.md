# 路由策略「每次发布都丢」：根因、config 基线与 AKS 发布

Author: Zeno Ren

日期：2026-10-26（北京时间）

## 现象与根因

运维报「自定义路由策略重新发布就丢失」。**不是丢失。**

`zeno-sea-mysql`（zeno-rg，Burstable B1ms）处于 **Stopped**，于是 `KeyPolicyStore.reload()`
每次启动都在 10s `IO_TIMEOUT` 上超时，降级规则照约定回落基线 —— 而 per-model 路由
偏好的基线是**硬编码的默认顺序**。每次发布因此都看起来像规则被清空，手工重配一次、
下次发布再丢一次。

证据链（全部线上实读）：

| 检查点 | 结果 |
|---|---|
| `az mysql flexible-server list` | `zeno-sea-mysql` → **Stopped** |
| Pod 启动日志 | `key_policy: load failed (TimeoutError); state=unavailable` |
| 同一日志 | `credentials: load failed (TimeoutError)`、`Failed to initialize usage data store` |
| `/stats.provider_policy` | `state: unavailable`、`last_loaded_at: null`、`model_preferences: 0` |
| ghost-blog（同一个库） | CrashLoopBackOff，`connect ETIMEDOUT` |
| Deployment volumes | 只有 config 与 copilot-auth，**无 PVC** → 库不可达时没有任何本地回落 |

启库 + 滚动重启后 overlay 原样回来：`accounts: 5`、`model_preferences: 2`、
`model_accounts: 2`，两条规则（`claude-opus-5-5`、`gpt-6-astra`）都是
`copilot` + `zenoren-icloud`。**一条都没丢。**

同一根因还有两个此前没被注意到的后果：config 声明 `enabled: false` 的 4 个
business seat 全部回落成 inert（只剩 0 个 claude 模型的 `ZenoRewn` 在服务 GHCP），
以及录入的上游凭据 `records=0` 完全没有在用。

## 修复（`c638187`，PR #68 → `a35b7a5`）

**一、per-model 路由有了 config 基线。** 其他每一层的基线都是运维写过的东西
（provider allowlist 回落「config 配了哪些 provider」、`allowed_models` 回落端点的
`models`、账户启用态回落 `enabled:`、面板密码回落 env），唯独这一层回落常量，所以
规则只有策略库一个落点。新增 `model_routing` 段，形状与
`PUT /admin/model-routing/{model}` 的请求体**刻意一致** —— 面板上读到的规则可以
原样抄进 ConfigMap。按**模型身份**匹配而非拼法，与存储里的规则共用 `rule_key`。

写错**拒绝启动**：provider id 拼错、段名拼错、同一个模型两种拼法各写一条，都只是
静默无效、没有症状可查 —— 与 `${ENV}` 引用同一个判断。代价同样写明：单副本 +
`maxSurge:0` 下拒绝启动等于停服到回滚，所以改 config 前必须先用校验 Pod 跑
`load_config`。

`DELETE` 从此是「回到基线」而不是「回到默认顺序」。`reset_to` 由服务端用**与生效
顺序同一个折算**算出；在前端重算一遍会在有基线的模型上给出另一个答案，而不一致的
那一份恰好出现在运维下决心前核对的确认框里。

**二、加载失败后重试到成功。** 这个 store 过去是 load-once：`reload()` 只在
`start()` 调一次，既无重试也无 admin 入口，所以数据库晚一分钟起来，进程就带着
「回落基线」跑到下次发布 —— 而那次重滚本身是一次停服，为的是取回一份从未丢失的
状态。`recover_until_loaded()` 指数退避重试（有上限），任务挂在 lifespan 上与其他
后台循环同生命周期。

两个有意为之的细节：

- **停止条件是 `fully_loaded` 而不是 `state == 'loaded'`。** overlay 不可达时一次
  成功写入会把 state 翻成 `loaded`，好让刚写的规则生效，但缓存此时只有那一条；
  照 state 停下来会把其余已存规则永久当成不存在。
- **凭据库不加这个重试。** 端点固定在启动时构建，晚到的凭据不会变成端点，重试只会
  让 `/admin/credentials` 报出一批并没有在服务的记录。

**三、诊断与判断同源。** `model_preference_resolution` / `model_accounts_resolution`
返回「值 + 来源」是主实现，普通读取由它派生 —— 一条 config 基线渲染成「已保存」会
让人去找一行并不存在的数据库记录。新增字段：`/stats.provider_policy` 的
`baseline_models` / `recovering`，`/admin/model-routing/{model}` 的
`providers_source` / `copilot_accounts_source` / `reset_to`。

## 验证

| 检查 | 结果 |
|---|---|
| 完整回归 `pytest tests/ -q` | 1735 passed、6 skipped、1408 subtests |
| 新增 `tests/test_routing_baseline.py` | 40 个（先在旧实现上红） |
| node UI 测试 5 个 | 全通过；路由 UI 48 → 56 |
| CI（3.11/3.12/3.13、docker-smoke、mysql-usage 8.0/8.4） | 全绿 |

**真实浏览器复核**（离线 fixture 全绿之后才发现的四处文案错误，全部只在真实页面
暴露）：状态行「模型优先序 0」与页面上两条正在生效的规则直接矛盾；分组叫「已保存
规则」而那两条从未保存过；「恢复整条默认会撤销放行」在基线仍指 Copilot 时是错的；
确认框标题说「默认」而实际回到基线。均已修并补测试。

**不可达数据库的端到端复现**：另起一个 `usage_storage.host` 指向 127.0.0.1:3307 的
实例，`state=unavailable` + `recovering=true` + `baseline_models=2`，两条规则仍由
config 生效（`source: config`，messages 面确实把 copilot 排到首位），面板显示
「配置基线 2 · 正在重试加载」，日志里退避重试持续在跑。

## AKS 发布

镜像 `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:4b3a40965a82e03aaefe5b88bb8eff630c52bd650ce1aaaef7ebd2e67028e5b3`，
`operations.build_identity` 构建，`source_tree_dirty: false`。运行文件清单未变（仍 23 个），
**控制器无需重建**。

release `lb-routing-baseline-c638187`：沿用上一次 profile，仅换镜像与源码提交；
`COPILOT_RESPONSES_ITEM_ID_MODE=stabilize` 与模型名单不变。

- 11 个阶段一次通过，维护窗口 **51.7 秒**（04:21:18 → 04:22:10 UTC，预算 300），
  含探针全程 92.4 秒。
- `image_preflight` 新旧镜像均 verified；`verify_backend` 的 imageID 与计划 digest 相符。
- 业务探针 4 条全部 `completed` + `marker_found`：`messages/databricks-claude-opus-5`
  （非流式）、`responses/gpt-6-astra`、`responses/grok-4.7`、`chat/gpt-5.6-luna`。
- 旧 writer exitCode 143，有终止证据；`restore_routes` 与 `cleanup` verified，
  `cleanup_errors: []`。
- 发布后 `/version` = `c638187` / `verification: matched`；Pod 内 `key_policy.py`
  `e478165d…`、`main.py` `7ef88a9f…`、`dashboard.html` `c8f85c4c…` 与构建清单逐一相符。

### ConfigMap 基线是**独立的第二步**

发布完成时 `baseline_models: 0` —— 镜像在线但 ConfigMap 还没有那一段，持久化这一半
等于没生效。顺序不能反：旧镜像忽略该段，新镜像会校验它。

1. 备份现网 ConfigMap 到 `.release-private/cm-backup/`（未跟踪）。
2. **集群内校验 Pod**：同一镜像、同一 `envFrom` Secret、同一 copilot-auth 挂载，
   执行 `load_config` → `load_config OK`，两条基线解析正确，5 个 Copilot 端点全部加载。
3. `kubectl patch` 只改 `config.yaml` 这一个 key（不整份 apply），读回与候选逐字节一致。
4. 滚动重启加载（配置在启动时读取），校验 Pod 与候选 ConfigMap 随后删除。

重启后：

```
[Routing] config baseline declares 2 per-model rule(s): claude-opus-5-5, gpt-6-astra
[Policy] provider allow-list store: backend=mysql state=loaded tenants=0 baseline_rules=2
provider_policy: baseline_models 2 · recovering false · model_preferences 2 · accounts 5
claude-opus-5-5 | providers ['copilot'] overlay | reset_to.source config ['copilot']
gpt-6-astra     | providers ['copilot'] overlay | reset_to.source config ['copilot']
```

overlay 仍然优先（规则来自数据库），基线在其下待命。

### 两个实操坑

- **本机 docker 构建必须走离线 wheel**：`--dependency-stage dependencies-offline`
  用 `.wheelhouse`。直连 PyPI 在构建容器里是 `SSLEOFError(UNEXPECTED_EOF_WHILE_READING)`，
  五次重试后 `pip install` 失败。
- **校验 Pod 必须设 `securityContext.fsGroup: 1000`**。Copilot token 缓存是 Secret
  挂载、模式 0440，容器以 uid 1000 运行；没有 fsGroup 时组属 root，4 个账户会报
  `No GitHub token resolved` 而被 skip —— 这是**校验环境的假阳性**，不是配置缺陷。
  第一轮校验就是这样误报的（先误判成 `HOME` 问题，加 `HOME=/home/app` 无效，
  加 fsGroup 后 0 个 skip）。拿那一轮结果当通过，会把「4 个账户加载失败」写进回执。

## 未验证边界

- **重试路径未在生产触发过。** 没有为了验收去停生产数据库；`recover_until_loaded()`
  的生产行为只能等下一次真实不可达时确认：
  `kubectl -n zeno-apps logs deploy/claude-lb | grep -E "\[Policy\]|key_policy: load failed"`
- **基线从未作为生效层服务过生产流量。** 线上 overlay 与基线内容相同，overlay 优先，
  所以「DB 停机时路由不变」是设计 + 本地复现，不是生产观测。
- **ConfigMap 那次滚动重启不走发布工具。** 它没有维护窗口、排空编排与业务探针，
  只有 Deployment 自带的 preStop（45s）与 readiness 门控；单副本 `maxSurge:0` 下仍有
  短暂中断。该重启后只做了只读核对（日志、`/stats`、`/admin/model-routing`），
  **没有重跑三协议业务探针**。
- **用量账本与凭据录入仍硬依赖同一个 MySQL**，没有本地回落（无 PVC）。本次只让路由
  策略这一层扛住了库停机，那两块没有改动。
- `az mysql flexible-server start` 恢复了该实例，但**为什么停**未追查（活动日志 30 天
  查询无该资源记录）。它是 Burstable B1ms 单实例，没有高可用。
