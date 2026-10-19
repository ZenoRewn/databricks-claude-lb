# 2026-10-18 上游凭据迁 Secret + `${ENV}` 解析闸门

Author: Zeno Ren

两步，顺序是必须的：先发布一道**行为无变化**的代码闸门，再搬凭据。
反过来做的话，这次迁移最可能的失败（变量名拼错）会是静默的。

| | |
|---|---|
| 闸门源码 | `1d00c5827e28a221b44702b0fcbcc3c81efd7346`（PR #58） |
| 闸门镜像 | `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:06fe4ae2…`（本地 image ID `c4aedf53…` 之后的新构建） |
| source manifest | `55c840241807030d7be8f81590ae33bb62602e339ee309c4e2df73faf3ba3ab1`（24 个运行文件，清单未变 → 控制器未重建） |
| 发布维护窗口 | **55.4 秒**，11 阶段一次通过；三协议探针 `completed` + `marker_found`，`persistence_verified: true`，3 条用量事件入账；`cleanup.verified` / `restore_routes.verified` 均 true |
| 迁移凭据数 | **17**（9 ADB token、4 Azure OpenAI key、1 Foundry key、1 GHCP token、`auth.api_key`、MySQL 密码） |
| 新 Secret | `claude-lb-upstream-credentials`（Opaque，17 键，**无** `last-applied` 注解） |
| 线上 | `1d00c58` / `matched`，Pod `claude-lb-c57445999-n2kz9`，1/1、0 重启 |

## 一、为什么闸门必须先上线

查下去发现这次迁移的首要失败模式**不是**「启动失败」，而是**静默成功**：

`expand_env_vars` 把缺失变量替换成空串，而没有任何一层拒绝空 token —— 端点照常构建、
照常打 `Loaded`、照常进 `/config/effective`，唯一症状是上游 401，而配置和日志里没有任何
线索指向那个拼写。凭据全部内联在 ConfigMap 时这无所谓（根本没有引用可以写错）。

收口在 `resolve_credential(value, field=...)`，用在 `auth.api_key`、`auth.api_keys[*]`、
Databricks / Foundry / Azure 三处端点凭据、`usage_storage.password` 与 `usage_data_dir`。
只有**引用**解析为空才报错 —— **字面空值仍合法**（多租户配置里 `auth.api_key` 本就缺省），
所以它对当时那份全内联的生产配置是**行为无变化的空操作**，这正是它能单独先发布的原因。
报错带 config 路径与变量名（都不是秘密，合起来就是全部修复指引），**不带**任何已解析出的
值 —— 包括同一份配置里已经解析成功的那些，有测试专门钉这一条。

**Copilot 的 `github_token` 是唯一例外**，不走闸门：它会回落到挂载的凭据缓存，而且代码里
早就明确警告「刚 rotate 的 Secret 没有在用」。那是**有落点**的韧性；其他 provider 没有
落点，空凭据只是坏的。

代价必须写明：单副本 + `maxSurge:0 / maxUnavailable:1` → **旧 Pod 先终止**。所以「拒绝启动」
就是停服直到回滚。这不是把闸门改回宽松的理由，而是校验 Pod 为什么是必须步骤的理由。

## 二、迁移怎么做的

**变量名全部由脚本从映射生成，两侧都不手敲。** 「任何一个名字拼错就是启动失败」是这件事
最大的风险，而消除它的办法不是小心，是让两侧（config 的 `${X}` 与 Secret 的键名）出自
同一份 `env-mapping.json`。命名规则 `<前缀>_<端点名大写，非字母数字转下划线>`：

| config 字段 | 环境变量 |
|---|---|
| `auth.api_key` | `LB_API_KEY` |
| `endpoints[*].token` ×9 | `ADB_TOKEN_<名>` |
| `azure_openai[*].api_key` ×4 | `AZURE_OPENAI_KEY_<名>` |
| `azure_anthropic[eastus2-foundry].api_key` | `AZURE_ANTHROPIC_KEY_EASTUS2_FOUNDRY` |
| `github_copilot[ZenoRewn].github_token` | `GITHUB_COPILOT_TOKEN_ZENOREWN` |
| `usage_storage.password` | `MYSQL_PASSWORD` |

生成前自检：变量名唯一、符合 `[A-Z_][A-Z0-9_]*`、每个凭据值在原文中**恰好出现一次**。

**替换按「键名 + 值同行」锚定**，不是裸值替换 —— MySQL 密码不到 16 字符，裸替换对短值不
安全。替换后做**结构化比对**：新旧 YAML 反序列化后必须逐字段相同，除那 17 个字段变为
`${ENV}`；并断言新配置里不残留任何原值。133 行 / 19 行注释前后一致，其余 116 行逐字未动。

**注解里是整份明文的副本。** ConfigMap 带着
`kubectl.kubernetes.io/last-applied-configuration` —— 清掉 `data` 不够，它也必须去掉，
否则凭据仍然可读。替换用 `kubectl replace`（不写该注解），不是 `apply`。
Secret 同样用 `kubectl create -f -` 走标准输入创建：值不进 argv、不落新文件、不被打印。

### 校验 Pod 是关卡，不是可选步骤

用生产同一个镜像、挂新 ConfigMap（临时名）+ 新 Secret 的 `envFrom`，跑一遍
`load_config` 并断言无空凭据，退出码 0 才动生产。结果：

```json
{"adb": 9, "foundry": 1, "azure": 4, "copilot": 5, "blank": [],
 "storage_type": "mysql", "storage_password_set": true,
 "copilot_sources": ["env", "file"], "api_keys": 1}
```

端点数与生产一致，`blank` 为空。`copilot_sources` 含 `env` 是关键一行：它证明 `ZenoRewn`
的 token 确实从 Secret 解析，**没有**静默回落到挂载缓存 —— 那一条是闸门唯一不保护的凭据，
所以必须单独拿证据。`declares github_token=${...}` 警告 0 条。

## 三、执行与一处操作失误

计划是「替换 ConfigMap + 打 envFrom 补丁一次完成，由后者驱动唯一一次滚动」。实际第一条
命令里**重定向与脚本写了同一个文件**，ConfigMap manifest 被覆盖成脚本的 stdout，
`kubectl replace` 解析失败；而 Deployment 的 envFrom 补丁已经生效。

这个中间态是安全的，而且正是我原先列为「零风险」的那一步：配置仍然全内联，多出来的 17 个
环境变量没有任何引用去读它们。确认 Pod 1/1、三个健康端点 200 之后补上 ConfigMap 替换，再
`rollout restart` 让新配置生效。**代价是多了一次滚动重启**（两次而不是一次）。

顺带：这个顺序（先 envFrom、后 ConfigMap）其实**消除了**原计划里那个几秒的窗口 ——
「新配置已就位但 envFrom 还没打上时 Pod 恰好自行重启」会触发闸门而停服。不是有意为之，
但结果更稳。

## 四、线上实测

端点加载：ADB **9**、Azure **4**、Foundry **1**、Copilot **5**，与迁移前一致。
`Loaded Copilot endpoint: ZenoRewn [in service] (models: all, source: env)` —— `source: env`。
静默回落警告 **0** 条。

健康端点不证明凭据可用，所以实打四条真实上游请求：

| 入口 | 结果 |
|---|---|
| `/v1/messages` `databricks-claude-opus-5` | `stop_reason: max_tokens`，20 in / 32 out（thinking 吃掉了 32 的预算，请求本身成功） |
| `/v1/responses` `gpt-5.4` | `status: completed` |
| `/v1/chat/completions` `gpt-5.6-luna` | `finish_reason: stop`，内容 `SECRET_OK` |
| `/v1/messages` `claude-opus-5-5`（Foundry） | `stop_reason: end_turn`，`model: claude-opus-5-5`，内容 `SECRET_OK` |

账本（证明 `MYSQL_PASSWORD` 解析正确）：`Usage storage: mysql`、
`[Credentials] backend=mysql state=loaded key=present`，今日总量 558 → **563** 请求，
按 provider 归因 `{azure_anthropic: 1, copilot: 24, databricks: 313, unknown: 225}`
（`unknown` 是归因上线前写入的历史行）。读写两条路径都通。

**残留扫描**：按那 17 个真实凭据值在 `configmap` / `deployment` / `replicaset` / `pod`
四类对象里逐个搜，`claude-lb` 的全部对象结果为 **无**。

## 五、一个本次未处理的发现

命名空间级扫描时 MySQL 密码仍然命中，来源**不是** `claude-lb`：

```
deployment  ghost-blog   ghost.env[database__connection__password]
pod         ghost-blog-f6f56684b-88nz6   同上
```

`ghost-blog` 共用同一个 MySQL 实例，把密码写成了 `env` 明文值。**本次未改动** —— 那是另一个
工作负载，不在本次授权范围内。但要说清后果：这一条凭据在命名空间里**仍然可读**，所以
对它而言本次迁移的收益是**部分**的，直到 `ghost-blog` 也改掉。另外轮换 MySQL 密码现在是
两处要一起改，漏一处会让 ghost 或网关之一连不上。

## 六、回滚

`.release-private/cm-backup/ROLLBACK.md`（0600，目录 0700，在 `.gitignore` 内）写了两条
命令与顺序约束：

1. `kubectl replace -f .../claude-lb-config.before-secret.json` —— 恢复明文 ConfigMap，
   含原注解，与变更前逐字一致；
2. `kubectl -n zeno-apps patch deploy claude-lb --type=json -p '[{"op":"remove","path":".../envFrom"}]'`
   —— 去掉 envFrom，触发滚动重启并加载上一步恢复的配置。

顺序不能换也不能只做一步：只做 1 不影响运行中的 Pod（config 是 subPath 挂载，ConfigMap
更新不传播）；只做 2 会让 Pod 带着 `${ENV}` 配置而没有变量启动 → 闸门拒绝启动。

`claude-lb-config.before-secret.json` 是迁移后**唯一**的明文副本，它同时是回滚材料。

## 验证

- `python3 -m pytest tests/ -q` → **1484 passed, 6 skipped, 1387 subtests**
- 新增 `tests/test_credential_env_refs.py`（13 条）。**有效性已验证**：实现前 9 条红
- 校验 Pod 退出码 0；发布 `plan` 阶段 `production_changed: false`、`server_dry_run_verified: true`

## 边界

- **凭据仍然在运行进程的内存里**，这一点没有改变；Secret 改变的是「谁能读到静态副本」
- `claude-lb-copilot-auth`（4 个 Copilot 账户的缓存 token）本来就是 Secret，未动
- `claude-lb-credential-key`（`LB_CREDENTIAL_KEY` / `LB_DASHBOARD_PASSWORD`）未动，
  与上游凭据分开是有意的：轮换节奏不同，而 `envFrom` 会把 Secret 的每个键都注入为环境变量
- **轮换现在是两步**：改 Secret，再滚动重启（端点只在启动时构建）。`/admin/copilot/reload`
  仍可单独重读 Copilot token
- 本次**没有**改 `deploy/k8s/` 下的示例清单，它仍是示例形态而非线上形态
- 进程内累计计数器随重启归零；本次共两次滚动重启
