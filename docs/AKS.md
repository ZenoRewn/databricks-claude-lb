# AKS 部署指南

Author: Zeno Ren

## 当前运行版本

最近一次已记录的 AKS 发布：**2026-10-19，应用源码 `53615db`**，维护 78.5 秒，11 个阶段一次通过，三协议业务探针、账本与清理验收通过。修的是一条**已经在生产发生过**的策略绕过：Anthropic Messages 面的候选判断经 `LoadBalancer.select_endpoint()`，而那里只看 config 声明的 `models` 列表 —— Copilot 账户声明 `models: []`（= all），谓词恒真，于是启用态 / `allowed_models` / per-model 账户选择 / 已发现能力**四层全部被跳过**。实测一次强制走 copilot 的 `/v1/messages` 落在运维从未启用的 `zenoren-msft` 上，而 `/admin/model-routing/{model}` 同时报它被排除。**首次提交在 preflight 以 403 中止**（`maintenance_started_at: null`，生产未被触碰）：2026-10-18 的凭据迁移给 Deployment 加了 `envFrom` 却漏了把该 Secret 名加进 `Role/lb-release-controller-target` —— 正是下面那条注意事项说过的情况，且 `plan` 阶段不会暴露它。补上后换新 release ID 重跑一次通过。本次运行文件清单未变（仍 24 个），**控制器不需要重建**。见 [本次回执](reviews/2026-10-19-messages-account-policy/REPORT.md)。

上一次（2026-10-17，`0a9f830`）是 Claude 三路径上线与 Foundry 凭据首次投产，当时**控制器必须重建**（新增运行文件让 `APP_FILES` 从 21 变 22，而 `validate_plan` 在控制器进程内断言文件集合完全相等）—— 新增运行时文件时记得这条，见 [上一次回执](reviews/2026-10-17-three-paths-release/REPORT.md)。

**历次发布记录不在本文维护，而在 [验证与发布索引](reviews/README.md)** —— 那里逐次记录范围、失败、修复与未验证边界。此处曾堆叠逐次发布的摘要，与索引重复、且在 2026-10-14 至 10-16 的四次发布中都没有被同步更新，于是「当前运行版本」这一行长期指向早已不在线上的 `5f2463e`。一份事实只在一个地方维护。

判断线上究竟跑什么**只能读 live 对象或 `/version`**：本仓库的文档、`deploy/k8s/` 示例与 Git 最新提交都不是生产事实（2026-10-16 我把示例 manifest 的 `LOG_FORMAT=json` 当成线上形态，由此得出的结论是反的）。配置、指标版本与兼容性变化见 [当前契约](OBSERVABILITY_AND_CONTEXT.md)。渠道能力 JSON 如需使用，应先审阅来源与有效期，再作为单独只读文件挂载；不从 `config/effective` 或 Git revision 注解单独推断运行文件一致。**本文的历史 live 表格不构成当前现场快照，也不授权整份覆盖生产 manifest。**

把 `databricks-claude-lb` 部署到 Azure Kubernetes Service，目标是**单 Pod 长期稳定运行 + GitHub Copilot token 完全自动刷新 + 无需重启 Pod 即可滚动 Secret**。

---

## 2026-09-20 事故后的发布约束

当日一次发布在排空脚本正常退出后遗留维护 selector，Service 无匹配后端，导致公网不可用。关闭流量、排空和恢复流量被拆成需要后续交互接力的独立脚本，是这次流程缺陷。原始复盘保留在本地未提交的 `docs/reviews/2026-09-20-service-optimization/deployment-r1/` 目录；本轮恢复工具的实现与演练结果见 [验收报告](reviews/2026-09-20-post-upgrade-hardening/VALIDATION.md)。

- 在维护前完成镜像、补丁、验证脚本、完整入口清单、可信观测基线和恢复材料；维护阶段不得等待下一轮 agent 回复或临时编写后续脚本。
- 同一受控发布任务负责整个切换及失败收尾。排空成功只是中间状态；路由恢复、EndpointSlice、公网业务和持久化验收完成后，才能记录发布成功。
- 保存完整原 selector，绑定对象身份、当前版本和本轮所有权；处理 null/空 EndpointSlice、API ACK 丢失及并发接管。日志失败不能阻断恢复检查。
- finally 不能覆盖执行器崩溃。独立恢复机制必须结合发布所有权、后端健康和数据兼容性判断，不得按 TTL 无条件开放流量。
- 现有应用 CI 不能替代发布器的中断、回滚和恢复测试。受控发布器及其隔离验证见 [发布操作指南](RELEASE_TOOL.md)；2026-09-30 已升级既有双协调器并完成本次发布，具体证据和边界见部署回执。历史临时脚本与补丁不可作为生产发布工具复用。

下文通用清单中的 apply 命令只适用于独立测试环境或已完成配置收敛的新部署，不能用于覆盖当前生产环境。

## 🚨 红线：禁止 `kubectl apply -f deploy/k8s/`

**本仓库的 manifest 描述的是「设计目标形态」，不是现网 live 形态。二者已结构性偏离，盲 apply 会打挂服务。**

具体后果：

| 盲 apply 的动作 | 后果 |
|---|---|
| 覆盖 ConfigMap | 命名约定已对齐，但 **storage 档不同**：示例是 `usage_data_dir`（JSON 单写者），live 是 `usage_storage.type: mysql`。整份覆盖会把 MySQL 连接配置整段抹掉 → 用量落到容器内的本地文件，**且 provider 归因失效**（那个主键只有 MySQL 后端有）。引用对不上时自 2026-10-18 起会**拒绝启动**而不是静默 401（`resolve_credential`），更容易发现，但单副本下仍是停服。注意早先这里写的「拿到字面量 `${...}` 当 token」不准确：`expand_env_vars` 替换成的是**空串**，不是字面量 |
| 覆盖 Deployment | config 挂载点从 live 的 `/app/config.yaml` 变成 repo 的 `/etc/claude-lb`，同时 `CONFIG_PATH` 被设成 `/etc/claude-lb/config.yaml` —— 二者自洽，但若只 apply 了 ConfigMap 没 apply Deployment（或反之）就找不到配置文件 |
| 创建 PVC + usage-data 卷 | live 走的是 MySQL 后端（`usage_storage.type: mysql`），凭空多一个 RWO PVC 且 `Recreate` 策略被带回 |
| 挂 copilot-cache | live 里那个 secret 叫 `claude-lb-copilot-auth` 而不是 repo 的 `claude-lb-copilot-cache`，直接 apply 会因 secret 不存在而 Pod Pending |

### live 与 repo 漂移对照

> 数据来源：2026-09-08 在有集群网络的机器上实测（本项目 repo 侧无法直连 —— 集群是私有 AKS，API server 在 privatelink 内网，外部机器 DNS 解析不到）。

| 项 | repo（本仓库） | live（zeno-apps/claude-lb） |
|---|---|---|
| ConfigMap 敏感字段 | `${LB_API_KEY}` / `${ADB_TOKEN_<端点名>}` 占位符 | ✅ **已收敛（2026-10-18）**：17 个字段均为 `${ENV}` 引用，`last-applied` 注解里那份整份明文副本也已清除。repo 示例同日改用同一套**按端点名生成**的命名，两侧不再是两套约定 |
| `envFrom` | `secretRef: claude-lb-upstream-credentials` | ✅ 同名（17 键） |
| `azure_anthropic`（Foundry） | ✅ 已补（2026-10-18）；此前示例里**完全没有这一段**，照抄装出来的网关会少掉第三条 Claude 路径 | 1 个端点、4 个 deployment |
| config 挂载点 | `/etc/claude-lb`（配 `CONFIG_PATH`） | `/app/config.yaml`，**`CONFIG_PATH` 已显式设置**（2026-10-18）。此前未设、靠 Dockerfile 的 `WORKDIR` 与挂载点恰好一致 —— 能用但是隐式依赖，`WORKDIR` 一改就断。两侧现在都是显式的，路径不同只是选择 |
| usage 存储 | `usage_data_dir` + RWO PVC | `usage_storage.type: mysql`，**无 PVC** |
| copilot-cache 卷 | `secret/claude-lb-copilot-cache`，挂 `/home/app/.config/databricks-claude-lb` | ✅ 已挂载，但 secret 名是 `claude-lb-copilot-auth`（卷名 `copilot-auth-accounts`）。5 个账户中 4 个的 token 来自这里，1 个（`ZenoRewn`）来自上面那个 Secret |
| 发布策略 | `Recreate`（RWO PVC 所致） | `RollingUpdate` + `maxUnavailable=1` / `maxSurge=0`（等价于先杀后拉，**同样零可用**） |
| preStop / grace | ✅ 已对齐（2026-10-18）：`python /app/gateway_lifecycle.py --wait-seconds 45` / 90s。此前示例是裸 `sleep 15` —— 只是等时间，**不落 drain 标记**，`/health/accepting` 不会转为「不接新流量」 | 同左 |
| PDB / HPA | 无（单副本不需要） | 无 |
| live 独有 env | — | `COPILOT_STREAM_HIGH_WATERMARK` / `_OVERLOAD_GRACE` / `_DISCONNECT_GRACE` / `_MONITOR_INTERVAL` / `IMG_MAX_COUNT` |

### 收敛步骤 —— 认证路径部分**已完成（2026-10-18）**

实际怎么做的、实测了什么、留了哪些边界，见
[凭据迁 Secret 回执](reviews/2026-10-18-credential-secret-migration/REPORT.md)。与这里
2026-09-08 写下的计划相比，有四处是执行时才看清的：

1. **必须先上一道代码闸门。** 原计划第 5 步说「缺任何一个 env 会让对应 endpoint 静默失败，
   Ready 仍是绿的」—— 这是对的，而且比描述的更糟：没有任何一层拒绝空 token，端点照常构建、
   照常打 `Loaded`、照常进 `/config/effective`。所以先发布 `resolve_credential`（引用解析
   为空就拒绝启动），它对当时全内联的配置是**行为无变化的空操作**，因此可以单独先发。
2. **`last-applied-configuration` 注解里是整份明文的副本。** 只换 `data` 不够，凭据仍然可读。
   用 `kubectl replace`（不写该注解），不是 `apply`。
3. **变量名两侧都由脚本从一份映射生成，不手敲。** 「任何一个名字拼错就是启动失败」的解法
   不是小心，是让 config 的 `${X}` 与 Secret 的键名出自同一份 `env-mapping.json`。
4. **动生产前先跑校验 Pod**：生产同一镜像 + 新 ConfigMap（临时名）+ 新 Secret 的 `envFrom`，
   跑 `load_config` 并断言无空凭据，退出码 0 才继续。单副本 + `maxSurge:0/maxUnavailable:1`
   意味着拒绝启动就是停服，所以这一步是关卡不是可选项 —— 与镜像预检同一种性质。

原计划第 4 步要求「换 ConfigMap 与加 `envFrom` 必须同一次 rollout」。执行时因一处命令失误
变成了两次（先 `envFrom`、后 ConfigMap）。那个中间态是安全的 —— 配置仍全内联，多出的环境
变量没有任何引用去读 —— 而且这个顺序反而**消除了**原计划里那个窗口：「新配置已就位但
`envFrom` 还没打上时 Pod 恰好自行重启」会触发闸门而停服。代价是多了一次滚动重启。

**同日补完的收尾（2026-10-18）**：

- live 的 `CONFIG_PATH` 已**显式**设为 `/app/config.yaml`（此前未设，靠 Dockerfile 的
  `WORKDIR` 与挂载点恰好一致 —— 能用，但 `WORKDIR` 一改就断）。一次滚动重启，端点
  9/4/1/5 与变更前一致、零 ERROR。挂载路径本身没有改：那是选择而不是缺陷，为它重启
  单副本服务换不到任何东西。
- `deploy/k8s/` 示例改用与线上**同一套**命名约定（按端点名生成）、同一个 Secret 名、
  同样的 `gateway_lifecycle.py` preStop 与 90s grace，并补上**此前完全缺失的
  `azure_anthropic`（Foundry）段**与 MySQL 账本档。示例此前教的是另一套约定，照抄装出来
  的网关既少一条 Claude 路径、也拿不到 provider 归因。

**仍然有意保留的差异**：示例是 `usage_data_dir`（JSON 单写者）+ RWO PVC + `Recreate`，
那是一个合法的简单起点；线上是 MySQL + 无 PVC + `RollingUpdate`。所以
**整份 apply 示例清单依然会出事** —— 现在的首要理由不再是名字对不上，而是会把 MySQL
连接配置整段抹掉。

**命名空间里还有一处明文**：`ghost-blog` 的 Deployment 把同一个 MySQL 的密码写成 `env`
明文值。那是另一个工作负载，本次未动，所以 MySQL 密码在命名空间内**仍然可读**，且轮换
它现在要两处一起改。

### 关于「每次发布必然断服」

live 的 `maxUnavailable=1 + maxSurge=0 + replicas=1` 与 repo 的 `Recreate` 效果相同：**先杀旧再拉新，中间零可用**。要真正消除这个窗口需要多副本，前置条件见 `deploy/k8s/deployment.yaml` 里「零断服变体」注释块与 `docs/RESILIENCE.md`「多副本语义」。

**关键一点**：多副本的阻断项**不是** Copilot 会话亲和性 —— LB 对 Copilot 无跨请求进程态，单账户下任意副本都能接任意一轮。真正的阻断项是 usage 持久化的多写安全（已在 `usage_store.py` 改为增量 upsert 解决）。别把这两件事搞混而白等。

短期不动副本数的话，加 preStop 至少能让进行中的 SSE 长流优雅排空，不再被硬断。

---

## 整体架构

```
                  ┌─────────────────────────────────────┐
                  │  AKS Pod (claude-lb)                │
   客户端          │  ┌─────────────────┐  ┌──────────┐ │      ┌───────────┐
 (Codex/Claude  ──┼──▶ FastAPI :8000   ├──┤ session  ├─┼──────▶ GitHub    │
  Code/openclaw)  │  │ Anthropic /     │  │ token    │ │      │ Copilot   │
                  │  │ OpenAI routes   │  │ cache    │ │      └───────────┘
                  │  │                 │  │ (mem)    │ │
                  │  │ background      │  └────┬─────┘ │      ┌───────────┐
                  │  │ refresh task    │       │       │      │ Databricks│
                  │  │ (5min scan)     │       │       ├──────▶ x9 region │
                  │  └────────┬────────┘       │       │      └───────────┘
                  │           │ readiness      │       │      ┌───────────┐
                  │           ▼                │       ├──────▶ Azure AI  │
                  │  /health/{live,ready}      │       │      │ x4 region │
                  │  /metrics  (Prometheus)    │       │      └───────────┘
                  └────┬───────────────────────┘       │
                       │                               │
              ┌────────┴────────┐         ┌────────────┴──────────┐
              │ ConfigMap       │         │ Secret claude-lb-     │
              │ claude-lb-config│         │  copilot-cache         │
              │ (config.yaml)   │         │ (long-lived OAuth     │
              └─────────────────┘         │   token JSON file)    │
                       ▲                  └────────────┬──────────┘
              ┌────────┴────────┐                      │
              │ Secret claude-  │       ┌──────────────▼──────────┐
              │  lb-secrets     │       │ kubelet 自动同步         │
              │ (env vars: ADB, │       │ (~1 min 周期)            │
              │  Azure, LB key) │       │ Pod 内 reload_github_   │
              └─────────────────┘       │  token() 自动 pick up   │
                                        └─────────────────────────┘
```

---

## 一、镜像构建与推送

### 1. 本地构建并推到 ACR

```bash
# 登录 ACR
ACR=myacr  # 你的 Azure Container Registry 名字
az acr login --name $ACR

# 构建（多架构推荐：AKS 默认 amd64；如果你用 ARM 节点改 --platform linux/arm64）
docker buildx build --platform linux/amd64 \
  -t $ACR.azurecr.io/databricks-claude-lb:$(git rev-parse --short HEAD) \
  -t $ACR.azurecr.io/databricks-claude-lb:latest \
  --push .
```

### 2. （可选）让 ACR 自动从 git 触发构建

```bash
az acr task create \
  --registry $ACR \
  --name build-claude-lb \
  --image databricks-claude-lb:{{.Run.ID}} \
  --image databricks-claude-lb:latest \
  --context https://github.com/<you>/databricks-claude-lb \
  --branch main --file Dockerfile \
  --git-access-token <PAT>
```

---

## 二、准备 Secret 和 ConfigMap

### 1. 一次性生成 GitHub Copilot long-lived token

**这一步必须在能开浏览器的机器上做（device flow 需要交互授权）**。生成后 token 永不过期（除非用户撤销），可重复用于所有 Pod。

```bash
# 项目目录下
python main.py --copilot-login --endpoint gh-account-1
# 终端会打印 verification URL + user code，去浏览器输入即可
# 完成后 token 落到 ~/.config/databricks-claude-lb/copilot-auth-gh-account-1.json (mode 0600)
```

### 2. 创建命名空间和 Secrets

```bash
NS=claude-lb
kubectl create ns $NS

# Secret 1：所有 ${ENV_VAR} 占位的实际值（API key、Databricks tokens、Azure keys）
kubectl create secret generic claude-lb-secrets -n $NS \
  --from-literal=LB_API_KEY="sk-xxxxx" \
  --from-literal=DATABRICKS_TOKEN_CENTRALUS="dapi_xxx" \
  --from-literal=DATABRICKS_TOKEN_EASTUS="dapi_xxx" \
  --from-literal=DATABRICKS_TOKEN_EASTUS2="dapi_xxx" \
  --from-literal=DATABRICKS_TOKEN_NORTHCENTRALUS="dapi_xxx" \
  --from-literal=DATABRICKS_TOKEN_SOUTHCENTRALUS="dapi_xxx" \
  --from-literal=DATABRICKS_TOKEN_WESTUS="dapi_xxx" \
  --from-literal=DATABRICKS_TOKEN_AUSTRALIAEAST="dapi_xxx" \
  --from-literal=DATABRICKS_TOKEN_CANADACENTRAL="dapi_xxx" \
  --from-literal=DATABRICKS_TOKEN_JAPANEAST="dapi_xxx" \
  --from-literal=AZURE_KEY_EASTUS2="..." \
  --from-literal=AZURE_KEY_SWEDENCENTRAL="..." \
  --from-literal=AZURE_KEY_POLANDCENTRAL="..." \
  --from-literal=AZURE_KEY_SOUTHCENTRALUS="..."

# Secret 2：Copilot OAuth token JSON（被 Pod 内 reload_github_token() 监听重读）
kubectl create secret generic claude-lb-copilot-cache -n $NS \
  --from-file=copilot-auth-gh-account-1.json=$HOME/.config/databricks-claude-lb/copilot-auth-gh-account-1.json
```

<a id="panel-password-and-the-controller-role"></a>

#### Dashboard 面板密码：加它之前先看控制器 Role

面板解锁密码有两个来源：**`LB_DASHBOARD_PASSWORD`（env，基线）** 与**存储里的覆盖值**
（运维在 Dashboard 的「面板密码」页写入，存 PBKDF2 单向验证器）。有覆盖值时它优先；
覆盖值缺失或策略存储未加载时回落 env；**两者都没有则面板密码登录整体不可用而不是放行**。
无论哪种来源都要求 ≥12 个**字符**（非字节）。

所以 Secret 仍然是该配的基线，三个理由：

1. 策略存储**只在启动时加载**，所以页内改的密码只对**当前副本**即时生效；其余副本在
   重启前仍然接受 env 里的那份。扩副本或滚动重启后，Secret 里是什么就以什么为准
2. 覆盖值被忘记时只能用 LB API Key 解锁后重设（验证器单向、找不回来）—— 有一份 env
   基线能少一次尴尬
3. 全新集群第一次打开面板时还没有覆盖值

**页内改完密码请同步更新 Secret，再滚动重启**，否则下次重启会静默回到旧密码。

示例 manifest 用 `envFrom.secretRef` 指向 `claude-lb-secrets`，所以在那条路径上**只需往
该 Secret 加一个 key**，Deployment 模板不变。

**live 集群（zeno-apps）不是这样**：它没有 `claude-lb-secrets`，`envFrom` 为空，env 秘密
各自用单独的 `secretKeyRef`（当前只有 `LB_CREDENTIAL_KEY` → `claude-lb-credential-key`）。
所以那里有两个选择，代价不同：

| 做法 | 要动 Deployment 模板 | 要同步控制器 Role |
|---|---|---|
| 把 key 加进既有的 `claude-lb-credential-key` | 加一个 env（`secretKeyRef.name` 不变） | **否** —— 该 Secret 名已在 `resourceNames` 白名单里 |
| 新建一个 Secret（如 `claude-lb-panel-auth`） | 加一个 env + 新的 `secretKeyRef.name` | **是，必须** |

> **给 Deployment 新增任何 ConfigMap/Secret 引用，都要把该对象名加进
> `Role/lb-release-controller-target` 的 `secrets`（或 `configmaps`）`resourceNames`。**
> preflight 的 `_verify_references()` 会逐个读取模板引用的每个对象并比对摘要，漏一个
> 就是 403 —— 2026-10-14 首轮发布就是这么中止的。而 **`plan` 阶段不会暴露这个问题**，
> 因为 plan 用的是运维自己的身份，不是控制器的 ServiceAccount。
>
> **别凭记忆核这份清单 —— 它可以算出来。** 安装命令里的 `--secret` / `--configmap`
> 是命令行参数，所以「当次正确的集合」并不留在仓库里；2026-10-19 的发布就是因为
> 2026-10-18 的凭据迁移加了一个 `envFrom` 而没同步 Role 才在 preflight 403 中止的。
> 下面这条命令从 live Deployment 推导实际引用，再与 Role 允许的集合比对，**空输出 = 一致**：
>
> ```bash
> kubectl -n YOUR_TARGET_NAMESPACE get deploy YOUR_DEPLOYMENT -o json \
>   | python3 -c '
> import sys,json
> d=json.load(sys.stdin);sp=d["spec"]["template"]["spec"];c=sp["containers"][0]
> cms=set();secs=set()
> for v in sp.get("volumes") or []:
>     if v.get("configMap"):cms.add(v["configMap"]["name"])
>     if v.get("secret"):secs.add(v["secret"]["secretName"])
> for e in c.get("env") or []:
>     f=e.get("valueFrom") or {}
>     if f.get("secretKeyRef"):secs.add(f["secretKeyRef"]["name"])
>     if f.get("configMapKeyRef"):cms.add(f["configMapKeyRef"]["name"])
> for e in c.get("envFrom") or []:
>     if e.get("secretRef"):secs.add(e["secretRef"]["name"])
>     if e.get("configMapRef"):cms.add(e["configMapRef"]["name"])
> print(json.dumps({"configmaps":sorted(cms),"secrets":sorted(secs)}))'
> ```
>
> 把结果与 `kubectl -n YOUR_TARGET_NAMESPACE get role lb-release-controller-target -o json`
> 里对应规则的 `resourceNames` 逐项对照。**2026-10-19 实测两侧一致**：
> configmaps `[claude-lb-config]`、secrets
> `[claude-lb-copilot-auth, claude-lb-credential-key, claude-lb-upstream-credentials]`。
> 改 Deployment 的引用之后、下一次发布之前跑一遍，比在 preflight 被 403 拦住便宜得多。
>
> 查法（`resourceNames` 必须一起打出来，否则 `can-i` 的结果看着自相矛盾）：
>
> ```bash
> kubectl -n zeno-apps get role lb-release-controller-target -o json \
>   | python3 -c "import json,sys;[print(r.get('resources'), r.get('verbs'), r.get('resourceNames','(unrestricted)')) for r in json.load(sys.stdin)['rules']]"
> ```
>
> 改 Role 用 `kubectl apply`（从它自己的 `last-applied-configuration` 重建后只改那一处），
> 不要用 JSON patch —— 该对象由 apply 管理，patch 会让 `last-applied` 失配，下一次
> apply 就把改动吞回去。

未配置或过短时面板登录整体不可用，页面回落到用 LB API Key 解锁，**不会退化为无密码放行**。

### 3. （可选）把 ConfigMap 的镜像引用替换成你的 ACR

`deploy/k8s/kustomization.yaml` 改：
```yaml
images:
  - name: <REGISTRY>/databricks-claude-lb
    newName: myacr.azurecr.io/databricks-claude-lb
    newTag: latest
```

---

## 三、部署

```bash
kubectl apply -k deploy/k8s/ -n claude-lb

# 观察启动
kubectl logs -n claude-lb -l app=claude-lb -f
# 期望日志（JSON 格式）：
#   {"ts":"...","level":"INFO","logger":"main","message":"Loaded Databricks endpoint: ..."}
#   {"ts":"...","level":"INFO","logger":"main","message":"[Copilot] resolved token for 'gh-account-1' from /home/app/.config/..."}
#   {"ts":"...","level":"INFO","logger":"main","message":"[Copilot] session token cached for gh-account-1: base=..., expires in 86340s"}
#     ↑ expires_in 由上游 expires_at 决定；本账户实测 ≈86400s（24h）。别按固定值断言
#   {"ts":"...","level":"INFO","logger":"main","message":"[Copilot] background refresh started: interval=300s, threshold=600s"}
```

### Service 暴露

默认 ClusterIP。对外访问的常见做法：

- **集群内消费者**（Codex CLI 跑在另一个 Pod 里）：直接 `http://claude-lb.claude-lb.svc.cluster.local:8000`
- **VPN / 私有 LB**：`kubectl patch svc claude-lb -n claude-lb -p '{"spec":{"type":"LoadBalancer"}}'`，配合 Azure Internal LB annotation
- **公网 + TLS**：上 Ingress（推荐 ingress-nginx 或 application gateway），配 cert-manager

#### Ingress 配置（重要）

仓库提供 `deploy/k8s/ingress.yaml` 模板，**4 个 annotation 都不能省**：

| Annotation | 推荐值 | 不设的后果 |
|---|---|---|
| `nginx.ingress.kubernetes.io/proxy-body-size` | `64m` | ingress 默认 1MB，带图请求在到达 LB 之前就被 413 拒，LB 的图片自动压缩没机会跑 |
| `nginx.ingress.kubernetes.io/proxy-buffering` | `off` | 默认 on 会把 SSE 流缓存到关闭再下发，客户端"一次性收到"或读超时断流 |
| `nginx.ingress.kubernetes.io/proxy-read-timeout` | `600` | 默认 60s，长 thinking + 慢上游会被 ingress 主动断 |
| `nginx.ingress.kubernetes.io/proxy-send-timeout` | `600` | 同上 |

`64m` 直接对齐 `main.py:MAX_RAW_REQUEST_SIZE=64MB`，让 Pillow 把超大图缩到 ≤4MB 再上行。如果业务上确实没大图，可以下调到 `8m` 减少滥用面。

**部署 + 验证**：
```bash
# 1. 改 host（lb.example.com → 你的域名），如果用 cert-manager 把 cluster-issuer 改成现网名
vim deploy/k8s/ingress.yaml

# 2. apply
kubectl apply -k deploy/k8s/

# 3. 确认 annotation 进了 ingress 对象
kubectl describe ingress -n claude-lb claude-lb | grep -E "proxy-body-size|proxy-buffering|proxy-(read|send)-timeout"

# 4. 确认 nginx-controller 已 reload（看到 backend reloaded 字样）
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx --tail=50 | grep "backend reloaded"

# 5. 端到端测大请求不再 413
curl -X POST https://<your-host>/v1/chat/completions \
  -H "Authorization: Bearer $LB_API_KEY" -H "Content-Type: application/json" \
  --data-binary @big_image_payload.json -i | head -3
# 期望：HTTP/2 200（之前是 413 + nginx HTML 错误页）
```

**集群级别 ConfigMap 兜底**：如果 ingress 上加了 annotation 还是 413，说明 ingress-nginx Controller 的全局 ConfigMap 设了更严的上限：
```bash
kubectl -n ingress-nginx get cm ingress-nginx-controller -o yaml | grep -E "body-size|proxy-buffering"
# 如有 proxy-body-size 字段且 < 64m，让集群管理员调高
```

**Cloudflare 在前面时**：Free/Pro plan body 上限 100MB、Business 200MB、Enterprise 自定义。64MB 在所有档位都安全。如果客户端报 `cf-ray` + nginx 错误页，几乎可以确定是 ingress 拦的（Cloudflare 自己拦会返回 Cloudflare 品牌的错误页）。

---

## 四、Token 自动刷新机制（核心）

### Session token（TTL 由上游决定，本账户实测 ≈24h）— 全自动

| 触发点 | 行为 |
|---|---|
| 启动 warmup | 初始化时为每个 endpoint 交换一次 |
| 后台 task | 每 `COPILOT_REFRESH_INTERVAL=300` 秒扫一遍，剩 `COPILOT_REFRESH_THRESHOLD=600` 秒就主动刷新 |
| 请求触发 | 每次请求前检查，剩 ≤60s 就刷新 |
| 上游 401 | 强制刷新一次后重试同一请求 |

→ **任何情况下都不需要人工干预**。

### Long-lived OAuth token — 配合 K8s Secret rotation

**正常场景**：long-lived token 永不过期，无需任何操作。

**异常场景**（你 GitHub 那边主动 revoke 了 token）：

1. 本地重新跑：
   ```bash
   python main.py --copilot-login --endpoint gh-account-1
   ```

2. 替换 K8s Secret：
   ```bash
   kubectl create secret generic claude-lb-copilot-cache -n claude-lb \
     --from-file=copilot-auth-gh-account-1.json=$HOME/.config/databricks-claude-lb/copilot-auth-gh-account-1.json \
     --dry-run=client -o yaml | kubectl apply -f -
   ```

3. **零重启生效**（推荐 — 但有 1 min 延迟）：等 kubelet 自动把新 Secret 同步到 mounted file，下次后台刷新（≤5min）会自然 pick up。

3'. **立刻生效**：
   ```bash
   # 调 admin endpoint 强制重读 + 重新交换
   curl -X POST http://<svc>:8000/admin/copilot/reload \
     -H "Authorization: Bearer $LB_API_KEY"
   # 返回 JSON 显示每个 endpoint 是否成功 reload
   ```

3''. **保底**：如果上述都不行（比如 Secret 同步异常），最后手段：
   ```bash
   kubectl rollout restart deployment/claude-lb -n claude-lb
   ```

---

## 五、监控告警

### Prometheus 抓取（已 annotated）

`/metrics` 暴露：

| Metric | 告警建议 |
|---|---|
| `copilot_session_token_remaining_seconds` | < 120 持续 5 min → P2 告警（后台刷新失效） |
| `copilot_token_refresh_failed_total` | rate 5min > 0 → P2 告警 |
| `copilot_endpoint_circuit_open` | == 1 → P1 告警（**先看下一行判断是不是真凭证问题**；熔断后一次干净的最小请求即可恢复，见 `docs/TROUBLESHOOTING.md` 第 15 节） |
| `copilot_upstream_401_total{scope="endpoint"}` | increase 5min > 0 → P1（真凭证/席位问题，需重新登录 + rotate Secret）。`scope="request"` 单独涨**不要告警** —— 那是客户端在回放坏状态，账户是好的 |
| `copilot_orphaned_item_id_events_total{stage="recovered"}` | **> 0 → P2** —— 这才是「还有别的 `input[*].id` 通道」的金丝雀（事后还能剥到 id）。**旧文档写的 `stage="detected"` 恒 0 已被生产反证**：`detected` 非 0 通常只说明上游拒绝了 reasoning blob 的归属，主动剥 id 结构上防不住，见 `docs/TROUBLESHOOTING.md` 第 17 节 |
| `copilot_stream_retry_budget_exhausted_total` | **> 0 → P1** —— 流式重试循环的不变量被破坏，客户端本会收到 HTTP 200 + 空 SSE 流（会毒化会话）。按回归处理，先 `COPILOT_OPAQUE_STATE_RECOVERY=false` 再排查 |
| `copilot_session_affinity_total{outcome="unavailable"}` | increase 1h > 0 → P3（亲和目标熔断，该会话这一轮跨账户、丢推理链；先看那个账户为什么熔断）。**单账户下 hit/unavailable 必须恒为 0** |
| `copilot_opaque_state_requests_total` | increase 1h > 0 → P3 监控（受影响的**唯一客户端请求**数；用户表现为「这个会话往后每轮都失败」）|
| `copilot_opaque_state_recovery_total{outcome="exhausted"}` | **increase 1h > 0 → P2** —— 恢复阶梯走完仍失败，客户端必须重建会话。与 `..._requests_total` 相除 = 救不回的比例 |
| `copilot_opaque_state_recovery_total{outcome="succeeded"}` | 不告警，看趋势：与 `{outcome="attempted"}` 相除 = 恢复命中率。长期接近 0 说明 rung 2 的前提不成立，应 `COPILOT_OPAQUE_STATE_RECOVERY=false` 回到根因调查 |
| `databricks_endpoint_circuit_open` | == 1 持续 5 min → P2 告警（单 endpoint 不影响整体） |
| `copilot_endpoint_errors_total` | rate 5min > 0.1 req/s → P3 监控 |
| `copilot_stream_connections_active` | > 400 持续 30s → P2；结合 forced release / PoolTimeout 判断 |
| `copilot_pool_timeout_total` | increase 5min > 0 → P1（等池分配超时；**不等于**本地池已满，也不等于上游建连失败。拿结构化日志的 `httpx_pool_observed_full` 定方向，见 TROUBLESHOOTING §9） |
| `copilot_stream_forced_releases_total` | increase 15min > 0 → P2（已自动回收确认断开的 SSE） |
| `copilot_stream_upstream_idle_max_seconds` | 只做诊断，不单独告警/回收；长 thinking 可合法长时间无 token |

### Azure Monitor managed Prometheus

AKS 启用 [Azure Monitor managed Prometheus](https://learn.microsoft.com/azure/azure-monitor/essentials/prometheus-metrics-overview)：
```bash
az aks update -n <cluster> -g <rg> --enable-azure-monitor-metrics
```

`prometheus.io/scrape=true` 注解已在 deployment.yaml 配好，自动被采集。

### 日志（JSON 格式 → Log Analytics）

```kusto
ContainerLogV2
| where PodName startswith "claude-lb-"
| extend p = parse_json(LogMessage)
| where p.level == "WARNING" or p.level == "ERROR"
| where p.message contains "[Copilot]"
| project TimeGenerated, p.level, p.message
```

### OpenTelemetry tracing（默认镜像用不了，必须先重建镜像）

`OTEL_ENABLED=true` **单独设进 Deployment 是无效的**。OTel 的 5 个依赖放在
`requirements-otel.txt` 且**故意不在 `requirements.txt` 里**（tracing 默认关闭，
不值得让所有部署都背上 protobuf/grpc 传递依赖），所以默认镜像里没有这些包。
在这种镜像上打开开关只会在启动日志里留一条：

```
[OTel] OTEL_ENABLED=true but opentelemetry packages are missing (ModuleNotFoundError: ...). Tracing DISABLED.
```

然后静默继续跑，**没有 trace 但服务完全正常** —— 很容易被当成 collector 配错而往
错方向排查。要真正启用：

```dockerfile
# Dockerfile 里 `RUN pip install -r requirements.txt` 之后追加
COPY requirements-otel.txt .
RUN pip install --no-cache-dir -r requirements-otel.txt
```

重建 + push 镜像，再设 `OTEL_ENABLED=true` 与 `OTEL_EXPORTER_OTLP_ENDPOINT`。
排查用这条 KQL 确认到底是缺包还是 collector 不通：

```kusto
ContainerLogV2
| where PodName startswith "claude-lb-"
| extend p = parse_json(LogMessage)
| where p.message startswith "[OTel]"
| project TimeGenerated, p.level, p.message
```

---

## 六、容量规划

| 资源 | requests | limits | 说明 |
|---|---|---|---|
| CPU | 200m | 2000m | 主要消耗在 SSE 流式转发；峰值低 |
| 内存 | 256Mi | 1Gi | usage_data 缓冲 + httpx 连接池；MySQL 后端可降到 512Mi |
| 网络出口 | — | — | 需访问 `*.databricks.net`、`*.openai.azure.com`、`api.github.com`、`api.githubcopilot.com`、`api.enterprise.githubcopilot.com` |

单 Pod 处理能力：约 50 RPS（瓶颈在上游 token 速率限制）。如需扩容：

- replicas → 多副本时把 `usage_storage` 切到 MySQL（避免 PVC 挂载冲突）
- HPA 基于 `kube_pod_container_resource_requests{resource="cpu"}` 或自定义 metric `copilot_endpoint_active_requests`

---

## 七、故障排查

### Pod 不 ready

```bash
kubectl describe pod -n claude-lb -l app=claude-lb
# 看 Readiness probe 失败原因
curl http://<pod-ip>:8000/health/ready
# 返回 issues 字段定位到底是哪个 provider 没就绪
```

常见 issues：

- `github_copilot: no healthy endpoint` → Copilot token 失效，按"四"重新登录 + rotate
- `databricks: no available endpoints` → 所有 ADB workspace 都熔断，检查 token 是否被旋转
- `azure_openai: no available endpoints` → 同上，4xx 不熔断，所以一般是 5xx 持续

### Copilot 突然 401

```bash
# 看是不是 long-lived token 失效
curl http://<svc>:8000/metrics | grep copilot_token_refresh_failed_total
# 上升 → 走"四"的 rotation 流程
```

### usage 数据丢失

PVC 没挂载或 storageClassName 错误：
```bash
kubectl get pvc -n claude-lb
# Status 应为 Bound；Pending → 检查 storageClassName 是否存在
```

切到 MySQL 后端：编辑 ConfigMap，`usage_storage.type=mysql` + 加连接配置；同时 requirements.txt 加 `aiomysql>=0.2.0` 重新构建镜像。

---

## 八、运维 Cheat Sheet

```bash
# 看实时日志
kubectl logs -n claude-lb -l app=claude-lb -f

# 进容器调试
kubectl exec -it -n claude-lb deploy/claude-lb -- /bin/bash

# 看当前 token 剩余时间（在 Pod 外，靠 metrics）
kubectl run -i --rm --restart=Never -n claude-lb tmp-curl --image=curlimages/curl -- \
  -s http://claude-lb:8000/metrics | grep copilot_session_token_remaining_seconds

# 强制 reload Copilot token
TOKEN=$(kubectl get secret -n claude-lb claude-lb-secrets -o jsonpath='{.data.LB_API_KEY}' | base64 -d)
kubectl run -i --rm --restart=Never -n claude-lb tmp-curl --image=curlimages/curl -- \
  -X POST -H "Authorization: Bearer $TOKEN" http://claude-lb:8000/admin/copilot/reload

# 查 dashboard（端口转发到本地）
kubectl port-forward -n claude-lb svc/claude-lb 8000:8000
# 浏览器开 http://localhost:8000/stats/dashboard
```
