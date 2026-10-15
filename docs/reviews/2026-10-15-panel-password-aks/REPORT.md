# 面板独立密码与会话 cookie AKS 发布回执

Author: Zeno Ren

2026-10-15 将 `db8845a1a4ef21feeac0183a666e7ed65b22a050`（功能提交 `272314f`，[#38](https://github.com/ZenoRewn/databricks-claude-lb/pull/38)）发布到 AKS。线上此前停在 `7229327`。

**一次成功**，北京时间 **07:52:05.442–07:52:56.147** 完成维护并恢复流量，**50.704 秒**，十项 action 全部 verified，三个业务探针通过且 `persistence_verified=True`（记录 3 条用量事件），`cleanup_errors` 为空。

## 本次发布改变了什么

Dashboard「密钥与渠道」页的解锁从 LB API Key 换成**独立密码 + HttpOnly 会话 cookie**。原来那把 key 驻留在页面脚本的闭包变量里，这一页上的 XSS 就能取走它，而刷新页面还要重输一遍。现在浏览器携带会话，**页面脚本不持有任何凭据** —— 不是只为省事。

密码取自 `LB_DASHBOARD_PASSWORD`（≥12 字符），未配置或过短时**面板登录整体不可用，绝不退化为无密码放行**：`/admin/*` 由没有 OAuth 的那个 ingress 提供，对公网可达。有效 API Key 也能换到同一个 cookie —— 否则回落路径仍得把 key 留在页面里，「脚本不持凭据」就只在一半情况下成立。

发布后走了一遍真实登录（从 Pod 内对公网 URL 发起，密码没有离开集群），**21 项全部通过**：cookie 带 `HttpOnly` / `SameSite=Strict` / `Path=/admin` / `Secure`，响应体不含密码也不含 token，cookie 单独授权四块，错密码 401 且不下发 cookie，登出后 admin 重新关闭，API Key 回落路径也能换到会话。

## 为什么是两步

面板密码要以 env 到达 Pod，而发布工具的 `plan` 从**计划时的实时快照** deep-copy pod template —— 所以 env 必须在 plan 之前就位，而加 env 本身就是一次滚动重启。顺序不可颠倒：

1. **加 env**（`kubectl patch --type=json`，先做 server-side dry-run 比对）→ generation 124，镜像未变，仍跑 `7229327`。Pod 内实测 env 到位、13 字符、`LB_CREDENTIAL_KEY` 未受影响
2. **plan → submit** → 切到 `db8845a`。已核对快照包含全部 7 个 env

密码写进**既有**的 Secret `claude-lb-credential-key` 而不是新建一个，正是为了绕开 2026-10-14 那次 403：它的名字已在控制器 Role 的 `secrets.resourceNames` 白名单里，所以**本次不需要动 RBAC**。发布前仍逐项复核了一遍 Role，那是上次翻车的位置。

密码由运维在自己的终端设定，值从未经过助手。顺带说明一件容易被误解的事：**Kubernetes Secret 的 `data` 只是 base64，不是加密**，有 namespace 读权限的人随时能解出来 —— 所以忘记密码不是风险，但它的保密性等同于 namespace 的访问控制，与 `LB_CREDENTIAL_KEY` 同一层。

## 我自己的三处错误

**一处差点造成灾难性回滚。** 私有准备记录里我写的是「这个 Deployment 不由 apply 管理，所以用 JSON patch」。它**是** apply 管理的 —— 命令开对了，理由写错了，而按那个理由行事会更糟：它的 `last-applied-configuration` 严重过时（image 是 `2f6fa30-full`、env 是**空数组**、探针与 lifecycle 全不一致，因为历次发布都走发布工具的 patch、从不走 apply）。如果我套用两天前改控制器 Role 时那套「从 last-applied 重建再 apply」的手法，会把镜像回滚到一个远古 tag、清空所有 env、重置全部探针。**动手前先把 last-applied 与 live 做了 diff**，才发现这件事。

教训是具体的：*apply 管理*与*last-applied 可信*是两件事。注解存在只说明某次用过 apply，不说明它还代表现状。

**两处是命令写错。** 设密码的命令用了 bash 的 `read -rsp`，而本机 shell 是 zsh —— 那里 `-p` 的意思是「从协程读取」，直接报错；换成 `stty` 写法并在两种 shell 下都验证过。核对 Secret 的命令用了 `-o jsonpath='{range $k,$v := .data}…'`，这个 kubectl 版本对 map 做 range 会**静默输出空**，一度看着像 Secret 里什么都没有；正确的工具是 `kubectl describe secret`，它只报字节数、不报值。

## 一个值得单独记的核验细节

Secret 的**字节数不等于字符数**，而网关的长度下限判的是字符数（Python `len(str)`）。这次 13 字节恰好是 13 个 ASCII 字符，合格；但同样字节数的多字节密码可能只有 4 个字符，会被静默判为「未配置」而面板登录始终打不开。所以没有只看 `describe` 的字节数，而是用网关自己的 `_panel_password()` 验了一遍。

## 构建

`linux/amd64`、`dependencies-offline`、工作树干净（`source_tree_dirty=false`、`out_of_tree=false`），镜像内 `source_attested=true`、23 个运行时文件齐全。**无新依赖**，`.wheelhouse` 直接复用；`operations/` 与 `APP_FILES` 集合自 `7229327` 起未变，所以**不需要重建发布控制器镜像**。

推送前在镜像内自检：build-info attestation、23 个运行时文件、**密码缺失与过短都判为不可用（绝不敞开）**、会话只存摘要不存 token、三条新路由存在、镜像里的 `dashboard.html` 是密码版且面板代码段只剩一处 `x-api-key`（即换 cookie 那一次）、`LB_SETTINGS` 不含密码。

## 发布后核验

公网 `/version` 报 `db8845a`、`verification: matched`；Pod `claude-lb-68f6b9978-695hc` Ready、重启数 0；`/health/accepting` 200；两个 Service selector 干净无发布门标签残留。`/admin/session` 报 `password_configured: true`、`auth_methods: [panel_password, api_key]`、`session_ttl_seconds: 1800` —— 这个端点不需要凭据，是确认 env 到位且长度合格最直接的一条。

`[Policy] backend=mysql state=loaded`、`[Credentials] backend=mysql state=loaded key=present`。9 个 Databricks、4 个 Azure、5 个 Copilot 端点加载。本批**不建表、不改列**，`model_copilot_account` 是上一次发布创建的。

**本次发布不改变任何路由行为。**

## 未完成

- **把路由决策在请求准入时冻结**仍未实现
- **Azure Foundry 凭据仍然全路径 401**，凭据本身得换 —— 运维明确推迟
- 4 个 Copilot 账户已加载可见但仍 inert —— 运维明确不启用
- **面板会话是进程内的**，重启或第二个副本都意味着要重新解锁。单副本下可接受，扩副本前值得重新考虑
- Copilot 价格刷新记了一条 `ValueError` 并保留了上一次的价格。既有 fail-safe 行为，未深究

机器可读回执见 [receipt.json](receipt.json)。
