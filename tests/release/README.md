# 发布工具的隔离验证

Author: Zeno Ren

普通 `python -m pytest tests/` 会执行纯状态机、Lease、计划约束和运行时回归。`kind_lab.py` 与 `run_kind_scenarios.py` 仅通过显式参数运行，不会被 pytest 自动连接集群。

两者拒绝非 `kind-lb-release-lab-*` context 和非 loopback API 地址。使用独立 kubeconfig；不得复用生产 context。所有模型请求均由 `upstream_fixture.py` 合成，MySQL 使用临时数据库与测试 CA，TLS 校验保持开启。

## 一次性环境

1. 创建三节点 Kind：一控制平面、两 worker，名称以 `lb-release-lab-` 开头，并用 `--kubeconfig` 保存到任务专属私有目录。
2. 创建仅监听 `127.0.0.1` 的本地 registry，连接 Kind Docker 网络；仅在这些临时节点的 containerd hosts 配置中声明该测试 registry。
3. 使用项目 Dockerfile 构建当前应用、`deploy/release/Dockerfile` 构建协调器。另制作仅增加测试 label 的应用镜像，以不同 digest 验证相同源码的重复升级。
4. 旧版 fixture 的十份应用运行文件取自 `521f45a`，与已核对的 892e397 源码树一致；它运行在本地固定依赖和架构上，不冒充原 AKS 镜像或原部署依赖。
5. 把这些测试镜像及 MySQL 8.4 放到本地 registry。`images.json` 使用 `old`、`new`、`alternate`、`controller`、`mysql` 五个不可变镜像引用。

安装独立测试依赖后，用下面入口创建应用、两条 Service、合成上游和验证 TLS 的 MySQL：

```bash
PYTHONPATH=. python tests/release/kind_lab.py \
  --context kind-lb-release-lab-YOUR_RUN \
  --kubeconfig YOUR_PRIVATE_DIRECTORY/kubeconfig \
  --images YOUR_PRIVATE_DIRECTORY/images.json \
  --directory YOUR_PRIVATE_DIRECTORY/lab
```

使用 `operations.release.manifests` 生成并安装**到上述临时集群**的双协调器清单，设置 `--lab`，目标 `lb-lab/claude-lb`，只读引用 Secret 为 `lab-config` 和 `lab-tls`。不要把 lab 清单用于 AKS。

## 场景入口

```bash
PYTHONPATH=. python tests/release/run_kind_scenarios.py \
  --context kind-lb-release-lab-YOUR_RUN \
  --kubeconfig YOUR_PRIVATE_DIRECTORY/kubeconfig \
  --images YOUR_PRIVATE_DIRECTORY/images.json \
  --directory YOUR_PRIVATE_DIRECTORY/scenarios \
  --output YOUR_EVIDENCE_DIRECTORY \
  --case UNIQUE_CASE_ID --mode crash --phase draining
```

可选模式：普通发布（省略 mode）、`ack_loss_after_gate`、`journal_failure`、`verification_failure`、`exec_ack_loss`、`crash`、`long_stream`、`manual_takeover`、`unavailable_rollback`、`scheduling`。

`crash` 覆盖 gating、pausing、draining、stopping、starting、verifying_backend、restoring_routes、verifying_business、finalizing。测试通过 CRI 仅终止协调器容器，等待 Lease 的新持有者/代次；不强删应用 Pod 来制造 writer 退出证据。

回执校验实际镜像、两条 Service 的完整 selector/Ready endpoint、终态以及合成供应商发送次数。ACK 丢失与 exec 回执丢失不能增加模型发送次数。长请求测试要求原 Pod 保留、请求正常完成；人工接管不能被覆盖；无健康恢复目标必须进入 needs_attention；调度测试只对任务自有 Kind worker 临时 cordon，并在 finally 恢复。

每个 case ID 只运行一次以保护原始材料。失败记录保留，修复后使用新 ID；不能重写成通过。测试结束后删除任务创建的 Kind 集群、本地 registry 和临时数据库容器，保留脱敏回执，私有目录不纳入 Git。

测试 profile 的全零 Git SHA 是明确的 lab fixture 标记；实际运行文件用逐项 hash 核对。它不是某个发布提交的证明。正式计划须从真实 Git SHA 读取源码，并用固定 registry digest 验证。
