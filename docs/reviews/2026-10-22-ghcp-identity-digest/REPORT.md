# GHCP 逐 delta 换 ID 导致长回答中断：修复与 AKS 发布

Author: Zeno Ren

日期：2026-10-22（北京时间）

## 现象与根因

Codex 报 `stream disconnected before completion: ... Response message identity could not be safely normalized; generation was not replayed.`（req `b881c0c8…`，`gpt-6-astra`，`zenoren-icloud`）。

线上 48 小时日志：3676 次 identity 处理，10 次失败，**全部是 `identity_mapping_limit`**，无一次真实身份冲突。失败样本均为 `identity_drift_items: 1`、`identity_aliases ≈ 5041`、`identity_rewritten_frames ≈ 5040`、`identity_peak_mapping_bytes ≈ 2,097,063`（上限 2 MiB）。

即 GHCP 在**每个 delta** 上换一个新的 message `item_id`（约 416 字节）。旧实现把每个 alias 按原始字符串保存，保留字节随回答长度线性增长，约 5000 个 delta 即触顶。`stabilize` 按设计不混用 ID、不重放，于是中断。

## 修复（`51507c5`）

- alias 表改存**每 attempt 随机 key 的 16 字节 BLAKE2b 摘要**；只有要写回下游的 canonical ID 与 item type 原样保存。保留字节随 delta 数增长，而不随上游 ID 长度增长，映射里也不再留存原始上游 ID。
- 跨 index 冲突检测语义不变；摘要碰撞只会导致误报冲突（fail closed），不会静默混用 ID。
- `max_aliases` 16384 → 131072；2 MiB 字节预算不变。
- 新增 3 条测试（先在旧实现上红）：2 万次逐帧换 ID 完整结束且下游只见一个 ID；长漂移后跨 index 复用仍被检出；alias key 不是原始 ID。一条容量边界用例阈值 9 → 30（每 alias 固定 16 B）。

## 验证

| 检查 | 结果 |
|---|---|
| 完整回归 `python3 -m pytest tests/ -q` | 1685 passed、6 skipped（需本地 MySQL）、1400 subtests |
| node UI 测试、`git diff --check` | 通过 |
| 本机合成测量（tracemalloc，非 RSS） | 线上失败样本规模 5041 alias：81 KB（旧 2.1 MB）；131000 alias：映射 2.0 MB、跟踪内存约 11 MiB、5.5 s |

## AKS 发布

- 镜像 `zenoseaacr.azurecr.io/databricks-claude-lb@sha256:1e5bcd897b3a7921ee35934637dc7e41217888a08a1321c56ce64707113ea3c1`，`operations.build_identity` 构建，源码干净。
- release `ghcp-id-digest-51507c5`：沿用上一次 profile，仅换镜像；`COPILOT_RESPONSES_ITEM_ID_MODE=stabilize` 与模型名单不变。运行文件集合未变，控制器无需重建。
- 首次 `submit` 因 zsh 未拆分参数变量而**根本未执行**（status 404，生产未触碰）；修正命令后提交，preflight → draining → verifying_backend → verifying_business → `succeeded`。
- 发布后：`/version` 为 `51507c5`、`verification: matched`；Pod 内 `responses_item_identity.py` sha256 `2c6cabef…` 与构建清单一致；15 分钟内 2 次 identity 处理均 `reason: none`。

## 未验证边界

- 发布后尚未出现超过 5000 delta 的真实长回答，线上「长输出不再中断」待实际流量确认：
  `kubectl -n zeno-apps logs deploy/claude-lb | grep copilot_response_identity | grep -v '"reason":"none"'`
- 131072 alias 的最坏内存仅为本机合成测量，不代表生产 RSS 上限（Pod limit 2Gi，发布前常驻约 293Mi）。
