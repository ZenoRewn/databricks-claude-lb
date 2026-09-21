# Main 合并前独立复核

Author: Zeno Ren

日期：2026-09-21。PR：[GitHub #5](https://github.com/ZenoRewn/databricks-claude-lb/pull/5)。独立复核由用户明确授权的只读 QA 子代理完成，未参与修复编辑；本记录由主代理整理，不代表人工签字。

初审对象为 `0490ff4`。初始 GitHub 六项 CI 全部通过，但独立复核发现三项未被原矩阵覆盖的问题。主代理先加入失败测试，再按以下独立提交修复；没有将首次失败记录改写为成功。

| 问题 | 修复 | 验证 |
|---|---|---|
| HTTP 200 JSON 上下文错误在流式路径惩罚健康端点 | `99a3d28`：三个 provider 使用已分类的请求局部原因做 neutral 结算 | 五个 provider/API 组合分别验证流式与非流式、一次发送、无成功结算、资源释放 |
| 可变旧镜像 tag 可能在回滚时漂移 | `f85c3a1`：准备、集群预检、旧版恢复与回滚开流前核对不可变单平台 digest 和 Pod imageID | 拒绝 tag、缺失/不匹配 imageID；错误镜像即使健康也不能恢复路由 |
| Histogram/Summary 部分分量重置仍可能报告完整增量 | `dc2981b`：任一分量重置时整族排除同一时间区间 | 整族统一 partial/unknown，并保留重置后的正常区间 |

旧镜像多平台 index 与运行时平台 manifest/config digest 的关系不能只凭 Pod 状态推断；v1 对不能直接核验的形态提前拒绝，不忽略不匹配，也不自动修改旧 Deployment 引用。

## 独立复核结论

复核源码：`dc2981b18522ef633805898d9b49a744f43a62e4`。

独立 QA 对三项修复重新审阅并复测原负例，运行 **52 passed、67 subtests passed**，`git diff --check` 通过。结论为：三项问题均关闭，在本次审阅范围内未发现剩余合并阻塞；建议在最新提交完整回归及 GitHub CI 通过后合并。

本报告之后的文档归档不改变上述运行源码。GitHub CI 的实际状态和最终合并回执以 PR 检查为准，本文件不提前宣告合并完成。

## 主代理验证记录

- [上下文错误负例](context-red.txt)：五个流式组合失败，非流式组合通过；[针对性回归](context-green.txt)：82 passed、87 subtests passed。
- [镜像身份和指标重置负例](safety-red.txt)；[发布器及报告回归](safety-green.txt)：43 passed、27 subtests passed。
- [完整本地回归](host-tests.txt)：Python 3.14，625 passed、6 skipped、553 subtests passed。六个 skip 为需显式隔离 MySQL 服务的集成用例；GitHub CI 分别运行 MySQL 8.0/8.4。
- [新协调器镜像测试](coordinator-image-tests.txt)：23 passed、23 subtests passed，未挂载 operations 源码；[镜像身份](coordinator-image.json) 与复核源码 hash 一致。

## 范围

本结论用于代码合并。本轮未连接或修改 AKS、未修改 OpenClaw、未调用真实模型供应商。没有重跑此次修订的完整 Kind 矩阵；之前的 Kind/候选应用镜像证据仍对应其原提交，见 [历史验收](../2026-09-20-post-upgrade-hardening/VALIDATION.md)。

生产部署需要为实际源码重新构建、核对不可变镜像，并完成当次环境、入口、供应商和运行验收。此处独立代码复核不能替代这些验收，也没有将生产 gate 解除。
