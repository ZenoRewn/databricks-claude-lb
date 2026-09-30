# 发布镜像与源码注解的一致性

Author: Zeno Ren

日期：2026-09-30。基线：d88d742。

发布准备时发现：旧发布器会更新 Deployment 镜像，却保留 Pod template 中较早的源码注解。实际运行文件可以已升级，监控仍会看到旧 SHA。本次将镜像、已核验的 source_revision 和副本恢复写入同一次 UID/resourceVersion/发布所有权条件补丁；回滚恢复原注解值或缺失状态，保留无关注解。

独立 QA 又指出，protected_spec 有意忽略允许变更的字段，不能独自证明 API Server 接受了目标注解。现在 CLI 计划、集群预检、每次实际写入前的 dry-run 以及正常 ACK 返回都核对完整目标 Pod spec 和注解。上游 admission 改写 image/source/probe 时拒绝通过；ACK 丢失后仅在实际模板匹配时作为已完成处理，不额外滚动。

新增回归覆盖正常切换、缺失注解、回滚、无关字段保留、ACK 读回、三层 dry-run 与正常 ACK 被改写。目标发布工具、业务探针、schema 和构建身份检查为 **54 passed、71 subtests passed**。失败与修复证据分别为：

- [最初失败](source-annotation-red.txt)
- [API 读回失败](source-readback-red.txt)
- [每次写入前检查失败](source-write-dry-run-red.txt)
- [最终目标回归](source-readback-final-green.txt)

应用运行模块未改变；此前已验证的应用镜像仍以其独立 runtime manifest 和 registry digest 为准。本报告仅覆盖发布工具变更及验证，不能替代具体 AKS 发布的备份、独立复核、双协调器就绪、业务和账本验收。独立 QA 在 11746b3 上通过（27 passed、29 subtests）；最终协调器镜像内 35 项测试通过，所有发布工具源文件与该提交匹配。主机完整回归为 730 passed、6 skipped、711 subtests；6 个 skip 为显式 opt-in 的隔离 MySQL 测试，应用存储代码未在本补丁改变。原始输出见 [主机回归](host-full.txt)、[协调器镜像回归](controller-image-tests.txt) 和 [结构化回执](validation.json)。GitHub 状态以本补丁 PR 的实时 Checks 与合并记录为准。
