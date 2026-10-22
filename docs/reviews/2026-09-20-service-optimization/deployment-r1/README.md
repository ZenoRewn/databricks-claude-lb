# 未完成发布的历史证据

Author: Zeno Ren

此目录记录 2026-09-20 一次未完成的发布及随后本地复盘。它不是生产发布成功回执。用户已在另一流程中恢复并完成升级；本目录历史补丁不得再次执行。

请先阅读 [事故补充复盘](INCIDENT_REVIEW.md)。

- `ci-r3.json`、`image-tests-r3.txt`、`source-hashes-r3.json`、`aks-image-preflight.json`：候选代码/镜像验证，不能替代生产切换成功。
- `mysql-migration.json`：已执行的追加表 DDL 回执。
- `maintenance-drain*.json`：实际维护与排空记录，不能证明流量最终恢复。
- `release-review.md` 和三个 patch 文件：事故前准备材料；现场已改变，不可直接执行。
- `gate-design.fixture.txt`、`reproduce_gate_failure.py`、`gate-failure-reproduction.json`：历史控制流的离线复现，不是可用发布器。

没有本任务生成的 `rollout.json`，因为本任务未执行后续镜像切换脚本。不要用另一执行流程的成功验收反向填补本次执行记录。
