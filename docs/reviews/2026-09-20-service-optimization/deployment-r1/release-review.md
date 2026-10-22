# Release review: 2026-09-20

Author: Zeno Ren

The release executor reviewed the existing candidate before deployment: request/lease/send ownership, timeout placement and cleanup, replay policy, compatible candidate selection and pinning, parameter compatibility, additive MySQL transactions and rollback, package contents, and the live deployment delta. This records a fresh technical review of the previously implemented candidate; it does not claim an external human approval or full client certification.

The first GitHub CI run exposed a Python 3.11 cancellation race in `asyncio.wait_for` when a queued slot was offered concurrently with cancellation. The existing regression test failed without modification. Commit `521f45a` switches to an owned `asyncio.timeout` around the shielded waiter, retaining the offered-lease cleanup. The test and its cancellation assertion were not weakened. The initially pushed ACR digest `ac91d5f618f44242c652afe448fff8ea0677e7008f3b9eefd4acfcc4016c9d30` must not be deployed; a new tested image is required.

Runtime publication uses GitHub noreply identities after GH007 rejected the first push. `commit-map.json` maps historical local evidence to identical published trees. The old local commits remain on a local backup branch; no existing remote history was rewritten.

Production data review confirmed MySQL 8.4.8-azure, InnoDB, the needed create/insert/update privileges, 90-day retention, Azure seven-day point-in-time backup retention, and a private aggregate-table snapshot outside Git. The additive ledger was created while the old service was running. Existing aggregate counts were unchanged across that DDL. Rollback retains this additive table and switches the application back; it does not restore an old aggregate snapshot over new usage.

Deployment changes are limited to the application image, `/health/live` liveness, `/health/accepting` readiness, startup probe, a 45-second preStop drain helper, and a 120-second termination grace. Existing authentication/configuration, mounted credentials, resources, storage, init container, replica count, and rollout strategy are preserved. No full repository Kubernetes apply is permitted.

The cluster has little schedulable CPU request headroom, so this release keeps the existing single-replica topology. A temporary Service selector gate stops new ingress work, existing requests are observed draining, and the old application's periodic usage flush is allowed to settle before replacement. The gate is removed only after the new pod passes local checks; a short service gap is expected. Failure requires restoring the old image/probes and removing the service gate. This is not a zero-downtime or high-availability claim.

Acceptance requires all GitHub CI jobs passing at the runtime commit, immutable ACR digest verification, live baked source hashes, readiness, clean usage persistence, and bounded real-provider checks. Full Mac/OpenClaw client E2E, sustained capacity tests, long thinking sessions, OTel export, and node maintenance remain explicit coverage limits. OpenClaw itself is not modified.
