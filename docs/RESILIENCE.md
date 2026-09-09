# Local admission, outcome accounting, and replay safety

## Scope and invariants

This change builds on `5dc6e12` response/transport ownership and `9dd80a3`
credential-exchange pool isolation. It does not change the HTTPX response adapter,
read/close task ownership, SSE parsing, terminal-event detection, provider model
mapping, request transformations, credentials, dependencies, limits, or deployment
probes. Existing balancing strategies and configured model eligibility remain.
No model response is synthesized to claim success.

`LoadBalancer.on_request_start` returns a `RequestAttempt`. Production handlers
pass that same identity to the streaming response or settle it themselves.
Retries acquire a new identity. Finalization is idempotent per identity and cannot
subtract another request's active slot. Direct low-level callers may omit the
identity only with exactly one outstanding admission on that endpoint; ambiguous
legacy completion fails rather than guessing. This is an internal Python API
extension, not an HTTP request/API-schema change.

## Circuit state machine

* **CLOSED:** accept normal traffic. Successful inference resets
  `consecutive_errors`; eligible endpoint failures increment it. Client rejection,
  pool contention, and cancellation are neutral to this streak.
* **OPEN:** reject new admissions with HTTP 503 and integer `Retry-After` until
  the monotonic cooldown expires. Preserve cumulative error telemetry.
* **HALF_OPEN:** after cooldown, allow exactly **one real request** per endpoint
  to claim the trial slot. Eligibility and readiness inspection never claim it or
  declare inference recovered. A successful trial closes the circuit; an upstream
  failure reopens it. An inconclusive (cancelled/local/client-rejected) trial also
  rearms cooldown but does not increment errors. This deliberately avoids a
  rapid cancel/reject/reprobe loop.

Check-and-claim contains no await and is atomic in the application's event loop.
This is **per-process**, not a distributed circuit across worker processes/pods.
Every trip, successful trial, or explicit administrative circuit reset advances
an endpoint generation. A result admitted in an older generation still settles
its own slot and cumulative telemetry, but cannot override the newer breaker
state or restart its cooldown. Within a CLOSED generation, streak ordering is
completion order (not request start order).

There is no timer that abandons a long-running trial lease. Normal request/stream
ownership must settle it, including cancellation of a never-iterated response.
A genuinely non-cooperative transport can therefore delay recovery; this change
does not weaken the earlier joined-cleanup guarantee with a forced lease expiry.
No readiness check, TCP reachability probe, or successful token exchange counts
as a successful inference. Token refresh can clear an auth-invalid indicator but
cannot clear the inference circuit or cumulative errors.

### Which upstream 401s are eligible failures (Copilot only)

An upstream 401 is not one thing. On Copilot it is either a broken credential
(endpoint-scoped) or a request carrying state the upstream will not accept
(request-scoped: cross-account opaque-state replay, or a connection-bound
`input[*].id` whose connection is gone). `CopilotProxy._classify_upstream_failure`
splits them on `_request_has_opaque_state`: **a 401 on a stateful request is
neutral to the streak; a 401 on a stateless request is an eligible failure.**

This is the same design as pinning, seen from the other side. `_select_endpoint`
returns `pinned if pinned in matched else None`, and `matched` already excludes
circuited endpoints — so for a stateful request, opening the breaker can only
hurt: it kills the one endpoint that conversation can ever use and takes the rest
of that account's traffic with it, while failover was already forbidden for that
request. Counting buys no availability.

Coverage does not narrow. A revoked seat still exchanges tokens successfully, so
`auth_unhealthy` never sets and `_mark_endpoint_unhealthy` never fires — only the
streak can trip that case. But a revoked seat 401s *everything*, including every
first turn and all Chat Completions traffic (`_request_has_opaque_state` is always
False for chat), and those are stateless, so they still count. A genuinely invalid
long-lived token is separate and faster: `_mark_endpoint_unhealthy` calls `_open()`
directly, with `consecutive_errors` untouched.

403 and 429 remain eligible failures regardless of statefulness: 429 is server
overload and 403 is typically CDN bot management, both endpoint-level signals.

Databricks and Azure keep the original inline predicate at all four of their
call sites — there a 401 really does mean the endpoint's own credential
(`dapi` token / `api-key`) is bad, so counting it is correct. The kill switch
`COPILOT_STATEFUL_401_NEUTRAL=false` restores the legacy behaviour.

## Compatibility and metrics

Existing endpoint `total_errors`, `total_requests`, `active_requests`,
`successful_requests`, `error_rate`, usage statistics, and `circuit_open` remain.
`total_errors` is now actually cumulative across automatic recovery/token refresh;
only the existing explicit statistics reset or process recreation may reset it.
Existing token invalidation can contribute a non-request auth error, so this
legacy counter is not a pure denominator-matched request failure rate.
`successful_requests` retains its usage-accounting meaning; new
`completed_requests` measures successfully settled attempts even if local usage
recording fails. No old key or Prometheus metric is renamed or deleted.

Additive fields expose `consecutive_errors`, `circuit_state`,
`half_open_in_flight`, `retry_after_seconds`, `completed_requests`,
`cancelled_requests`, `neutral_requests`, `rejected_requests`, and
`usage_record_errors`. Numeric Prometheus fields have provider/endpoint labels
only; no request IDs, prompt/user text, request headers, or credentials are added.
`circuit_open=true` includes HALF_OPEN (not yet proven recovered), even when a
real trial is admissible. Rejections counted by `rejected_requests` are atomic
admission rejections, not every routing-precheck rejection.

Local usage-record failures are reported as an aggregate and exception **type**,
not retried or misrepresented as upstream inference failure. Usage persistence
is best-effort on this request path; a failure may leave partial usage totals and
requires operational attention. Payload/model/tool/context processing is unchanged.

`copilot_upstream_401_total{scope="request|endpoint"}` exposes the 401 split above.
A high `request` with `endpoint` at 0 means clients are replaying state the
upstream rejects while the account itself is fine; a climbing `endpoint` means a
real credential or seat problem, and the streak will trip the breaker as before.

### Label counters carry a zero baseline

`copilot_orphaned_item_id_events_total`, `copilot_stateful_request_pinned_total`,
`copilot_upstream_html_events_by_status_total`, and `copilot_upstream_401_total`
are backed by dicts that are empty on a healthy process. Rendering samples straight
from those dicts emitted HELP/TYPE with **no sample line at all**, so a healthy
scrape carried no `0` series and an operator could not distinguish "no events" from
"metric never shipped" — worst of all for the `stage="detected"` canary, which is
required to stay 0. `_labeled_counter_samples` now fills the known label set with
zeros at exposition time, so alerts can be written as `> 0` rather than `absent()`.
The dicts themselves are **not** pre-seeded, so their runtime semantics are
unchanged, and labels outside the known set still appear. Model-labelled counters
such as `copilot_stream_truncated_no_completion_by_model_total` have an open label
domain and are deliberately not pre-seeded.

## Routing and retries

* Known configured model with no available endpoint: **503**, not unsupported
  model **404**. Azure deployment eligibility is distinguished from availability.
  Copilot wildcard configuration expresses eligibility, not proof the upstream
  supports every arbitrary model; explicit upstream unsupported-model rejection
  retains 404/fallback behavior.
* OPEN `Retry-After` is ceil(remaining monotonic seconds). A busy HALF_OPEN trial
  yields a one-second advisory minimum, **not** a promised recovery deadline.
* Only pre-execution connection acquisition/setup failures (`PoolTimeout`,
  `ConnectTimeout`, `ConnectError`) and explicit admission rejection (429 without
  `Retry-After`; existing 401 session refresh) can take bounded existing retry
  paths. Existing maximum is three attempts with bounded exponential delay plus
  small jitter. Single-endpoint Copilot pool saturation still fails fast.
* Read/write timeout/error, remote protocol failure, malformed successful JSON,
  and internal exceptions must not replay POST, even before the client sees
  content. Generic ambiguous buffered failure returns 502. Returned upstream
  HTTP status is preserved; 5xx is not assumed to prove non-execution.
* A supplied upstream `Retry-After` suppresses internal 429 replay; buffered HTTP
  errors preserve that header (both delay-seconds and HTTP-date). Once streaming
  HTTP 200 is committed it cannot be changed; the existing SSE error envelope is
  returned, not an invented new HTTP status. Streaming Retry-After metadata is
  deferred to protocol coordination.
* Any upstream content forbids streaming replay. Databricks uses any forwarded
  content rather than relying solely on parsed `message_start` for this guard.
* Cross-provider fallback after an arbitrary 503 is prohibited. Only a locally
  classified pre-admission `endpoint_unavailable`, or the existing explicit
  unsupported-model rejection, may follow the existing configured Azure route.
  No new provider/account/model substitution policy is added.

HTTP semantics reference: RFC 9110 sections 9.2.2, 10.2.3, 15.5.5, 15.6.4.
In particular, no received-content heuristic establishes POST idempotency.
Circuit-pattern reference:
https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker

## Health endpoints

`/health` remains the backwards-compatible process-health alias; `/health/live`
checks process responsiveness only. `/health/ready` preserves the current
all-configured-critical-providers policy, but recovery-eligible HALF_OPEN state
can be ready without lying that inference recovered. A leased trial does not
make readiness oscillate. Copilot also requires a valid cached session and no
known auth-invalid indicator. Readiness is a local admission signal, **not** a
live provider SLA check; it sends no model request and acquires no trial lease.

Do not silently change Kubernetes probes. The inspected deployed probes both
use `/health`, so production currently treats them as process health. Moving a
single multiprovider pod to strict `/health/ready` could remove healthy sibling
routes when one provider fails. Decide all-critical versus partial-capability
readiness explicitly, then independently review any deployment change.

## Unresolved attribution and policy

The observed 10 partial-output truncations at two similar input byte lengths do
not prove a hard size threshold, timeout, token failure, or endpoint-wide outage.
This change removes unsupported timeout/size causal claims in client error text.
It deliberately retains existing truncation failure detection/accounting; it does
not disable a real upstream-failure signal based only on request-size correlation.
As a consequence, a sufficiently long eligible failure streak can still trip an
endpoint shared by sibling models. This stage is **not** the final solution to
request/model-specific failure isolation.

Parent coordination must choose after protocol forensics: (1) separate
endpoint/model/API circuits with parent endpoint protection for shared transport,
auth and overload failures; or (2) a narrowly proven request-local classification
that records truncation but does not trip the endpoint. Do not silently select
an arbitrary body-size cutoff, suppress all EOF failures, or blacklist request
fingerprints/prompts. VS Code and Mac client equivalence is unverified.

## Multi-replica semantics

What is and is not safe when `replicas > 1`. This section exists because the
intuitive worry is the wrong one, and acting on it blocks a real fix.

### Copilot opaque state is NOT process-affine

A recurring misreading is that a Codex session is pinned to an LB process, so a
second replica would cause `input item does not belong to this connection`. It
would not. The LB holds **no cross-request session state for Copilot**:

- `CopilotProxy._request_has_opaque_state(body, api_type)` is a `@staticmethod`
  that inspects only the request body (`previous_response_id`, or any
  `input[*].encrypted_content`). It reads nothing process-local.
- `pinned_endpoint` in `_proxy` is a local variable. It constrains endpoint
  switching **within one HTTP request's retries** and is discarded on return.
- There is no `response_id → endpoint` map anywhere in the codebase.
- Measured 2026-09-08: force-refreshing an endpoint's session token
  (`POST /admin/copilot/reload`) does **not** invalidate previously minted
  `encrypted_content` — the state is bound to the **account**, not to the
  session token and not to the process. GHCP additionally rejects
  `previous_response_id` outright (HTTP 400 `previous_response_id is not
  supported`), so `encrypted_content` is the only opaque-state channel in play.

Consequence: **with a single Copilot account, any replica can serve any turn.**
Routing is replica-agnostic.

**With two or more Copilot accounts this changes.** `least_requests` would split
consecutive turns of one conversation across accounts, and cross-account
`encrypted_content` returns 401. Replicas share no memory, so the fix must be
*stateless and deterministic* — e.g. a consistent hash over a stable session key
so every replica maps a given conversation to the same account. This is a
**prerequisite for adding a second Copilot account**, independent of replica
count (it is already a latent issue at `replicas=1` for the same reason). Not
implemented; `copilot_stateful_request_pinned_total{reason}` staying at 0 is
consistent with single-account operation.

### The real blocker was usage persistence

`UsageDataStore` accumulates the running day total in `_today_cache` (seeded
from the backend by `_load_day` at startup) and flushes every 30s. Persisting
that *cumulative* value is only correct for a single writer:

- Two replicas load the same starting point, each accumulate their own share,
  and each overwrite the row with `starting_point + own_share`. Whichever
  flushes last wins; the other replica's tokens are permanently lost. With a
  30s flush cycle this repeats indefinitely, so the stored day total collapses
  to roughly one replica's private view.

Fixed by splitting the persistence hook:

- `UsageDataStore._save_day_delta(d, delta, cumulative)` — `delta` carries only
  the current flush batch. The default implementation writes `cumulative` via
  `_save_day`, i.e. the original single-writer whole-day overwrite.
- `MysqlUsageStore` overrides it and writes `delta` with
  `col = col + VALUES(col)`, so InnoDB accumulates under the row lock and the
  stored row equals the sum across all replicas.
- `JsonUsageStore` keeps the default path. A file cannot be incremented
  atomically across processes, so **the JSON backend is single-replica only,
  permanently.** Multi-replica requires `usage_storage.type: mysql`.

Regression guard: `tests/test_usage_store.py::test_two_writers_sum_instead_of_clobber`
simulates two writers from a shared starting point and asserts the persisted
deltas sum to the true total. It fails against the overwrite implementation.

Secondary benefit at `replicas=1`: under the old scheme a transient `_load_day`
returning empty would let `_flush` rebuild the cache from zero and overwrite the
whole day row with just that batch. Delta writes have no such path.

### Still per-process under multiple replicas

These are per-process by design and are *degraded observability*, not
correctness bugs — but they will surprise anyone reading a dashboard:

| State | Multi-replica behaviour |
|---|---|
| `GlobalStats`, `ClaudeProxy.today_model_stats` | Each replica counts only its own traffic. `/stats` and the dashboard reflect whichever pod the request landed on, not the fleet. Prometheus scrapes every pod, so `/metrics` aggregates correctly — trust `/metrics`, not `/stats`, once `replicas > 1`. |
| Circuit breaker state (`circuit_open`, HALF_OPEN leases) | Each replica learns endpoint health independently. A dead endpoint trips N times instead of once; admission fairness holds per replica. |
| Copilot session token cache | N replicas perform N token exchanges. Harmless but multiplies calls to `copilot_internal/v2/token`; the background refresh interval applies per replica. |
| HTML soft cooldown | Per replica; one pod's cooldown does not steer another pod away. |

Deployment-side preconditions and the exact `RollingUpdate` block are documented
in `deploy/k8s/deployment.yaml`; live-vs-repo drift is in `docs/AKS.md`.

## Tests and review gate

New tests cover time-controlled 100-contender HALF_OPEN admission, stale results,
neutral/cancelled trials, long trial ownership, cooldown boundaries, pure status
inspection, cross-endpoint independence, explicit attempt identity, POST replay
safety, Retry-After, route status/fallback, and real streaming trial cancellation.
Existing transport/ownership tests are unchanged. Six existing test methods have
explicit policy/fixture adaptations: four no longer expect replay after malformed
2xx or local usage failure, one monitor fixture creates a real admission, and
one token-refresh test no longer expects inference-circuit/telemetry reset. Their
original source and non-PASS run are preserved in stage evidence; do not report
the literal unmodified suite as passing this changed contract.

All publication/deployment remains gated on the parent's independent review,
including the request/model-specific classification and full Mac/provider E2E
coverage gaps. This document is not authorization to deploy.
