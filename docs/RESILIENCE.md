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
