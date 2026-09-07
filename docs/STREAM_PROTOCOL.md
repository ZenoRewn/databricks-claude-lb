# Streaming protocol, retained-byte policy and API eligibility

This contract extends [RESILIENCE.md](RESILIENCE.md). It does not diagnose any original client incident, set model token/context limits, authorize account enrollment, or change deployment probes. Existing ownership/close adapters and token-exchange isolation are retained.

## Wire contract

All three native streaming providers use one complete-frame framer. It recognizes WHATWG CRLF, CR and LF, preserves complete wire frames, parses UTF-8 with replacement semantics and ignores one initial BOM. It retains only the unfinished frame, not the entire response. A final CR is a valid line ending; an unfinished event at EOF is discarded without appending a delimiter. Independent local errors are emitted only at known frame boundaries. No completed/Anthropic message_stop event is manufactured to hide a missing terminal. Empty and comment-only EOF are not success.

A complete frame is offered promptly once its blank-line delimiter arrives. CR at a chunk boundary waits for the following byte or EOF so a split CRLF remains intact. This is intentionally different from immediate arbitrary HTTP chunk forwarding: clients receive no partial event. Heartbeats remain independent SSE comments. If a local heartbeat precedes the first upstream BOM, that BOM is removed from the first forwarded frame because it would no longer be at stream offset zero. Without that exception, complete frames remain byte-identical. No additional buffering of completed events is introduced beyond the existing bounded pump and downstream backpressure.

The first recognized valid API terminal settles the attempt before being offered downstream. The remaining upstream transport is closed, not drained indefinitely. Bytes after that terminal are not forwarded. A later transport EOF/reset/close cannot override a terminal generation outcome or cause a second settlement. This also accommodates clients that stop reading at completion. Actual failures before a valid terminal remain conservative; no automatic POST replay is allowed after receiving response headers/body, including when a partial event has not yet been forwarded.

### Terminal and usage interpretation

WHATWG specifies framing, **not** the application JSON schema. For Responses, the JSON `type` discriminator takes precedence over an SSE event header. Header-only, malformed JSON, non-object data, missing/wrong completed ID and wrongly typed usage/optional fields do not certify successful completion. The required/optional fields follow the pinned Codex implementation below, including i64 usage fields, nested cached/reasoning tokens, optional end_turn and usage_metadata.amount. Contradictory response.status/error/incomplete_details cannot count as success. These integer/schema checks are not model-specific token ceilings.

- Responses `response.completed`: valid completed object → successful generation and usage.
- Responses `response.failed` / `response.incomplete`: generation failure, never successful usage accounting. Like pinned Codex, these discriminators surface errors even without optional error/reason details. Only explicit `invalid_prompt`, `context_length_exceeded`, `max_output_tokens` error codes or incomplete reason `max_output_tokens` are request-neutral. Unknown/server/auth/quota reasons are not exempted.
- Chat: exact assembled data `[DONE]` ends a successful stream; a valid error object is preserved as failure, without an extra local truncation.
- Anthropic: JSON `message_stop` ends a successful stream; JSON `error` is preserved as failure. Usage is read from message_start/message_delta.
- Native JSON error events are preserved once. Unknown EOF/malformed terminals are not evidence of any particular timeout/size cause.
- A local observer exception is `local_observer_error`: neutral/inconclusive, with no automatic POST replay.

Failed/incomplete/error terminals and local limits release the same exactly-once admission lease. A local-pressure HALF_OPEN trial rearms cooldown once without endpoint-error increments. HTTP 401/403, 429 and 5xx remain health evidence; the existing one-time Copilot 401 session repair is retained, but an unrepaired/second rejection is a real failure. A streaming 401 repair keeps the same admission across its single auth retry, avoiding opening the inference breaker before that repair can run. Token exchange itself never resets inference health/history.

## Explicit operational resource policy

Environment variables are parsed at startup/import:

| Setting | Default | Meaning |
|---|---:|---|
| `PER_PENDING_EVENT` | 8,388,608 bytes (8 MiB) | Maximum pending content-decoded SSE frame bytes, including fields, BOM and delimiters |
| `PER_PROCESS_TOTAL_RETAINED_STREAM_BUFFER` | 67,108,864 bytes (64 MiB) | Process-wide retained decoded pump/current-delivery plus pending/yielded-frame accounting, shared by all providers |

Both must be decimal positive integers no greater than `sys.maxsize`; event budget must not exceed aggregate budget. Empty, zero, negative, nonintegral, nonfinite or inconsistent settings fail startup. Limits are overrideable operational choices, **not provider-advertised cutoffs**. Operators must size overrides for process/container capacity. There is no claim of zero compatibility impact: an event larger than the approved cap, or a stream arriving during aggregate pressure, fails locally. No disk spooling is used.

The stream's total lifetime bytes are unlimited: every completed/released frame returns its accounting. The 64-item queue remains a secondary backpressure bound; byte reservations also cover a producer waiting to enqueue and the current delivery while its frames are consumed. The current delivery and its copied pending/yielded frame are both charged while both are retained. Thus aggregate admission can reject before the sum of logical pending-event sizes alone reaches 64 MiB. This conservative accounting is intentional, not silent observer discard.

On an event cap or aggregate pressure, fail only the impacted stream with outcome/message `local_resource_limit` and SSE error code `protocol_buffer_limit`. Discard its unfinished frame, close/cancel its upstream, return all reservations, settle neutral/inconclusive and never automatically replay its POST. Siblings already admitted are not evicted. In particular, no old 64 KiB observer reset is reused.

### Decompression and the limit of the guarantee

For streaming single `Content-Encoding: gzip`, the code uses supported HTTPX `aiter_raw()` with an incremental stdlib zlib decoder, requesting at most 16 KiB decoded output per step, including concatenated gzip members. The original `_OwnedResponseStream` remains installed on the response: raw reads and EOF/error cleanup still cross its cancellation-safe boundary. This avoids HTTPX's otherwise potentially huge single gzip-decoded allocation. Setting `aiter_bytes(chunk_size=...)` alone would not avoid that allocation, so it is not used as a false decompression guarantee.

Other/stacked encodings, preconsumed responses and nonstandard response implementations retain HTTPX's supported `aiter_bytes()` path. An oversized decoded delivery there may **already be allocated**. Count/check it immediately before retention, discard the rejected delivery/traceback references, cancel/close the impacted upstream, and drain its queued references on aggregate pressure rather than waiting for a slow consumer to free a queue slot. This is a transient allocation limitation, not a hard whole-process RSS guarantee.

Accounting excludes compressed raw input, Python object/bytearray overhead, brief conversion/JSON/UTF-8/decompression copies, HTTPX/httpcore internal buffers, buffered non-stream responses/HTTP-error body reads, request serialization, downstream ASGI/socket/kernel buffers and unrelated application state. Complete-frame JSON parsing may temporarily allocate multiples of frame size. A huge raw transport chunk can also already exist before application inspection. The budget bounds the documented retained stream payloads, **not total RSS**, and is per process rather than a distributed multi-worker memory coordinator. Bounded tests/benchmarks do not certify sustained production capacity.

## Copilot model + API eligibility

Copilot endpoints accept an optional `api_types` allowlist:

```yaml
name: example-responses-account
weight: 1
models: [gpt-6-astra, gpt-5.6-sol]
api_types: [responses]
```

- Missing field: backward-compatible `responses` and `chat`.
- Explicit field: nonempty list of unique exact values `responses` / `chat`. Null, empty, strings, duplicates and unknown values fail closed at startup, **before token resolution**.
- Eligibility, availability, selection, retries, routing diagnostics and unavailable/unsupported status filter model **and actual upstream API before trial admission**. Inspection does not claim a HALF_OPEN trial.
- Existing `models: []` wildcard behavior is unchanged. API eligibility is separate; a catalog name alone does not certify Chat support.
- Existing configured Chat→Responses adapters select using actual upstream `responses`. Their inherited buffered payload transformations are unchanged. No Astra membership, identity headers, model swapping or context/tool/history dropping is introduced.
- `least_requests` retains active/weight then total/weight tie-breaking. Four equal-weight model-matching Responses-only accounts distribute evenly in synthetic sequential/held-concurrent tests and are never selected for actual Chat.
- The startup loader rejects duplicate resolved credential strings, including several new entries resolving through the same legacy cache. It never logs credential values. This intentionally rejects a previously possible duplicate-credential configuration; distinct strings are not by themselves proof of distinct account identity. External account validation/projection remains the installer's responsibility. Missing credentials retain the old skipped-endpoint behavior; mandatory Secret projection/init validation is a separate deployment guard.

No new accounts, Secret/config append, global settings, runtime deployment, model catalog, identity headers or adapter model sets are installed by this source change.

## Diagnostics and regression compatibility

Copilot `input_bytes` is now `len(req.content)` on each actually built HTTPX request, not a second JSON character-count estimate. Responses `input[].content[].type=input_image` is detected as well as Chat `image_url`; the existing vision indicator is reused without inventing identity headers. Terminal logs distinguish completed/failed/incomplete/error/truncated/local_resource_limit/local_observer_error. Additive decoded_bytes, decoded_deliveries, frames, pending_eof_bytes and peak_pending_bytes fields are scalar only. The legacy chunks field now counts offered complete frames; decoded_deliveries explicitly names content-decoder deliveries (bounded zlib deliveries on gzip), not TCP chunks or tokens. These counters do not establish the cause of any historical incident.

The prior resilience policy changes remain: no ambiguous/5xx POST replay, retained cumulative errors, bounded HALF_OPEN and best-effort local usage persistence. The literal old-policy suite is retained as a comparison, not silently declared passing. Existing success fixtures now provide a real JSON discriminator, completed ID and valid usage (including total_tokens); Anthropic ownership fixtures use their native terminal. No ownership/cancellation assertions are removed; successful ownership completion additionally checks the completed-admission counter (the fixture stubs its usage writer). The named-event method keeps its identity but no longer claims that WHATWG makes header-only JSON a client-valid Responses completion. New tests explicitly reject those old malformed fixtures. Fake route/admission objects implement the new API-aware call signatures/availability surface.

## References

- WHATWG SSE parsing/interpretation: https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream
- Codex pinned `5ecb3afd1bf405149e2159bfda50093b0c1b5fab`, event/usage/terminal handling: https://github.com/openai/codex/blob/5ecb3afd1bf405149e2159bfda50093b0c1b5fab/codex-rs/codex-api/src/sse/responses.rs (112–181, 417–507, 575–677).
- Same pinned `ResponseUsageMetadata`: https://github.com/openai/codex/blob/5ecb3afd1bf405149e2159bfda50093b0c1b5fab/codex-rs/protocol/src/response_usage.rs.

These references are not proof of any installed Mac client version, actual upstream framing of an old request, client/account entitlement parity, model context limit, or original root cause. Local tests do not replace the parent’s independent review and separately authorized combined deployment.
