# Streaming protocol, retained-byte policy and API eligibility

Author: Zeno Ren

This contract extends [RESILIENCE.md](RESILIENCE.md). It does not diagnose any original client incident, set model token/context limits, authorize account enrollment, or change deployment probes. Existing ownership/close adapters and token-exchange isolation are retained.

## Wire contract

All three native streaming providers use one complete-frame framer. It recognizes WHATWG CRLF, CR and LF, preserves complete wire frames, parses UTF-8 with replacement semantics and ignores one initial BOM. It retains only the unfinished frame, not the entire response. A final CR is a valid line ending; an unfinished event at EOF is discarded without appending a delimiter. Independent local errors are emitted only at known frame boundaries. No completed/Anthropic message_stop event is manufactured to hide a missing terminal. Empty and comment-only EOF are not success.

A complete frame is offered promptly once its blank-line delimiter arrives. CR at a chunk boundary waits for the following byte or EOF so a split CRLF remains intact. This is intentionally different from immediate arbitrary HTTP chunk forwarding: clients receive no partial event. Heartbeats remain independent SSE comments. If a local heartbeat precedes the first upstream BOM, that BOM is removed from the first forwarded frame because it would no longer be at stream offset zero. Complete frames otherwise remain byte-identical, except for the explicitly enabled Copilot message-ID normalization below. No additional buffering of completed events is introduced beyond the existing bounded pump and downstream backpressure.

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

## Opt-in Copilot Responses message identity

`responses_item_identity.py` provides an attempt-local compatibility transform for native Copilot/GHCP Responses streams. Its purpose is to keep one output item's ID consistent when upstream IDs drift between added, delta, done and terminal snapshots. It never compares or deduplicates text: two messages containing the same words remain two messages. **Every output item type is covered** — message, reasoning and function_call alike. The first implementation stabilized messages only, and on 2026-10-27 `gpt-6.1-sol` was measured drifting its reasoning ID across added, done and the terminal snapshot while its message ID stayed put, which duplicates a thinking block through the same client mechanism the message fix addresses. Only the top-level `id` is ever rewritten; item content, `call_id`, arguments and encrypted state are preserved, and rewriting is safe for the same reason it was for messages — the downstream ID never travels back upstream, because input-ID stripping removes the top-level `id` from every input item. Foundry Responses, Messages, Chat, non-streaming Responses and the buffered Chat→Responses adapter retain their existing behavior.

These environment variables are read at startup/import; they are not YAML fields or hot switches:

| Setting | Default | Meaning |
|---|---|---|
| `COPILOT_RESPONSES_ITEM_ID_MODE` | `off` | `off`, `observe` or `stabilize` |
| `COPILOT_RESPONSES_ITEM_ID_MODELS` | empty | Comma-separated exact model allowlist; empty enables no models, with no wildcard or prefix matching |
| `COPILOT_STRIP_INPUT_ITEM_IDS` | `true` | Existing next-turn input-ID stripping; must remain enabled when mode is `stabilize` |

Mode is trimmed and lowercased; CSV model entries are trimmed and deduplicated, with case-sensitive exact matching. Unknown modes and `stabilize` with input-ID stripping disabled reject startup, even if the model allowlist is empty. `off` and nonmatching models retain the original path. `observe` tracks ID drift and mapping capacity while returning the original frame bytes; an observer failure stops observation for that attempt and does not change output, retry, terminal or account-health behavior. `stabilize` applies only the supported references below. Mode and model selection are frozen for the request, and each upstream attempt gets fresh mappings, including the existing authorized 401-repair attempt. No mapping crosses requests or persists in a database/cache.

The first valid `response.output_item.added` binds a nonnegative integer `output_index`, item type and nonempty canonical `item.id`. Booleans are not valid indices. A later ID for the same item is an alias of the first ID. A missing added event is not synthesized: that unbound event is forwarded unchanged and recorded as `missing_added`, so that identity is not claimed as repaired. The transform runs after complete-frame parsing and before downstream semantic observation and successful settlement; it does not buffer a full answer or generate extra terminal events.

| Supported event | Field that may change for an established output item |
|---|---|
| `response.output_item.added`, `response.output_item.done` | `item.id` |
| `response.content_part.added`, `response.content_part.done` | Top-level `item_id` |
| `response.output_text.delta`, `response.output_text.done` | Top-level `item_id` |
| `response.refusal.delta`, `response.refusal.done` | Top-level `item_id` |
| `response.output_text.annotation.added` | Top-level `item_id` |
| `response.completed`, `response.failed`, `response.incomplete` | Established message IDs in `response.output`, only when the snapshot can be safely correlated |

The official annotation event includes `output_index`, `content_index` and `annotation_index`; its nested annotation is preserved. Official text delta/done events carry a `logprobs` array, which is preserved with their text; no speculative standalone logprob event is accepted by name. Top-level `response.id`, text/refusal content, phase, role, content/sequence indices, tool data, usage, status and error details are never rewritten. There is no recursive ID substitution.

Unchanged frames keep their original bytes. A changed frame replaces only its JSON data payload, preserving the relative order and bytes of SSE `event`, `id`, `retry`, comment and other non-data lines. Multiple data lines are assembled using SSE rules and rebuilt as valid data; SSE `id:` is never a JSON item ID. Invalid JSON, duplicate identity keys, nonfinite numbers and invalid Unicode are not repaired by serialization. Unknown events without an affected identity retain the existing forwarding contract.

Conflicting types/indices, duplicate added events, ambiguous terminal snapshots or unsupported events that require rewriting an already normalized identity cannot silently fall back to mixed IDs. A snapshot must match the complete contiguous set of observed indices, list length and item types; a known ID at another index is a conflict. After any drift, an absent snapshot is also a conflict. Stabilize reports a bounded compatibility error and ends that attempt; it never replays the inference POST. `identity_conflict`, `identity_mapping_limit` and `identity_transform_error` distinguish protocol conflict, mapping capacity and local transformation failure. Local transform/resource failures do not penalize the upstream account. If upstream usage was already available, the existing idempotent usage path preserves that fact even when local delivery fails; this is not a completed downstream generation.

When every snapshot ID is new and several slots have the same item type, correlation relies on the Responses schema's ordered `output` array contract. An arbitrary upstream reorder of those unknown IDs cannot be independently detected from identity/type information alone. The checks above detect observable alias/type conflicts, not every possible upstream ordering violation; real GHCP snapshot ordering remains part of live acceptance.

Per-attempt mapping limits are 4,096 output items, 131,072 ID aliases (including canonical IDs), 8 KiB per ID and 2 MiB of retained mapping bytes. GHCP can issue a fresh message ID on **every delta** (2026-10-21 production: one ~5,000-delta answer carried ~5,000 distinct ~416-byte IDs and exhausted the former raw-string budget, ending the attempt with `identity_mapping_limit`). Aliases are therefore retained as 16-byte BLAKE2b digests keyed with a per-attempt random key, so retained bytes grow with delta count rather than upstream ID length; only canonical IDs (written back downstream) and per-index item types are kept raw. Conflict detection is unchanged: a digest hit at a different index still fails. A digest collision could only produce a false `identity_conflict` (failing closed), never a silent mix of IDs. Active mappings are not evicted to continue a stream. These are local engineering limits, not provider API limits or a total RSS guarantee. A rewritten frame must fit `PER_PENDING_EVENT`; its bytes also reserve the existing retained-stream budget while offered downstream. Completion, error and cancellation release mappings and byte reservations.

Diagnostics record bounded counts and reasons without prompts, answer text, raw item IDs, encrypted state or credentials. Drift counts are per output item; rewritten frames are counted per frame. `copilot_response_identity_{requests,drift_items,rewritten_frames,rewritten_requests,unsupported_requests}_total` uses only `mode` labels (`observe` or `stabilize`). `copilot_response_identity_failures_total` adds the finite `reason` label above. `copilot_response_identity_mapping_bytes` measures current mapping strings across active attempts; `copilot_response_identity_peak_mapping_bytes{mode}` is the maximum per-attempt mapping size observed among finished attempts, not RSS.

One safe request summary uses `kind=copilot_response_identity`, mode/model/request context, drift items, rewritten frames, mapped items, aliases, peak mapping bytes and unsupported events/reasons. Request summaries may correlate with the LB request ID, but IDs, hashes and output indices are not metric labels. Observation or rewrite counts alone do not prove client acceptance: full terminal semantics, single desktop rendering and multi-turn/tool behavior require separate verification.

To disable normalization, set mode to `off` and use the controlled restart/rollout procedure; an in-flight request keeps its frozen mode until it ends. This returns new requests to original forwarding and can re-expose the original duplicate-display symptom. Do not discard an active mapping midway through a stream, replay a request, or roll back usage data. Clients using `item_reference` or item retrieval need separate compatibility validation; the current scope assumes full-history input replay with the existing input-ID stripping.

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
- Configured Chat→Responses adapters select using actual upstream `responses`. The 2026-09-30 [adapter contract](OBSERVABILITY_AND_CONTEXT.md) preserves schema, tool-call/result IDs, roles, refusal and incomplete outcomes while declaring `X-LB-Stream-Mode: buffered-adapter`. It does not add native upstream streaming, Astra membership, identity headers or model swapping.
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
- Official OpenAI Responses event schemas, checked 2026-10-20: [text delta](https://developers.openai.com/api/reference/resources/responses/streaming-events#response.output_text.delta), [text done](https://developers.openai.com/api/reference/resources/responses/streaming-events#response.output_text.done), [annotation added](https://developers.openai.com/api/reference/resources/responses/streaming-events#response.output_text.annotation.added).

These references are not proof of any installed Mac client version, actual upstream framing of an old request, client/account entitlement parity, model context limit, or original root cause. Local tests do not replace the parent’s independent review and separately authorized combined deployment.
