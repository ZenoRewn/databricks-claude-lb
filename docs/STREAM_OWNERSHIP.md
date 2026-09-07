# Streaming response ownership and pool timeout semantics

## Ownership invariant

`client.send(..., stream=True)` returns an open HTTPX response. Before response
headers are handed to the proxy generator, `_await_with_heartbeat` owns the send
task and its result. If downstream stops consuming while a heartbeat is yielded,
headers may finish in that task. Finalization must join the task and close any
untransferred HTTPX response, including a task that returns a response while
handling cancellation. An unfinished task is cancelled, not left in the pool.

All three providers explicitly finalize the header iterator using `aclosing`.
The helper marks ownership transferred immediately before yielding the result;
each caller assigns it without an intervening await. After transfer, only the
caller closes it. The helper must not close a legitimate long-lived response.

The send owner installs `_OwnedResponseStream` on the public HTTPX `stream`
attribute immediately after headers return, before any await or ownership
transfer. Its first `aclose()` retains a single task closing the original bound
stream; every later stream close joins the same task, including its failure.
This covers HTTPX's implicit EOF close from `aiter_bytes()` and error-body
`aread()`. HTTPX's early `is_closed` flag is not a cleanup-completion signal;
explicit finalization also joins the adapter even when `Response.aclose()` skips.

httpcore additionally self-closes *inside* stream iteration when a read fails or
is cancelled. A close-only outer adapter cannot protect that internal operation.
Each underlying `anext()` therefore runs in a retained worker with an AnyIO
cancel scope. Raw caller cancellation cancels that scope (not the worker task),
then joins the worker before re-raising. httpcore's internal close shields honor
scope cancellation and can finish releasing the connection. Normal silent reads
remain cancellable; this does not shield the entire pump, a queue wait, or an
unlimited read against cancellation. No cancel scope spans a yield. The extra
per-read task/shield scheduling has not been benchmarked at production load.

Cleanup uses a joined task shielded against repeated `asyncio.Task.cancel()` and
an AnyIO shielded scope against level cancellation. No cancel scope spans a
streaming yield. Body-pump cancellation and response close are shielded together,
including when cancellation is raised inside `__anext__`, not just at an ASGI
send. Iterator close and business lease release are also joined before returning.
Databricks now uses the same idempotent lease/ASGI-finalization pattern as Azure
and Copilot; send failure must not leave its iterator or active count behind.

No pool size, keepalive, HTTP/2, read timeout, token exchange, or production
configuration tuning is part of this change. There is no forced cleanup timeout:
a non-cooperative custom transport could delay shutdown; the tested HTTPX/httpcore
transport cooperates. Direct raw cancellation of a private read/close worker is
not a supported cancellation path and can defeat httpcore's shields; callers
cancel the pump/iterator, which retains and joins those workers. No code resets
closed flags or removes private pool entries. Process kill cannot run Python cleanup.

## Diagnostic/API compatibility

- `PoolTimeout` means pool assignment/acquisition timed out. It does **not** prove
  a TCP/TLS failure or that a new socket was being established. A later TCP probe
  cannot establish the earlier cause.
- Streaming error `error.code` changes from `upstream_connect_stalled` to
  **`pool_acquire_timeout`**. Structured `classification` and stream-end `outcome`
  use the neutral name too. Consumers filtering the old literal must migrate.
  Responses API still emits `response.failed`; chat still emits its error frame
  and `[DONE]`. Request ID prefix/metadata/headers and HTTP statuses are unchanged.
  Buffered requests retain their existing generic HTTP 503 error envelope.
- A single-endpoint pool timeout now always fails fast. Previously this was gated
  by the misleading business-count classification; the old high-business-count
  branch could retry even the sole endpoint. Multi-endpoint retry limits/backoff
  and the no-replay-after-model-content guard remain unchanged.
- Existing metric keys remain. `pool_timeout_total` is the authoritative count.
  `pool_timeout_saturated_total` now means the HTTPX snapshot was observed full,
  **not** a proven cause. `pool_timeout_upstream_stall_total` is deprecated and
  receives no new increments; it remains present for scraper compatibility.
- `requests_waiting` counts `is_queued()` pool entries, not all assigned streamed
  responses. This is best-effort existing private diagnostic inspection, not a
  new public per-request metrics interface.
- Disconnect chunk/event fields are updated before yielding body bytes. These
  count chunks **offered** downstream, not confirmed network delivery. Heartbeats
  no longer update the timestamp labelled upstream activity. No prompts, headers,
  tokens, or response bodies are added to diagnostics.

## Regression tests

`python -m unittest discover -s tests -v` runs without pytest. Alternatively:

```sh
python -m pytest -q tests
```

`tests/test_stream_response_ownership.py` uses an event-controlled loopback
HTTP/1.1 upstream with real HTTPX/httpcore. It checks pool request removal and
active connection release **before** closing the client, plus server-observed EOF
for abandoned responses, exactly-once business lease end, and Copilot registry
removal. Fully consumed responses may legitimately retain an idle reusable socket.
Coverage includes paused heartbeats, pre-header/body cancellation, ASGI 2.4 send
errors, ASGI 2.3 disconnect cancellation, repeated cancellation during cleanup,
finish/cancel races, multiple origins, repeated disconnect/recovery cycles,
normal silent streams, result transfer, and neutral diagnostics.

`tests/test_stream_transport_close.py` additionally exercises implicit EOF close,
error-body EOF close, and internal close after truncated or cancelled reads for
all providers. It holds the real httpcore state lock (and separately tests its
uncontended checkpoint), cancels the original pump/consumer repeatedly, and
releases the lock independently. Assertions require joined cancellation,
exactly-once transport/business finalization, zero pool requests/active
connections, and a completed request to a second origin before client teardown.
Concurrent close callers, cancelled AnyIO scopes, retained close errors and the
send-finish/cancel race are covered as well.

The ASGI tests call the real Starlette response with controlled `send`/`receive`;
they are not a claim to test the complete external ingress/client chain.
Accelerated heartbeat tests exercise silent-stream logic, not hours of load.

## Lifecycle references

- [HTTPX manual async streaming](https://www.python-httpx.org/async/#streaming-responses):
  the caller must eventually call `Response.aclose()`.
- [httpcore 1.0.9 pool](https://github.com/encode/httpcore/blob/1.0.9/httpcore/_async/connection_pool.py):
  assigned requests stay in `_requests` until `PoolByteStream.aclose()`; queued
  entries expose `is_queued()`. TCP/TLS work follows pool assignment.
- [ASGI HTTP disconnect semantics](https://asgi.readthedocs.io/en/stable/specs/www.html):
  spec 2.4 send on closed connection should raise an `OSError`; `http.disconnect`
  is a separate receive event and may arrive after send failure.
- [AnyIO cancellation/finalization](https://anyio.readthedocs.io/en/stable/cancellation.html):
  protect awaited cleanup in a shielded scope and preserve cancellation.

## Release caution

Tests establish a repaired ownership boundary, not that every historical stall
had this cause. In particular, the deleted pre-restart logs for incident request
`b8e8a1e0ba91d5963613eacf55ff88ab` do not permit conclusive attribution.

Review the implementation commit before building/publishing a production image.
Use an immutable image digest for rollout and preserve the live Deployment, not
the example manifest (its namespace/container/strategy may differ). The current
single-replica rollout can interrupt active long streams; do not use restart or
reset-pool as a diagnostic. Verify real terminal events and post-disconnect
recovery, not only readiness or HTTP 200. Retain the previous image digest for
rollback. Full rollout/rollback commands and tested image provenance are in the
implementation-stage handoff report; no deployment is performed by these tests.
