# Independent review regression fixtures

These five Python files are byte-identical copies of the 2026-09-07 independent
review or its inherited real-socket harness. Original assertions remain intact.
`test_real_lock.py` and `test_adversarial.py` retain their historical standalone
hash checks for blocked commit e3902379b87b1a083043d162c382a5e28c473c95. Do not
change those checks to imply that the original negative-control image passed.

Run the successor regressions through `tests/test_revision2_review.py`, which
imports the original TestCase classes without executing their standalone guards.
It restores import-time logging changes so existing logging-safety assertions
remain effective. The release runner independently verifies `/app/main.py` and
the exact immutable image; only test files are mounted, never application code.

Sources:
- `/openclaw/tmp/lb-independent-20260907/test_real_lock.py`
- `/openclaw/tmp/lb-independent-20260907/test_adversarial.py`
- `/openclaw/tmp/lb-independent-20260907/test_wire_edges.py`
- `/tmp/lb-integrated-20260907/protocol_socket_all.py`
- `/tmp/lb-integrated-20260907/test_protocol_all.py`

Additional committed checks:
- `test_revision2_buffered_auth.py`: actual token-lock/ordinary Sol adapter 499,
  joined repeated cancellation, same-lease CLOSED/HALF_OPEN 401 repair on both
  APIs, failed refresh/second 401, no ambiguous 503/read replay. Resource and
  attribution assertions precede client teardown.
- `test_revision2_json.py`: both Responses providers over real upstream and
  downstream sockets, invalid-scalar rejection, all 12 root typed duplicate
  fields, finite Unicode/nested Value/typed-usage controls, exact wire prefix.

The JSON duplicate policy follows the pinned Codex ResponsesStreamEvent:
`codex-rs/codex-api/src/sse/responses.rs` at
`5ecb3afd1bf405149e2159bfda50093b0c1b5fab`. Its initial typed root rejects duplicate
fields; nested `response` is first decoded as serde_json::Value and preserves
last-wins semantics before typed completed/usage deserialization. The repair
changes observation, not upstream wire serialization. This is component
compatibility evidence, not a full Codex binary or production certification.
