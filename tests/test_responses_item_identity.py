"""Synthetic, offline GHCP Responses message identity regressions."""

import json

import pytest

from responses_item_identity import IdentityNormalizationError, ResponsesItemIdentity


def frame(payload, newline=b"\n"):
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + newline * 2


def payload(wire):
    data = []
    for line in wire.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    return json.loads("\n".join(data))


def added(index=0, ident="A", kind="message", **extra):
    return frame({"type": "response.output_item.added", "output_index": index,
                  "item": {"type": kind, "id": ident, **extra}})


def reference(ident="B", index=0, event="response.output_text.delta", **extra):
    return frame({"type": event, "output_index": index, "item_id": ident,
                  "content_index": 0, **extra})


def terminal(items, event="response.completed", **extra):
    return frame({"type": event, "response": {"id": "response-stays", "output": items, **extra}})


def assert_failure(normalizer, wire, reason="identity_conflict"):
    with pytest.raises(IdentityNormalizationError) as caught:
        normalizer.transform(wire)
    assert caught.value.reason == reason
    assert normalizer.summary()["failure_reason"] == reason
    assert normalizer.summary()["mapping_bytes"] == 0
    assert "sensitive" not in str(caught.value)


def test_drift_lifecycle_retains_one_item_without_changing_text_or_usage():
    normalizer = ResponsesItemIdentity()
    start = added(phase="commentary", role="assistant", content=[])
    assert normalizer.transform(start) is start
    same = reference("A", delta="same words A B C")
    assert normalizer.transform(same) is same
    delta = payload(normalizer.transform(reference(delta="same words A B C", sequence_number=3)))
    assert delta["item_id"] == "A"
    assert delta["delta"] == "same words A B C"
    assert delta["sequence_number"] == 3
    done = {"type": "response.output_item.done", "output_index": 0,
            "item": {"type": "message", "id": "B", "phase": "commentary",
                     "role": "assistant", "content": [{"type": "output_text", "text": "same words A B C"}]}}
    end_item = payload(normalizer.transform(frame(done)))["item"]
    assert end_item == {**done["item"], "id": "A"}
    usage = {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    end = payload(normalizer.transform(terminal([{**done["item"], "id": "C"}], status="completed", usage=usage)))
    assert end["response"]["id"] == "response-stays"
    assert end["response"]["output"] == [end_item]
    assert end["response"]["usage"] == usage
    # Mirrors clients replacing an existing item by ID, without text deduplication.
    rendered = {payload(start)["item"]["id"]: payload(start)["item"]}
    rendered[end_item["id"]] = end_item
    assert len(rendered) == 1
    summary = normalizer.summary()
    assert summary["drift_items"] == 1
    assert summary["rewritten_frames"] == 3
    assert summary["aliases"] == 3
    normalizer.close()
    assert normalizer.summary()["mapping_bytes"] == 0
    assert normalizer.summary()["peak_mapping_bytes"] == summary["peak_mapping_bytes"]
    assert normalizer.summary()["mapped_items"] == 1


@pytest.mark.parametrize("event", [
    "response.content_part.added", "response.content_part.done",
    "response.output_text.delta", "response.output_text.done",
    "response.refusal.delta", "response.refusal.done",
    "response.output_text.annotation.added",
])
def test_supported_message_references_preserve_all_other_fields(event):
    normalizer = ResponsesItemIdentity()
    normalizer.transform(added())
    original = payload(reference(event=event, text="B", delta="B", content_index=4,
                                 annotation={"type": "url_citation", "title": "B", "url": "https://example.invalid/B"},
                                 logprobs=[{"token": "B", "logprob": -0.25}]))
    actual = payload(normalizer.transform(frame(original)))
    assert actual == {**original, "item_id": "A"}


def test_same_text_messages_and_interleaved_nonmessage_items_stay_distinct():
    normalizer = ResponsesItemIdentity()
    items = [
        {"type": "message", "id": "A", "phase": "commentary", "content": [{"text": "same"}]},
        {"type": "reasoning", "id": "R", "encrypted_content": "B-sensitive"},
        {"type": "function_call", "id": "T", "call_id": "B", "name": "tool", "arguments": '{"id":"B"}'},
        {"type": "message", "id": "D", "phase": "final_answer", "content": [{"text": "same"}]},
    ]
    for index, item in enumerate(items):
        raw = frame({"type": "response.output_item.added", "output_index": index, "item": item})
        assert normalizer.transform(raw) is raw
    for index in (1, 2):
        raw = frame({"type": "response.output_item.done", "output_index": index, "item": items[index]})
        assert normalizer.transform(raw) is raw
    final = payload(normalizer.transform(terminal([
        {**items[0], "id": "B"}, items[1], items[2], {**items[3], "id": "E"},
    ])))
    assert final["response"]["output"] == items
    assert normalizer.summary()["drift_items"] == 2


@pytest.mark.parametrize("newline", [b"\n", b"\r", b"\r\n"])
@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
def test_rewrite_preserves_sse_control_lines_and_multiline_data(newline, bom):
    normalizer = ResponsesItemIdentity()
    # BOM is meaningful on the first upstream frame only.
    start = bom + added()
    normalizer.transform(start)
    lines = [b": heartbeat B", b"event: transport-type", b"data: {\"type\":\"response.output_text.delta\",",
             b"id: B", b"retry: 1234", b"data: \"output_index\":0,\"item_id\":\"B\",\"delta\":\"B\"}",
             b"custom: B", b"", b""]
    wire = newline.join(lines)
    changed = normalizer.transform(wire)
    assert payload(changed)["item_id"] == "A"
    assert payload(changed)["delta"] == "B"
    controls = [b": heartbeat B", b"event: transport-type", b"id: B", b"retry: 1234", b"custom: B"]
    assert [line for line in changed.split(newline) if line in controls] == controls
    assert changed.endswith(newline * 2)
    assert b"id: B" + newline in changed


def test_unchanged_frames_are_exact_bytes_and_instances_are_isolated():
    left, right = ResponsesItemIdentity(), ResponsesItemIdentity()
    left.transform(added(ident="left"))
    right.transform(added(ident="right"))
    assert payload(left.transform(reference("alias")))["item_id"] == "left"
    assert payload(right.transform(reference("alias")))["item_id"] == "right"
    for wire in [b": hello\r\r", b"data: [DONE]\n\n", b'data: {"type":"unrelated","text":"alias"}\n\n']:
        assert left.transform(wire) is wire


@pytest.mark.parametrize("index", [None, -1, True, 1.5, "0"])
def test_invalid_unbound_index_and_missing_added_are_unsupported_passthrough(index):
    normalizer = ResponsesItemIdentity()
    wire = reference(index=index)
    assert normalizer.transform(wire) is wire
    assert normalizer.summary()["unsupported_events"] == 1
    assert normalizer.summary()["failure_reason"] is None


def test_missing_added_is_not_fabricated():
    normalizer = ResponsesItemIdentity()
    for wire in [reference(), frame({"type": "response.output_item.done", "output_index": 0,
                                    "item": {"type": "message", "id": "B"}}),
                 terminal([{"type": "message", "id": "B"}])]:
        assert normalizer.transform(wire) is wire
    assert normalizer.summary()["mapped_items"] == 0
    assert normalizer.summary()["unsupported_reason"] == "missing_added"


@pytest.mark.parametrize("wire", [
    added(),
    added(index=1, ident="A"),
    frame({"type": "response.output_item.done", "output_index": 0,
           "item": {"type": "reasoning", "id": "B"}}),
    reference(index=True),
    reference(ident=""),
])
def test_conflicting_id_type_or_bound_identity_fails(wire):
    normalizer = ResponsesItemIdentity()
    normalizer.transform(added())
    assert_failure(normalizer, wire)


def test_known_alias_cannot_cross_indices():
    normalizer = ResponsesItemIdentity()
    normalizer.transform(added())
    normalizer.transform(added(index=1, ident="X"))
    normalizer.transform(reference("B"))
    assert_failure(normalizer, reference("B", index=1))


@pytest.mark.parametrize("event", ["response.completed", "response.failed", "response.incomplete"])
def test_all_snapshot_terminal_types_preserve_outcome(event):
    normalizer = ResponsesItemIdentity()
    normalizer.transform(added())
    wire = terminal([{"type": "message", "id": "B"}], event, status="failed",
                    error={"code": "upstream_error", "message": "B"})
    actual = payload(normalizer.transform(wire))
    assert actual["type"] == event
    assert actual["response"]["status"] == "failed"
    assert actual["response"]["error"] == {"code": "upstream_error", "message": "B"}
    assert actual["response"]["output"][0]["id"] == "A"


@pytest.mark.parametrize("items", [
    [{"type": "message", "id": "X"}, {"type": "message", "id": "A"}],
    [{"type": "message", "id": "B"}],
    [{"type": "message", "id": "B"}, {"type": "reasoning", "id": "X"}],
])
def test_reordered_partial_or_type_conflicting_snapshots_fail_after_rewrite(items):
    normalizer = ResponsesItemIdentity()
    normalizer.transform(added())
    normalizer.transform(added(index=1, ident="X"))
    normalizer.transform(reference())
    assert_failure(normalizer, terminal(items))


def test_missing_snapshot_keeps_existing_shape_but_cannot_hide_rewrite_dependency():
    raw = frame({"type": "response.completed", "response": {"id": "response-stays"}})
    untouched = ResponsesItemIdentity()
    untouched.transform(added())
    assert untouched.transform(raw) is raw
    rewritten = ResponsesItemIdentity()
    rewritten.transform(added())
    rewritten.transform(reference())
    assert_failure(rewritten, raw)


def test_unknown_event_is_unchanged_unless_reference_needs_rewrite():
    normalizer = ResponsesItemIdentity()
    normalizer.transform(added())
    normalizer.transform(reference())
    stable = reference("A", event="response.future.delta", delta="B")
    assert normalizer.transform(stable) is stable
    assert_failure(normalizer, reference("B", event="response.future.delta"))


def test_unknown_event_without_an_identity_field_needs_no_rewrite():
    normalizer = ResponsesItemIdentity()
    normalizer.transform(added())
    normalizer.transform(reference())
    wire = frame({"type": "response.future.notice", "output_index": 0, "text": "B"})
    assert normalizer.transform(wire) is wire


def test_unknown_event_known_alias_without_index_cannot_escape_rewrite_check():
    normalizer = ResponsesItemIdentity()
    normalizer.transform(added())
    normalizer.transform(reference())
    assert_failure(normalizer, frame({"type": "response.future.notice", "item_id": "B"}))


@pytest.mark.parametrize("item", [{}, {"id": "other"}, {"id": "A"}])
def test_unknown_event_item_object_cannot_shadow_top_level_alias(item):
    normalizer = ResponsesItemIdentity()
    normalizer.transform(added())
    normalizer.transform(reference())
    assert_failure(normalizer, frame({"type": "response.future.notice", "item_id": "B", "item": item}))


def test_known_alias_cannot_be_assigned_to_a_new_index():
    normalizer = ResponsesItemIdentity()
    normalizer.transform(added())
    normalizer.transform(reference())
    assert_failure(normalizer, added(index=1, ident="B"))


def test_unsafe_added_cannot_establish_a_canonical_id():
    normalizer = ResponsesItemIdentity()
    wire = b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"A","id":"B"}}\n\n'
    assert normalizer.transform(wire) is wire
    assert normalizer.summary()["mapped_items"] == 0
    assert normalizer.transform(reference()) == reference()


def test_invalid_utf8_is_not_repaired_when_an_identity_rewrite_is_needed():
    normalizer = ResponsesItemIdentity()
    normalizer.transform(added())
    wire = b'data: {"type":"response.output_text.delta","output_index":0,"item_id":"B","delta":"\xff"}\n\n'
    assert_failure(normalizer, wire, "identity_transform_error")


def test_valid_unicode_surrogate_pair_and_finite_numbers_survive_rewrite():
    normalizer = ResponsesItemIdentity()
    normalizer.transform(added(ident="汉😀"))
    wire = b'data: {"type":"response.output_text.delta","output_index":0,"item_id":"B","delta":"\\ud83d\\ude00","value":1e308}\n\n'
    result = payload(normalizer.transform(wire))
    assert result["item_id"] == "汉😀"
    assert result["delta"] == "😀"
    assert result["value"] == 1e308


@pytest.mark.parametrize("raw_json", [
    '{"type":"response.output_text.delta","type":"response.output_text.delta","output_index":0,"item_id":"B"}',
    '{"type":"response.output_text.delta","output_index":1,"output_index":0,"item_id":"B"}',
    '{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"sensitive","id":"B"}}',
    '{"type":"response.output_text.delta","output_index":0,"item_id":"B","delta":"\\ud800"}',
    '{"type":"response.output_text.delta","output_index":0,"item_id":"B","other":NaN}',
    '{"type":"response.output_text.delta","output_index":0,"item_id":"B","other":1e400}',
    '{"type":"response.output_text.delta","output_index":0,"item_id":"B","other":{"value":"\\udfff","value":1}}',
    '{"type":"response.output_text.delta","output_index":0,"item_id":"B","other":{"x":1,"x":2}}',
])
def test_unsafe_json_is_never_laundered_by_rewriting(raw_json):
    normalizer = ResponsesItemIdentity()
    normalizer.transform(added())
    assert_failure(normalizer, b"data: " + raw_json.encode() + b"\n\n", "identity_transform_error")


def test_invalid_json_without_rewrite_keeps_previous_wire_contract():
    normalizer = ResponsesItemIdentity()
    wire = b'data: {"type":"response.completed","response":{"id":"r","other":NaN}}\n\n'
    assert normalizer.transform(wire) is wire
    assert normalizer.summary()["failure_reason"] is None


@pytest.mark.parametrize("kwargs,first,next_frame", [
    ({"max_items": 1}, added(), added(index=1, ident="B")),
    ({"max_aliases": 1}, added(), reference()),
    ({"max_id_bytes": 2}, added(), reference("汉")),
    ({"max_mapping_bytes": 9}, added(), reference("BCD")),
    ({"max_frame_bytes": 130}, added(ident="A" * 100), reference("B")),
])
def test_mapping_and_rebuilt_frame_limits_fail_without_eviction(kwargs, first, next_frame):
    normalizer = ResponsesItemIdentity(**kwargs)
    normalizer.transform(first)
    assert_failure(normalizer, next_frame, "identity_mapping_limit")


def test_observe_returns_original_bytes_and_stops_safely_on_conflict():
    normalizer = ResponsesItemIdentity(mode="observe")
    for wire in [added(), reference(), terminal([{"type": "message", "id": "C"}])]:
        assert normalizer.transform(wire) is wire
    assert normalizer.summary()["drift_items"] == 1
    assert normalizer.summary()["rewritten_frames"] == 0
    assert normalizer.summary()["would_rewrite_frames"] == 2
    collision = added()
    assert normalizer.transform(collision) is collision
    before = normalizer.summary()
    assert before["failure_reason"] == "identity_conflict"
    assert before["mapping_bytes"] == 0
    assert normalizer.transform(added(index=2, ident="new")) == added(index=2, ident="new")
    assert normalizer.summary() == before


def test_stabilize_cannot_resume_passthrough_after_failure():
    normalizer = ResponsesItemIdentity()
    normalizer.transform(added())
    normalizer.transform(reference())
    assert_failure(normalizer, added())
    assert_failure(normalizer, reference("C"))


def test_observe_unexpected_transform_exception_never_changes_serving(monkeypatch):
    normalizer = ResponsesItemIdentity(mode="observe")
    monkeypatch.setattr(normalizer, "_transform", lambda wire: 1 / 0)
    wire = added()
    assert normalizer.transform(wire) is wire
    assert normalizer.summary()["failure_reason"] == "identity_transform_error"
    normalizer.close()
    normalizer.close()


def test_summary_never_contains_ids_or_content_and_close_releases_state():
    normalizer = ResponsesItemIdentity()
    normalizer.transform(added(ident="sensitive-canonical"))
    normalizer.transform(reference("sensitive-alias", delta="sensitive-answer"))
    summary = normalizer.summary()
    assert "sensitive" not in json.dumps(summary)
    assert summary["mapping_bytes"] > 0
    normalizer.close()
    assert normalizer.summary()["mapping_bytes"] == 0
    assert normalizer.summary()["aliases"] == summary["aliases"]
    assert normalizer.summary()["peak_mapping_bytes"] == summary["peak_mapping_bytes"]


def test_off_is_noop_even_for_conflicts():
    normalizer = ResponsesItemIdentity(mode="off")
    for wire in [added(), added(), reference()]:
        assert normalizer.transform(wire) is wire
    assert normalizer.summary()["mapped_items"] == 0
