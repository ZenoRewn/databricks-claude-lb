"""Attempt-local GHCP Responses message identity compatibility.

Author: Zeno Ren

This module owns no network, retries, stream buffers, or account health. Call it
only with complete SSE frames, and close it when that upstream attempt ends.
Only a message's first valid output_item.added establishes its downstream ID.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
import re


_ERROR_MESSAGES = {
    "identity_conflict": "Responses output item identity is inconsistent.",
    "identity_mapping_limit": "Responses output item identity capacity exceeded.",
    "identity_transform_error": "Responses output item identity could not be safely transformed.",
}
_REFERENCE_EVENTS = frozenset({
    "response.content_part.added", "response.content_part.done",
    "response.output_text.delta", "response.output_text.done",
    "response.refusal.delta", "response.refusal.done",
    # Official streaming-event schema: this event has a top-level item_id.
    # Text delta/done carry logprobs themselves; no guessed logprob event.
    "response.output_text.annotation.added",
})
_TERMINAL_EVENTS = frozenset({
    "response.completed", "response.failed", "response.incomplete",
})
_LINE_ENDING = re.compile(rb"\r\n|\r|\n")
_BOM = b"\xef\xbb\xbf"
_DIGEST_BYTES = 16


class IdentityNormalizationError(Exception):
    """Finite, content-free reason suitable for local error handling."""

    def __init__(self, reason):
        self.reason = reason if reason in _ERROR_MESSAGES else "identity_transform_error"
        super().__init__(_ERROR_MESSAGES[self.reason])


def _unsafe_value(value):
    """Check discarded duplicate values too; never turn invalid JSON valid."""
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            try:
                current.encode("utf-8")
            except UnicodeEncodeError:
                return True
        elif isinstance(current, float) and not math.isfinite(current):
            return True
        elif isinstance(current, _JSONObject):
            if current.unsafe:
                return True
        elif isinstance(current, list):
            pending.extend(current)
    return False


class _JSONObject(dict):
    """Retain an unsafe flag even for invalid/duplicate overwritten values."""

    def __init__(self, pairs):
        super().__init__()
        self.unsafe = False
        for key, value in pairs:
            self.unsafe |= key in self or _unsafe_value(key) or _unsafe_value(value)
            self[key] = value


@dataclass
class _Binding:
    kind: str
    canonical: str
    drifted: bool = False


@dataclass
class _Frame:
    lines: list
    data_indices: list
    data: bytes
    first_bom: bool

    @classmethod
    def parse(cls, frame, first_frame):
        lines, indices, data = [], [], []
        start = 0
        for match in _LINE_ENDING.finditer(frame):
            lines.append((frame[start:match.start()], match.group()))
            start = match.end()
        if start < len(frame):
            lines.append((frame[start:], b""))
        first_bom = first_frame and frame.startswith(_BOM)
        for index, (raw, _) in enumerate(lines):
            line = raw[len(_BOM):] if index == 0 and first_bom else raw
            key, _, value = line.partition(b":")
            if key != b"data":
                continue
            indices.append(index)
            data.append(value[1:] if value.startswith(b" ") else value)
        return cls(lines, indices, b"\n".join(data), first_bom)

    def rebuild(self, data):
        first = self.data_indices[0]
        data_indices = set(self.data_indices)
        output = []
        for index, (line, ending) in enumerate(self.lines):
            if index == first:
                prefix = _BOM if index == 0 and self.first_bom else b""
                output.append(prefix + b"data: " + data + ending)
            elif index not in data_indices:
                output.append(line + ending)
        return b"".join(output)


class ResponsesItemIdentity:
    """Bounded per-attempt message ID mappings; observe is serving-neutral.

    aliases counts distinct observed IDs, including canonical IDs. GHCP can
    issue a fresh ID on every delta, so aliases are retained as fixed-size
    keyed digests rather than raw strings: retained bytes grow with the delta
    count, not with upstream ID length. Canonical IDs (which are written back
    downstream) and item types are kept raw. mapping_bytes counts digest,
    canonical-ID and type bytes; Python object overhead is not covered. Counters/peaks survive close(), while current mappings do not.
    The caller accounts rewritten frame bytes in its retained-buffer budget;
    max_frame_bytes also prevents growth beyond its per-event policy.
    """

    def __init__(self, mode="stabilize", *, max_items=4096, max_aliases=131072,
                 max_id_bytes=8192, max_mapping_bytes=2 * 1024 * 1024,
                 max_frame_bytes=8 * 1024 * 1024):
        if mode not in ("off", "observe", "stabilize"):
            raise ValueError("Invalid Responses identity mode")
        limits = (max_items, max_aliases, max_id_bytes, max_mapping_bytes, max_frame_bytes)
        if any(type(limit) is not int or limit <= 0 for limit in limits):
            raise ValueError("Responses identity limits must be positive integers")
        self.mode = mode
        self.max_items = max_items
        self.max_aliases = max_aliases
        self.max_id_bytes = max_id_bytes
        self.max_mapping_bytes = max_mapping_bytes
        self.max_frame_bytes = max_frame_bytes
        self._bindings = {}
        self._aliases = {}
        # Per-attempt key: digests are never comparable across attempts/logs.
        self._digest_key = os.urandom(16)
        self._first_frame = True
        self._stopped = False
        self._mapped_items = 0
        self._tracked_items = 0
        self._alias_count = 0
        self._mapping_bytes = 0
        self._peak_mapping_bytes = 0
        self._drift_items = 0
        self._rewritten_frames = 0
        self._would_rewrite_frames = 0
        self._failure_reason = None
        self._unsupported_reason = None
        self._unsupported_events = 0

    def summary(self):
        return {
            "mode": self.mode,
            "drift_items": self._drift_items,
            "rewritten_frames": self._rewritten_frames,
            "would_rewrite_frames": self._would_rewrite_frames,
            "mapped_items": self._mapped_items,
            "tracked_items": self._tracked_items,
            "aliases": self._alias_count,
            "mapping_bytes": self._mapping_bytes,
            "peak_mapping_bytes": self._peak_mapping_bytes,
            "failure_reason": self._failure_reason,
            "unsupported_reason": self._unsupported_reason,
            "unsupported_events": self._unsupported_events,
        }

    def close(self):
        self._bindings.clear()
        self._aliases.clear()
        self._mapping_bytes = 0
        self._stopped = True

    def transform(self, frame):
        if self._failure_reason is not None and self.mode == "stabilize":
            raise IdentityNormalizationError(self._failure_reason)
        if self._stopped or self.mode == "off":
            return frame
        try:
            return self._transform(frame)
        except Exception as exc:
            reason = exc.reason if isinstance(exc, IdentityNormalizationError) else "identity_transform_error"
            self._failure_reason = reason
            self.close()
            if self.mode == "observe":
                return frame
            # Never expose parser errors, raw payloads, opaque IDs or tracebacks
            # through the public compatibility error message.
            raise IdentityNormalizationError(reason) from None

    def _unsupported(self, reason):
        self._unsupported_events += 1
        if self._unsupported_reason is None:
            self._unsupported_reason = reason

    @staticmethod
    def _index(value):
        return type(value) is int and value >= 0

    @staticmethod
    def _id(value):
        return isinstance(value, str) and bool(value)

    def _fail(self, reason="identity_conflict"):
        raise IdentityNormalizationError(reason)

    def _reserve_strings(self, amount):
        if amount > self.max_mapping_bytes - self._mapping_bytes:
            self._fail("identity_mapping_limit")
        self._mapping_bytes += amount
        self._peak_mapping_bytes = max(self._peak_mapping_bytes, self._mapping_bytes)

    def _key(self, ident):
        return hashlib.blake2b(ident.encode("utf-8", "surrogatepass"), digest_size=_DIGEST_BYTES,
                               key=self._digest_key).digest()

    def _lookup(self, ident):
        return self._aliases.get(self._key(ident)) if isinstance(ident, str) else None

    def _alias(self, ident, index):
        key = self._key(ident)
        existing = self._aliases.get(key)
        if existing is not None:
            if existing != index:
                self._fail()
            return
        if len(ident.encode("utf-8", "surrogatepass")) > self.max_id_bytes or len(self._aliases) >= self.max_aliases:
            self._fail("identity_mapping_limit")
        self._reserve_strings(_DIGEST_BYTES)
        self._aliases[key] = index
        self._alias_count += 1

    def _binding(self, index, ident):
        if not self._index(index) or not self._id(ident):
            # An invalid index cannot safely address any established message.
            if self._mapped_items:
                self._fail()
            self._unsupported("invalid_identity")
            return None
        other_index = self._lookup(ident)
        if other_index is not None and other_index != index:
            self._fail()
        binding = self._bindings.get(index)
        if binding is None:
            self._unsupported("missing_added")
        return binding

    def _canonical(self, binding, index, ident):
        self._alias(ident, index)
        if binding.kind != "message" or ident == binding.canonical:
            return ident
        if not binding.drifted:
            binding.drifted = True
            self._drift_items += 1
        return binding.canonical

    def _needs_rewrite(self, event):
        """Conservative relevance probe before touching unsafe JSON values."""
        index = event.get("output_index")
        binding = self._bindings.get(index) if self._index(index) else None
        item = event.get("item")
        identities = []
        if "item_id" in event:
            identities.append(event["item_id"])
        if isinstance(item, dict) and "id" in item:
            identities.append(item["id"])
        for ident in identities:
            if binding is not None and binding.kind == "message" and ident != binding.canonical:
                return True
            if isinstance(ident, str):
                alias_index = self._lookup(ident)
                if alias_index is not None:
                    known = self._bindings[alias_index]
                    if known.kind == "message" and ident != known.canonical:
                        return True
        response = event.get("response")
        output = response.get("output") if isinstance(response, dict) else None
        if isinstance(output, list):
            for index, item in enumerate(output):
                binding = self._bindings.get(index)
                if binding is not None and binding.kind == "message" and (
                        not isinstance(item, dict) or item.get("id") != binding.canonical):
                    return True
        return False

    def _transform(self, frame):
        parsed_frame = _Frame.parse(frame, self._first_frame)
        self._first_frame = False
        if not parsed_frame.data_indices or parsed_frame.data == b"[DONE]":
            return frame
        invalid_encoding = False
        try:
            text = parsed_frame.data.decode("utf-8")
        except UnicodeDecodeError:
            invalid_encoding = True
            text = parsed_frame.data.decode("utf-8", errors="replace")
        try:
            event = json.loads(text, object_pairs_hook=_JSONObject)
        except (ValueError, RecursionError):
            if self._drift_items:
                self._fail("identity_transform_error")
            self._unsupported("unsafe_json")
            return frame
        if not isinstance(event, dict):
            return frame
        if invalid_encoding or event.unsafe:
            if self._needs_rewrite(event):
                self._fail("identity_transform_error")
            self._unsupported("unsafe_json")
            return frame
        kind = event.get("type")
        if kind == "response.output_item.added":
            changed = self._added(event)
        elif kind == "response.output_item.done":
            changed = self._done(event)
        elif isinstance(kind, str) and kind in _REFERENCE_EVENTS:
            changed = self._reference(event)
        elif isinstance(kind, str) and kind in _TERMINAL_EVENTS:
            changed = self._snapshot(event)
        else:
            changed = self._unknown(event)
        if not changed:
            return frame
        self._would_rewrite_frames += 1
        if self.mode == "observe":
            return frame
        data = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        rewritten = parsed_frame.rebuild(data)
        if len(rewritten) > self.max_frame_bytes:
            self._fail("identity_mapping_limit")
        self._rewritten_frames += 1
        return rewritten

    def _added(self, event):
        index, item = event.get("output_index"), event.get("item")
        if not self._index(index) or not isinstance(item, dict) or not self._id(item.get("id")) or not self._id(item.get("type")):
            if self._mapped_items:
                self._fail()
            self._unsupported("invalid_identity")
            return False
        if index in self._bindings:
            self._fail()  # Never turn duplicate added events into event deduplication.
        if len(self._bindings) >= self.max_items:
            self._fail("identity_mapping_limit")
        self._alias(item["id"], index)
        self._reserve_strings(len(item["type"].encode("utf-8")) + len(item["id"].encode("utf-8", "surrogatepass")))
        self._bindings[index] = _Binding(item["type"], item["id"])
        self._tracked_items += 1
        self._mapped_items += item["type"] == "message"
        return False

    def _done(self, event):
        item = event.get("item")
        if not isinstance(item, dict):
            if self._mapped_items:
                self._fail()
            self._unsupported("invalid_identity")
            return False
        index = event.get("output_index")
        ident = item.get("id")
        binding = self._binding(index, ident)
        if binding is None:
            return False
        if item.get("type") != binding.kind:
            self._fail()
        canonical = self._canonical(binding, index, ident)
        if canonical == ident:
            return False
        item["id"] = canonical
        return True

    def _reference(self, event):
        index, ident = event.get("output_index"), event.get("item_id")
        binding = self._binding(index, ident)
        if binding is None:
            return False
        canonical = self._canonical(binding, index, ident)
        if canonical == ident:
            return False
        event["item_id"] = canonical
        return True

    def _snapshot(self, event):
        response = event.get("response")
        output = response.get("output") if isinstance(response, dict) else None
        if output is None:
            if self._drift_items:
                self._fail()
            if self._mapped_items:
                self._unsupported("missing_snapshot")
            return False
        # The API's ordered output array is usable only with the complete set
        # of observed added slots. Known aliases also catch same-type reorder.
        complete = isinstance(output, list) and len(output) == len(self._bindings) and all(
            index in self._bindings for index in range(len(output)))
        if not complete:
            if self._drift_items or self._needs_rewrite(event):
                self._fail()
            self._unsupported("missing_added" if not self._bindings else "ambiguous_snapshot")
            return False
        for index, item in enumerate(output):
            binding = self._bindings[index]
            if not isinstance(item, dict) or item.get("type") != binding.kind or not self._id(item.get("id")):
                self._fail()
            existing = self._lookup(item["id"])
            if existing is not None and existing != index:
                self._fail()
        changed = False
        for index, item in enumerate(output):
            ident = item["id"]
            canonical = self._canonical(self._bindings[index], index, ident)
            if canonical != ident:
                item["id"] = canonical
                changed = True
        return changed

    def _unknown(self, event):
        # Inspect identity-shaped top-level fields only. Never recursively
        # replace strings in text, annotations, tool arguments, or opaque data.
        if self._needs_rewrite(event):
            self._fail()
        return False
