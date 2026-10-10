"""Operator-sourced channel capabilities and observe-first budgets. Author: Zeno Ren.

No built-in model limits, remote discovery, tokenization claims or content edits.
Input estimates never authorize rejection; only verified explicit output/feature
contracts can be enforced, with an opt-in server setting.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import HTTPException
from request_telemetry import CURRENT, current_request_id, log_event, note_failure
from safe_diagnostics import safe_identifier

MODE = os.getenv('LB_CONTEXT_BUDGET_MODE', 'observe')
if MODE not in ('off', 'observe', 'enforce'):
    raise ValueError('LB_CONTEXT_BUDGET_MODE must be off, observe or enforce')
LARGE_INPUT_BYTES = int(os.getenv('LB_CONTEXT_LARGE_INPUT_BYTES', '262144'))
if not 1 <= LARGE_INPUT_BYTES <= 64 * 1024 * 1024:
    raise ValueError('LB_CONTEXT_LARGE_INPUT_BYTES must be between 1 and 67108864')
# Louder advisory tier above LARGE_INPUT_BYTES. The 1 MiB default is a heuristic
# chosen so the 1.58/2.20/3.21 MB bodies seen in the 2026-10-09/10 incidents stop
# sharing a label with a 256 KiB request. It is NOT an evidence-derived limit —
# large requests also succeed — so it never rejects, trims or summarises input.
ELEVATED_INPUT_BYTES = int(os.getenv('LB_CONTEXT_ELEVATED_INPUT_BYTES', '1048576'))
if not 1 <= ELEVATED_INPUT_BYTES <= 64 * 1024 * 1024:
    raise ValueError('LB_CONTEXT_ELEVATED_INPUT_BYTES must be between 1 and 67108864')
if ELEVATED_INPUT_BYTES <= LARGE_INPUT_BYTES:
    raise ValueError('LB_CONTEXT_ELEVATED_INPUT_BYTES must exceed LB_CONTEXT_LARGE_INPUT_BYTES')
LIMITS = ('input_tokens', 'context_tokens', 'output_tokens')
FEATURES = ('tools', 'images', 'structured_output', 'opaque_state')
SEMANTIC_FIELDS = {'messages', 'input', 'system', 'instructions', 'tools', 'tool_choice',
                   'response_format', 'text', 'output_config', 'context_management', 'previous_response_id',
                   'conversation', 'conversation_id'}


class CapabilityRejection(HTTPException):
    """A verified local contract conflict, including during endpoint selection."""


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError('Capability timestamps require an explicit timezone')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Capability timestamps require an explicit timezone')
    return parsed.astimezone(timezone.utc)


class CapabilityCatalog:
    def __init__(self, document=None, *, now=None):
        document = {'schema_version': 1, 'entries': []} if document is None else copy.deepcopy(document)
        if (not isinstance(document, dict) or set(document) - {'schema_version', 'entries', 'author'}
                or type(document.get('schema_version')) is not int or document['schema_version'] != 1
                or not isinstance(document.get('entries'), list) or len(document['entries']) > 256):
            raise ValueError('Invalid capability catalog schema')
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.entries = {}
        allowed = {'provider', 'api_type', 'model', 'endpoint_alias', 'model_version', 'verification',
                   'verified_at', 'expires_at', 'source', 'limits', 'features'}
        for entry in document['entries']:
            if not isinstance(entry, dict) or set(entry) != allowed:
                raise ValueError('Invalid capability entry fields')
            if (entry['provider'] not in ('databricks', 'azure_openai', 'copilot') or entry['api_type'] not in ('messages', 'responses', 'chat')
                    or not isinstance(entry['model'], str) or safe_identifier(entry['model'], None) is None
                    or entry['endpoint_alias'] is not None and safe_identifier(entry['endpoint_alias'], None) is None
                    or safe_identifier(entry['model_version'], None) is None):
                raise ValueError('Invalid channel, model or endpoint identity')
            if not isinstance(entry['limits'], dict) or set(entry['limits']) != set(LIMITS):
                raise ValueError('Capability limits require explicit known or null values')
            if any(v is not None and (type(v) is not int or not 1 <= v < 2**63) for v in entry['limits'].values()):
                raise ValueError('Capability limits must be positive integers or null')
            if not isinstance(entry['features'], dict) or set(entry['features']) != set(FEATURES) or any(
                    v is not None and type(v) is not bool for v in entry['features'].values()):
                raise ValueError('Capability features must be boolean or unknown')
            if entry['verification'] not in ('unverified', 'operator_verified'):
                raise ValueError('Capability verification must be explicit')
            source = entry['source']
            if source is not None:
                if not isinstance(source, dict) or set(source) != {'kind', 'url'} or source['kind'] not in ('channel_contract', 'channel_test'):
                    raise ValueError('Invalid capability source')
                if not isinstance(source['url'], str) or len(source['url']) > 2048:
                    raise ValueError('Invalid capability source URL')
                url = urlsplit(source['url'])
                if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query:
                    raise ValueError('Capability source must be HTTPS without credentials or query secrets')
            if entry['verification'] == 'operator_verified':
                if source is None or timestamp(entry['expires_at']) <= timestamp(entry['verified_at']):
                    raise ValueError('Verified capabilities require a source and finite validity window')
            elif any(value is not None for value in (entry['verified_at'], entry['expires_at'])):
                for value in (entry['verified_at'], entry['expires_at']):
                    if value is not None:
                        timestamp(value)
            key = (entry['provider'], entry['api_type'], entry['model'], entry['endpoint_alias'])
            if key in self.entries:
                raise ValueError('Duplicate capability route')
            self.entries[key] = entry
        self.sha256 = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def _status(self, entry):
        if entry['verification'] != 'operator_verified':
            return 'unverified'
        now = self.now()
        if now < timestamp(entry['verified_at']):
            return 'unverified'
        return 'verified' if now < timestamp(entry['expires_at']) else 'expired'

    def lookup(self, provider, api_type, model, endpoint_alias=None):
        entry = self.entries.get((provider, api_type, model, endpoint_alias))
        if entry is None:
            entry = self.entries.get((provider, api_type, model, None))
        if entry is None:
            return {'capability_status': 'unknown', 'limits': dict.fromkeys(LIMITS), 'features': dict.fromkeys(FEATURES)}
        return {**copy.deepcopy(entry), 'capability_status': self._status(entry)}

    def public_view(self):
        return {'schema_version': 1, 'author': 'Zeno Ren', 'catalog_sha256': self.sha256,
                'unknown_policy': 'observe_without_input_rejection', 'input_estimate_is_authoritative': False,
                'verification_authority': 'operator attestation, not automatic provider certification',
                'entries': [{**copy.deepcopy(value), 'capability_status': self._status(value)} for value in self.entries.values()]}

    @classmethod
    def from_path(cls, path):
        if not path:
            return cls()
        with Path(path).open('rb') as source:
            raw = source.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError('Capability catalog exceeds 1 MiB')
        return cls(json.loads(raw))


CATALOG = CapabilityCatalog.from_path(os.getenv('LB_MODEL_CAPABILITIES_PATH'))


def _utf8_size(value):
    # Avoid a second giant allocation for long histories or tool schemas.
    return sum(len(value[i:i + 65536].encode('utf-8', errors='replace')) for i in range(0, len(value), 65536))


def estimate_payload(payload):
    """Size-based estimate of supplied semantic input, not an exact tokenizer.

    Unknown modalities/state stay explicit. Iterators bound auxiliary memory;
    excess depth/node count yields a partial estimate rather than a guessed total.
    """
    total, images, visited, unknown = 2, 0, 0, set()
    definitions = {'tools', 'response_format', 'text', 'output_config'}
    def children(value, definition):
        for key, item in value.items() if isinstance(value, dict) else enumerate(value):
            yield key, item, definition
    roots = ((key, value, key in definitions) for key, value in payload.items() if key in SEMANTIC_FIELDS) if isinstance(payload, dict) else iter(())
    stack = [iter(roots)]
    while stack:
        try:
            key, value, definition = next(stack[-1])
        except StopIteration:
            stack.pop(); continue
        visited += 1
        if visited > 100000 or len(stack) > 128:
            unknown.add('traversal_limit'); break
        if not definition and key in ('encrypted_content',):
            unknown.add('opaque_state'); continue
        if not definition and key in ('previous_response_id', 'conversation', 'conversation_id') and value:
            unknown.add('prior_state'); continue
        total += _utf8_size(key) + 4 if isinstance(key, str) else 1
        if isinstance(value, dict):
            kind = value.get('type')
            if not definition and isinstance(kind, str) and kind in ('image', 'image_url', 'input_image'):
                images += 1; unknown.add('images'); continue
            if not definition and isinstance(kind, str) and kind in ('file', 'input_file', 'input_audio', 'audio'):
                unknown.add('files' if kind in ('file', 'input_file') else 'audio'); continue
            if not definition and kind == 'item_reference':
                unknown.add('prior_state'); continue
            total += 2
            stack.append(children(value, definition))
        elif isinstance(value, list):
            total += 2
            stack.append(children(value, definition))
        elif isinstance(value, str):
            if not definition and value.startswith('data:image/'):
                images += 1; unknown.add('images')
            else:
                total += _utf8_size(value) + 2
        elif value is None or type(value) is bool:
            total += 5
        elif type(value) in (int, float):
            total += len(str(value))
        else:
            unknown.add('unsupported_content')
    return {'estimated_input_tokens': math.ceil(total / 4), 'text_bytes': total, 'image_count': images,
            'estimate_method': 'utf8-json-estimate-v1', 'estimate_confidence': 'low',
            'estimate_complete': not unknown, 'unknown_components': sorted(unknown), 'enforcement_allowed': False}


def output_budget(payload, api_type):
    names = ('max_output_tokens',) if api_type == 'responses' else ('max_tokens',) if api_type == 'messages' else ('max_completion_tokens', 'max_tokens')
    for name in names:
        value = payload.get(name)
        if type(value) is int and value > 0:
            return value
    return None


def context_advice(estimate, limits=None, reserved=None):
    """Advisory size/estimated utilization; never an input admission decision."""
    limits = limits or {}
    ratios = []
    if limits.get('input_tokens'):
        ratios.append(estimate['estimated_input_tokens'] / limits['input_tokens'])
    if limits.get('context_tokens'):
        ratios.append((estimate['estimated_input_tokens'] + (reserved or 0)) / limits['context_tokens'])
    if any(ratio > 1 for ratio in ratios):
        return 'estimated_over_limit'
    if any(ratio >= .8 for ratio in ratios):
        return 'estimated_near_limit'
    # Size tiers are reported even when the token estimate is incomplete: bytes on
    # the wire are known regardless of unknown image/opaque token counts.
    if estimate['text_bytes'] >= ELEVATED_INPUT_BYTES:
        return 'elevated_input'
    return 'large_input' if estimate['text_bytes'] >= LARGE_INPUT_BYTES else 'none'


def requested_features(payload):
    """Inspect protocol positions, never treat schema examples as live input."""
    images = False
    opaque = any(bool(payload.get(key)) for key in ('previous_response_id', 'conversation', 'conversation_id'))
    roots = [payload.get(key) for key in ('messages', 'input') if isinstance(payload.get(key), list)]
    stack, visited = [iter(items) for items in roots], 0
    while stack and visited < 100000:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop(); continue
        visited += 1
        if not isinstance(item, dict):
            continue
        kind = item.get('type')
        if isinstance(kind, str):
            images = images or kind in ('image', 'image_url', 'input_image')
            opaque = opaque or kind == 'item_reference' or kind == 'reasoning' and bool(item.get('encrypted_content'))
        if len(stack) < 128 and isinstance(item.get('content'), list):
            stack.append(iter(item['content']))
        if kind == 'function_call_output' and len(stack) < 128 and isinstance(item.get('output'), list):
            stack.append(iter(item['output']))
    formats = [payload.get('response_format')]
    formats.extend(payload[key].get('format') for key in ('text', 'output_config') if isinstance(payload.get(key), dict))
    return {'tools': bool(payload.get('tools')), 'images': images, 'opaque_state': bool(opaque),
            'structured_output': any(isinstance(fmt, dict) and fmt.get('type') in ('json_schema', 'json_object') for fmt in formats)}


def evaluate_budget(catalog, provider, api_type, model, payload, *, endpoint_alias=None, mode='observe'):
    if mode not in ('off', 'observe', 'enforce'):
        raise ValueError('Invalid context budget mode')
    capability = catalog.lookup(provider, api_type, model, endpoint_alias)
    estimate = estimate_payload(payload)
    verified = capability['capability_status'] == 'verified'
    limits = capability['limits'] if verified else dict.fromkeys(LIMITS)
    reserved = output_budget(payload, api_type)
    over = ((limits['input_tokens'] is not None and estimate['estimated_input_tokens'] > limits['input_tokens']) or
            (limits['context_tokens'] is not None and reserved is not None and estimate['estimated_input_tokens'] + reserved > limits['context_tokens']))
    result = {**estimate, 'capability_status': capability['capability_status'], 'budget_mode': mode,
              'context_status': 'estimated_over' if over else 'observed' if verified else 'unknown',
              'reserved_output_tokens': reserved, 'input_limit': limits['input_tokens'],
              'context_limit': limits['context_tokens'], 'output_limit': limits['output_tokens']}
    result['context_advice'] = context_advice(estimate, limits, reserved)
    if mode == 'enforce' and verified:
        present = requested_features(payload)
        unsupported = [name for name in FEATURES if present[name] and capability['features'][name] is False]
        output_over = reserved is not None and limits['output_tokens'] is not None and reserved > limits['output_tokens']
        if unsupported or output_over:
            code = 'output_budget_exceeded' if output_over else 'model_capability_mismatch'
            note_failure('invalid_input', origin='local')
            raise CapabilityRejection(status_code=400, detail={'error': {'code': code,
                'message': 'The requested output budget or feature conflicts with the configured verified channel contract.',
                'unsupported_features': unsupported, 'output_limit': limits['output_tokens'],
                'lb_request_id': current_request_id(), 'retryable': False}})
    return result


def observe_route_budget(provider, api_type, model, payload, endpoint_alias=None):
    if MODE == 'off':
        return
    # No real provider access; source confidence/freshness is never inferred.
    result = evaluate_budget(CATALOG, provider, api_type, model, payload, endpoint_alias=endpoint_alias, mode=MODE)
    record = CURRENT.get()
    if record is not None:
        record.context_budget = result
    log_event({'kind': 'lb_context_budget', 'provider': provider, 'api_type': api_type,
               'forwarded_model': model, 'endpoint_alias': endpoint_alias, **result})
