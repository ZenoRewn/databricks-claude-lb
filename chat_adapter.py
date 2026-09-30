"""Explicit Chat -> Responses transformations; no provider capability claims.

Author: Zeno Ren
Schema reference: https://developers.openai.com/api/docs/guides/migrate-to-responses
"""
import os
import re
from request_telemetry import note_parameter_transforms, reject_parameters

MODE = os.getenv('LB_CHAT_ADAPTER_CONTRACT', 'preserve')
if MODE not in ('preserve', 'text-only'):
    raise ValueError('LB_CHAT_ADAPTER_CONTRACT must be preserve or text-only')
_models = os.getenv('LB_CHAT_ADAPTER_PRESERVE_MODELS')
PRESERVE_MODELS = None if _models is None else frozenset(x.strip().lower() for x in _models.split(',') if x.strip())
if PRESERVE_MODELS is not None and (len(PRESERVE_MODELS) > 64 or any(len(x) > 128 for x in PRESERVE_MODELS)):
    raise ValueError('Adapter model selection exceeds the configuration bound')


def reject(field):
    reject_parameters({field})


def content(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        reject('messages.content')
    result = []
    for part in value:
        if not isinstance(part, dict):
            reject('messages.content')
        kind = part.get('type')
        if kind == 'text' and isinstance(part.get('text'), str) and set(part) <= {'type', 'text', 'prompt_cache_breakpoint'}:
            result.append({**part, 'type': 'input_text'})
        elif kind == 'image_url' and isinstance(part.get('image_url'), dict):
            image = part['image_url']
            if not isinstance(image.get('url'), str) or set(image) - {'url', 'detail'} or set(part) - {'type', 'image_url'}:
                reject('messages.content')
            result.append({'type': 'input_image', 'image_url': image['url'],
                           **({'detail': image['detail']} if 'detail' in image else {})})
        elif kind == 'file' and isinstance(part.get('file'), dict) and set(part) <= {'type', 'file'}:
            file = part['file']
            if set(file) - {'file_id', 'file_data', 'filename'} or not ('file_id' in file or 'file_data' in file):
                reject('messages.content')
            result.append({'type': 'input_file', **file})
        else:
            reject('messages.content')
    note_parameter_transforms({'messages.content'})
    return result


def messages_to_items(messages):
    if not isinstance(messages, list):
        reject('messages.content')
    converted, pending, seen = [], set(), set()
    for message in messages:
        if not isinstance(message, dict):
            reject('messages.content')
        role = message.get('role')
        if role not in ('system', 'developer', 'user', 'assistant', 'tool'):
            reject('messages.role')
        if set(message) - {'role', 'content', 'tool_calls', 'tool_call_id'}:
            reject('messages.content')
        if role != 'tool' and 'tool_call_id' in message or role == 'tool' and 'tool_calls' in message:
            reject('messages.tool_calls')
        if role == 'tool':
            call_id = message.get('tool_call_id')
            if not isinstance(call_id, str) or call_id not in pending:
                reject('messages.tool_results')
            pending.remove(call_id)
            output = message.get('content')
            if not isinstance(output, str):
                # Chat tool message text blocks can be concatenated losslessly.
                if not isinstance(output, list) or not all(isinstance(p, dict) and set(p) == {'type', 'text'}
                    and p['type'] == 'text' and isinstance(p['text'], str) for p in output):
                    reject('messages.tool_results')
                output = ''.join(p['text'] for p in output)
            converted.append({'type': 'function_call_output', 'call_id': call_id, 'output': output})
            note_parameter_transforms({'messages.tool_results'})
            continue
        if pending:
            reject('messages.tool_results')
        calls = message.get('tool_calls')
        if calls is not None:
            if role != 'assistant' or not isinstance(calls, list):
                reject('messages.tool_calls')
            if message.get('content') not in (None, '', []):
                converted.append({'role': role, 'content': content(message['content'])})
            for call in calls:
                if not isinstance(call, dict) or call.get('type') != 'function' or set(call) - {'id', 'type', 'function'}:
                    reject('messages.tool_calls')
                call_id, function = call.get('id'), call.get('function')
                if (not isinstance(call_id, str) or not call_id or call_id in seen or not isinstance(function, dict)
                        or set(function) != {'name', 'arguments'} or not isinstance(function['name'], str)
                        or not function['name'] or not isinstance(function['arguments'], str)):
                    reject('messages.tool_calls')
                seen.add(call_id); pending.add(call_id)
                converted.append({'type': 'function_call', 'call_id': call_id, **function})
            note_parameter_transforms({'messages.tool_calls'})
        else:
            converted.append({'role': role, 'content': content(message.get('content'))})
    if pending:
        reject('messages.tool_results')
    return converted


def tools_to_items(tools):
    if tools is None:
        return None
    if not isinstance(tools, list):
        reject('tools')
    result = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get('type') != 'function' or set(tool) != {'type', 'function'}:
            reject('tools')
        function = tool['function']
        if (not isinstance(function, dict) or set(function) - {'name', 'description', 'parameters', 'strict'}
                or not isinstance(function.get('name'), str) or not function['name']
                or 'parameters' in function and not isinstance(function['parameters'], dict)
                or function.get('strict') is not None and type(function['strict']) is not bool):
            reject('tools')
        result.append({'type': 'function', **function, 'strict': bool(function.get('strict', False)),
                       'parameters': function.get('parameters', {'type': 'object', 'properties': {}})})
    if result:
        note_parameter_transforms({'tools', 'tools.strict'})
    return result or None


def token_limit(value, field):
    if value is None:
        return None
    if isinstance(value, str) and re.fullmatch(r'[0-9]{1,18}', value):
        value = int(value)
    if type(value) is not int:
        reject(field)
    return value if value > 0 else None


def build_payload(body, *, removed_sampling=()):
    supported = {'model', 'messages', 'stream', 'stream_options', 'max_tokens', 'max_completion_tokens',
                 'temperature', 'top_p', 'tools', 'tool_choice', 'response_format', 'reasoning_effort',
                 'parallel_tool_calls', 'store', 'metadata', 'prompt_cache_key', 'safety_identifier',
                 'prompt_cache_retention', 'prompt_cache_options', 'service_tier', 'user', 'verbosity', 'n'}
    if set(body) - supported:
        reject('adapter.unsupported')
    if 'n' in body and body['n'] is not None and (type(body['n']) is not int or body['n'] != 1):
        reject('adapter.unsupported')
    options = body.get('stream_options')
    if options is not None and (not isinstance(options, dict) or set(options) - {'include_usage'}
                               or 'include_usage' in options and type(options['include_usage']) is not bool):
        reject('stream_options')
    messages = body.get('messages', [])
    if not isinstance(messages, list):
        reject('messages.content')
    preserve = MODE == 'preserve' and (PRESERVE_MODELS is None or str(body.get('model', '')).lower() in PRESERVE_MODELS)
    if not preserve and (any(body.get(k) is not None for k in ('tools', 'tool_choice', 'response_format', 'reasoning_effort'))
            or any(isinstance(m, dict) and (m.get('tool_calls') or m.get('role') == 'tool' or isinstance(m.get('content'), list))
                   for m in messages)):
        reject('adapter.unsupported')
    result = {'model': body.get('model', 'unknown'), 'input': messages_to_items(messages), 'stream': False}
    legacy = token_limit(body.get('max_tokens'), 'max_tokens')
    modern = token_limit(body.get('max_completion_tokens'), 'max_completion_tokens')
    if legacy is not None and modern is not None and legacy != modern:
        reject('max_completion_tokens')
    if legacy is not None or modern is not None:
        result['max_output_tokens'] = modern if modern is not None else legacy
        note_parameter_transforms({'max_completion_tokens' if modern is not None else 'max_tokens'})
    for key in ('temperature', 'top_p', 'store', 'metadata', 'parallel_tool_calls', 'prompt_cache_key',
                'safety_identifier', 'service_tier', 'user', 'prompt_cache_retention', 'prompt_cache_options'):
        if key in body and body[key] is not None and key not in removed_sampling:
            if key in ('store', 'parallel_tool_calls') and type(body[key]) is not bool:
                reject('adapter.unsupported')
            result[key] = body[key]
    if body.get('reasoning_effort') is not None:
        effort = body['reasoning_effort']
        if not isinstance(effort, str) or not re.fullmatch(r'[a-z_]{1,32}', effort):
            reject('reasoning_effort')
        result['reasoning'] = {'effort': effort}
        note_parameter_transforms({'reasoning_effort'})
    if body.get('response_format') is not None:
        fmt = body['response_format']
        if not isinstance(fmt, dict):
            reject('response_format')
        if fmt.get('type') in ('text', 'json_object') and set(fmt) == {'type'}:
            converted = dict(fmt)
        elif fmt.get('type') == 'json_schema' and set(fmt) == {'type', 'json_schema'}:
            schema = fmt['json_schema']
            if (not isinstance(schema, dict) or set(schema) - {'name', 'description', 'schema', 'strict'}
                    or not isinstance(schema.get('name'), str) or not schema['name']
                    or not isinstance(schema.get('schema'), dict)
                    or schema.get('strict') is not None and type(schema['strict']) is not bool):
                reject('response_format')
            converted = {'type': 'json_schema', **schema, 'strict': bool(schema.get('strict', False))}
        else:
            reject('response_format')
        result['text'] = {'format': converted}
        note_parameter_transforms({'response_format'})
    if body.get('verbosity') is not None:
        result.setdefault('text', {})['verbosity'] = body['verbosity']
    tools = tools_to_items(body.get('tools'))
    if tools:
        result['tools'] = tools
    choice = body.get('tool_choice')
    if choice is not None:
        if isinstance(choice, str) and choice in ('auto', 'none', 'required'):
            converted = choice
        elif (isinstance(choice, dict) and set(choice) == {'type', 'function'} and choice['type'] == 'function'
                and isinstance(choice['function'], dict) and set(choice['function']) == {'name'}
                and isinstance(choice['function']['name'], str)):
            converted = {'type': 'function', 'name': choice['function']['name']}
            if not tools or converted['name'] not in {t['name'] for t in tools}:
                reject('tool_choice')
        else:
            reject('tool_choice')
        result['tool_choice'] = converted
        note_parameter_transforms({'tool_choice'})
    return result
