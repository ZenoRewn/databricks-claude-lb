"""Preserve the supported native effort field without widening legacy passthrough. Author: Zeno Ren."""

def preserve_native_effort(body):
    output = body.pop("output_config", None)
    if body.get("model", "").lower() == "databricks-claude-opus-5" and isinstance(output, dict) and "effort" in output:
        # Leave value validation to the actual upstream. In particular, do not
        # silently default invalid effort values to high.
        body["output_config"] = {"effort": output["effort"]}


def effort_response_headers(body):
    output = body.get("output_config")
    effort = output.get("effort") if isinstance(output, dict) else None
    return {"x-claude-effort-forwarded": effort} if isinstance(effort, str) and effort in ("low", "medium", "high", "xhigh", "max") else {}


def databricks_parameter_drops(body, *, adaptive_supported=False):
    """Describe only fields the existing local compatibility policy removes."""
    fields = set()
    if 'context_management' in body:
        fields.add('context_management')
    if 'output_config' in body:
        output = body['output_config']
        if not isinstance(output,dict) or not output:
            fields.add('output_config')
        else:
            for key in output:
                if key=='effort' and body.get('model','').lower()=='databricks-claude-opus-5':
                    continue
                fields.add('output_config.'+key if key in ('effort','format') else 'output_config.other')
    blocks = [b for b in body.get('system',[]) if isinstance(b,dict)] if isinstance(body.get('system'),list) else []
    for tool in body.get('tools',[]) if isinstance(body.get('tools'),list) else []:
        if not isinstance(tool,dict):
            continue
        blocks.append(tool)
        for key in ('defer_loading','input_examples'):
            if key in tool or isinstance(tool.get('custom'),dict) and key in tool['custom']:
                fields.add('tools.'+key)
    for message in body.get('messages',[]) if isinstance(body.get('messages'),list) else []:
        if not isinstance(message,dict) or not isinstance(message.get('content'),list):
            continue
        for block in message['content']:
            if not isinstance(block,dict):
                continue
            blocks.append(block)
            nested = block.get('content') if block.get('type')=='tool_result' else []
            candidates = [block]+([v for v in nested if isinstance(v,dict)] if isinstance(nested,list) else [])
            blocks.extend(candidates[1:])
            if any(v.get('type')=='tool_reference' for v in candidates):
                fields.add('messages.tool_reference')
    if any(isinstance(b.get('cache_control'),dict) and any(k!='type' for k in b['cache_control']) for b in blocks):
        fields.add('cache_control.extras')
    thinking = body.get('thinking')
    if adaptive_supported and isinstance(thinking,dict) and thinking.get('type')=='adaptive' and 'budget_tokens' in thinking:
        fields.add('thinking.budget_tokens')
    return sorted(fields)
