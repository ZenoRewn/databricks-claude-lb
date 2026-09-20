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
