"""One Claude model, three provider spellings. Author: Zeno Ren.

Three upstreams serve Claude over the Anthropic Messages protocol, and each
names the same model differently. Verified against live upstreams on
2026-10-17 (docs/reviews/2026-10-17-claude-three-paths/EVIDENCE.md):

    Opus 5.5   databricks        databricks-claude-opus-5-5
               copilot           claude-opus-5.5        <- dots
               azure_anthropic   claude-opus-5-5        <- dashes

The difference is load-bearing, not cosmetic: Foundry answers
`claude-opus-5.5` with 404 `DeploymentNotFound`. So every failover hop has to
rewrite the model id, and that rewrite lives here and nowhere else.

**Identity is not availability.** This module answers "what is this model
called over there", never "can it serve it". Availability keeps its existing
owners: config for Databricks, the discovered capability snapshot for Copilot,
operator-declared deployments for Foundry — which has no Models API at all
(`GET /anthropic/v1/models` returns 401). Encoding a point-in-time deployment
inventory here would turn one afternoon's observation into a permanent claim,
which is the mistake the capability catalog exists to prevent.
"""
import re
from typing import Optional

# Providers that speak the Anthropic Messages protocol. `azure_openai` is
# absent on purpose: it is the *other* face of the same Foundry resource and
# speaks the Azure OpenAI shape, so it cannot serve /v1/messages.
MESSAGES_PROVIDERS = ('databricks', 'copilot', 'azure_anthropic')

# Operator policy, set 2026-10-17: ADB first, then GHCP, then Foundry. A
# per-model rule in the Dashboard overrides this; nothing else does.
DEFAULT_CLAUDE_ORDER = ('databricks', 'copilot', 'azure_anthropic')

# How each provider spells a canonical id. `version_separator` is the whole
# difference between the three: Databricks and Foundry use dashes, Copilot uses
# dots. Foundry's value is the *default* deployment name — an operator may
# rename a deployment, which is why the config can override it per endpoint.
_SPELLING = {
    'databricks': {'prefix': 'databricks-claude-', 'version_separator': '-'},
    'copilot': {'prefix': 'claude-', 'version_separator': '.'},
    'azure_anthropic': {'prefix': 'claude-', 'version_separator': '-'},
}

# A canonical id is `<family>-<version>[-<suffix>]` with dashes throughout:
# `opus-5-5`, `opus-5`, `opus-4-8-fast`, `fable-5-1`. Families are listed
# explicitly rather than matched loosely, so a future non-Claude model that
# happens to contain "claude" in its name cannot be claimed by accident.
_FAMILIES = ('opus', 'sonnet', 'haiku', 'fable', 'mythos')

# `claude-opus-5.5`, `databricks-claude-opus-5-5`, `claude-opus-4.8-fast`.
# The version needs at least one digit; a bare `claude-opus` names no specific
# model and must not resolve, because picking a version for the caller is
# exactly what this repository's model resolution never does.
_PARSE = re.compile(
    r'^(?:databricks-)?claude-(?P<family>' + '|'.join(_FAMILIES) + r')'
    r'-(?P<version>\d+(?:[.\-]\d+)*)'
    r'(?P<suffix>(?:-[a-z0-9]+)*)$'
)


def canonical(model) -> Optional[str]:
    """The provider-independent id for a Claude model, or None.

    Accepts any of the three provider spellings. Returns None for every
    non-Claude model, so callers can treat "not Claude" and "unknown" alike
    without a separate membership test.
    """
    if not isinstance(model, str):
        return None
    match = _PARSE.match(model.strip().lower())
    if match is None:
        return None
    version = match.group('version').replace('.', '-')
    return f"{match.group('family')}-{version}{match.group('suffix')}"


def provider_model_id(canonical_id: str, provider: str) -> Optional[str]:
    """How `canonical_id` is spelled on `provider`, or None if not a Claude id.

    Raises on an unknown provider rather than guessing a spelling: a silent
    wrong id reaches the upstream as a 404 on a failover hop, which is the
    least debuggable moment to discover a typo.
    """
    if provider not in _SPELLING:
        raise ValueError(f'provider {provider!r} does not speak Anthropic Messages; '
                         f'expected one of {MESSAGES_PROVIDERS}')
    if not isinstance(canonical_id, str):
        return None
    match = re.match(r'^(?P<family>' + '|'.join(_FAMILIES) + r')'
                     r'-(?P<version>\d+(?:-\d+)*)'
                     r'(?P<suffix>(?:-[a-z0-9]+)*)$', canonical_id.strip().lower())
    if match is None:
        return None
    rule = _SPELLING[provider]
    version = match.group('version').replace('-', rule['version_separator'])
    return f"{rule['prefix']}{match.group('family')}-{version}{match.group('suffix')}"


def rewrite(model: str, provider: str) -> str:
    """Spell `model` the way `provider` expects, or return it untouched.

    A non-Claude model has no Claude identity to rewrite, so it passes through
    — callers on a mixed routing path do not need to branch.
    """
    canonical_id = canonical(model)
    if canonical_id is None:
        return model
    return provider_model_id(canonical_id, provider) or model
