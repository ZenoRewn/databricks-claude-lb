"""Runtime policy store: per-tenant provider allow-lists and Copilot account
enablement. Author: Zeno Ren.

Two policies live here because they share one cache, one degradation contract
and one status surface; splitting them would mean two stores to reason about
during an outage.

Copilot accounts carry two layers. `capability` is discovered from upstream and
read-only — it records what an account actually serves. `allowed_models` is the
operator's subset of it. A write naming a model outside capability is refused,
so an account can never be told to serve something it does not have. A
hand-maintained list would drift; a discovered one cannot.

Scope and deliberate limits:

- The store keys on **tenant name**, never on the client API key, so no
  credential material is persisted here. Listing elsewhere exposes a mask and a
  sha256 prefix, never plaintext.
- An allow-list only removes routing candidates. It cannot make a provider
  serve a model it does not support, and it does not reorder the existing
  Copilot-first → Azure-fallback policy.
- Enforcement reads an in-memory cache. When the backend was never loaded the
  gateway applies **no** restriction and reports `unavailable`: a restriction
  layer must not turn a database outage into a total outage. The state is
  surfaced rather than hidden so the gap is visible.
- Its own small connection pool, not the usage ledger's: authorisation data
  should not become unavailable because the usage ledger is unhealthy.
"""
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Optional

logger = logging.getLogger(__name__)

IO_TIMEOUT = 10

PROVIDER_IDS = ('databricks', 'copilot', 'azure_openai')
PROVIDER_LABELS = {'databricks': 'ADB', 'copilot': 'GHCP', 'azure_openai': 'Foundry'}
ALL_PROVIDERS = frozenset(PROVIDER_IDS)

# Why an account is not a candidate, as a closed set rather than prose. The
# Dashboard has to tell an operator *which layer* dropped an account, and a
# free-text reason could neither be branched on nor tested. `main` adds the
# reasons for the layers it owns (config-declared enablement and the configured
# `models` list) and publishes the union.
#
# The distinction that matters most here is `not_in_capability` versus
# `excluded_by_allowed_models`: the first is upstream fact and is fixed by
# re-discovering, the second is operator intent and is fixed by a checkbox.
# Collapsing them would send people to the wrong control.
EXCLUSION_REASONS = frozenset({
    'account_disabled',
    'not_in_capability',
    'excluded_by_allowed_models',
    'not_selected_for_model',
})

MASK_MIN_LENGTH = 12

# Distinguishes "caller omitted allowed_models" from "caller asked to clear it".
_UNSET = object()


def normalise_providers(value) -> tuple:
    """Validate an operator-supplied provider list into canonical order.

    A string is rejected on purpose: "copilot" would otherwise iterate into
    characters. An empty result is rejected too — a tenant allowed nothing
    would be locked out with no way to tell it apart from a mistake.
    """
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError('providers must be a list of provider ids')
    unknown = sorted({str(item) for item in value} - ALL_PROVIDERS)
    if unknown:
        raise ValueError(f'unknown provider ids: {", ".join(unknown)}')
    ordered = tuple(pid for pid in PROVIDER_IDS if pid in set(value))
    if not ordered:
        raise ValueError('providers must name at least one provider')
    return ordered


def ordered_providers(value) -> tuple:
    """Validate an ordered provider preference.

    Unlike `normalise_providers` this keeps the caller's order — the order *is*
    the meaning. Duplicates are an error rather than collapsed, because a
    repeated entry means the caller believes order matters somewhere it cannot.
    """
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError('providers must be an ordered list of provider ids')
    items = [str(item) for item in value]
    if not items:
        raise ValueError('providers must name at least one provider')
    if len(set(items)) != len(items):
        raise ValueError('providers must not repeat; the order is the preference')
    unknown = sorted(set(items) - ALL_PROVIDERS)
    if unknown:
        raise ValueError(f'unknown provider ids: {", ".join(unknown)}')
    return tuple(items)


def normalise_accounts(value) -> tuple:
    """Validate a per-model account selection into a sorted set.

    Empty (or None) is **accepted** and means "not limited" — the opposite of
    `normalise_providers`, where empty would lock a tenant out. Here the empty
    selection is the baseline, so rejecting it would leave the Dashboard with no
    way to undo a limit.

    Order is dropped because session affinity, not the operator, picks among the
    selected accounts. A bare string is still rejected: it would iterate into
    characters and silently select nothing that exists.
    """
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError('copilot_accounts must be a list of account names')
    names = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError('copilot_accounts entries must be non-empty strings')
        names.append(item.strip())
    return tuple(sorted(set(names)))


def fingerprint(key: str) -> str:
    return hashlib.sha256((key or '').encode()).hexdigest()[:12]


def mask(key: str) -> str:
    """First and last four characters, or nothing for a short key.

    Showing 8 of 10 characters would hand over most of the secret, so short
    keys fall back to the fingerprint alone.
    """
    if not key or len(key) < MASK_MIN_LENGTH:
        return ''
    return f'{key[:4]}…{key[-4:]}'


class KeyPolicyStore:
    """Tenant → allowed providers, cached in memory and persisted per backend."""

    def __init__(self, storage_config: Optional[dict] = None):
        config = storage_config or {}
        self._config = config
        requested = config.get('type', 'memory')
        self.backend = requested if requested in ('mysql', 'json') else 'memory'
        self._policies: dict = {}
        self._accounts: dict = {}
        self._preferences: dict = {}
        self._model_accounts: dict = {}
        # One-way verifier for the panel password, or None for "no overlay".
        self._panel_password: Optional[str] = None
        self._state = 'unavailable'
        self._last_loaded_at: Optional[float] = None
        self._last_error_type: Optional[str] = None
        self._pool = None
        self._lock = asyncio.Lock()
        self._path = None
        if self.backend == 'json':
            # usage_storage.path is a directory, same key and semantics the JSON
            # usage backend uses.
            self._path = Path(config.get('path') or './usage_data') / 'api_key_policy.json'

    # -- lifecycle -------------------------------------------------------

    async def start(self):
        if self.backend == 'memory':
            self._state = 'loaded'
            self._last_loaded_at = time.time()
            return
        await self.reload()

    async def close(self):
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    async def reload(self) -> dict:
        """Refresh the cache. A failure keeps the last known policy."""
        try:
            async with asyncio.timeout(IO_TIMEOUT):
                loaded = await self._backend_load()
                accounts = await self._backend_load_accounts()
                preferences = await self._backend_load_preferences()
                model_accounts = await self._backend_load_model_accounts()
                panel_password = await self._backend_load_panel_password()
        except Exception as exc:
            self._last_error_type = type(exc).__name__
            # Never loaded → no restriction; loaded before → keep enforcing it.
            self._state = 'unavailable' if self._last_loaded_at is None else 'stale'
            logger.warning('key_policy: load failed (%s); state=%s', self._last_error_type, self._state)
            return self.status()
        self._policies = loaded
        self._accounts = accounts
        self._preferences = preferences
        self._model_accounts = model_accounts
        self._panel_password = panel_password
        self._state = 'loaded'
        self._last_loaded_at = time.time()
        self._last_error_type = None
        return self.status()

    # -- enforcement -----------------------------------------------------

    def allowed(self, tenant: str) -> frozenset:
        if self._state == 'unavailable':
            return ALL_PROVIDERS
        return frozenset(self._policies.get(tenant, ALL_PROVIDERS))

    def restricted(self, tenant: str) -> bool:
        return self.allowed(tenant) != ALL_PROVIDERS

    def snapshot(self) -> dict:
        return {tenant: tuple(providers) for tenant, providers in self._policies.items()}

    def status(self) -> dict:
        return {
            'state': self._state,
            'backend': self.backend,
            'tenants': len(self._policies),
            'accounts': len(self._accounts),
            'model_preferences': len(self._preferences),
            'model_accounts': len(self._model_accounts),
            'last_loaded_at': self._last_loaded_at,
            'last_error_type': self._last_error_type,
            'unrestricted_when_unavailable': True,
        }

    # -- per-model provider preference -----------------------------------

    def model_preference(self, model: str) -> tuple:
        """Ordered preference for one model; empty means «use the default order»."""
        if self._state == 'unavailable':
            return ()
        return tuple(self._preferences.get(model, ()))

    def model_preferences_snapshot(self) -> dict:
        return {model: tuple(order) for model, order in self._preferences.items()}

    async def set_model_preference(self, model: str, providers) -> tuple:
        if not isinstance(model, str) or not model.strip():
            raise ValueError('model must be a non-empty string')
        order = ordered_providers(providers)
        async with self._lock:
            async with asyncio.timeout(IO_TIMEOUT):
                await self._backend_write_preference(model.strip(), order)
            self._preferences[model.strip()] = order
            if self._state == 'unavailable':
                self._state = 'loaded'
                self._last_loaded_at = time.time()
        return order

    async def clear_model_preference(self, model: str) -> None:
        async with self._lock:
            async with asyncio.timeout(IO_TIMEOUT):
                await self._backend_delete_preference(model)
            self._preferences.pop(model, None)

    # -- per-model Copilot account selection -----------------------------
    #
    # The reverse of `allowed_models`: that one says which models an account may
    # serve, this one says which accounts a model may use. Both subtract, and the
    # effective candidate set is their intersection.
    #
    # Order is *not* preserved here, unlike `model_preference`. Which of the
    # selected accounts a request lands on is session affinity's decision, so an
    # operator-visible ordering would be a promise this layer cannot keep.

    def model_accounts(self, model: str) -> tuple:
        """Accounts this model is limited to; empty means «not limited».

        Same degradation rule as every other layer: an unloaded overlay yields
        the config baseline, which here is "no limit".
        """
        if self._state == 'unavailable':
            return ()
        return tuple(self._model_accounts.get(model, ()))

    def model_accounts_exclusion_reason(self, model: str, account: str) -> Optional[str]:
        """Why this model's selection drops this account, or None if it does not."""
        chosen = self.model_accounts(model)
        if not chosen or account in chosen:
            return None
        return 'not_selected_for_model'

    def model_allows_account(self, model: str, account: str) -> bool:
        # Derived from the reason, never computed beside it: the diagnostic and
        # the decision have to be the same code or the diagnostic will drift and
        # be believed anyway.
        return self.model_accounts_exclusion_reason(model, account) is None

    def model_accounts_snapshot(self) -> dict:
        return {model: tuple(accounts) for model, accounts in self._model_accounts.items()}

    async def set_model_accounts(self, model: str, accounts) -> tuple:
        """Limit one model to a set of accounts; empty/None clears the limit.

        Clearing has to mean "not limited" rather than "no account may serve
        this": the Dashboard expresses the limit as checkboxes, so unchecking
        the last one is the natural undo and must not remove the model instead.
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError('model must be a non-empty string')
        key = model.strip()
        selection = normalise_accounts(accounts)
        async with self._lock:
            async with asyncio.timeout(IO_TIMEOUT):
                if selection:
                    await self._backend_write_model_accounts(key, selection)
                else:
                    await self._backend_delete_model_accounts(key)
            if selection:
                self._model_accounts[key] = selection
            else:
                self._model_accounts.pop(key, None)
            if self._state == 'unavailable':
                self._state = 'loaded'
                self._last_loaded_at = time.time()
        return selection

    async def clear_model_accounts(self, model: str) -> None:
        async with self._lock:
            async with asyncio.timeout(IO_TIMEOUT):
                await self._backend_delete_model_accounts(model)
            self._model_accounts.pop(model, None)

    # -- Copilot account enablement --------------------------------------

    def account_policy(self, name: str) -> dict:
        """Effective record for one account; defaults keep existing config intact.

        ``enabled`` is three-state because this store holds operator intent
        while ``capability`` holds upstream fact — the separation the rest of
        this module already keeps. A record exists as soon as capability is
        discovered, so "has a record" cannot mean "an operator decided to use
        this account"; reading it that way would let a read-only discovery call
        put an account into service. ``enabled_decided`` is the only signal
        that an operator actually chose, and callers merging in a
        config-declared baseline must branch on it rather than on
        ``configured``.
        """
        record = self._accounts.get(name) or {}
        return {
            'enabled': bool(record.get('enabled', True)),
            'enabled_decided': 'enabled' in record,
            'allowed_models': record.get('allowed_models'),
            'capability': tuple(record.get('capability') or ()),
            'capability_at': record.get('capability_at'),
            'configured': name in self._accounts,
        }

    def account_exclusion_reason(self, name: str, model: str) -> Optional[str]:
        """哪一层把这个账户从该模型上摘掉，没摘掉则 None。

        有效集 = capability ∩ allowlist，只做减法。capability 本身就是外层边界：
        上游说这个账户没有该模型，把请求路由过去没有意义。未发现过 capability
        （空）时不据此限制，避免「还没发现」被当成「什么都没有」。无记录或存储
        未加载时一律放行 —— 后者是全项目那条「回落配置基线」的实例。

        顺序即诊断的优先级：停用压过一切，上游事实压过运维意图。两个账户侧的
        reason 必须分开报，它们指向不同的修法（刷新能力 vs 改勾选）。
        """
        if self._state == 'unavailable':
            return None
        record = self._accounts.get(name)
        if not record:
            return None
        if not record.get('enabled', True):
            return 'account_disabled'
        capability = record.get('capability') or ()
        if capability and model not in capability:
            return 'not_in_capability'
        allowed = record.get('allowed_models')
        if allowed is None or model in allowed:
            return None
        return 'excluded_by_allowed_models'

    def account_allows(self, name: str, model: str) -> bool:
        # Derived, not parallel — see model_allows_account for why.
        return self.account_exclusion_reason(name, model) is None

    # -- panel password overlay ------------------------------------------
    #
    # Stores a **verifier**, never the password and never recoverable
    # ciphertext. Upstream credentials are sealed because they have to be
    # replayed to the upstream; this one is only ever compared against, so a
    # one-way hash is strictly stronger and needs no encryption key at all.
    #
    # Same degradation rule as every other layer: no row, or a store that never
    # loaded, means "no overlay" and the caller falls back to its config
    # baseline — which for this layer is the environment variable. The caller is
    # responsible for the part that is not negotiable: when neither exists,
    # panel login is *unavailable*, not open.

    def panel_password_verifier(self) -> Optional[str]:
        if self._state == 'unavailable':
            return None
        return self._panel_password

    async def set_panel_password_verifier(self, verifier: str) -> None:
        if not isinstance(verifier, str) or not verifier.strip():
            raise ValueError('verifier must be a non-empty string')
        async with self._lock:
            async with asyncio.timeout(IO_TIMEOUT):
                await self._backend_write_panel_password(verifier)
            self._panel_password = verifier
            if self._state == 'unavailable':
                self._state = 'loaded'
                self._last_loaded_at = time.time()

    async def clear_panel_password_verifier(self) -> None:
        async with self._lock:
            async with asyncio.timeout(IO_TIMEOUT):
                await self._backend_delete_panel_password()
            self._panel_password = None

    def accounts_snapshot(self) -> dict:
        return {name: self.account_policy(name) for name in self._accounts}

    async def set_account_capability(self, name: str, models) -> tuple:
        """Record what upstream says this account serves. Read-only to operators."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError('account must be a non-empty string')
        discovered = tuple(sorted({str(m) for m in (models or []) if str(m)}))
        async with self._lock:
            record = dict(self._accounts.get(name.strip()) or {})
            record['capability'] = discovered
            record['capability_at'] = time.time()
            # Deliberately no `enabled` default: discovery is a read of upstream,
            # not a decision to use the account. Writing one here would enable an
            # account that config declared inert.
            record.setdefault('allowed_models', None)
            # Capability can shrink; drop allow-list entries upstream no longer
            # offers rather than keeping a selection that cannot be served.
            if record.get('allowed_models') is not None:
                kept = tuple(m for m in record['allowed_models'] if m in discovered)
                record['allowed_models'] = kept or None
            async with asyncio.timeout(IO_TIMEOUT):
                await self._backend_write_account(name.strip(), record)
            self._accounts[name.strip()] = record
            if self._state == 'unavailable':
                self._state = 'loaded'
                self._last_loaded_at = time.time()
        return discovered

    async def set_account(self, name: str, *, enabled=None, allowed_models=_UNSET) -> dict:
        """Toggle an account and/or set its allow-list within capability."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError('account must be a non-empty string')
        key = name.strip()
        current = dict(self._accounts.get(key) or {})
        capability = tuple(current.get('capability') or ())
        record = {
            'allowed_models': current.get('allowed_models'),
            'capability': capability,
            'capability_at': current.get('capability_at'),
        }
        # Carry an existing decision forward, but do not invent one: setting only
        # an allow-list must not silently enable an account declared inert.
        if 'enabled' in current:
            record['enabled'] = current['enabled']
        if enabled is not None:
            if not isinstance(enabled, bool):
                raise ValueError('enabled must be a boolean')
            record['enabled'] = enabled
        if allowed_models is not _UNSET:
            if allowed_models is None:
                record['allowed_models'] = None
            else:
                if not isinstance(allowed_models, (list, tuple, set, frozenset)):
                    raise ValueError('allowed_models must be a list of model ids or null')
                chosen = tuple(sorted({str(m) for m in allowed_models if str(m)}))
                if not chosen:
                    # Disabling is the explicit way to stop using an account; an
                    # empty allow-list would do it silently under another name.
                    raise ValueError('allowed_models must name at least one model, or be null')
                if capability:
                    outside = sorted(set(chosen) - set(capability))
                    if outside:
                        raise ValueError(
                            f'models outside this account\'s discovered capability: {", ".join(outside)}')
                record['allowed_models'] = chosen
        async with self._lock:
            async with asyncio.timeout(IO_TIMEOUT):
                await self._backend_write_account(key, record)
            self._accounts[key] = record
            if self._state == 'unavailable':
                self._state = 'loaded'
                self._last_loaded_at = time.time()
        return self.account_policy(key)

    # -- writes ----------------------------------------------------------

    async def set_providers(self, tenant: str, providers) -> tuple:
        """Persist first, then update the cache.

        Updating the cache first would leave the enforced policy disagreeing
        with the stored one after a write failure.
        """
        if not isinstance(tenant, str) or not tenant.strip():
            raise ValueError('tenant must be a non-empty string')
        ordered = normalise_providers(providers)
        async with self._lock:
            async with asyncio.timeout(IO_TIMEOUT):
                await self._backend_write(tenant.strip(), ordered)
            self._policies[tenant.strip()] = ordered
            if self._state == 'unavailable':
                self._state = 'loaded'
                self._last_loaded_at = time.time()
        return ordered

    async def clear_providers(self, tenant: str) -> None:
        async with self._lock:
            async with asyncio.timeout(IO_TIMEOUT):
                await self._backend_delete(tenant)
            self._policies.pop(tenant, None)

    # -- backends --------------------------------------------------------

    async def _backend_load(self) -> dict:
        if self.backend == 'mysql':
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute('SELECT tenant, providers FROM api_key_policy')
                    rows = await cur.fetchall()
            return {row[0]: normalise_providers(json.loads(row[1])) for row in rows}
        if self.backend == 'json':
            if not self._path.exists():
                return {}
            raw = json.loads(self._path.read_text() or '{}')
            return {name: normalise_providers(value) for name, value in raw.items()}
        return dict(self._policies)

    async def _backend_load_preferences(self) -> dict:
        if self.backend == 'mysql':
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute('SELECT model, providers FROM model_provider_preference')
                    rows = await cur.fetchall()
            return {row[0]: ordered_providers(json.loads(row[1])) for row in rows}
        if self.backend == 'json':
            path = self._preferences_path()
            if not path.exists():
                return {}
            raw = json.loads(path.read_text() or '{}')
            return {model: ordered_providers(value) for model, value in raw.items()}
        return dict(self._preferences)

    async def _backend_write_preference(self, model: str, order: tuple) -> None:
        if self.backend == 'mysql':
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        'INSERT INTO model_provider_preference (model, providers) VALUES (%s, %s) '
                        'ON DUPLICATE KEY UPDATE providers = VALUES(providers)',
                        (model, json.dumps(list(order))))
            return
        if self.backend == 'json':
            merged = {**self.model_preferences_snapshot(), model: order}
            self._atomic_write_json(self._preferences_path(),
                                    {m: list(o) for m, o in merged.items()})

    async def _backend_delete_preference(self, model: str) -> None:
        if self.backend == 'mysql':
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute('DELETE FROM model_provider_preference WHERE model = %s', (model,))
            return
        if self.backend == 'json':
            merged = {m: o for m, o in self.model_preferences_snapshot().items() if m != model}
            self._atomic_write_json(self._preferences_path(),
                                    {m: list(o) for m, o in merged.items()})

    def _preferences_path(self):
        return self._path.parent / 'model_provider_preference.json'

    async def _backend_load_model_accounts(self) -> dict:
        if self.backend == 'mysql':
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute('SELECT model, accounts FROM model_copilot_account')
                    rows = await cur.fetchall()
            return {row[0]: normalise_accounts(json.loads(row[1])) for row in rows}
        if self.backend == 'json':
            path = self._model_accounts_path()
            if not path.exists():
                return {}
            raw = json.loads(path.read_text() or '{}')
            return {model: normalise_accounts(value) for model, value in raw.items()}
        return dict(self._model_accounts)

    async def _backend_write_model_accounts(self, model: str, accounts: tuple) -> None:
        if self.backend == 'mysql':
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        'INSERT INTO model_copilot_account (model, accounts) VALUES (%s, %s) '
                        'ON DUPLICATE KEY UPDATE accounts = VALUES(accounts)',
                        (model, json.dumps(list(accounts))))
            return
        if self.backend == 'json':
            merged = {**self.model_accounts_snapshot(), model: accounts}
            self._atomic_write_json(self._model_accounts_path(),
                                    {m: list(a) for m, a in merged.items()})

    async def _backend_delete_model_accounts(self, model: str) -> None:
        if self.backend == 'mysql':
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute('DELETE FROM model_copilot_account WHERE model = %s', (model,))
            return
        if self.backend == 'json':
            merged = {m: a for m, a in self.model_accounts_snapshot().items() if m != model}
            self._atomic_write_json(self._model_accounts_path(),
                                    {m: list(a) for m, a in merged.items()})

    def _model_accounts_path(self):
        return self._path.parent / 'model_copilot_account.json'

    async def _backend_load_panel_password(self) -> Optional[str]:
        if self.backend == 'mysql':
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute('SELECT verifier FROM panel_password WHERE id = 1')
                    row = await cur.fetchone()
            return row[0] if row else None
        if self.backend == 'json':
            path = self._panel_password_path()
            if not path.exists():
                return None
            raw = json.loads(path.read_text() or '{}')
            verifier = raw.get('verifier')
            return verifier if isinstance(verifier, str) and verifier else None
        return self._panel_password

    async def _backend_write_panel_password(self, verifier: str) -> None:
        if self.backend == 'mysql':
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        'INSERT INTO panel_password (id, verifier) VALUES (1, %s) '
                        'ON DUPLICATE KEY UPDATE verifier = VALUES(verifier)', (verifier,))
            return
        if self.backend == 'json':
            self._atomic_write_json(self._panel_password_path(), {'verifier': verifier})

    async def _backend_delete_panel_password(self) -> None:
        if self.backend == 'mysql':
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute('DELETE FROM panel_password WHERE id = 1')
            return
        if self.backend == 'json':
            path = self._panel_password_path()
            if path.exists():
                path.unlink()

    def _panel_password_path(self):
        return self._path.parent / 'panel_password.json'

    async def _backend_load_accounts(self) -> dict:
        if self.backend == 'mysql':
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute('SELECT account, enabled, allowed_models, capability, '
                                      'UNIX_TIMESTAMP(capability_at) FROM copilot_account_policy')
                    rows = await cur.fetchall()
            return {row[0]: self._decode_account(row[1], row[2], row[3], row[4]) for row in rows}
        if self.backend == 'json':
            path = self._accounts_path()
            if not path.exists():
                return {}
            raw = json.loads(path.read_text() or '{}')
            return {name: self._normalise_account(value) for name, value in raw.items()}
        return dict(self._accounts)

    @staticmethod
    def _decode_account(enabled, allowed, capability, capability_at) -> dict:
        allowed_list = json.loads(allowed) if allowed else None
        record = {
            'allowed_models': tuple(allowed_list) if allowed_list else None,
            'capability': tuple(json.loads(capability) if capability else ()),
            'capability_at': float(capability_at) if capability_at is not None else None,
        }
        # NULL is "no operator decision recorded"; it must survive a round trip
        # or a restart would turn an inert account into an enabled one.
        if enabled is not None:
            record['enabled'] = bool(enabled)
        return record

    @staticmethod
    def _normalise_account(value: dict) -> dict:
        allowed = value.get('allowed_models')
        record = {
            'allowed_models': tuple(allowed) if allowed else None,
            'capability': tuple(value.get('capability') or ()),
            'capability_at': value.get('capability_at'),
        }
        if value.get('enabled') is not None:
            record['enabled'] = bool(value['enabled'])
        return record

    async def _backend_write_account(self, account: str, record: dict) -> None:
        if self.backend == 'mysql':
            await self._ensure_pool()
            allowed = record.get('allowed_models')
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        'INSERT INTO copilot_account_policy '
                        '(account, enabled, allowed_models, capability, capability_at) '
                        'VALUES (%s, %s, %s, %s, FROM_UNIXTIME(%s)) '
                        'ON DUPLICATE KEY UPDATE enabled = VALUES(enabled), '
                        'allowed_models = VALUES(allowed_models), '
                        'capability = VALUES(capability), capability_at = VALUES(capability_at)',
                        (account, (1 if record['enabled'] else 0) if 'enabled' in record else None,
                         json.dumps(list(allowed)) if allowed else None,
                         json.dumps(list(record.get('capability') or ())),
                         record.get('capability_at')))
            return
        if self.backend == 'json':
            merged = {**{n: dict(v) for n, v in self._accounts.items()}, account: dict(record)}
            self._atomic_write_json(self._accounts_path(), {
                name: {'enabled': bool(value['enabled']) if 'enabled' in value else None,
                       'allowed_models': list(value['allowed_models']) if value.get('allowed_models') else None,
                       'capability': list(value.get('capability') or ()),
                       'capability_at': value.get('capability_at')}
                for name, value in merged.items()})

    def _accounts_path(self):
        return self._path.parent / 'copilot_account_policy.json'

    async def _backend_write(self, tenant: str, providers: tuple) -> None:
        if self.backend == 'mysql':
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        'INSERT INTO api_key_policy (tenant, providers) VALUES (%s, %s) '
                        'ON DUPLICATE KEY UPDATE providers = VALUES(providers)',
                        (tenant, json.dumps(list(providers))))
            return
        if self.backend == 'json':
            merged = {**self.snapshot(), tenant: providers}
            self._atomic_write(merged)

    async def _backend_delete(self, tenant: str) -> None:
        if self.backend == 'mysql':
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute('DELETE FROM api_key_policy WHERE tenant = %s', (tenant,))
            return
        if self.backend == 'json':
            merged = {name: value for name, value in self.snapshot().items() if name != tenant}
            self._atomic_write(merged)

    def _atomic_write(self, policies: dict) -> None:
        self._atomic_write_json(self._path, {name: list(value) for name, value in policies.items()})

    def _atomic_write_json(self, path, payload: dict) -> None:
        # Single writer only, same constraint the JSON usage backend documents.
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(payload, sort_keys=True, indent=1)
        handle, temp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
        try:
            with os.fdopen(handle, 'w') as fh:
                fh.write(body)
            os.replace(temp, path)
        except Exception:
            Path(temp).unlink(missing_ok=True)
            raise

    async def _ensure_pool(self):
        if self._pool is not None:
            return
        import ssl as _ssl

        import aiomysql
        self._pool = await aiomysql.create_pool(
            host=self._config.get('host', 'localhost'),
            port=self._config.get('port', 3306),
            user=self._config.get('user', 'root'),
            password=self._config.get('password', ''),
            db=self._config.get('database', 'claude_lb'),
            minsize=1,
            maxsize=2,
            autocommit=True,
            charset='utf8mb4',
            ssl=_ssl.create_default_context(),
            connect_timeout=IO_TIMEOUT,
        )
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS api_key_policy (
                        tenant VARCHAR(190) NOT NULL,
                        providers VARCHAR(255) NOT NULL,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        PRIMARY KEY (tenant)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS copilot_account_policy (
                        account VARCHAR(190) NOT NULL,
                        -- NULL = 运维尚未决定；启用态由 config 的初始声明兜底。
                        enabled TINYINT NULL DEFAULT NULL,
                        allowed_models TEXT NULL,
                        capability TEXT NULL,
                        capability_at TIMESTAMP NULL DEFAULT NULL,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        PRIMARY KEY (account)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_provider_preference (
                        model VARCHAR(190) NOT NULL,
                        providers VARCHAR(255) NOT NULL,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        PRIMARY KEY (model)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                # Its own table rather than a column on model_provider_preference:
                # that table already exists in production, so widening it would be
                # a real ALTER. A new table is created lazily like every other one
                # here, so this ships with no migration.
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_copilot_account (
                        model VARCHAR(190) NOT NULL,
                        -- 只存「限定到哪些账户」。空选择不落行 —— 删行就是回到基线。
                        accounts TEXT NOT NULL,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        PRIMARY KEY (model)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                # 单行表（id 恒为 1）。存的是**单向验证器**，不是密码，也不是
                # 可解开的密文 —— 面板密码只需要验证，所以哈希比封装严格更好：
                # 即使 LB_CREDENTIAL_KEY 泄露也推不出密码。没有行就是没有
                # overlay，于是回落到 env 基线。
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS panel_password (
                        id TINYINT NOT NULL,
                        verifier VARCHAR(255) NOT NULL,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        PRIMARY KEY (id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
