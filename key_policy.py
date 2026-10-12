"""Per-tenant provider allow-lists. Author: Zeno Ren.

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

MASK_MIN_LENGTH = 12


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
        except Exception as exc:
            self._last_error_type = type(exc).__name__
            # Never loaded → no restriction; loaded before → keep enforcing it.
            self._state = 'unavailable' if self._last_loaded_at is None else 'stale'
            logger.warning('key_policy: load failed (%s); state=%s', self._last_error_type, self._state)
            return self.status()
        self._policies = loaded
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
            'last_loaded_at': self._last_loaded_at,
            'last_error_type': self._last_error_type,
            'unrestricted_when_unavailable': True,
        }

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
        # Single writer only, same constraint the JSON usage backend documents.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({name: list(value) for name, value in policies.items()},
                             sort_keys=True, indent=1)
        handle, temp = tempfile.mkstemp(dir=str(self._path.parent), suffix='.tmp')
        try:
            with os.fdopen(handle, 'w') as fh:
                fh.write(payload)
            os.replace(temp, self._path)
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
