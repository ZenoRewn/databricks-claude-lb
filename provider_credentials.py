"""Operator-entered upstream credentials, sealed before storage. Author: Zeno Ren.

This is a deliberate posture change. The gateway used to never store upstream
credentials — it read them from config, env or a mounted Secret. It now accepts
and persists them, because on AKS the Secret mount is read-only tmpfs,
config.yaml is read-only, and the application's ServiceAccount has no secret
write permission. The usage database is the only writable durable store it has.

Because of that, sealing is mandatory rather than optional:

- AES-256-GCM with a key supplied out of band via ``LB_CREDENTIAL_KEY``, so the
  key lives in the Kubernetes Secret and not in the database that holds the
  ciphertext. A database dump alone is not enough.
- A fresh 12-byte nonce per write; identical secrets never produce identical
  ciphertext.
- ``provider|account`` as additional authenticated data, so a ciphertext cannot
  be relocated to another row and opened there.
- Without a key the store refuses to write at all. Quietly persisting plaintext
  tokens next to the usage ledger would contradict the rest of this project,
  where even release receipts carry only fingerprints.

What sealing does **not** solve, and must not be claimed: the running process
holds plaintext in memory, so anyone able to exec in the pod can still read it,
and losing the key makes stored credentials unrecoverable — there is no managed
KMS and no automatic rotation here.
"""
import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

IO_TIMEOUT = 10
KEY_ENV = 'LB_CREDENTIAL_KEY'
NONCE_BYTES = 12
KEY_BYTES = 32

PROVIDERS = ('databricks', 'azure_openai', 'copilot')

# Which field of an endpoint spec the sealed secret becomes, per provider.
SECRET_FIELD = {'databricks': 'token', 'azure_openai': 'api_key', 'copilot': 'github_token'}


class CredentialKeyError(RuntimeError):
    """The sealing key is absent or unusable."""


class CredentialSealError(RuntimeError):
    """Ciphertext failed authentication: wrong key, wrong row, or tampered."""


class DeviceFlowExpired(RuntimeError):
    """The device authorisation window closed before it completed."""


def _key_bytes(key: Optional[str]) -> bytes:
    if not key or not str(key).strip():
        raise CredentialKeyError(f'{KEY_ENV} is not set')
    try:
        raw = base64.b64decode(str(key).strip(), validate=True)
    except Exception as exc:
        raise CredentialKeyError(f'{KEY_ENV} must be base64') from exc
    if len(raw) != KEY_BYTES:
        raise CredentialKeyError(f'{KEY_ENV} must decode to {KEY_BYTES} bytes, got {len(raw)}')
    return raw


def key_fingerprint(key: Optional[str]) -> str:
    """Identify which key sealed a record without revealing the key."""
    return hashlib.sha256(b'lb-credential-key:' + _key_bytes(key)).hexdigest()[:12]


def secret_fingerprint(secret: str) -> str:
    return hashlib.sha256((secret or '').encode()).hexdigest()[:12]


def _aad(provider: str, account: str) -> bytes:
    return f'{provider}|{account}'.encode()


def seal(key: Optional[str], provider: str, account: str, secret: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    raw = _key_bytes(key)
    nonce = secrets.token_bytes(NONCE_BYTES)
    ciphertext = AESGCM(raw).encrypt(nonce, (secret or '').encode(), _aad(provider, account))
    return base64.b64encode(nonce + ciphertext).decode()


def unseal(key: Optional[str], provider: str, account: str, sealed: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    raw = _key_bytes(key)
    try:
        blob = base64.b64decode(sealed, validate=True)
    except Exception as exc:
        raise CredentialSealError('sealed value is not base64') from exc
    if len(blob) <= NONCE_BYTES:
        raise CredentialSealError('sealed value is too short')
    try:
        plaintext = AESGCM(raw).decrypt(blob[:NONCE_BYTES], blob[NONCE_BYTES:], _aad(provider, account))
    except Exception as exc:
        # Never echo the cause: it can carry cipher internals.
        raise CredentialSealError('authentication failed: wrong key, wrong record, or tampered') from exc
    return plaintext.decode()


class DeviceFlowRegistry:
    """In-process device authorisation flows.

    The ``device_code`` is the bearer of the pending authorisation, so it stays
    here only — never stored, never logged, never returned to a caller.
    """

    def __init__(self):
        self._flows: dict = {}

    def start(self, payload: dict) -> str:
        handle = uuid.uuid4().hex
        self._flows[handle] = {
            'device_code': payload['device_code'],
            'user_code': payload.get('user_code'),
            'verification_uri': payload.get('verification_uri'),
            'interval': int(payload.get('interval') or 5),
            'expires_at': time.time() + int(payload.get('expires_in') or 0),
            'started_at': time.time(),
        }
        return handle

    def view(self, handle: str) -> dict:
        flow = self._flows[handle]
        return {
            'handle': handle,
            'user_code': flow['user_code'],
            'verification_uri': flow['verification_uri'],
            'interval': flow['interval'],
            'expires_in': max(0, int(flow['expires_at'] - time.time())),
        }

    def device_code(self, handle: str) -> str:
        flow = self._flows[handle]
        if time.time() >= flow['expires_at']:
            self._flows.pop(handle, None)
            raise DeviceFlowExpired('device authorisation window closed; start a new flow')
        return flow['device_code']

    def finish(self, handle: str) -> None:
        self._flows.pop(handle, None)

    def prune(self) -> None:
        now = time.time()
        for handle in [h for h, f in self._flows.items() if now >= f['expires_at']]:
            self._flows.pop(handle, None)


class CredentialStore:
    """Sealed upstream credentials, materialised into endpoints at startup only."""

    def __init__(self, storage_config: Optional[dict] = None):
        config = storage_config or {}
        self._config = config
        requested = config.get('type', 'memory')
        self.backend = requested if requested in ('mysql', 'json') else 'memory'
        self._records: dict = {}
        self._state = 'unavailable'
        self._last_error_type: Optional[str] = None
        self._pool = None
        self._lock = asyncio.Lock()
        self._key = os.environ.get(KEY_ENV) or None
        self._key_fingerprint: Optional[str] = None
        if self._key:
            try:
                self._key_fingerprint = key_fingerprint(self._key)
            except CredentialKeyError as exc:
                logger.error('credentials: %s — entry disabled', exc)
                self._key = None
        self._path = (Path(config.get('path') or './usage_data') / 'provider_credentials.json'
                      if self.backend == 'json' else None)

    # -- lifecycle -------------------------------------------------------

    @property
    def sealing_available(self) -> bool:
        return self._key is not None

    async def start(self):
        if self.backend == 'memory':
            self._state = 'loaded'
            return
        try:
            async with asyncio.timeout(IO_TIMEOUT):
                self._records = await self._backend_load()
            self._state = 'loaded'
        except Exception as exc:
            self._last_error_type = type(exc).__name__
            self._state = 'unavailable'
            logger.warning('credentials: load failed (%s); stored credentials are not in use',
                           self._last_error_type)
            return
        # A stored credential that cannot be opened silently removes an endpoint,
        # so say so loudly rather than letting the account just disappear.
        unopenable = [f"{r['provider']}/{r['account']}" for r in self._records.values()
                      if self._key_state(r) != 'ok']
        if unopenable:
            logger.warning('credentials: %d record(s) cannot be unsealed with the current key '
                           '(%s); their endpoints will NOT be created',
                           len(unopenable), ', '.join(sorted(unopenable)[:5]))

    async def close(self):
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    def status(self) -> dict:
        return {
            'state': self._state,
            'backend': self.backend,
            'key': 'present' if self._key else 'missing',
            'key_fingerprint': self._key_fingerprint,
            'records': len(self._records),
            'unopenable': sum(1 for r in self._records.values() if self._key_state(r) != 'ok'),
            'last_error_type': self._last_error_type,
            'activation_requires_restart': True,
        }

    # -- reads -----------------------------------------------------------

    def _key_state(self, record: dict) -> str:
        if not self._key:
            return 'no_key'
        if record.get('key_fingerprint') != self._key_fingerprint:
            return 'sealed_with_other_key'
        return 'ok'

    def listing(self) -> list:
        """Identity only. Never includes the sealed blob or the plaintext."""
        items = []
        for record in self._records.values():
            items.append({
                'provider': record['provider'],
                'account': record['account'],
                'secret_fingerprint': record.get('secret_fingerprint'),
                'key_state': self._key_state(record),
                'meta': dict(record.get('meta') or {}),
                'validated_at': record.get('validated_at'),
                'validation': record.get('validation'),
                'updated_at': record.get('updated_at'),
            })
        return sorted(items, key=lambda i: (i['provider'], i['account']))

    def resolve(self, provider: str, account: str) -> Optional[str]:
        """Plaintext for internal use (endpoint construction, validation)."""
        record = self._records.get((provider, account))
        if not record or self._key_state(record) != 'ok':
            return None
        try:
            return unseal(self._key, provider, account, record['sealed'])
        except CredentialSealError:
            return None

    def endpoint_specs(self, provider: str) -> list:
        """Endpoint kwargs for every openable record of one provider.

        Records that cannot be unsealed are skipped — never substituted with a
        placeholder, which would produce an endpoint that fails every request.
        """
        specs = []
        for (stored_provider, account), record in sorted(self._records.items()):
            if stored_provider != provider:
                continue
            secret = self.resolve(provider, account)
            if secret is None:
                continue
            meta = dict(record.get('meta') or {})
            spec = {'name': account, SECRET_FIELD[provider]: secret,
                    'weight': int(meta.get('weight') or 1)}
            if provider == 'databricks':
                spec['api_base'] = meta.get('api_base')
            elif provider == 'azure_openai':
                spec['api_base'] = meta.get('api_base')
                spec['deployments'] = list(meta.get('deployments') or [])
            specs.append(spec)
        return specs

    # -- writes ----------------------------------------------------------

    async def put(self, provider: str, account: str, secret: str, meta: Optional[dict] = None,
                  *, validation: Optional[dict] = None) -> dict:
        if provider not in PROVIDERS:
            raise ValueError(f'unknown provider: {provider!r}')
        if not isinstance(account, str) or not account.strip():
            raise ValueError('account must be a non-empty string')
        if not isinstance(secret, str) or not secret.strip():
            raise ValueError('secret must be a non-empty string')
        if not self.sealing_available:
            raise CredentialKeyError(
                f'{KEY_ENV} is not configured; refusing to store an unsealed credential')
        account = account.strip()
        record = {
            'provider': provider,
            'account': account,
            'sealed': seal(self._key, provider, account, secret),
            'key_fingerprint': self._key_fingerprint,
            'secret_fingerprint': secret_fingerprint(secret),
            'meta': dict(meta or {}),
            'validated_at': time.time() if validation else None,
            'validation': validation,
            'updated_at': time.time(),
        }
        async with self._lock:
            async with asyncio.timeout(IO_TIMEOUT):
                await self._backend_write(record)
            self._records[(provider, account)] = record
        return self.listing_for(provider, account)

    def listing_for(self, provider: str, account: str) -> dict:
        return next((i for i in self.listing()
                     if i['provider'] == provider and i['account'] == account), {})

    async def delete(self, provider: str, account: str) -> None:
        async with self._lock:
            async with asyncio.timeout(IO_TIMEOUT):
                await self._backend_delete(provider, account)
            self._records.pop((provider, account), None)

    async def note_validation(self, provider: str, account: str, validation: dict) -> dict:
        record = self._records.get((provider, account))
        if not record:
            raise KeyError(f'{provider}/{account}')
        updated = {**record, 'validation': validation, 'validated_at': time.time()}
        async with self._lock:
            async with asyncio.timeout(IO_TIMEOUT):
                await self._backend_write(updated)
            self._records[(provider, account)] = updated
        return self.listing_for(provider, account)

    # -- backends --------------------------------------------------------

    async def _backend_load(self) -> dict:
        if self.backend == 'mysql':
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute('SELECT provider, account, sealed, key_fingerprint, '
                                      'secret_fingerprint, meta, UNIX_TIMESTAMP(validated_at), '
                                      'validation, UNIX_TIMESTAMP(updated_at) '
                                      'FROM provider_credential')
                    rows = await cur.fetchall()
            return {(r[0], r[1]): {
                'provider': r[0], 'account': r[1], 'sealed': r[2], 'key_fingerprint': r[3],
                'secret_fingerprint': r[4], 'meta': json.loads(r[5]) if r[5] else {},
                'validated_at': float(r[6]) if r[6] is not None else None,
                'validation': json.loads(r[7]) if r[7] else None,
                'updated_at': float(r[8]) if r[8] is not None else None} for r in rows}
        if self.backend == 'json':
            if not self._path.exists():
                return {}
            raw = json.loads(self._path.read_text() or '{}')
            return {(v['provider'], v['account']): v for v in raw.values()}
        return dict(self._records)

    async def _backend_write(self, record: dict) -> None:
        if self.backend == 'mysql':
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        'INSERT INTO provider_credential (provider, account, sealed, '
                        'key_fingerprint, secret_fingerprint, meta, validated_at, validation) '
                        'VALUES (%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),%s) '
                        'ON DUPLICATE KEY UPDATE sealed=VALUES(sealed), '
                        'key_fingerprint=VALUES(key_fingerprint), '
                        'secret_fingerprint=VALUES(secret_fingerprint), meta=VALUES(meta), '
                        'validated_at=VALUES(validated_at), validation=VALUES(validation)',
                        (record['provider'], record['account'], record['sealed'],
                         record['key_fingerprint'], record['secret_fingerprint'],
                         json.dumps(record.get('meta') or {}), record.get('validated_at'),
                         json.dumps(record['validation']) if record.get('validation') else None))
            return
        if self.backend == 'json':
            merged = {**{f"{p}/{a}": v for (p, a), v in self._records.items()},
                      f"{record['provider']}/{record['account']}": record}
            self._atomic_write(merged)

    async def _backend_delete(self, provider: str, account: str) -> None:
        if self.backend == 'mysql':
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute('DELETE FROM provider_credential WHERE provider=%s AND account=%s',
                                      (provider, account))
            return
        if self.backend == 'json':
            merged = {f'{p}/{a}': v for (p, a), v in self._records.items()
                      if (p, a) != (provider, account)}
            self._atomic_write(merged)

    def _atomic_write(self, records: dict) -> None:
        import tempfile
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(records, sort_keys=True, indent=1)
        handle, temporary = tempfile.mkstemp(dir=str(self._path.parent), suffix='.tmp')
        try:
            with os.fdopen(handle, 'w') as fh:
                fh.write(payload)
            os.chmod(temporary, 0o600)
            os.replace(temporary, self._path)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
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
            minsize=1, maxsize=2, autocommit=True, charset='utf8mb4',
            ssl=_ssl.create_default_context(), connect_timeout=IO_TIMEOUT)
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS provider_credential (
                        provider VARCHAR(32) NOT NULL,
                        account VARCHAR(190) NOT NULL,
                        sealed TEXT NOT NULL,
                        key_fingerprint CHAR(12) NOT NULL,
                        secret_fingerprint CHAR(12) NOT NULL,
                        meta TEXT NULL,
                        validated_at TIMESTAMP NULL DEFAULT NULL,
                        validation TEXT NULL,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        PRIMARY KEY (provider, account)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
