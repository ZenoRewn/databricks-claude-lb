"""Bound upstream diagnostic bodies before HTTPX buffers them. Author: Zeno Ren."""
import asyncio
from collections import Counter
import json
import math
import os
import zlib

import httpx

MAX_BYTES = int(os.getenv('UPSTREAM_ERROR_BODY_MAX_BYTES', '65536'))
TIMEOUT = float(os.getenv('UPSTREAM_ERROR_BODY_TIMEOUT_SECONDS', '2'))
if not 256 <= MAX_BYTES <= 16 * 1024 * 1024 or not math.isfinite(TIMEOUT) or TIMEOUT <= 0:
    raise ValueError('Upstream error body limits must be finite and positive (256 bytes to 16 MiB)')
RESULTS = Counter()


class BodyLimit(Exception):
    def __init__(self, reason):
        self.reason = reason


async def _read(response, max_bytes):
    # Preconsumed responses occur with MockTransport and some low-level callers.
    if isinstance(response, httpx.Response) and hasattr(response, '_content'):
        if len(response.content) > max_bytes:
            raise BodyLimit('too_large')
        return response.content
    if not isinstance(response, httpx.Response):
        body=await response.aread()
        if len(body) > max_bytes:
            raise BodyLimit('too_large')
        return body
    encoding=response.headers.get('content-encoding','').strip().lower()
    if encoding not in ('','identity','gzip','deflate'):
        raise BodyLimit('unsupported_encoding')
    decoder = zlib.decompressobj(16+zlib.MAX_WBITS if encoding=='gzip' else zlib.MAX_WBITS) if encoding in ('gzip','deflate') else None
    body=bytearray()
    encoded=0
    first=True
    async for raw in response.aiter_raw(chunk_size=8192):
        encoded += len(raw)
        if encoded > max(max_bytes*2,16384):
            raise BodyLimit('too_large')
        if decoder is None:
            if len(raw) > max_bytes-len(body):
                raise BodyLimit('too_large')
            body.extend(raw)
            continue
        pending=raw
        while pending:
            if decoder.eof:
                if encoding != 'gzip':
                    raise BodyLimit('invalid_encoding')
                decoder=zlib.decompressobj(16+zlib.MAX_WBITS)
            try:
                decoded=decoder.decompress(pending,max_bytes-len(body)+1)
            except zlib.error:
                if encoding=='deflate' and first:
                    decoder=zlib.decompressobj(-zlib.MAX_WBITS)
                    decoded=decoder.decompress(pending,max_bytes-len(body)+1)
                else:
                    raise
            first=False
            if len(decoded)>max_bytes-len(body):
                raise BodyLimit('too_large')
            body.extend(decoded)
            pending=decoder.unused_data if decoder.eof else decoder.unconsumed_tail
    if decoder is not None and not decoder.eof:
        raise BodyLimit('invalid_encoding')
    return bytes(body)


async def read_error_body(response, *, max_bytes=MAX_BYTES, timeout=TIMEOUT, close=None):
    """Cache a bounded decoded body; preserve status/Retry-After, discard unsafe prefixes.

    HTTPX's response hook runs before its automatic aread(). Setting _content is
    the same cache operation as Response.aread; this is covered by a real HTTPX
    post() test. The caller may supply its existing shielded cleanup owner.
    Cleanup is outside the read deadline and is never abandoned by that deadline.
    """
    if isinstance(response,httpx.Response) and response.extensions.get('lb_error_body_guarded'):
        return response.content
    reason='complete'
    try:
        async with asyncio.timeout(timeout):
            body=await _read(response,max_bytes)
    except TimeoutError:
        reason='timeout'
    except BodyLimit as exc:
        reason=exc.reason
    except (httpx.HTTPError,zlib.error):
        reason='read_error'
    finally:
        if close is not None:
            await close(response)
        else:
            await response.aclose()
    if reason != 'complete':
        # Never let a partial error string trigger token/state recovery policies.
        message=f'Upstream error body discarded: {reason}.'
        if response.headers.get('content-type','').lower().startswith('text/html'):
            body=f'<html>{message}</html>'.encode()
        else:
            body=json.dumps({'error':{'code':f'upstream_error_body_{reason}',
                                     'message':message,'upstream_status':response.status_code}}).encode()
    if isinstance(response,httpx.Response):
        response._content=body
        if reason != 'complete':
            response.encoding='utf-8'
        response.extensions['lb_error_body_guarded']=reason
    RESULTS[reason]+=1
    return body


async def error_body_hook(response, **kwargs):
    if response.status_code >= 400 or response.headers.get('content-type','').lower().startswith('text/html'):
        await read_error_body(response,**kwargs)


def render_metrics():
    name='lb_upstream_error_body_reads_total'
    lines=[f'# HELP {name} Bounded diagnostic body reads; no inference replay',f'# TYPE {name} counter']
    for reason in ('complete','timeout','too_large','unsupported_encoding','invalid_encoding','read_error'):
        lines.append(f'{name}{{result="{reason}"}} {RESULTS[reason]}')
    return '\n'.join(lines)+'\n'
