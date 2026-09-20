"""Monotonic inference budgets with protocol-correct timeout results. Author: Zeno Ren."""
import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
import json
import math
import os
import time

import httpx

TOTAL_TIMEOUT = float(os.getenv('INFERENCE_TOTAL_TIMEOUT_SECONDS', '1800'))
STARTUP_TIMEOUT = float(os.getenv('UPSTREAM_STARTUP_TIMEOUT_SECONDS', '180'))
if not all(math.isfinite(v) and v > 0 for v in (TOTAL_TIMEOUT, STARTUP_TIMEOUT)):
    raise ValueError('Inference and upstream startup deadlines must be finite positive seconds')

CURRENT_DEADLINE = ContextVar('inference_deadline', default=None)
HEADER_TIMER = ContextVar('upstream_header_timer', default=None)
ROUTES = {'/v1/messages':'messages', '/v1/responses':'responses', '/v1/chat/completions':'chat'}


class UpstreamStartupTimeout(httpx.ReadTimeout):
    """Headers did not arrive in budget; execution is unknown, never replay-safe."""


def remaining_seconds():
    deadline = CURRENT_DEADLINE.get()
    return max(0, deadline-time.monotonic()) if deadline is not None else None


def headers_received():
    timer = HEADER_TIMER.get()
    if timer is not None and not timer.expired():
        timer.reschedule(None)


@asynccontextmanager
async def startup_budget(seconds=None):
    timer = asyncio.timeout(STARTUP_TIMEOUT if seconds is None else seconds)
    token = HEADER_TIMER.set(timer)
    try:
        async with timer:
            yield
    except TimeoutError as exc:
        if timer.expired():
            raise UpstreamStartupTimeout('Upstream headers did not arrive within startup budget; execution may have occurred. Not replayed.') from exc
        raise
    finally:
        HEADER_TIMER.reset(token)


class RequestBudgetMiddleware:
    def __init__(self, app, *, error_frame_factory, timeout=None):
        self.app = app
        self.timeout = TOTAL_TIMEOUT if timeout is None else timeout
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('Request timeout must be finite and positive')
        self.error_frame_factory = error_frame_factory

    async def __call__(self, scope, receive, send):
        api = ROUTES.get(scope.get('path'))
        if scope['type'] != 'http' or scope.get('method') != 'POST' or api is None:
            return await self.app(scope,receive,send)
        attempted_start = False
        headers_sent = False
        body_complete = False
        is_sse = False

        async def observed_send(message):
            nonlocal attempted_start, headers_sent, body_complete, is_sse
            if message['type'] == 'http.response.start':
                attempted_start = True
                is_sse = dict(message.get('headers',[])).get(b'content-type',b'').lower().startswith(b'text/event-stream')
            await send(message)
            if message['type'] == 'http.response.start':
                headers_sent = True
            elif message['type'] == 'http.response.body' and not message.get('more_body',False):
                body_complete = True

        token = CURRENT_DEADLINE.set(time.monotonic()+self.timeout)
        timer = asyncio.timeout(self.timeout)
        try:
            async with timer:
                await self.app(scope,receive,observed_send)
        except TimeoutError:
            if not timer.expired():
                raise
            # A completed response must not receive another terminal during cleanup.
            if body_complete:
                return
            scope.setdefault('state',{})['lb_deadline_exceeded'] = True
            code = 'request_deadline_exceeded'
            message = 'Inference deadline exceeded; upstream execution may have occurred. Not replayed.'
            # Once response-start is attempted, never emit a second status line or
            # append SSE to a partial JSON response. Let the server close that case.
            if attempted_start and (not headers_sent or not is_sse):
                raise
            async with asyncio.timeout(1):
                if headers_sent:
                    await send({'type':'http.response.body',
                                'body':self.error_frame_factory(api,code,message), 'more_body':False})
                else:
                    body = json.dumps({'error':{'code':code,'message':message,
                                               'execution_certainty':'unknown','retryable':False}}).encode()
                    await send({'type':'http.response.start','status':504,
                                'headers':[(b'content-type',b'application/json'),(b'content-length',str(len(body)).encode())]})
                    await send({'type':'http.response.body','body':body,'more_body':False})
        finally:
            CURRENT_DEADLINE.reset(token)
