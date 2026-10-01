"""Weekly price refresh contracts. Author: Zeno Ren."""
import asyncio
from datetime import date, datetime, timedelta, timezone
import json
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

import copilot_pricing as p

HTML = (Path(__file__).parent / 'fixtures/copilot-pricing-20261001.html').read_text()
NOW = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)


def async_test(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return run


@pytest.fixture(autouse=True)
def isolate_catalog():
    with patch.object(p, '_active', None, create=True), patch.object(p, '_refresh', None, create=True):
        yield


def test_official_snapshot_adds_six_models_and_preserves_boundaries():
    catalog = p.parse_pricing_html(HTML, '2026-10-01')
    assert len(catalog['models']) == 35
    assert len(catalog['long_context']) == 12
    assert catalog['models'] == p.MODELS
    assert catalog['long_context'] == p.LONG_CONTEXT
    assert catalog['promotion_end'] == p.PROMOTION_END
    p.install_catalog(catalog)
    for name in ('gpt-6-luna', 'gpt-6-sol', 'gpt-6.1-sol', 'claude-opus-5.5', 'claude-sonnet-5.5', 'grok-4.7'):
        assert p.get_pricing(name)
    assert p.get_pricing('gpt-6.1-sol', 272000)['cache_read'] == .1
    assert p.get_pricing('GPT-6-1-SOL', 272001)['cache_read'] == .2
    assert p.get_pricing('grok-4.7', 200001)['input'] == 4
    assert p.get_pricing('claude-opus-4.8-fast')['input'] == 10
    assert p.get_pricing('gemini-3.8-flash', at=date(2027, 1, 1)) is None
    assert p.get_pricing('gpt-6.1-sol-pro') is None


@pytest.mark.parametrize('old,new', [
    ('$0.125', '$NaN'), ('$0.125', '$-1'), ('$0.125', 'TBD'),
    ('per 1 million tokens', 'per 1 thousand tokens'),
    ('1 AI credit = $0.01 USD', '1 AI credit = $1 USD'),
    ('&gt; 272K', '&gt; 300K'), ('Long context', 'Special tier'),
    ('December 31, 2026', 'until further notice'),
    ('#user-content-fn-gemini-flash-promo', '#user-content-fn-new-rule'),
    ('<td>GPT-6 Luna</td>', '<td>GPT-6 Sol</td>'),
    ('Cache write', 'New billing dimension'),
    ('data-footnote-ref=""', 'data-changed-ref=""'),
    ('All prices are', 'Some prices are'),
])
def test_ambiguous_pages_are_rejected(old, new):
    assert old in HTML
    with pytest.raises(ValueError):
        p.parse_pricing_html(HTML.replace(old, new), '2026-10-01')


def test_missing_table_and_truncated_page_are_rejected():
    import re
    partial = re.sub(r'<table aria-labelledby="xai".*?</table>', '', HTML, flags=re.S)
    for text in (partial, HTML[:HTML.index('</table>')], '<html>Service unavailable</html>'):
        with pytest.raises(ValueError):
            p.parse_pricing_html(text, '2026-10-01')


def test_price_updates_do_not_revalue_existing_estimates():
    stats = {'requests': 1}
    p.record_estimate(stats, 'gpt-5-mini', 1000000, 0, 0, 0)
    before = p.cost_view(stats)
    p.install_catalog(p.parse_pricing_html(HTML.replace('$0.25', '$0.30'), '2026-10-01'))
    assert p.cost_view(stats) == before
    assert p.estimate_request('gpt-5-mini', 1000000, 0)['estimated_cost_usd'] == .30


@async_test
async def test_refresh_persists_and_new_instance_waits_until_seven_days(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        assert str(request.url) == p.SOURCE_URL
        assert 'authorization' not in request.headers
        return httpx.Response(200, text=HTML, headers={'content-type': 'text/html'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        worker = p.PricingRefresh(tmp_path / 'prices.json', clock=lambda: NOW)
        assert await worker.refresh_once(client)
        assert len(calls) == 1
        assert worker.next_due == NOW + timedelta(days=7)
        cached = json.loads(worker.cache_path.read_text())
        assert cached['source_url'] == p.SOURCE_URL
        again = p.PricingRefresh(worker.cache_path, clock=lambda: NOW + timedelta(days=1))
        await again.load_cache()
        assert again.next_due == NOW + timedelta(days=7)
        assert p.get_metadata()['checked_on'] == '2026-10-01'
        assert p.get_metadata()['refresh']['last_success_at'] == NOW.isoformat()
        assert p.get_metadata()['refresh']['stale'] is False


@async_test
@pytest.mark.parametrize('failure', ['http', 'html', 'disk', 'oversize', 'redirect'])
async def test_refresh_failure_retains_memory_and_cache(tmp_path, failure):
    worker = p.PricingRefresh(tmp_path / 'prices.json', clock=lambda: NOW)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=HTML))) as client:
        assert await worker.refresh_once(client)
    old = worker.cache_path.read_bytes()
    old_price = p.get_pricing('gpt-6.1-sol')
    response = {
        'http': httpx.Response(503), 'html': httpx.Response(200, text='broken'),
        'disk': httpx.Response(200, text=HTML.replace('$0.125', '$0.130')),
        'oversize': httpx.Response(200, content=b'x' * (p.MAX_PRICING_BYTES + 1)),
        'redirect': httpx.Response(302, headers={'location': 'https://example.invalid'}),
    }[failure]
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: response)) as client:
        if failure == 'disk':
            with patch.object(p, '_save_cache', side_effect=OSError('sensitive local path')):
                assert await worker.refresh_once(client) is False
        else:
            assert await worker.refresh_once(client) is False
    assert p.get_pricing('gpt-6.1-sol') == old_price
    assert worker.cache_path.read_bytes() == old
    assert worker.next_due == NOW + timedelta(hours=1)
    assert worker.status()['last_error']
    assert 'sensitive' not in json.dumps(worker.status())


@async_test
@pytest.mark.parametrize('bad', ['broken', 'future', 'source', 'hash'])
async def test_invalid_cache_is_not_installed(tmp_path, bad):
    worker = p.PricingRefresh(tmp_path / 'prices.json', clock=lambda: NOW)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=HTML))) as client:
        await worker.refresh_once(client)
    value = json.loads(worker.cache_path.read_text())
    if bad == 'future': value['fetched_at'] = (NOW + timedelta(days=1)).isoformat()
    if bad == 'source': value['source_url'] = 'https://example.invalid'
    if bad == 'hash': value['html'] = value['html'].replace('$0.125', '$9.125')
    worker.cache_path.write_text('broken' if bad == 'broken' else json.dumps(value))
    p._active = None
    fresh = p.PricingRefresh(worker.cache_path, clock=lambda: NOW)
    await fresh.load_cache()
    assert p._active is None
    assert fresh.next_due == NOW
    assert fresh.status()['last_error']


@async_test
async def test_cancelled_fetch_leaves_previous_catalog_and_cache(tmp_path):
    started = asyncio.Event()
    async def wait(request):
        started.set()
        await asyncio.Event().wait()
    worker = p.PricingRefresh(tmp_path / 'prices.json', clock=lambda: NOW)
    async with httpx.AsyncClient(transport=httpx.MockTransport(wait)) as client:
        task = asyncio.create_task(worker.refresh_once(client))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
    assert p._active is None
    assert not worker.cache_path.exists()


@async_test
async def test_disabled_worker_does_not_fetch(tmp_path):
    worker = p.PricingRefresh(tmp_path / 'prices.json', enabled=False, clock=lambda: NOW)
    with patch.object(p, 'httpx') as network:
        await worker.run()
        network.AsyncClient.assert_not_called()
    assert worker.status()['enabled'] is False


@async_test
async def test_run_waits_for_weekly_schedule_and_recovers_after_failure(tmp_path):
    now = NOW
    worker = p.PricingRefresh(tmp_path / 'prices.json', clock=lambda: now)
    calls, sleeps = [], []
    def response(request):
        calls.append(request)
        return httpx.Response(503) if len(calls) == 2 else httpx.Response(200, text=HTML)
    async def sleep(seconds):
        nonlocal now
        sleeps.append(seconds)
        if len(sleeps) == 4:
            raise asyncio.CancelledError()
        now += timedelta(seconds=seconds)
    client = httpx.AsyncClient(transport=httpx.MockTransport(response))
    with patch.object(p.httpx, 'AsyncClient', return_value=client), patch.object(p.asyncio, 'sleep', side_effect=sleep):
        with pytest.raises(asyncio.CancelledError): await worker.run()
    assert sleeps == [0, 7 * 86400, 3600, 7 * 86400]
    assert len(calls) == 3
    assert worker.status()['last_error'] is None


@async_test
async def test_stale_cache_is_used_while_refresh_is_due(tmp_path):
    worker = p.PricingRefresh(tmp_path / 'prices.json', clock=lambda: NOW)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=HTML))) as client:
        await worker.refresh_once(client)
    old = p.PricingRefresh(worker.cache_path, clock=lambda: NOW + timedelta(days=8))
    await old.load_cache()
    assert old.status()['stale'] is True
    assert old.next_due < old.clock()
    assert p.get_pricing('gpt-6.1-sol')['input'] == 2


@async_test
async def test_runtime_owns_and_cancels_pricing_worker(tmp_path):
    import main
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    started, stopped = asyncio.Event(), asyncio.Event()
    async def run():
        started.set()
        try: await asyncio.Event().wait()
        finally: stopped.set()
    base = SimpleNamespace(load_balancer=SimpleNamespace(endpoints=[]), close=AsyncMock())
    cp = SimpleNamespace(load_balancer=SimpleNamespace(endpoints=[]), close=AsyncMock(),
                         warmup=AsyncMock(), background_refresh_loop=AsyncMock(), connection_monitor_loop=AsyncMock())
    store = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), get_today_data=lambda: {})
    with patch.object(main, 'load_config', return_value=(base, None, cp, {'type': 'json', 'path': str(tmp_path)})), \
            patch.object(main, 'create_usage_store', return_value=store), patch('otel_setup.setup_tracing'), \
            patch.object(main, 'PricingRefresh') as factory, patch.object(main, 'proxy', None), \
            patch.object(main, 'azure_proxy', None), patch.object(main, 'copilot_proxy', None), patch.object(main, 'usage_store', None), \
            patch.dict('os.environ', {'COPILOT_PRICING_AUTO_REFRESH': 'true', 'COPILOT_PRICING_CACHE_PATH': str(tmp_path / 'prices.json')}):
        factory.return_value.run.side_effect = run
        async with main.lifespan(main.app):
            await asyncio.wait_for(started.wait(), 1)
            assert not stopped.is_set()
            factory.assert_called_once_with(str(tmp_path / 'prices.json'), enabled=True)
        assert stopped.is_set()
        cp.close.assert_awaited_once()
