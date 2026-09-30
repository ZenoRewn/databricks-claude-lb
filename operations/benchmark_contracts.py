"""Synthetic microbenchmark of context observation on the candidate. Author: Zeno Ren."""
import argparse
import asyncio
import json
import logging
import math
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time


async def worker(mode, size, samples):
    import httpx
    import main
    import model_capabilities
    from safe_diagnostics import BoundedLogHandler
    class Sink(logging.Handler):
        def __init__(self):
            super().__init__(); self.events = self.bytes = 0
            self.setFormatter(main._JsonLogFormatter())
        def emit(self, record):
            self.events += 1; self.bytes += len(self.format(record).encode())
    sink = Sink(); handler = BoundedLogHandler(sink)
    old_handlers = logging.getLogger().handlers
    logging.getLogger().handlers = [handler]
    for old in old_handlers: old.close()
    model_capabilities.MODE = mode
    ep = main.WorkspaceEndpoint('synthetic', 'https://fixture.invalid', 'synthetic')
    proxy = main.ClaudeProxy(main.LoadBalancer([ep]), 'synthetic')
    hooks = proxy.client.event_hooks
    await proxy.client.aclose()
    calls = []
    async def upstream(request):
        calls.append(1)
        return httpx.Response(200, json={'type': 'message', 'stop_reason': 'end_turn',
            'content': [{'type': 'text', 'text': 'synthetic'}], 'usage': {'input_tokens': 1, 'output_tokens': 1}})
    proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream), event_hooks=hooks, trust_env=False)
    main.proxy, main.azure_proxy, main.copilot_proxy, main.usage_store = proxy, None, None, None
    payload = {'model': 'claude-opus-5', 'messages': [{'role': 'user', 'content': 'x' * size}]}
    durations = []
    cpu, began = time.process_time(), time.perf_counter()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app), base_url='http://local', trust_env=False) as client:
            async def request():
                start = time.perf_counter()
                response = await client.post('/v1/messages', json=payload, headers={'Authorization': 'Bearer synthetic'})
                if response.status_code != 200 or response.json().get('stop_reason') != 'end_turn':
                    raise RuntimeError('Synthetic generation did not complete')
                durations.append(time.perf_counter()-start)
            for offset in range(0, samples, 4):
                await asyncio.gather(*(request() for _ in range(min(4, samples-offset))))
        elapsed, cpu = time.perf_counter()-began, time.process_time()-cpu
        if not handler.drain(5): raise RuntimeError('Synthetic sink failed to drain')
        if len(calls) != samples or ep.active_requests: raise RuntimeError('Unexpected sends or leaked admission')
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {'mode': mode, 'payload_text_bytes': size, 'samples': samples, 'concurrency': 4,
                'completed': samples, 'upstream_sends': len(calls), 'active_after': ep.active_requests,
                'p50_ms': statistics.median(durations)*1000,
                'p95_ms': sorted(durations)[max(0, math.ceil(len(durations)*.95)-1)]*1000,
                'elapsed_seconds': elapsed, 'cpu_seconds': cpu,
                'process_peak_rss_bytes': peak if sys.platform == 'darwin' else peak*1024,
                'log_events': sink.events, 'log_bytes': sink.bytes, 'diagnostic_drops': dict(handler.dropped)}
    finally:
        await proxy.close(); handler.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('off', 'observe'))
    parser.add_argument('--bytes', type=int, default=4096)
    parser.add_argument('--samples', type=int, default=64)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(asyncio.run(worker(args.worker, args.bytes, args.samples)), allow_nan=False))
    else:
        results = []
        for size, samples in ((4096, 64), (2*1024*1024, 16)):
            for mode in ('off', 'observe'):
                run = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', mode,
                                      '--bytes', str(size), '--samples', str(samples)],
                                     capture_output=True, text=True, check=True, timeout=60)
                results.append(json.loads(run.stdout))
        print(json.dumps({'author': 'Zeno Ren', 'scope': 'synthetic candidate microbenchmark',
            'comparison': 'Context observation off versus observe; both retain the new diagnostics and phase tracking.',
            'limits': 'MockTransport/ASGI, no real provider, no production capacity or latency claim; fresh process per arm.',
            'results': results}, indent=2, allow_nan=False))
