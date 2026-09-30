"""Local synthetic Dashboard preview, with no provider or database access. Author: Zeno Ren."""
import argparse
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


def model(requests=620, cost=12.84):
    return dict(requests=requests, input_tokens=3_840_000, output_tokens=740_000,
                cache_creation_tokens=180_000, cache_read_tokens=1_960_000,
                estimated_cost_usd=cost, estimated_ai_credits=cost * 100 if cost is not None else None,
                known_cost_subtotal_usd=cost or 0, priced_requests=requests if cost is not None else 0,
                unpriced_requests=0 if cost is not None else requests)


def totals(models):
    return {key: sum(item.get(key, 0) for item in models.values()) for key in
            ('requests', 'input_tokens', 'output_tokens', 'cache_creation_tokens', 'cache_read_tokens')}


def endpoint(name, models, *, state='CLOSED', active=2, latency=2400):
    return dict(name=name, circuit_state=state, circuit_open=state != 'CLOSED', active_requests=active,
                total_requests=sum(m['requests'] for m in models.values()), error_rate=0.4 if state == 'CLOSED' else 6.2,
                avg_response_time_ms=latency, total_tokens=4_580_000, model_stats=models,
                deployments=list(models), models=list(models), session_token_expires_at=time.time() + 72_000)


def stats():
    summary = dict(total_requests=4280, total_input_tokens=11_520_000, total_output_tokens=2_220_000,
                   total_tokens=13_740_000, total_cache_creation_tokens=540_000, total_cache_read_tokens=5_880_000,
                   avg_response_time_ms=2870, requests_per_minute=1.4, estimated_total_cost_usd=42.6, uptime_seconds=183_620)
    db = [endpoint('workspace-us-east', {'databricks-claude-opus-5': model()}, latency=2680),
          endpoint('workspace-eu-west', {'databricks-claude-sonnet-5': model(420, 6.72)}, state='HALF_OPEN', active=1, latency=3240),
          endpoint('workspace-asia', {'databricks-claude-opus-5': model(820, 18.72)}, latency=1820)]
    return {'global': summary, 'endpoints': db, 'today_date': '2026-09-30',
            'today_model_stats': {'databricks-claude-opus-5': model(1620, 31.56), 'databricks-claude-sonnet-5': model(740, 11.04)},
            'azure_openai': {'global': summary, 'endpoints': [endpoint('azure-eastus', {'gpt-5.4': model(360, 4.56)})]},
            'github_copilot': {'global': {**summary, 'estimated_total_cost_usd': None,
                'estimated_ai_credits': None, 'known_cost_subtotal_usd': 3.84, 'priced_requests': 240,
                'unpriced_requests': 3, 'pricing_status': 'partial'},
                'endpoints': [endpoint('demo-account', {'gpt-5.6-luna': model(240, 3.84), 'demo-unpriced-model': model(3, None)})],
                'pool': dict(active_requests=8, max_connections=200, utilization_pct=4, pool_timeout_total=0,
                             stream_connections_active=3, stream_high_watermark=80, max_keepalive_connections=100,
                             read_timeout_seconds=None, acquire_timeout_seconds=30)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scenario = {'name': 'normal'}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value, content_type='application/json'):
            body = value.encode() if isinstance(value, str) else json.dumps(value).encode()
            self.send_response(status); self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store')
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path); query = parse_qs(url.query)
            if url.path in ('/', '/stats/dashboard'):
                scenario['name'] = query.get('scenario', ['normal'])[0]
                html = (root / 'dashboard.html').read_text()
                if scenario['name'] == 'no-charts':
                    import re
                    html = re.sub(r'<script src=.*?</script>', '', html, flags=re.S)
                html = html.replace('<div class="brand-caption">多渠道模型网关</div>',
                                    '<div class="brand-caption">本地预览 · 合成数据</div>')
                return self.reply(200, html, 'text/html; charset=utf-8')
            if url.path == '/stats':
                if scenario['name'] == 'error': return self.reply(503, {'error': 'synthetic unavailable'})
                data = stats()
                if scenario['name'] == 'empty':
                    data['endpoints'] = []; data['today_model_stats'] = {}
                    data.pop('azure_openai'); data.pop('github_copilot')
                    data['global'] = {k: 0 for k in data['global']}
                return self.reply(200, data)
            if url.path == '/version':
                if scenario['name'] == 'version-error': return self.reply(503, {})
                if scenario['name'] == 'unknown':
                    return self.reply(200, dict(version=None, source_revision=None, verification='unknown'))
                return self.reply(200, dict(version='aaaaaaa', source_revision='a' * 40,
                    verification='modified' if scenario['name'] == 'modified' else 'matched'))
            if url.path == '/stats/history':
                if scenario['name'] == 'history-error': return self.reply(503, {})
                days = max(1, min(365, int(query.get('days', ['7'])[0])))
                history = []
                for i in range(days):
                    models = {'databricks-claude-opus-5': model(320 + i * 50, 6.4 + i * .3), 'gpt-5.4': model(240, 3.2)}
                    if i == 1: models['demo-unpriced-model'] = model(1, None)
                    history.append({'date': str(date(2026, 9, 30) - timedelta(days=days - i - 1)),
                                    'models': models, 'totals': totals(models)})
                return self.reply(200, dict(history=history, days=days))
            return self.reply(404, {})

        def do_DELETE(self):
            # Simulation only: no fixture, disk, database or application data is deleted.
            if self.headers.get('x-api-key') != 'synthetic-preview-key':
                return self.reply(401, {'detail': 'Synthetic key required'})
            return self.reply(200, {'deleted': 3})

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'http://127.0.0.1:{server.server_port}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
