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


# ---- synthetic 「密钥与渠道」 fixtures ----------------------------------------
# Identity-shaped only: masks, fingerprints and account names, exactly what the
# real endpoints return. No plaintext key, token or password exists here, and
# every write below is simulated — nothing is stored, forwarded or deleted.

PREVIEW_ACCOUNTS = ['gh-biz-1', 'gh-biz-2', 'gh-ent-1', 'gh-fresh']
PREVIEW_CAPABILITY = {
    'gh-biz-1': ['gpt-4o', 'gpt-6-astra', 'kimi-k3'],
    'gh-biz-2': ['gpt-4o', 'gpt-6-astra', 'kimi-k3'],
    'gh-ent-1': ['gpt-4o', 'gpt-5.4', 'gpt-5.5'],
    'gh-fresh': [],
}


def preview_keys():
    return {
        'keys': [
            dict(tenant='default', masked='lb-p…9f2a', fingerprint='3b9ac2f10e45',
                 providers=['databricks', 'copilot', 'azure_openai'], restricted=False,
                 requests=418, last_used_at=time.time() - 240),
            dict(tenant='team-research', masked='lb-r…1c77', fingerprint='c41d7e0ab332',
                 providers=['copilot'], restricted=True, requests=57,
                 last_used_at=time.time() - 9_000),
            dict(tenant='team-batch', masked='', fingerprint='77aa10bd4e9c',
                 providers=['databricks'], restricted=True, requests=0, last_used_at=None),
        ],
        'providers': ['databricks', 'copilot', 'azure_openai'],
        'provider_labels': {'databricks': 'ADB', 'copilot': 'GHCP', 'azure_openai': 'Foundry'},
        'policy': preview_policy(), 'counters_are_process_local': True,
    }


def preview_policy():
    return dict(state='loaded', backend='mysql', tenants=2, accounts=4,
                model_preferences=2, model_accounts=1, last_loaded_at=time.time() - 600,
                last_error_type=None, unrestricted_when_unavailable=True)


def preview_account(name, *, enabled, source='config', selection=None):
    capability = PREVIEW_CAPABILITY[name]
    # Idle-but-enabled is the state worth seeing in the preview: gh-biz-2 is on,
    # yet a per-model selection keeps it out of every model.
    serves = 0 if not enabled else len([m for m in capability
                                        if not selection or name in selection.get(m, [name])])
    return dict(account=name, enabled=enabled, enabled_in_config=name != 'gh-fresh',
                enablement_source=source, configured_models=[], api_types=['responses', 'chat'],
                capability=capability, capability_at=time.time() - 86_400 if capability else None,
                allowed_models=None, token_source='file', requests=120 if enabled else 0,
                circuit_open=False, serves_models=serves, receiving_traffic=serves > 0)


def preview_reason(name, model_name, selection):
    """Same closed set the gateway publishes, so the matrix renders for real."""
    if name == 'gh-fresh':
        return 'account_never_enabled'
    chosen = selection.get(model_name)
    if chosen and name not in chosen:
        return 'not_selected_for_model'
    if PREVIEW_CAPABILITY[name] and model_name not in PREVIEW_CAPABILITY[name]:
        return 'not_in_capability'
    return None


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
                'pricing': {'checked_on': '2026-10-01', 'model_count': 35, 'refresh': {
                    'enabled': True, 'status': 'ok', 'last_success_at': '2026-10-01T09:00:00+00:00',
                    'next_refresh_at': '2026-10-08T09:00:00+00:00', 'stale': False}},
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
    # Panel state lives here so the preview can be walked end to end: lock,
    # unlock, edit a rule, change the password. All in memory, all synthetic.
    panel = {'authenticated': False, 'password_source': 'env',
             'selection': {'gpt-4o': ['gh-biz-1']},
             'preferences': {'gpt-4o': ['copilot'], 'claude-opus-5.5': ['copilot']},
             'enabled': {'gh-biz-1': True, 'gh-biz-2': True, 'gh-ent-1': True, 'gh-fresh': False}}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value, content_type='application/json'):
            body = value.encode() if isinstance(value, str) else json.dumps(value).encode()
            self.send_response(status); self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store')
            self.end_headers(); self.wfile.write(body)

        # ---- synthetic panel helpers ----
        def routing_view(self, model_name):
            order = panel['preferences'].get(model_name) or []
            default = ['copilot', 'azure_openai']
            effective = (order + [p for p in default if p not in order]) if order else default
            selected = panel['selection'].get(model_name) or []
            live = [name for name in PREVIEW_ACCOUNTS
                    if panel['enabled'].get(name)
                    and preview_reason(name, model_name, panel['selection']) is None]
            return dict(model=model_name, providers=order, effective_openai_order=effective,
                        copilot_accounts=selected, effective_copilot_accounts=live,
                        copilot_candidates_empty=not live,
                        opens_claude_on_openai_entry_points='claude' in model_name and bool(order))

        def panel_guard(self):
            if panel['authenticated']:
                return True
            self.reply(401, {'detail': {'error': {'code': 'unauthorized',
                                                  'message': 'Unlock the panel first.'}}})
            return False

        def do_GET(self):
            url = urlsplit(self.path); query = parse_qs(url.query)
            if url.path == '/admin/session':
                return self.reply(200, dict(
                    authenticated=panel['authenticated'],
                    expires_in=1800 if panel['authenticated'] else 0,
                    password_configured=panel['password_source'] != 'none',
                    password_source=panel['password_source'], session_ttl_seconds=1800,
                    auth_methods=['panel_password', 'api_key']))
            if url.path.startswith('/admin/'):
                if not self.panel_guard():
                    return None
            if url.path == '/admin/api-keys':
                return self.reply(200, preview_keys())
            if url.path == '/admin/copilot/accounts':
                if scenario['name'] == 'no-copilot':
                    return self.reply(404, {'detail': {'error': {'message': 'not configured'}}})
                return self.reply(200, dict(
                    accounts=[preview_account(name, enabled=panel['enabled'][name],
                                              source='operator' if name == 'gh-biz-1' else 'config',
                                              selection=panel['selection'])
                              for name in PREVIEW_ACCOUNTS],
                    catalogue_models=9, policy=preview_policy(), counters_are_process_local=True))
            if url.path == '/admin/model-routing':
                return self.reply(200, dict(
                    preferences=[self.routing_view(m) for m in
                                 sorted(set(panel['preferences']) | set(panel['selection']))],
                    providers=['databricks', 'copilot', 'azure_openai'],
                    provider_labels={'databricks': 'ADB', 'copilot': 'GHCP',
                                     'azure_openai': 'Foundry'},
                    default_openai_order=['copilot', 'azure_openai'],
                    known_models=['gpt-4o', 'gpt-5.4', 'gpt-5.5', 'gpt-6-astra', 'kimi-k3',
                                  'claude-opus-5.5'],
                    copilot_accounts_available=[dict(account=n, enabled=panel['enabled'][n])
                                                for n in PREVIEW_ACCOUNTS],
                    policy=preview_policy()))
            if url.path.startswith('/admin/model-routing/'):
                name = url.path.rsplit('/', 1)[-1]
                from urllib.parse import unquote
                name = unquote(name)
                rows = []
                for account in PREVIEW_ACCOUNTS:
                    enabled = panel['enabled'][account]
                    reason = (None if enabled else
                              ('account_never_enabled' if account == 'gh-fresh' else 'account_disabled'))
                    reason = reason or preview_reason(account, name, panel['selection'])
                    ignoring = (None if enabled else reason)
                    if enabled:
                        ignoring = preview_reason(account, name, {})
                    rows.append(dict(account=account, enabled=enabled,
                                     enablement_source='operator' if account == 'gh-biz-1' else 'config',
                                     selected=account in (panel['selection'].get(name) or []),
                                     serves_model=reason is None, reason=reason,
                                     serves_model_ignoring_selection=ignoring is None,
                                     reason_ignoring_selection=ignoring,
                                     capability_known=bool(PREVIEW_CAPABILITY[account]),
                                     requests=120 if enabled else 0, circuit_open=False))
                return self.reply(200, {**self.routing_view(name), 'accounts': rows,
                                        'policy': preview_policy(),
                                        'counters_are_process_local': True})
            if url.path == '/admin/credentials':
                if scenario['name'] == 'no-credential-store':
                    return self.reply(503, {'detail': {'error': {
                        'code': 'credential_store_unavailable',
                        'message': 'Credential storage is not configured.'}}})
                return self.reply(200, dict(
                    store=dict(key='present', key_fingerprint='9d41ab20cc7e', unopenable=0),
                    credentials=[dict(provider='copilot', account='gh-biz-1',
                                      secret_fingerprint='4f1b90ce22ad', key_state='ok',
                                      validation=dict(ok=True, model_count=38),
                                      active=True),
                                 dict(provider='azure_openai', account='foundry-west',
                                      secret_fingerprint='bb27d5e1908f', key_state='ok',
                                      validation=dict(ok=False, reason='invalid_deployment'),
                                      active=False)]))
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
                if scenario['name'] == 'pricing-error':
                    data['github_copilot']['pricing']['refresh'].update(status='error', stale=True)
                if scenario['name'] == 'pricing-disabled':
                    data['github_copilot']['pricing']['refresh'].update(status='disabled', enabled=False)
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
                    if scenario['name'] == 'history-pricing':
                        import sys
                        sys.path.insert(0, str(root))
                        from main import historical_model_cost
                        names = ('databricks-claude-opus-5', 'gpt-6-astra', 'gemini-3.8-flash', 'grok-4.6', 'gpt-6-luna', 'gpt-6-sol')
                        models = {name: historical_model_cost(name, model(100 + i, None)) for name in names}
                    history.append({'date': str(date(2026, 9, 30) - timedelta(days=days - i - 1)),
                                    'models': models, 'totals': totals(models)})
                return self.reply(200, dict(history=history, days=days,
                    pricing={'copilot': {'checked_on': '2026-10-01'}}))
            return self.reply(404, {})

        def body(self):
            length = int(self.headers.get('Content-Length') or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length) or b'{}')
            except Exception:
                return {}

        def do_POST(self):
            # Simulation only: nothing is stored, forwarded or sent upstream.
            url = urlsplit(self.path)
            payload = self.body()
            if url.path == '/admin/session':
                supplied = payload.get('password')
                if supplied is None and self.headers.get('x-api-key'):
                    panel['authenticated'] = True
                    return self.reply(200, {'authenticated': True, 'expires_in': 1800})
                # Any 12+ character password opens the synthetic panel; a shorter
                # one exercises the same refusal the gateway gives.
                if isinstance(supplied, str) and len(supplied) >= 12:
                    panel['authenticated'] = True
                    return self.reply(200, {'authenticated': True, 'expires_in': 1800})
                return self.reply(401, {'detail': {'error': {'code': 'invalid_panel_password',
                                                             'message': 'Incorrect password.'}}})
            if not self.panel_guard():
                return None
            if url.path.endswith('/discover'):
                return self.reply(200, {'discovered': 38, 'policy': preview_policy(),
                                        'account': preview_account(
                                            url.path.split('/')[-2], enabled=True)})
            if url.path == '/admin/api-keys/reload':
                return self.reply(200, {'reloaded': True, 'keys': 3,
                                        'tenants': ['default', 'team-batch', 'team-research'],
                                        'env_interpolated': True})
            return self.reply(200, {'ok': True})

        def do_PUT(self):
            url = urlsplit(self.path)
            payload = self.body()
            if not self.panel_guard():
                return None
            if url.path == '/admin/panel-password':
                panel['password_source'] = 'store'
                return self.reply(200, {'configured': True, 'password_source': 'store',
                                        'other_sessions_invalidated': True})
            if url.path.startswith('/admin/model-routing/'):
                from urllib.parse import unquote
                name = unquote(url.path.rsplit('/', 1)[-1])
                if 'providers' in payload:
                    panel['preferences'][name] = payload['providers']
                if 'copilot_accounts' in payload:
                    chosen = payload['copilot_accounts'] or []
                    if chosen:
                        panel['selection'][name] = chosen
                    else:
                        panel['selection'].pop(name, None)
                return self.reply(200, {**self.routing_view(name), 'policy': preview_policy()})
            if url.path.startswith('/admin/copilot/accounts/'):
                name = url.path.rsplit('/', 1)[-1]
                if name in panel['enabled'] and 'enabled' in payload:
                    panel['enabled'][name] = bool(payload['enabled'])
                return self.reply(200, {'account': preview_account(
                    name, enabled=panel['enabled'].get(name, True), source='operator',
                    selection=panel['selection']), 'policy': preview_policy()})
            if url.path.startswith('/admin/api-keys/'):
                return self.reply(200, {'tenant': url.path.split('/')[3],
                                        'providers': payload.get('providers', []),
                                        'policy': preview_policy()})
            return self.reply(200, {'ok': True})

        def do_DELETE(self):
            # Simulation only: no fixture, disk, database or application data is deleted.
            url = urlsplit(self.path)
            if url.path == '/admin/session':
                panel['authenticated'] = False
                return self.reply(200, {'authenticated': False})
            if url.path.startswith('/admin/'):
                if not self.panel_guard():
                    return None
                if url.path == '/admin/panel-password':
                    panel['password_source'] = 'env'
                    return self.reply(200, {'configured': True, 'password_source': 'env',
                                            'other_sessions_invalidated': True})
                if url.path.startswith('/admin/model-routing/'):
                    from urllib.parse import unquote
                    name = unquote(url.path.rsplit('/', 1)[-1])
                    panel['preferences'].pop(name, None)
                    panel['selection'].pop(name, None)
                    return self.reply(200, {**self.routing_view(name), 'policy': preview_policy()})
                if url.path.startswith('/admin/api-keys/'):
                    return self.reply(200, {'tenant': url.path.split('/')[3],
                                            'providers': ['databricks', 'copilot', 'azure_openai'],
                                            'policy': preview_policy()})
                return self.reply(200, {'ok': True})
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
