// Dashboard operations fail honestly and version links stay on GitHub. Author: Zeno Ren.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require('node:path').join(__dirname, '../dashboard.html'), 'utf8');
function section(from, to) {
  assert.ok(html.includes(from) && html.includes(to), `Missing section ${from}`);
  return html.slice(html.indexOf(from), html.indexOf(to));
}
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {textContent: '', innerHTML: '', value: '', hidden: false,
    dataset: {}, classList: {add() {}, remove() {}, toggle() {}}, setAttribute(k, v) {this[k] = v;},
    removeAttribute(k) {delete this[k];}});
  return elements.get(id);
}
const context = {document: {getElementById: element}, URL, AbortController, setTimeout, clearTimeout,
  console, historyLoaded: false, loadHistory: async () => {}, confirm: () => true,
  alert: () => {throw Error('Use inline feedback');}};
vm.createContext(context);
vm.runInContext(section('// ---------- Formatters', '// ---------- Number Ticker') +
  section('// ---------- Release identity', '// ---------- Refresh'), context);

(async () => {
  const partial = context.pricingSummary({known: {estimated_cost_usd: 1}, future: {requests: 1}});
  assert.equal(partial.estimated_cost_usd, null);
  assert.equal(partial.known_cost_subtotal_usd, 1);
  assert.equal(partial.pricing_status, 'partial');
  assert.match(context.fmtModelCost(partial), /未计价/);
  assert.equal(context.pricingSummary({}).estimated_cost_usd, 0);
  assert.equal(context.pricingSummary({future: {requests: 1}}).pricing_status, 'unknown');
  assert.match(context.circuitBadge({}), /未知/);
  assert.match(context.circuitBadge({circuit_open: false}), /Closed/);
  assert.match(context.circuitBadge({circuit_state: 'HALF_OPEN', circuit_open: true}), /Half-open/);
  assert.equal(context.escapeHtml('<img src=x onerror="oops">'), '&lt;img src=x onerror=&quot;oops&quot;&gt;');
  let requests = [];
  context.fetch = async (url, options) => {
    requests.push({url, options});
    return {ok: false, status: 401, json: async () => ({detail: 'Invalid API key'})};
  };
  element('cleanDays').value = '30'; element('managementKey').value = '';
  await context.cleanupHistory();
  assert.equal(requests.length, 0, 'Empty credentials must not issue a delete');
  element('managementKey').value = 'synthetic-test-key';
  await context.cleanupHistory();
  assert.equal(requests.length, 1);
  assert.equal(requests[0].options.headers['x-api-key'], 'synthetic-test-key');
  assert.equal(requests[0].options.headers.Authorization, undefined, 'Do not replace an ingress Authorization header');
  assert.equal(requests[0].options.redirect, 'error', 'Management DELETE must never follow redirects');
  assert.match(element('cleanupStatus').textContent, /401|凭据|鉴权/);
  assert.doesNotMatch(element('cleanupStatus').textContent, /已删除/);
  assert.equal(element('managementKey').value, '', 'Credentials are cleared after use');
  for (const days of ['0', '-1', '1.5', '4000', 'bad']) {
    element('cleanDays').value = days; element('managementKey').value = 'synthetic-test-key';
    await context.cleanupHistory();
  }
  assert.equal(requests.length, 1, 'Invalid retention must not issue a delete');
  element('cleanDays').value = '30';
  context.confirm = () => false;
  await context.cleanupHistory();
  assert.equal(requests.length, 1, 'Cancelled confirmation must not issue a delete');
  context.confirm = () => true;
  element('managementKey').value = 'synthetic-test-key';
  context.fetch = async () => ({ok: true, status: 200, json: async () => ({error: 'Storage unavailable'})});
  await context.cleanupHistory();
  assert.doesNotMatch(element('cleanupStatus').textContent, /已删除/);
  for (const deleted of [0, 3]) {
    element('managementKey').value = 'synthetic-test-key';
    context.fetch = async () => ({ok: true, status: 200, json: async () => ({deleted})});
    await context.cleanupHistory();
    assert.match(element('cleanupStatus').textContent, new RegExp(`已删除 ${deleted} 条`));
    assert.equal(element('managementKey').value, '');
  }
  let redirectedCalls = 0;
  element('managementKey').value = 'synthetic-test-key';
  context.fetch = async (_, options) => {
    redirectedCalls++;
    assert.equal(options.redirect, 'error');
    throw new TypeError('Synthetic redirect rejected');
  };
  await context.cleanupHistory();
  assert.equal(redirectedCalls, 1, 'An uncertain delete is not replayed');
  assert.match(element('cleanupStatus').textContent, /未能确认清理结果/);
  assert.equal(element('managementKey').value, '');

  context.renderRelease({source_revision: 'a'.repeat(40), version: 'aaaaaaa', verification: 'matched',
    commit_url: 'https://evil.invalid/steal', compare_url: 'javascript:alert(1)'});
  assert.equal(element('releaseCommit').href, 'https://github.com/ZenoRewn/databricks-claude-lb/commit/' + 'a'.repeat(40));
  assert.match(element('releaseState').textContent, /文件一致/);
  assert.doesNotMatch(element('releaseState').textContent, /最新/);
  context.renderRelease({source_revision: '<img onerror=bad>', verification: 'matched'});
  assert.equal(element('releaseCommit').href, undefined);
  assert.match(element('releaseState').textContent, /未标注|未知/);
  vm.runInContext(section('// ---------- Refresh ----------', '// ---------- Refresh controls'), context);
  context.refreshRelease = async () => {};
  context.fetch = async () => ({ok: false, status: 503});
  element('globalGrid').innerHTML = '正在读取渠道用量…';
  await context.refresh(true);
  assert.match(element('globalGrid').innerHTML, /暂时无法读取/);
  assert.match(element('endpointBody').innerHTML, /暂时无法读取/);
  assert.match(element('azureContent').innerHTML, /暂时无法读取/);
  element('globalGrid').dataset.built = '1';
  element('globalGrid').innerHTML = 'last successful snapshot';
  element('endpointBody').innerHTML = 'last successful endpoints';
  await context.refresh(true);
  assert.equal(element('globalGrid').innerHTML, 'last successful snapshot');
  assert.equal(element('endpointBody').innerHTML, 'last successful endpoints');
  // ---- API keys & provider allow-lists ----
  // The panel must never render a plaintext key, must refuse an empty
  // allow-list, and must say the enforced policy is unchanged when a write
  // fails. Rotation must state that the gateway does not issue keys.
  const nodes = [];
  context.document.querySelectorAll = selector => {
    if (selector.includes('[data-save]')) return [];
    return nodes.filter(node => node.selector === selector);
  };
  assert.match(html, /id="nav-keys"/, 'key panel needs a nav entry');
  assert.match(html, /id="tab-keys"/, 'key panel needs a tab panel');
  assert.match(html, /type="password"[^>]*id="keysAdminKey"|id="keysAdminKey"[^>]*type="password"/,
    'the admin key field must be a password input');
  assert.match(html, /不显示也不返回明文/, 'panel must state plaintext is never shown');
  assert.match(html, /本进程/, 'panel must label counters as process-local');

  context.renderKeys({providers: ['databricks', 'copilot'],
    provider_labels: {databricks: 'ADB', copilot: 'GHCP'},
    policy: {state: 'loaded', backend: 'mysql', tenants: 1},
    keys: [{tenant: 'team-a', masked: 'abcd…wxyz', fingerprint: 'deadbeefcafe',
            providers: ['copilot'], restricted: true, requests: 3, last_used_at: 1700000000}]});
  assert.match(element('keysContent').innerHTML, /team-a/);
  assert.match(element('keysContent').innerHTML, /abcd…wxyz/);
  assert.match(element('keysContent').innerHTML, /deadbeefcafe/);
  assert.match(element('keysPolicyState').innerHTML, /已加载/);

  context.renderKeys({providers: [], policy: {state: 'unavailable', backend: 'mysql', tenants: 0}, keys: []});
  assert.match(element('keysPolicyState').innerHTML, /不施加任何渠道限制/,
    'an unloaded policy store must be reported, not hidden');
  assert.match(element('keysContent').innerHTML, /没有已注册/);

  context.renderKeys({providers: ['copilot'], provider_labels: {},
    policy: {state: 'loaded', backend: 'json', tenants: 0},
    keys: [{tenant: '<img onerror=bad>', masked: '', fingerprint: 'f00d', providers: [], requests: 0,
            last_used_at: null}]});
  assert.doesNotMatch(element('keysContent').innerHTML, /<img onerror/, 'tenant names must be escaped');
  assert.match(element('keysContent').innerHTML, /过短，不展示掩码/);
  assert.match(element('keysContent').innerHTML, /本进程内未使用/);

  element('keysAdminKey').value = 'lb-key-for-panel-tests';
  requests = [];
  context.fetch = async (url, options) => {
    requests.push({url, options});
    return {ok: false, status: 503, json: async () => ({detail: {error: {code: 'policy_write_failed'}}})};
  };
  nodes.length = 0;
  nodes.push({selector: '#keysTable [data-tenant="team-a"]', checked: true,
              dataset: {provider: 'copilot'}});
  await context.saveTenantProviders('team-a');
  assert.match(element('keysStatus').textContent, /执行中的策略保持不变/,
    'a failed write must not claim the policy changed');
  assert.equal(element('keysAdminKey').value, '', 'the key field must be cleared after use');

  element('keysAdminKey').value = 'lb-key-for-panel-tests';
  nodes.length = 0;
  nodes.push({selector: '#keysTable [data-tenant="team-a"]', checked: false,
              dataset: {provider: 'copilot'}});
  requests = [];
  await context.saveTenantProviders('team-a');
  assert.equal(requests.length, 0, 'an empty allow-list must not reach the server');
  assert.match(element('keysStatus').textContent, /至少要允许一个渠道/);

  element('keysAdminKey').value = 'lb-key-for-panel-tests';
  context.fetch = async (url, options) => {
    requests.push({url, options});
    return {ok: true, status: 200,
            json: async () => ({keys: 2, tenants: ['default', 'team-a'], env_interpolated: true})};
  };
  requests = [];
  await context.reloadKeysFromSource();
  assert.equal(requests[0].url, '/admin/api-keys/reload');
  assert.equal(requests[0].options.method, 'POST');
  assert.equal(requests[0].options.headers['x-api-key'], 'lb-key-for-panel-tests');
  assert.match(element('keysStatus').textContent, /滚动重启/,
    'env-sourced values still need a restart and the panel must say so');
  assert.equal(element('keysAdminKey').value, '');
  assert.ok(!requests.some(r => r.url.includes('lb-key-for-panel-tests')),
    'the key must never travel in a URL');

  // ---- GHCP accounts ----
  assert.match(html, /id="accountsContent"/, 'account area needs a container');
  assert.match(html, /capability 是向上游的时点发现/, 'capability must be labelled as a snapshot');

  context.renderAccounts({policy: {state: 'loaded', backend: 'mysql', tenants: 0},
    accounts: [{account: 'biz', enabled: true, token_source: 'file', requests: 7,
                capability: ['gpt-4o', 'kimi-k3'], capability_at: 1700000000,
                allowed_models: ['gpt-4o']}]});
  assert.match(element('accountsContent').innerHTML, /biz/);
  assert.match(element('accountsContent').innerHTML, /已启用/);
  assert.match(element('accountsContent').innerHTML, /kimi-k3/);
  assert.doesNotMatch(element('accountsContent').innerHTML, /ghu_|Bearer/, 'no token material');

  context.renderAccounts({policy: {state: 'loaded', backend: 'json', tenants: 0},
    accounts: [{account: '<img onerror=bad>', enabled: false, token_source: null, requests: 0,
                capability: [], capability_at: null, allowed_models: null}]});
  assert.doesNotMatch(element('accountsContent').innerHTML, /<img onerror/, 'account names escaped');
  assert.match(element('accountsContent').innerHTML, /尚未发现能力/);
  assert.match(element('accountsContent').innerHTML, /已停用/);

  // An account shipped inert reads differently from one an operator switched
  // off: both are out of service, but only the second was a decision.
  context.renderAccounts({policy: {state: 'loaded', backend: 'mysql', tenants: 0},
    accounts: [{account: 'fresh', enabled: false, enabled_in_config: false,
                enablement_source: 'config', token_source: 'file', requests: 0,
                configured_models: [], capability: [], capability_at: null, allowed_models: null}]});
  let inert = element('accountsContent').innerHTML;
  assert.match(inert, /配置声明不投入服务/, 'an inert account must not read as operator-disabled');
  assert.doesNotMatch(inert, /已停用/);
  assert.match(inert, /启用前先「刷新能力」/, 'enabling it blind would make it a candidate for everything');
  assert.match(inert, /data-needs-capability="1"/, 'the guard must be reachable from the control');

  // Once capability is known the warning goes away; discovery alone must not
  // flip the account to enabled.
  context.renderAccounts({policy: {state: 'loaded', backend: 'mysql', tenants: 0},
    accounts: [{account: 'fresh', enabled: false, enabled_in_config: false,
                enablement_source: 'config', token_source: 'file', requests: 0,
                configured_models: [], capability: ['gpt-4o'], capability_at: 1700000000,
                allowed_models: null}]});
  inert = element('accountsContent').innerHTML;
  assert.match(inert, /配置声明不投入服务/, 'discovery is not an enablement');
  assert.doesNotMatch(inert, /data-needs-capability/, 'capability is known now');
  assert.doesNotMatch(inert, /启用前先「刷新能力」/);

  // Unchecking every model must not reach the server — disabling is a separate control.
  const accountNodes = [];
  context.document.querySelectorAll = selector => {
    if (selector.includes('[data-save]') || selector.includes('[data-save-account]')
        || selector.includes('[data-discover]')) return [];
    return accountNodes.filter(n => n.selector === selector);
  };
  element('keysAdminKey').value = 'lb-key-for-account-tests';
  accountNodes.push({selector: '#accountsTable [data-enable="biz"]', checked: true, dataset: {}},
    {selector: '#accountsTable [data-account="biz"]', checked: false, dataset: {model: 'gpt-4o'}});
  requests = [];
  await context.saveAccount('biz');
  assert.equal(requests.length, 0, 'an all-unchecked save must be refused locally');
  assert.match(element('keysStatus').textContent, /取消「启用」/);
  assert.equal(element('keysAdminKey').value, '');

  // Enabling an inert account with no capability must not reach the server:
  // it would be a wildcard candidate for every model until discovery runs.
  element('keysAdminKey').value = 'lb-key-for-account-tests';
  accountNodes.length = 0;
  accountNodes.push({selector: '#accountsTable [data-enable="fresh"]', checked: true,
                     dataset: {needsCapability: '1'}});
  requests = [];
  await context.saveAccount('fresh');
  assert.equal(requests.length, 0, 'enabling without capability must be refused locally');
  assert.match(element('keysStatus').textContent, /先点「刷新能力」/);
  assert.equal(element('keysAdminKey').value, '');

  // Disabling that same account is always allowed — the guard is about enabling.
  element('keysAdminKey').value = 'lb-key-for-account-tests';
  accountNodes.length = 0;
  accountNodes.push({selector: '#accountsTable [data-enable="fresh"]', checked: false,
                     dataset: {needsCapability: '1'}});
  requests = [];
  context.fetch = async (url, options) => {
    requests.push({url, options});
    return {ok: true, status: 200, json: async () => ({account: {account: 'fresh', enabled: false,
      enabled_in_config: false, enablement_source: 'operator', capability: []},
      policy: {state: 'loaded', backend: 'mysql', tenants: 0}})};
  };
  await context.saveAccount('fresh');
  assert.equal(requests.length, 1, 'an explicit disable must still be sendable');
  assert.equal(JSON.parse(requests[0].options.body).enabled, false);
  assert.equal(element('keysAdminKey').value, '');

  element('keysAdminKey').value = 'lb-key-for-account-tests';
  context.fetch = async (url, options) => {
    requests.push({url, options});
    return {ok: true, status: 200, json: async () => ({discovered: 38, policy: {state: 'loaded', backend: 'mysql', tenants: 0}})};
  };
  requests = [];
  await context.discoverAccount('biz');
  assert.equal(requests[0].url, '/admin/copilot/accounts/biz/discover');
  assert.equal(requests[0].options.method, 'POST');
  assert.match(element('keysStatus').textContent, /非能力认证/);
  assert.equal(element('keysAdminKey').value, '');
  assert.ok(!requests.some(r => r.url.includes('lb-key-for-account-tests')));

  element('keysAdminKey').value = 'lb-key-for-account-tests';
  context.fetch = async (url, options) => {
    requests.push({url, options});
    return {ok: false, status: 503, json: async () => ({detail: {error: {code: 'policy_write_failed'}}})};
  };
  requests = [];
  await context.discoverAccount('biz');
  assert.match(element('keysStatus').textContent, /已有快照保持不变/,
    'a failed discovery must not claim the snapshot changed');

  // ---- model → provider preference ----
  assert.match(html, /id="routingContent"/, 'routing area needs a container');
  assert.match(html, /只重排不筛除/, 'reorder-not-filter must be stated');
  assert.match(html, /仍然只走 ADB/, 'the /v1/messages limit must be stated');

  context.renderRouting({policy: {state: 'loaded', backend: 'mysql', tenants: 0},
    default_openai_order: ['copilot', 'azure_openai'],
    preferences: [{model: 'gpt-5.4', providers: ['azure_openai'],
                   effective_openai_order: ['azure_openai', 'copilot']},
                  {model: 'claude-opus-5.5', providers: ['copilot'],
                   effective_openai_order: ['copilot', 'azure_openai']}]});
  assert.match(element('routingContent').innerHTML, /azure_openai → copilot/,
    'the effective order must be shown, not just the stored one');
  assert.match(element('routingContent').innerHTML, /已在 OpenAI 风格入口放行/,
    'a claude model opened by preference must be flagged');
  const gptRow = element('routingContent').innerHTML.split('<tr>').find(r => r.includes('gpt-5.4'));
  assert.doesNotMatch(gptRow, /已在 OpenAI 风格入口放行/, 'a non-claude model must not be flagged');

  context.renderRouting({policy: {state: 'loaded', backend: 'json', tenants: 0},
    default_openai_order: ['copilot', 'azure_openai'],
    preferences: [{model: '<img onerror=bad>', providers: ['copilot'], effective_openai_order: []}]});
  assert.doesNotMatch(element('routingContent').innerHTML, /<img onerror/, 'model names escaped');

  context.renderRouting({policy: {state: 'loaded', backend: 'json', tenants: 0},
    default_openai_order: ['copilot', 'azure_openai'], preferences: []});
  assert.match(element('routingContent').innerHTML, /全部按默认顺序/);

  // A missing model or provider list must not reach the server.
  element('routeModel').value = '';
  element('routeProviders').value = 'copilot';
  element('keysAdminKey').value = 'lb-key-for-routing-tests';
  requests = [];
  await context.saveRouting();
  assert.equal(requests.length, 0, 'an incomplete routing form must be refused locally');
  assert.match(element('keysStatus').textContent, /至少一个 Provider/);

  element('routeModel').value = 'claude-opus-5.5';
  element('routeProviders').value = ' copilot , azure_openai ';
  element('keysAdminKey').value = 'lb-key-for-routing-tests';
  context.fetch = async (url, options) => {
    requests.push({url, options});
    return {ok: true, status: 200, json: async () => ({
      effective_openai_order: ['copilot', 'azure_openai'],
      opens_claude_on_openai_entry_points: true,
      policy: {state: 'loaded', backend: 'mysql', tenants: 0}})};
  };
  requests = [];
  await context.saveRouting();
  assert.equal(requests[0].url, '/admin/model-routing/claude-opus-5.5');
  assert.equal(requests[0].options.method, 'PUT');
  assert.deepEqual(JSON.parse(requests[0].options.body).providers, ['copilot', 'azure_openai'],
    'whitespace around the comma-separated order must be trimmed');
  assert.match(element('keysStatus').textContent, /已在 OpenAI 风格入口放行/);
  assert.equal(element('keysAdminKey').value, '');
  assert.ok(!requests.some(r => r.url.includes('lb-key-for-routing-tests')));

  element('keysAdminKey').value = 'lb-key-for-routing-tests';
  context.fetch = async (url, options) => {
    requests.push({url, options});
    return {ok: false, status: 503, json: async () => ({detail: {error: {code: 'policy_write_failed'}}})};
  };
  requests = [];
  await context.clearRouting('claude-opus-5.5');
  assert.equal(requests[0].options.method, 'DELETE');
  assert.match(element('keysStatus').textContent, /偏好保持不变/,
    'a failed clear must not claim the preference changed');

  // ---- upstream credentials ----
  assert.match(html, /id="credentialsContent"/, 'credential area needs a container');
  assert.match(html, /不会退化为存明文/, 'the no-plaintext-fallback rule must be stated');
  assert.match(html, /不显示也不返回明文/, 'the panel must state plaintext is never shown');
  assert.match(html, /需一次滚动重启才生效/, 'activation semantics must be stated');
  assert.match(html, /type="password"[^>]*id="credSecret"|id="credSecret"[^>]*type="password"/,
    'the credential field must be a password input');

  context.renderCredentials({store: {key: 'present', key_fingerprint: 'abc123def456', unopenable: 0},
    credentials: [{provider: 'databricks', account: 'ws-9', secret_fingerprint: 'deadbeefcafe',
                   key_state: 'ok', active: false,
                   validation: {ok: true, model_count: 3}, meta: {api_base: 'https://x.invalid'}}]});
  assert.match(element('credentialsContent').innerHTML, /ws-9/);
  assert.match(element('credentialsContent').innerHTML, /deadbeefcafe/);
  assert.match(element('credentialsContent').innerHTML, /待重启生效/,
    'a stored-but-inactive credential must not look live');
  assert.doesNotMatch(element('credentialsContent').innerHTML, /dapi-|ghu_|sealed/,
    'no secret or ciphertext may be rendered');

  context.renderCredentials({store: {key: 'missing', unopenable: 0}, credentials: []});
  assert.match(element('credentialsContent').innerHTML, /不会退化为存明文/,
    'a missing sealing key must be reported prominently');

  context.renderCredentials({store: {key: 'present', key_fingerprint: 'k', unopenable: 2}, credentials: []});
  assert.match(element('credentialsContent').innerHTML, /无法用当前密钥解开/,
    'records sealed with another key must be surfaced, not hidden');

  context.renderCredentials({store: {key: 'present', key_fingerprint: 'k', unopenable: 0},
    credentials: [{provider: 'copilot', account: '<img onerror=bad>', secret_fingerprint: 'f',
                   key_state: 'sealed_with_other_key', active: false, validation: {ok: false, reason: 'rejected'}}]});
  assert.doesNotMatch(element('credentialsContent').innerHTML, /<img onerror/, 'account names escaped');
  assert.match(element('credentialsContent').innerHTML, /由另一把密钥封装/);

  // A submitted credential must not linger in the DOM, and a 409 must be explained.
  element('credProvider').value = 'databricks';
  element('credAccount').value = 'ws-config';
  element('credSecret').value = 'dapi-should-not-persist';
  element('credApiBase').value = 'https://x.invalid';
  element('credDeployments').value = '';
  element('keysAdminKey').value = 'lb-key-for-cred-tests';
  requests = [];
  context.fetch = async (url, options) => {
    requests.push({url, options});
    return {ok: false, status: 409, json: async () => ({detail: {error: {code: 'account_declared_in_config'}}})};
  };
  await context.submitCredential();
  assert.equal(requests[0].options.method, 'PUT');
  assert.equal(JSON.parse(requests[0].options.body).secret, 'dapi-should-not-persist');
  assert.match(element('keysStatus').textContent, /静默遮蔽/, 'a 409 must explain the shadowing rule');
  assert.equal(element('credSecret').value, '', 'the credential field must be cleared after submit');
  assert.equal(element('keysAdminKey').value, '');
  assert.ok(!requests.some(r => r.url.includes('dapi-should-not-persist')),
    'the credential must never travel in a URL');

  // Device flow: pending is not a failure, and expiry resets the handle.
  element('keysAdminKey').value = 'lb-key-for-cred-tests';
  context.fetch = async (url, options) => {
    requests.push({url, options});
    return {ok: true, status: 200, json: async () => ({handle: 'h1', user_code: 'WXYZ-9876',
      verification_uri: 'https://github.com/login/device', expires_in: 899})};
  };
  requests = [];
  await context.startDeviceFlow();
  assert.match(element('deviceStatus').textContent, /WXYZ-9876/);
  assert.match(element('deviceStatus').textContent, /github\.com\/login\/device/);

  element('deviceAccount').value = 'gh-account-5';
  element('keysAdminKey').value = 'lb-key-for-cred-tests';
  context.fetch = async () => ({ok: true, status: 200, json: async () => ({pending: true, interval: 5})});
  await context.pollDeviceFlow();
  assert.match(element('deviceStatus').textContent, /尚未授权完成/,
    'authorization_pending must read as waiting, not as an error');

  element('keysAdminKey').value = 'lb-key-for-cred-tests';
  context.fetch = async () => ({ok: false, status: 410, json: async () => ({detail: {error: {code: 'device_flow_expired'}}})});
  await context.pollDeviceFlow();
  assert.match(element('deviceStatus').textContent, /已过期/);

  element('keysAdminKey').value = 'lb-key-for-cred-tests';
  element('deviceAccount').value = 'gh-account-5';
  requests = [];
  context.fetch = async (url, options) => { requests.push({url, options}); return {ok: true, status: 200, json: async () => ({}) }; };
  await context.pollDeviceFlow();
  assert.equal(requests.length, 0, 'an expired flow must not keep polling with a stale handle');

  console.log('Dashboard UI: authenticated cleanup, honest errors, safe release identity, key panel, accounts, routing, credentials passed.');
})().catch(error => {console.error(error); process.exitCode = 1;});
