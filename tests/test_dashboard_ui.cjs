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
  let nodes = [];
  const noop = () => {};
  context.document.querySelectorAll = selector =>
    nodes.filter(node => node.selector === selector)
      .map(node => Object.assign({addEventListener: noop}, node));
  assert.match(html, /id="nav-keys"/, 'key panel needs a nav entry');
  assert.match(html, /id="tab-keys"/, 'key panel needs a tab panel');
  assert.match(html, /type="password"[^>]*id="keysAdminKey"|id="keysAdminKey"[^>]*type="password"/,
    'the admin key field must be a password input');
  assert.match(html, /不显示也不返回明文/, 'panel must state plaintext is never shown');
  assert.match(html, /本进程/, 'panel must label counters as process-local');

  // The key is now held in page memory while unlocked, which is a change from
  // the one-request contract — so the page has to say so, and must still never
  // reach for a browser store or a URL.
  assert.match(html, /留在本页内存里/, 'the retention change must be stated, not silent');
  assert.match(html, /自动锁定/, 'the idle timeout must be stated');
  // Scoped to this panel's own code: the theme toggle legitimately persists a
  // preference, so a whole-file check would be meaningless. Matching usage
  // (`x.`) rather than the bare word keeps the prose that names these stores.
  const keyPanelJs = section('// ---------- API keys & provider allow-lists',
                             '// ---------- Refresh ----------');
  assert.doesNotMatch(keyPanelJs, /localStorage\s*\.|sessionStorage\s*\.|document\s*\.\s*cookie/,
    'the panel key must never reach a browser store');

  context.renderKeys({providers: ['databricks', 'copilot'],
    provider_labels: {databricks: 'ADB', copilot: 'GHCP'},
    policy: {state: 'loaded', backend: 'mysql', tenants: 1},
    keys: [{tenant: 'team-a', masked: 'abcd…wxyz', fingerprint: 'deadbeefcafe',
            providers: ['copilot'], restricted: true, requests: 3, last_used_at: 1700000000}]});
  assert.match(element('keysContent').innerHTML, /team-a/);
  assert.match(element('keysContent').innerHTML, /abcd…wxyz/);
  assert.match(element('keysContent').innerHTML, /deadbeefcafe/);
  assert.match(element('keysPolicyState').innerHTML, /已加载/);
  assert.match(element('keysContent').innerHTML, /data-reset="team-a"/,
    'a restricted tenant needs a way back to the config baseline');

  // Per-layer counts: "did my rule persist?" is the first question when a
  // change appears not to take effect.
  context.renderPolicyState({state: 'loaded', backend: 'mysql', tenants: 2,
    model_preferences: 3, model_accounts: 1, accounts: 4});
  assert.match(element('keysPolicyState').innerHTML, /模型优先序 3/);
  assert.match(element('keysPolicyState').innerHTML, /账户限定 1/);
  context.renderPolicyState({state: 'loaded', backend: 'json'});
  assert.match(element('keysPolicyState').innerHTML, /账户限定 0/,
    'a missing count reads as zero, not as undefined');

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
  assert.doesNotMatch(element('keysContent').innerHTML, /data-reset/,
    'an unrestricted tenant has nothing to reset');

  // ---- unlock / lock ----
  // One unlock reads all four sections; a write afterwards must not need the
  // key retyped. That is the whole point of the change.
  const PANEL_URLS = ['/admin/api-keys', '/admin/model-routing',
                      '/admin/copilot/accounts', '/admin/credentials'];
  const PANEL_PAYLOADS = {
    '/admin/api-keys': {providers: ['copilot'], provider_labels: {}, keys: [
      {tenant: 'team-a', masked: 'abcd…wxyz', fingerprint: 'f1', providers: ['copilot'],
       restricted: true, requests: 1, last_used_at: null}],
      policy: {state: 'loaded', backend: 'mysql', tenants: 1}},
    '/admin/model-routing': {default_openai_order: ['copilot', 'azure_openai'],
      known_models: ['gpt-4o', 'kimi-k3'],
      copilot_accounts_available: [{account: 'acct-a', enabled: true},
                                   {account: 'acct-b', enabled: false}],
      preferences: [], policy: {state: 'loaded', backend: 'mysql', tenants: 1}},
    '/admin/copilot/accounts': {accounts: [{account: 'acct-a', enabled: true,
      token_source: 'file', requests: 0, capability: ['gpt-4o'], capability_at: null,
      allowed_models: null}], policy: {state: 'loaded', backend: 'mysql', tenants: 1}},
    '/admin/credentials': {store: {key: 'present', key_fingerprint: 'kf', unopenable: 0},
      credentials: []},
  };
  const panelFetch = (overrides = {}) => async (url, options) => {
    requests.push({url, options});
    if (overrides[url]) return overrides[url];
    const payload = PANEL_PAYLOADS[url];
    if (!payload) return {ok: false, status: 404, json: async () => ({detail: 'no stub'})};
    return {ok: true, status: 200, json: async () => payload};
  };

  element('keysAdminKey').value = 'lb-panel-key';
  requests = [];
  context.fetch = panelFetch();
  await context.unlockPanels();
  assert.deepEqual(requests.map(r => r.url).sort(), [...PANEL_URLS].sort(),
    'unlocking reads every section exactly once');
  assert.ok(requests.every(r => r.options.headers['x-api-key'] === 'lb-panel-key'));
  assert.ok(!requests.some(r => r.url.includes('lb-panel-key')), 'the key must never travel in a URL');
  assert.equal(element('keysAdminKey').value, '', 'the field is cleared; the key lives in memory now');
  assert.match(element('keysStatus').textContent, /已解锁/);
  assert.match(element('keysContent').innerHTML, /team-a/);
  assert.match(element('accountsContent').innerHTML, /acct-a/);
  assert.match(element('ruleModel').innerHTML, /gpt-4o/, 'the model dropdown is populated on unlock');
  assert.match(element('ruleAccounts').innerHTML, /acct-a/, 'the account checkboxes are populated');

  // A write now reuses the held key without the field being touched.
  nodes = [{selector: '#keysTable [data-tenant="team-a"]', checked: true, dataset: {provider: 'copilot'}}];
  requests = [];
  context.fetch = panelFetch({'/admin/api-keys/team-a/providers': {ok: true, status: 200,
    json: async () => ({tenant: 'team-a', providers: ['copilot'],
                        policy: {state: 'loaded', backend: 'mysql', tenants: 1}})}});
  await context.saveTenantProviders('team-a');
  assert.equal(requests[0].url, '/admin/api-keys/team-a/providers');
  assert.equal(requests[0].options.headers['x-api-key'], 'lb-panel-key',
    'the held key is reused; nobody retypes it');
  assert.equal(requests[1].url, '/admin/api-keys',
    'the section re-reads itself so the result is visible without a button');

  // Reset is a DELETE and must be confirmed.
  requests = [];
  context.confirm = () => false;
  await context.resetTenantProviders('team-a');
  assert.equal(requests.length, 0, 'a cancelled reset must not reach the server');
  context.confirm = () => true;
  requests = [];
  context.fetch = panelFetch({'/admin/api-keys/team-a/providers': {ok: true, status: 200,
    json: async () => ({tenant: 'team-a', providers: ['databricks', 'copilot', 'azure_openai'],
                        restricted: false, policy: {state: 'loaded', backend: 'mysql', tenants: 0}})}});
  await context.resetTenantProviders('team-a');
  assert.equal(requests[0].options.method, 'DELETE');
  assert.match(element('keysStatus').textContent, /配置基线/,
    'reset must name what it returns to, not just say "done"');

  requests = [];
  context.fetch = panelFetch({'/admin/api-keys/team-a/providers': {ok: false, status: 503,
    json: async () => ({detail: {error: {code: 'policy_write_failed'}}})}});
  await context.resetTenantProviders('team-a');
  assert.match(element('keysStatus').textContent, /原有限制仍然生效/,
    'a failed reset must not claim the tenant is now unrestricted');

  nodes = [{selector: '#keysTable [data-tenant="team-a"]', checked: true, dataset: {provider: 'copilot'}}];
  requests = [];
  context.fetch = panelFetch({'/admin/api-keys/team-a/providers': {ok: false, status: 503,
    json: async () => ({detail: {error: {code: 'policy_write_failed'}}})}});
  await context.saveTenantProviders('team-a');
  assert.match(element('keysStatus').textContent, /执行中的策略保持不变/,
    'a failed write must not claim the policy changed');

  nodes = [{selector: '#keysTable [data-tenant="team-a"]', checked: false, dataset: {provider: 'copilot'}}];
  requests = [];
  await context.saveTenantProviders('team-a');
  assert.equal(requests.length, 0, 'an empty allow-list must not reach the server');
  assert.match(element('keysStatus').textContent, /至少要允许一个渠道/);
  assert.match(element('keysStatus').textContent, /重置/, 'it must point at the control that does work');

  // Rotating the source may invalidate the very key held in memory, so the
  // panel locks rather than keeping something that may no longer work.
  requests = [];
  context.fetch = panelFetch({'/admin/api-keys/reload': {ok: true, status: 200,
    json: async () => ({keys: 2, tenants: ['default', 'team-a'], env_interpolated: true})}});
  await context.reloadKeysFromSource();
  assert.equal(requests[0].url, '/admin/api-keys/reload');
  assert.equal(requests[0].options.method, 'POST');
  assert.match(element('keysStatus').textContent, /滚动重启/,
    'env-sourced values still need a restart and the panel must say so');
  assert.match(element('keysStatus').textContent, /已自动锁定/);
  assert.match(element('keysContent').innerHTML, /已锁定/, 'locking clears the identity material');
  nodes = [{selector: '#keysTable [data-tenant="team-a"]', checked: true, dataset: {provider: 'copilot'}}];
  requests = [];
  await context.saveTenantProviders('team-a');
  assert.equal(requests.length, 0, 'a locked panel must not issue writes');
  assert.match(element('keysStatus').textContent, /请先输入 API Key/);

  // Lock clears every section, not just the key.
  element('keysAdminKey').value = 'lb-panel-key';
  context.fetch = panelFetch();
  await context.unlockPanels();
  assert.match(element('accountsContent').innerHTML, /acct-a/);
  context.lockPanels();
  for (const id of ['keysContent', 'routingContent', 'accountsContent', 'credentialsContent']) {
    assert.match(element(id).innerHTML, /已锁定/, id + ' must be cleared on lock');
  }
  assert.equal(element('keysPolicyState').innerHTML, '');
  assert.equal(element('ruleAccounts').innerHTML, '');

  // An idle timeout locks on the next action instead of sending a stale key.
  let clock = 5_000_000;
  context.panelClock = () => clock;
  element('keysAdminKey').value = 'lb-panel-key';
  context.fetch = panelFetch();
  await context.unlockPanels();
  clock += 11 * 60 * 1000;
  nodes = [{selector: '#keysTable [data-tenant="team-a"]', checked: true, dataset: {provider: 'copilot'}}];
  requests = [];
  await context.saveTenantProviders('team-a');
  assert.equal(requests.length, 0, 'an expired key must not be sent');
  assert.match(element('keysStatus').textContent, /闲置超过 10 分钟/);
  assert.match(element('keysContent').innerHTML, /已锁定/);

  // Activity inside the window renews it rather than expiring on a fixed clock.
  clock = 6_000_000;
  element('keysAdminKey').value = 'lb-panel-key';
  context.fetch = panelFetch();
  await context.unlockPanels();
  clock += 9 * 60 * 1000;
  requests = [];
  context.fetch = panelFetch({'/admin/api-keys/team-a/providers': {ok: true, status: 200,
    json: async () => ({tenant: 'team-a', providers: ['copilot'], policy: {state: 'loaded'}})}});
  await context.saveTenantProviders('team-a');
  assert.ok(requests.length > 0, 'activity within the window must be allowed');
  clock += 9 * 60 * 1000;
  requests = [];
  await context.saveTenantProviders('team-a');
  assert.ok(requests.length > 0, 'the idle window is renewed by activity, not fixed at unlock');
  context.panelClock = () => Date.now();

  // A section that fails must not take the rest of the page down with it.
  element('keysAdminKey').value = 'lb-panel-key';
  requests = [];
  context.fetch = panelFetch({'/admin/credentials': {ok: false, status: 503,
    json: async () => ({detail: 'storage down'})}});
  await context.unlockPanels();
  assert.match(element('keysContent').innerHTML, /team-a/, 'the healthy sections still render');
  assert.match(element('credentialsContent').innerHTML, /暂时无法读取/);
  assert.match(element('keysStatus').textContent, /其余各块可用/);

  // A channel that is simply not configured reads as a fact, not a failure.
  element('keysAdminKey').value = 'lb-panel-key';
  context.fetch = panelFetch({'/admin/copilot/accounts': {ok: false, status: 404,
    json: async () => ({detail: {error: {message: 'GitHub Copilot is not configured'}}})}});
  await context.unlockPanels();
  assert.match(element('accountsContent').innerHTML, /该渠道未配置/);
  assert.doesNotMatch(element('keysStatus').textContent, /读取失败/);

  // A storage layer that was never set up answers 503 with a code; a broken one
  // answers 503 without. Treating both as a failure made every deployment
  // without credential storage show a warning on every unlock.
  element('keysAdminKey').value = 'lb-panel-key';
  context.fetch = panelFetch({'/admin/credentials': {ok: false, status: 503,
    json: async () => ({detail: {error: {code: 'credential_store_unavailable',
                                         message: 'Credential storage is not configured.'}}})}});
  await context.unlockPanels();
  assert.match(element('credentialsContent').innerHTML, /该渠道未配置/,
    'an unconfigured store is not a failure to report');
  assert.doesNotMatch(element('keysStatus').textContent, /读取失败/);

  element('keysAdminKey').value = 'lb-panel-key';
  context.fetch = panelFetch({'/admin/credentials': {ok: false, status: 503,
    json: async () => ({detail: {error: {code: 'storage_unreachable'}}})}});
  await context.unlockPanels();
  assert.match(element('credentialsContent').innerHTML, /暂时无法读取/,
    'a genuinely broken store must still be reported');
  assert.match(element('keysStatus').textContent, /读取失败/);

  // A wholly invalid key must not be retained, or every later write repeats it.
  element('keysAdminKey').value = 'wrong-key';
  const unauthorised = {ok: false, status: 401, json: async () => ({detail: 'Invalid API key'})};
  context.fetch = panelFetch(Object.fromEntries(PANEL_URLS.map(u => [u, unauthorised])));
  await context.unlockPanels();
  assert.match(element('keysStatus').textContent, /API Key 无效/);
  requests = [];
  await context.saveTenantProviders('team-a');
  assert.equal(requests.length, 0, 'a rejected key must not be kept');

  // Back to a usable state for the sections below.
  element('keysAdminKey').value = 'lb-panel-key';
  context.fetch = panelFetch();
  await context.unlockPanels();

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
  nodes = [{selector: '#accountsTable [data-enable="biz"]', checked: true, dataset: {}},
           {selector: '#accountsTable [data-account="biz"]', checked: false, dataset: {model: 'gpt-4o'}}];
  requests = [];
  await context.saveAccount('biz');
  assert.equal(requests.length, 0, 'an all-unchecked save must be refused locally');
  assert.match(element('keysStatus').textContent, /取消「启用」/);

  // Enabling an inert account with no capability must not reach the server:
  // it would be a wildcard candidate for every model until discovery runs.
  nodes = [{selector: '#accountsTable [data-enable="fresh"]', checked: true,
            dataset: {needsCapability: '1'}}];
  requests = [];
  await context.saveAccount('fresh');
  assert.equal(requests.length, 0, 'enabling without capability must be refused locally');
  assert.match(element('keysStatus').textContent, /先点「刷新能力」/);

  // Disabling that same account is always allowed — the guard is about enabling.
  nodes = [{selector: '#accountsTable [data-enable="fresh"]', checked: false,
            dataset: {needsCapability: '1'}}];
  requests = [];
  context.fetch = panelFetch({'/admin/copilot/accounts/fresh': {ok: true, status: 200,
    json: async () => ({account: {account: 'fresh', enabled: false, enabled_in_config: false,
      enablement_source: 'operator', capability: []},
      policy: {state: 'loaded', backend: 'mysql', tenants: 0}})}});
  await context.saveAccount('fresh');
  assert.equal(requests[0].url, '/admin/copilot/accounts/fresh',
    'an explicit disable must still be sendable');
  assert.equal(JSON.parse(requests[0].options.body).enabled, false);
  assert.ok(requests.some(r => r.url === '/admin/model-routing'),
    'an account change moves the effective candidates, so routing re-reads too');

  requests = [];
  context.fetch = panelFetch({'/admin/copilot/accounts/biz/discover': {ok: true, status: 200,
    json: async () => ({discovered: 38, policy: {state: 'loaded', backend: 'mysql', tenants: 0}})}});
  await context.discoverAccount('biz');
  assert.equal(requests[0].url, '/admin/copilot/accounts/biz/discover');
  assert.equal(requests[0].options.method, 'POST');
  assert.match(element('keysStatus').textContent, /非能力认证/);
  assert.ok(!requests.some(r => r.url.includes('lb-panel-key')));

  requests = [];
  context.fetch = panelFetch({'/admin/copilot/accounts/biz/discover': {ok: false, status: 503,
    json: async () => ({detail: {error: {code: 'policy_write_failed'}}})}});
  await context.discoverAccount('biz');
  assert.match(element('keysStatus').textContent, /已有快照保持不变/,
    'a failed discovery must not claim the snapshot changed');

  // ---- routing rules: model → provider → GHCP account ----
  assert.match(html, /id="routingContent"/, 'routing area needs a container');
  assert.match(html, /只重排不筛除/, 'reorder-not-filter must be stated for the provider segment');
  assert.match(html, /仍然只走 ADB/, 'the /v1/messages limit must be stated');
  assert.match(html, /没有列出的模型维持原本的规则/, 'the baseline for unlisted models must be stated');
  assert.match(html, /id="ruleModel"/, 'the model segment must be a control, not free text only');
  assert.match(html, /id="ruleProvider"/, 'the provider segment must be a control');
  assert.match(html, /id="ruleAccounts"/, 'the account segment needs a container');
  // ADB does not belong in the dropdown: the preference only reorders the
  // OpenAI-style entry points, so offering it would be an inert choice.
  const providerSelect = html.slice(html.indexOf('id="ruleProvider"'),
                                   html.indexOf('</select>', html.indexOf('id="ruleProvider"')));
  assert.doesNotMatch(providerSelect, /databricks/, 'a preference cannot reorder ADB');
  assert.match(providerSelect, /copilot/);
  assert.match(providerSelect, /azure_openai/);

  context.renderRouting({policy: {state: 'loaded', backend: 'mysql', tenants: 0},
    default_openai_order: ['copilot', 'azure_openai'], known_models: ['gpt-5.4'],
    copilot_accounts_available: [{account: 'acct-a', enabled: true}],
    preferences: [{model: 'gpt-5.4', providers: ['azure_openai'],
                   effective_openai_order: ['azure_openai', 'copilot'],
                   copilot_accounts: [], effective_copilot_accounts: ['acct-a'],
                   copilot_candidates_empty: false,
                   opens_claude_on_openai_entry_points: false},
                  {model: 'claude-opus-5.5', providers: ['copilot'],
                   effective_openai_order: ['copilot', 'azure_openai'],
                   copilot_accounts: ['acct-a'], effective_copilot_accounts: ['acct-a'],
                   copilot_candidates_empty: false,
                   opens_claude_on_openai_entry_points: true}]});
  assert.match(element('routingContent').innerHTML, /azure_openai → copilot/,
    'the effective order must be shown, not just the stored one');
  assert.match(element('routingContent').innerHTML, /已在 OpenAI 风格入口放行/,
    'a claude model opened by preference must be flagged');
  const gptRow = element('routingContent').innerHTML.split('<tr>').find(r => r.includes('gpt-5.4'));
  assert.doesNotMatch(gptRow, /已在 OpenAI 风格入口放行/, 'a non-claude model must not be flagged');
  assert.match(gptRow, /不限定/, 'no account limit must read as "not limited", not as empty');

  // An emptied candidate set is the one outcome an operator will not expect.
  context.renderRouting({policy: {state: 'loaded', backend: 'mysql', tenants: 0},
    default_openai_order: ['copilot', 'azure_openai'], known_models: [],
    copilot_accounts_available: [],
    preferences: [{model: 'gpt-5.4', providers: [], effective_openai_order: ['copilot', 'azure_openai'],
                   copilot_accounts: ['acct-z'], effective_copilot_accounts: [],
                   copilot_candidates_empty: true,
                   opens_claude_on_openai_entry_points: false}]});
  assert.match(element('routingContent').innerHTML, /GHCP 无候选/,
    'an emptied candidate set must be called out in the listing');

  context.renderRouting({policy: {state: 'loaded', backend: 'json', tenants: 0},
    default_openai_order: ['copilot', 'azure_openai'], known_models: ['<img onerror=bad>'],
    copilot_accounts_available: [{account: '<img onerror=bad>', enabled: true}],
    preferences: [{model: '<img onerror=bad>', providers: ['copilot'], effective_openai_order: [],
                   copilot_accounts: ['<img onerror=bad>'], effective_copilot_accounts: []}]});
  assert.doesNotMatch(element('routingContent').innerHTML, /<img onerror/, 'model names escaped');
  assert.doesNotMatch(element('ruleModel').innerHTML, /<img onerror/, 'dropdown options escaped');
  assert.doesNotMatch(element('ruleAccounts').innerHTML, /<img onerror/, 'account labels escaped');

  context.renderRouting({policy: {state: 'loaded', backend: 'json', tenants: 0},
    default_openai_order: ['copilot', 'azure_openai'], preferences: [],
    known_models: ['gpt-4o'], copilot_accounts_available: [{account: 'acct-a', enabled: true},
                                                           {account: 'acct-b', enabled: false}]});
  assert.match(element('routingContent').innerHTML, /全部模型按默认顺序/);
  assert.match(element('ruleAccounts').innerHTML, /一个都不勾 = 不限定/,
    'the empty-selection semantics must be on screen, not just in the API');
  assert.match(element('ruleAccounts').innerHTML, /未启用/,
    'an inert account must be labelled in the picker');

  // No model named at all must not reach the server.
  element('ruleModel').value = '';
  element('ruleModelCustom').value = '';
  requests = [];
  await context.saveRule();
  assert.equal(requests.length, 0, 'a rule with no model must be refused locally');
  assert.match(element('keysStatus').textContent, /手输模型名/);

  // Accounts alone, with no provider change, is a legitimate rule.
  element('ruleModel').value = 'gpt-4o';
  element('ruleProvider').value = '';
  nodes = [{selector: '#ruleAccounts [data-rule-account]', checked: true, dataset: {ruleAccount: 'acct-a'}}];
  requests = [];
  context.fetch = panelFetch({'/admin/model-routing/gpt-4o': {ok: true, status: 200,
    json: async () => ({model: 'gpt-4o', providers: [],
      effective_openai_order: ['copilot', 'azure_openai'],
      copilot_accounts: ['acct-a'], effective_copilot_accounts: ['acct-a'],
      copilot_candidates_empty: false, policy: {state: 'loaded'}})}});
  await context.saveRule();
  assert.equal(requests[0].url, '/admin/model-routing/gpt-4o');
  assert.equal(requests[0].options.method, 'PUT');
  let body = JSON.parse(requests[0].options.body);
  assert.deepEqual(body.copilot_accounts, ['acct-a']);
  assert.equal(body.providers, undefined, 'an unchanged provider segment must not be sent');
  assert.match(element('keysStatus').textContent, /生效候选/);

  // Unchecking everything must still be sent — that is how a limit is undone.
  nodes = [{selector: '#ruleAccounts [data-rule-account]', checked: false, dataset: {ruleAccount: 'acct-a'}}];
  requests = [];
  await context.saveRule();
  body = JSON.parse(requests[0].options.body);
  assert.deepEqual(body.copilot_accounts, [],
    'an empty selection must be sent explicitly, or clearing a limit is impossible');

  // The free-text field wins, so a model missing from the catalogue is reachable.
  element('ruleModel').value = 'gpt-4o';
  element('ruleModelCustom').value = 'gpt-7-unreleased';
  element('ruleProvider').value = 'copilot';
  nodes = [];
  requests = [];
  context.fetch = panelFetch({'/admin/model-routing/gpt-7-unreleased': {ok: true, status: 200,
    json: async () => ({model: 'gpt-7-unreleased', providers: ['copilot'],
      effective_openai_order: ['copilot', 'azure_openai'], copilot_accounts: [],
      effective_copilot_accounts: ['acct-a'], copilot_candidates_empty: false,
      policy: {state: 'loaded'}})}});
  await context.saveRule();
  assert.equal(requests[0].url, '/admin/model-routing/gpt-7-unreleased',
    'the free-text model name must take precedence over the dropdown');
  assert.deepEqual(JSON.parse(requests[0].options.body).providers, ['copilot']);
  assert.equal(element('ruleModelCustom').value, '', 'the one-off field is cleared after use');

  // A rule that empties the candidate set is applied but must be called out.
  element('ruleModelCustom').value = 'gpt-5.4';
  nodes = [{selector: '#ruleAccounts [data-rule-account]', checked: true, dataset: {ruleAccount: 'acct-b'}}];
  requests = [];
  context.fetch = panelFetch({'/admin/model-routing/gpt-5.4': {ok: true, status: 200,
    json: async () => ({model: 'gpt-5.4', providers: ['copilot'],
      effective_openai_order: ['copilot', 'azure_openai'], copilot_accounts: ['acct-b'],
      effective_copilot_accounts: [], copilot_candidates_empty: true,
      policy: {state: 'loaded'}})}});
  await context.saveRule();
  assert.match(element('keysStatus').textContent, /没有任何候选账户/,
    'an emptied candidate set must be reported even though the write succeeded');
  assert.match(element('keysStatus').textContent, /Foundry/, 'it must say where traffic goes instead');
  assert.equal(element('keysStatus').dataset.state, 'warn',
    'applied-but-probably-unintended is a warning, not a success');

  // A claude model pointed at GHCP changes both admission and pricing.
  element('ruleModelCustom').value = 'claude-opus-5.5';
  element('ruleProvider').value = 'copilot';
  nodes = [];
  let confirmText = '';
  context.confirm = text => { confirmText = text; return true; };
  requests = [];
  context.fetch = panelFetch({'/admin/model-routing/claude-opus-5.5': {ok: true, status: 200,
    json: async () => ({model: 'claude-opus-5.5', providers: ['copilot'],
      effective_openai_order: ['copilot', 'azure_openai'], copilot_accounts: [],
      effective_copilot_accounts: ['acct-a'], copilot_candidates_empty: false,
      opens_claude_on_openai_entry_points: true, policy: {state: 'loaded'}})}});
  await context.saveRule();
  assert.match(confirmText, /放行/, 'opening claude must be confirmed explicitly');
  assert.match(confirmText, /GHCP 分档价/, 'the pricing change must be in the confirmation');
  assert.match(element('keysStatus').textContent, /已在 OpenAI 风格入口放行/);
  context.confirm = () => true;

  // A rejected account name must read as "nothing was applied".
  element('ruleModelCustom').value = 'gpt-4o';
  nodes = [{selector: '#ruleAccounts [data-rule-account]', checked: true, dataset: {ruleAccount: 'typo'}}];
  requests = [];
  context.fetch = panelFetch({'/admin/model-routing/gpt-4o': {ok: false, status: 400,
    json: async () => ({detail: {error: {code: 'unknown_copilot_account',
      message: 'Not configured Copilot endpoints: typo'}}})}});
  await context.saveRule();
  assert.match(element('keysStatus').textContent, /规则未生效/,
    'a rejected rule must not look partially applied');

  requests = [];
  context.fetch = panelFetch({'/admin/model-routing/claude-opus-5.5': {ok: false, status: 503,
    json: async () => ({detail: {error: {code: 'policy_write_failed'}}})}});
  await context.clearRouting('claude-opus-5.5');
  assert.equal(requests[0].options.method, 'DELETE');
  assert.match(element('keysStatus').textContent, /规则保持不变/,
    'a failed clear must not claim the rule changed');

  requests = [];
  context.confirm = text => { confirmText = text; return true; };
  context.fetch = panelFetch({'/admin/model-routing/gpt-4o': {ok: true, status: 200,
    json: async () => ({model: 'gpt-4o', providers: [],
      effective_openai_order: ['copilot', 'azure_openai'], copilot_accounts: [],
      effective_copilot_accounts: ['acct-a'], copilot_candidates_empty: false,
      policy: {state: 'loaded'}})}});
  await context.clearRouting('gpt-4o');
  assert.match(confirmText, /整条规则/, 'clearing removes both segments and must say so');
  assert.match(element('keysStatus').textContent, /账户不限定/);
  context.confirm = () => true;

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
  // The upstream secret keeps the one-request contract even though the panel
  // key is now held: holding that one is a convenience, this one is a secret.
  element('credProvider').value = 'databricks';
  element('credAccount').value = 'ws-config';
  element('credSecret').value = 'dapi-should-not-persist';
  element('credApiBase').value = 'https://x.invalid';
  element('credDeployments').value = '';
  requests = [];
  context.fetch = panelFetch({'/admin/credentials/databricks/ws-config': {ok: false, status: 409,
    json: async () => ({detail: {error: {code: 'account_declared_in_config'}}})}});
  await context.submitCredential();
  assert.equal(requests[0].options.method, 'PUT');
  assert.equal(JSON.parse(requests[0].options.body).secret, 'dapi-should-not-persist');
  assert.match(element('keysStatus').textContent, /静默遮蔽/, 'a 409 must explain the shadowing rule');
  assert.equal(element('credSecret').value, '', 'the credential field must be cleared after submit');
  assert.ok(!requests.some(r => r.url.includes('dapi-should-not-persist')),
    'the credential must never travel in a URL');

  // Device flow: pending is not a failure, and expiry resets the handle.
  requests = [];
  context.fetch = panelFetch({'/admin/credentials/copilot/device/start': {ok: true, status: 200,
    json: async () => ({handle: 'h1', user_code: 'WXYZ-9876',
      verification_uri: 'https://github.com/login/device', expires_in: 899})}});
  await context.startDeviceFlow();
  assert.match(element('deviceStatus').textContent, /WXYZ-9876/);
  assert.match(element('deviceStatus').textContent, /github\.com\/login\/device/);

  element('deviceAccount').value = 'gh-account-5';
  context.fetch = panelFetch({'/admin/credentials/copilot/device/poll': {ok: true, status: 200,
    json: async () => ({pending: true, interval: 5})}});
  await context.pollDeviceFlow();
  assert.match(element('deviceStatus').textContent, /尚未授权完成/,
    'authorization_pending must read as waiting, not as an error');

  context.fetch = panelFetch({'/admin/credentials/copilot/device/poll': {ok: false, status: 410,
    json: async () => ({detail: {error: {code: 'device_flow_expired'}}})}});
  await context.pollDeviceFlow();
  assert.match(element('deviceStatus').textContent, /已过期/);

  element('deviceAccount').value = 'gh-account-5';
  requests = [];
  context.fetch = panelFetch();
  await context.pollDeviceFlow();
  assert.equal(requests.length, 0, 'an expired flow must not keep polling with a stale handle');

  // Locking must also stop the credential controls, not just the visible tables.
  context.lockPanels();
  element('credSecret').value = 'dapi-after-lock';
  requests = [];
  await context.submitCredential();
  assert.equal(requests.length, 0, 'a locked panel must not submit credentials');
  assert.equal(element('credSecret').value, '', 'the secret is cleared even on the refusal path');
  requests = [];
  await context.startDeviceFlow();
  assert.equal(requests.length, 0, 'a locked panel must not start a device flow');
  assert.match(element('deviceStatus').textContent, /请先解锁/);

  console.log('Dashboard UI: authenticated cleanup, honest errors, safe release identity, unlock/lock, key panel, accounts, routing rules, credentials passed.');
})().catch(error => {console.error(error); process.exitCode = 1;});
