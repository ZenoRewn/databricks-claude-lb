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
function clearElements(ids) { for (const id of ids) element(id).innerHTML = ''; }
function element(id) {
  if (!elements.has(id)) elements.set(id, {textContent: '', innerHTML: '', value: '', hidden: false,
    dataset: {}, classList: {add() {}, remove() {}, toggle() {}}, setAttribute(k, v) {this[k] = v;},
    removeAttribute(k) {delete this[k];}, addEventListener() {}, focus() {}, scrollIntoView() {},
    querySelector() {return null;}, querySelectorAll() {return [];},
    get options() {return [...this.innerHTML.matchAll(/<option[^>]*value="([^"]*)"/g)].map(m => ({value: m[1]}));}});
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
  // allow-list, and must report uncertainty when a write is not acknowledged.
  // Rotation must state that the gateway does not issue keys.
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

  // The session now rides an HttpOnly cookie, so the page has to say what it
  // holds (nothing) and where the credential comes from.
  assert.match(html, /HttpOnly cookie/, 'the session mechanism must be stated');
  assert.match(html, /页面脚本不持有任何凭据/, 'the no-credential property must be stated');
  assert.match(html, /LB_DASHBOARD_PASSWORD/, 'the password source must be named');
  assert.match(html, /不会退化为无密码放行/, 'the no-fallback-to-open rule must be stated');
  assert.match(html, /id="panelPassword"/, 'the panel needs a password field');
  assert.match(html, /type="password"[^>]*id="panelPassword"|id="panelPassword"[^>]*type="password"/,
    'the panel password must be a password input');
  // Scoped to this panel's own code: the theme toggle legitimately persists a
  // preference, so a whole-file check would be meaningless. Matching usage
  // (`x.`) rather than the bare word keeps the prose that names these stores.
  const keyPanelJs = section('// ---------- API keys & provider allow-lists',
                             '// ---------- Refresh ----------');
  assert.doesNotMatch(keyPanelJs, /localStorage\s*\.|sessionStorage\s*\.|document\s*\.\s*cookie/,
    'the panel must never reach a browser store');
  // The strongest form of "scripts hold no credential" is that there is no
  // variable to hold one. Writes must not attach an api-key header either —
  // except the one login call that trades a key for a cookie.
  assert.doesNotMatch(keyPanelJs, /let\s+adminKey|adminKeyAt/,
    'no variable may retain a credential any more');
  const keyHeaderUses = keyPanelJs.match(/'x-api-key'/g) || [];
  assert.equal(keyHeaderUses.length, 1,
    'only the login exchange may send an api-key header; found ' + keyHeaderUses.length);
  assert.match(keyPanelJs.slice(keyPanelJs.indexOf("'x-api-key'") - 400,
                                keyPanelJs.indexOf("'x-api-key'")),
    /\/admin\/session/, 'that single use must be the session exchange');

  context.renderKeys({providers: ['databricks', 'copilot'],
    provider_labels: {databricks: 'ADB', copilot: 'GHCP'},
    policy: {state: 'loaded', backend: 'mysql', tenants: 1},
    keys: [{tenant: 'team-a', masked: 'abcd…wxyz', fingerprint: 'deadbeefcafe',
            providers: ['copilot'], restricted: true, requests: 3, last_used_at: 1700000000}]});
  assert.match(element('keysContent').innerHTML, /team-a/);
  assert.match(element('keysContent').innerHTML, /abcd…wxyz/);
  assert.match(element('keysContent').innerHTML, /deadbeefcafe/);
  assert.match(element('keysPolicyState').innerHTML, /已加载/);
  assert.match(element('keysContent').innerHTML, /data-edit-key="team-a"/,
    'a restricted tenant needs an explicit edit action');
  context.openKeyDetail('team-a');
  assert.match(element('keyDetail').innerHTML, /data-reset="team-a"/,
    'the detail keeps the path back to the config baseline');

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

  // ---- panel session: unlock / lock ----
  // Unlocking trades a credential for an HttpOnly cookie once, then reads all
  // four sections. Nothing afterwards carries a credential: the browser owns
  // the session and these scripts cannot read it.
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
    // The candidate diagnostic: one row per account naming the layer that
    // decided. acct-b is enabled and capable but not selected; acct-c is
    // selected-eligible on paper but has no capability for this model.
    '/admin/model-routing/gpt-4o': {model: 'gpt-4o',
      effective_openai_order: ['copilot', 'azure_openai'],
      copilot_accounts: ['acct-a'], effective_copilot_accounts: ['acct-a'],
      copilot_candidates_empty: false,
      accounts: [
        {account: 'acct-a', enabled: true, enablement_source: 'operator', selected: true,
         serves_model: true, reason: null, serves_model_ignoring_selection: true,
         reason_ignoring_selection: null, capability_known: true},
        {account: 'acct-b', enabled: true, enablement_source: 'config', selected: false,
         serves_model: false, reason: 'not_selected_for_model',
         serves_model_ignoring_selection: true, reason_ignoring_selection: null,
         capability_known: true}],
      policy: {state: 'loaded', backend: 'mysql', tenants: 1}},
  };
  // Mirrors the server: GET reports state, POST mints a session, DELETE drops it.
  let session = {authenticated: false, password_configured: true};
  const ok = payload => ({ok: true, status: 200, json: async () => payload});
  const panelFetch = (overrides = {}) => async (url, options = {}) => {
    requests.push({url, options});
    if (overrides[url]) return overrides[url];
    if (url === '/admin/session') {
      const method = (options.method || 'GET').toUpperCase();
      if (method === 'POST') { session = {...session, authenticated: true}; return ok({authenticated: true, expires_in: 1800}); }
      if (method === 'DELETE') { session = {...session, authenticated: false}; return ok({authenticated: false}); }
      return ok({authenticated: session.authenticated, expires_in: session.authenticated ? 1800 : 0,
                 password_configured: session.password_configured,
                 password_source: session.password_source
                   || (session.password_configured ? 'env' : 'none')});
    }
    if (!session.authenticated) return {ok: false, status: 401, json: async () => ({detail: 'no session'})};
    const payload = PANEL_PAYLOADS[url];
    if (!payload) return {ok: false, status: 404, json: async () => ({detail: 'no stub'})};
    return ok(payload);
  };

  // A password unlock sends the password once and never a key header.
  session = {authenticated: false, password_configured: true};
  element('panelPassword').value = 'panel-password-value';
  element('keysAdminKey').value = '';
  requests = [];
  context.fetch = panelFetch();
  await context.unlockPanels(false);
  const login = requests[0];
  assert.equal(login.url, '/admin/session');
  assert.equal(login.options.method, 'POST');
  assert.equal(JSON.parse(login.options.body).password, 'panel-password-value');
  assert.equal(login.options.headers['x-api-key'], undefined,
    'a password login must not also send a key');
  assert.ok(!requests.some(r => r.url.includes('panel-password-value')),
    'the password must never travel in a URL');
  assert.deepEqual(requests.filter(r => PANEL_URLS.includes(r.url)).map(r => r.url).sort(),
    [...PANEL_URLS].sort(), 'unlocking reads every section exactly once');
  assert.ok(requests.filter(r => PANEL_URLS.includes(r.url))
    .every(r => !(r.options.headers || {})['x-api-key']),
    'section reads ride the cookie, not a header');
  assert.equal(element('panelPassword').value, '', 'the password field is cleared after use');
  assert.match(element('panelSessionBar').textContent, /已解锁/);
  assert.match(element('keysContent').innerHTML, /team-a/);
  assert.match(element('accountsContent').innerHTML, /acct-a/);
  assert.match(element('ruleModel').innerHTML, /gpt-4o/, 'the model dropdown is populated on unlock');
  // The matrix is per model, so with none chosen it says so rather than showing
  // verdicts that belong to no model.
  assert.doesNotMatch(element('ruleAccounts').innerHTML, /data-rule-account|data-global-enable/,
    'the matrix must not render verdicts before a model is chosen');
  await context.loadRuleIntoEditor('gpt-4o');
  assert.match(element('ruleAccounts').innerHTML, /acct-a/, 'the matrix lists every account');
  assert.match(element('ruleAccounts').innerHTML, /acct-b/);
  assert.match(element('panelSessionBar').textContent, /已解锁/, 'the session bar shows the state');
  assert.match(element('panelSessionBar').textContent, /分钟/, 'and how long is left');

  // A write afterwards carries no credential at all.
  nodes = [{selector: '#keyDetail [data-tenant]', checked: true, dataset: {tenant: 'team-a', provider: 'copilot'}}];
  requests = [];
  context.fetch = panelFetch({'/admin/api-keys/team-a/providers': {ok: true, status: 200,
    json: async () => ({tenant: 'team-a', providers: ['copilot'],
                        policy: {state: 'loaded', backend: 'mysql', tenants: 1}})}});
  await context.saveTenantProviders('team-a');
  assert.equal(requests[0].url, '/admin/api-keys/team-a/providers');
  assert.equal((requests[0].options.headers || {})['x-api-key'], undefined,
    'the cookie authorises the write; no credential is in the page');
  assert.equal(requests[1].url, '/admin/api-keys',
    'the section re-reads itself so the result is visible without a button');

  // A live session survives a reload: no field, no click, contents restored.
  session = {authenticated: true, password_configured: true};
  clearElements(['keysContent', 'routingContent', 'accountsContent', 'credentialsContent']);
  requests = [];
  context.fetch = panelFetch();
  const restored = await context.refreshPanelSession();
  assert.equal(restored, true, 'a live session must be detected on load');
  await context.loadAllPanels();
  assert.match(element('keysContent').innerHTML, /team-a/,
    'the panel restores itself without any credential being entered');
  assert.ok(!requests.some(r => (r.options.headers || {})['x-api-key']));

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
  assert.match(element('clientkeysFeedback').textContent, /配置基线/,
    'reset must name what it returns to, not just say "done"');

  requests = [];
  context.fetch = panelFetch({'/admin/api-keys/team-a/providers': {ok: false, status: 503,
    json: async () => ({detail: {error: {code: 'policy_write_failed'}}})}});
  await context.resetTenantProviders('team-a');
  assert.match(element('clientkeysFeedback').textContent, /未能确认.*结果/,
    'a failed reset must report that the enforced policy needs verification');
  assert.doesNotMatch(element('clientkeysFeedback').textContent, /原有限制仍然生效|已回到配置基线/,
    'a failed reset must not guess the enforced policy');

  nodes = [{selector: '#keyDetail [data-tenant]', checked: true, dataset: {tenant: 'team-a', provider: 'copilot'}}];
  requests = [];
  context.fetch = panelFetch({'/admin/api-keys/team-a/providers': {ok: false, status: 503,
    json: async () => ({detail: {error: {code: 'policy_write_failed'}}})}});
  await context.saveTenantProviders('team-a');
  assert.match(element('clientkeysFeedback').textContent, /未能确认.*结果/,
    'a failed write must not guess whether a lost acknowledgement changed policy');
  assert.doesNotMatch(element('clientkeysFeedback').textContent, /执行中的策略保持不变/,
    'a transport failure is not proof that the prior policy remains enforced');

  nodes = [{selector: '#keyDetail [data-tenant]', checked: false, dataset: {tenant: 'team-a', provider: 'copilot'}}];
  requests = [];
  await context.saveTenantProviders('team-a');
  assert.equal(requests.length, 0, 'an empty allow-list must not reach the server');
  assert.match(element('clientkeysFeedback').textContent, /至少要允许一个渠道/);
  assert.match(element('clientkeysFeedback').textContent, /恢复配置渠道|重置/, 'it must point at the control that does work');

  // Rotating the source may invalidate the key behind the session, so the
  // panel locks rather than keeping something that may no longer work.
  requests = [];
  context.fetch = panelFetch({'/admin/api-keys/reload': {ok: true, status: 200,
    json: async () => ({keys: 2, tenants: ['default', 'team-a'], env_interpolated: true})}});
  await context.reloadKeysFromSource();
  assert.equal(requests[0].url, '/admin/api-keys/reload');
  assert.equal(requests[0].options.method, 'POST');
  assert.match(element('panelLockFeedback').textContent, /滚动重启/,
    'env-sourced values still need a restart and the panel must say so');
  assert.match(element('panelLockFeedback').textContent, /已自动锁定/);
  assert.equal(element('keysContent').innerHTML, '', 'locking clears the identity material');
  assert.ok(requests.some(r => r.url === '/admin/session' && (r.options.method || '') === 'DELETE'),
    'locking must invalidate the session server-side, not just locally');

  nodes = [{selector: '#keyDetail [data-tenant]', checked: true, dataset: {tenant: 'team-a', provider: 'copilot'}}];
  requests = [];
  await context.saveTenantProviders('team-a');
  assert.equal(requests.filter(r => r.url !== '/admin/session').length, 0,
    'a locked panel must not issue writes');
  assert.match(element('clientkeysFeedback').textContent, /请先解锁/);

  // Lock clears every section, not just the session.
  session = {authenticated: false, password_configured: true};
  element('panelPassword').value = 'panel-password-value';
  context.fetch = panelFetch();
  await context.unlockPanels(false);
  assert.match(element('accountsContent').innerHTML, /acct-a/);
  await context.lockPanels();
  for (const id of ['keysContent', 'routingContent', 'accountsContent', 'credentialsContent']) {
    assert.equal(element(id).innerHTML, '', id + ' must be cleared on lock');
  }
  assert.equal(element('keysPolicyState').innerHTML, '');
  assert.equal(element('ruleAccounts').innerHTML, '');

  // Where no password is configured, the API Key is the way in — and it is
  // exchanged for the same cookie rather than kept in the page.
  session = {authenticated: false, password_configured: false};
  await context.refreshPanelSession();
  element('panelPassword').value = '';
  element('keysAdminKey').value = 'lb-fallback-key';
  requests = [];
  context.fetch = panelFetch();
  await context.unlockPanels(true);
  assert.equal(requests[0].url, '/admin/session');
  assert.equal(requests[0].options.headers['x-api-key'], 'lb-fallback-key');
  assert.equal(element('keysAdminKey').value, '', 'the key field is cleared after the exchange');
  assert.ok(requests.filter(r => PANEL_URLS.includes(r.url))
    .every(r => !(r.options.headers || {})['x-api-key']),
    'after the exchange even the fallback path holds nothing');
  assert.match(element('keysContent').innerHTML, /team-a/);

  // Neither field filled in: refuse locally and name the one that applies.
  session = {authenticated: false, password_configured: false};
  await context.refreshPanelSession();
  element('panelPassword').value = '';
  element('keysAdminKey').value = '';
  requests = [];
  await context.unlockPanels(false);
  assert.equal(requests.filter(r => r.url !== '/admin/session').length, 0);
  assert.match(element('panelLockFeedback').textContent, /LB API Key/,
    'with no password configured the prompt must point at the key');
  session = {authenticated: false, password_configured: true};
  await context.refreshPanelSession();
  await context.unlockPanels(false);
  assert.match(element('panelLockFeedback').textContent, /面板密码/,
    'with a password configured the prompt must point at the password');

  // A rejected password must leave the panel locked and say which it was.
  element('panelPassword').value = 'wrong-password';
  context.fetch = panelFetch({'/admin/session': {ok: false, status: 401,
    json: async () => ({detail: {error: {code: 'invalid_panel_password'}}})}});
  await context.unlockPanels(false);
  assert.match(element('panelLockFeedback').textContent, /密码不正确/);
  requests = [];
  nodes = [{selector: '#keyDetail [data-tenant]', checked: true, dataset: {tenant: 'team-a', provider: 'copilot'}}];
  await context.saveTenantProviders('team-a');
  assert.equal(requests.filter(r => r.url !== '/admin/session').length, 0,
    'a rejected login must not leave the panel usable');

  // A lockout is a distinct, actionable message rather than "invalid".
  element('panelPassword').value = 'wrong-password';
  context.fetch = panelFetch({'/admin/session': {ok: false, status: 429,
    json: async () => ({detail: {error: {code: 'too_many_attempts'}}})}});
  await context.unlockPanels(false);
  assert.match(element('panelLockFeedback').textContent, /过于频繁/);

  // A password typed where none is configured must explain the fallback.
  element('panelPassword').value = 'panel-password-value';
  context.fetch = panelFetch({'/admin/session': {ok: false, status: 503,
    json: async () => ({detail: {error: {code: 'panel_password_not_configured'}}})}});
  await context.unlockPanels(false);
  assert.match(element('panelLockFeedback').textContent, /LB API Key/);

  // A section that fails must not take the rest of the page down with it.
  session = {authenticated: false, password_configured: true};
  element('panelPassword').value = 'panel-password-value';
  requests = [];
  context.fetch = panelFetch({'/admin/credentials': {ok: false, status: 503,
    json: async () => ({detail: 'storage down'})}});
  await context.unlockPanels(false);
  assert.match(element('keysContent').innerHTML, /team-a/, 'the healthy sections still render');
  assert.match(element('credentialsContent').innerHTML, /暂时无法读取/);
  assert.match(element('clientkeysFeedback').textContent, /其余各块可用/);

  // A channel that is simply not configured reads as a fact, not a failure.
  session = {authenticated: false, password_configured: true};
  element('panelPassword').value = 'panel-password-value';
  context.fetch = panelFetch({'/admin/copilot/accounts': {ok: false, status: 404,
    json: async () => ({detail: {error: {message: 'GitHub Copilot is not configured'}}})}});
  await context.unlockPanels(false);
  assert.match(element('accountsContent').innerHTML, /该渠道未配置/);
  assert.ok(element('clientkeysFeedback').hidden || !/读取失败/.test(element('clientkeysFeedback').textContent));

  // A storage layer that was never set up answers 503 with a code; a broken one
  // answers 503 without. Treating both as a failure made every deployment
  // without credential storage show a warning on every unlock.
  session = {authenticated: false, password_configured: true};
  element('panelPassword').value = 'panel-password-value';
  context.fetch = panelFetch({'/admin/credentials': {ok: false, status: 503,
    json: async () => ({detail: {error: {code: 'credential_store_unavailable',
                                         message: 'Credential storage is not configured.'}}})}});
  await context.unlockPanels(false);
  assert.match(element('credentialsContent').innerHTML, /该渠道未配置/,
    'an unconfigured store is not a failure to report');
  assert.ok(element('clientkeysFeedback').hidden || !/读取失败/.test(element('clientkeysFeedback').textContent));

  session = {authenticated: false, password_configured: true};
  element('panelPassword').value = 'panel-password-value';
  context.fetch = panelFetch({'/admin/credentials': {ok: false, status: 503,
    json: async () => ({detail: {error: {code: 'storage_unreachable'}}})}});
  await context.unlockPanels(false);
  assert.match(element('credentialsContent').innerHTML, /暂时无法读取/,
    'a genuinely broken store must still be reported');
  assert.match(element('clientkeysFeedback').textContent, /读取失败/);

  // Back to a usable state for the sections below.
  session = {authenticated: true, password_configured: true};
  await context.refreshPanelSession();
  context.fetch = panelFetch();
  await context.loadAllPanels();


  // ---- the gate ----
  // Locked means the content is not rendered, not merely blurred. Blurring
  // leaves it in the tab order and screen readers still read it in full; the
  // usual patch for that — aria-hidden on the wrapper — is itself a WCAG 4.1.2
  // failure (axe `aria-hidden-focus`) because the wrapper holds focusable
  // controls. So `hidden` (display:none) is the primitive, and the frosted
  // glass is only the card's styling.
  assert.match(html, /id="panelLock"/, 'the gate needs its own card');
  assert.match(html, /id="panelBody"[^>]*hidden/,
    'the gated content must start hidden, not merely styled as unavailable');
  const panelBodyMarkup = html.slice(html.indexOf('id="panelBody"'),
                                     html.indexOf('</section>', html.indexOf('id="sub-panel"')));
  // aria-hidden is fine on a decorative span; it is forbidden on a focusable
  // element or on any ancestor of one. So the rule to enforce is that every use
  // inside the gated area is a self-contained decorative span, and that the
  // containers themselves never carry it.
  for (const tag of (panelBodyMarkup.match(/<[a-z]+[^>]*aria-hidden="true"[^>]*>/g) || [])) {
    assert.match(tag, /^<span class="(step-no|lock-mark)"/,
      'aria-hidden may only mark a decorative span, never a container: ' + tag);
  }
  assert.doesNotMatch(panelBodyMarkup.slice(0, panelBodyMarkup.indexOf('>')), /aria-hidden/,
    'aria-hidden on a container holding focusable controls is a WCAG 4.1.2 failure');
  assert.doesNotMatch(html.slice(html.indexOf('id="panelLock"'), html.indexOf('id="panelBody"')),
    /aria-modal/, 'this is a page section, not a dialog, so it must not claim modality');
  // The blur is decoration on the card itself, never the gate.
  assert.match(html, /\.lock-card[^}]*backdrop-filter/,
    'the frosted glass belongs to the card');

  session = {authenticated: false, password_configured: true};
  context.fetch = panelFetch();
  await context.refreshPanelSession();
  assert.equal(element('panelLock').hidden, false, 'locked shows the card');
  assert.equal(element('panelBody').hidden, true, 'locked hides everything else');
  assert.match(element('panelLockTitle').textContent, /面板密码/);
  session = {authenticated: true, password_configured: true};
  await context.refreshPanelSession();
  assert.equal(element('panelLock').hidden, true, 'unlocked hides the card');
  assert.equal(element('panelBody').hidden, false, 'unlocked reveals the content');

  // With no password configured the key is the only way in, so it must not be
  // buried behind a closed disclosure.
  session = {authenticated: false, password_configured: false};
  await context.refreshPanelSession();
  assert.match(element('panelLockTitle').textContent, /LB API Key/);
  assert.equal(element('panelKeyFallback').open, true,
    'the only usable field must be open');
  assert.equal(element('unlockPanels').hidden, true,
    'a password button is pointless where no password exists');
  session = {authenticated: false, password_configured: true};
  await context.refreshPanelSession();
  assert.equal(element('panelKeyFallback').open, false,
    'with a password configured the key is a fallback, not the headline');

  // ---- sub-tabs ----
  // Four surfaces used to share one screen and one feedback line at the top, so
  // acting on the bottom section put the result out of view.
  for (const id of ['sub-clientkeys', 'sub-routing', 'sub-accounts', 'sub-credentials',
                    'sub-panel']) {
    assert.match(html, new RegExp('id="' + id + '"'), id + ' must exist');
  }
  assert.match(html, /role="tablist"[^>]*aria-label="密钥与渠道"/,
    'the sub-tabs must be a labelled tablist');
  for (const id of ['clientkeysFeedback', 'routingFeedback', 'accountsFeedback',
                    'credentialsFeedback', 'panelFeedback']) {
    assert.match(html, new RegExp('id="' + id + '"'), id + ' must exist');
  }
  assert.doesNotMatch(html, /id="keysStatus"/,
    'the shared status line is what put feedback off-screen; it must be gone');
  // Observable behaviour rather than the module-level `let`: the selected tab
  // is the one whose button reports aria-selected="true".
  let subtabSelected = [];
  const subtabStubs = ['clientkeys', 'routing', 'accounts', 'credentials', 'panel'].map(name => ({
    selector: '.subtab-btn', dataset: {subtab: name},
    classList: {add() {}, remove() {}, toggle() {}},
    setAttribute(key, value) {
      if (key === 'aria-selected' && value === 'true') subtabSelected.push(name);
    },
  }));
  nodes = [...subtabStubs,
           {selector: '.subtab-panel', id: 'sub-routing',
            classList: {add() {}, remove() {}, toggle() {}}}];
  context.switchSubtab('routing');
  assert.deepEqual(subtabSelected, ['routing'], 'exactly one tab may be selected');
  subtabSelected = [];
  context.switchSubtab('not-a-tab');
  assert.deepEqual(subtabSelected, [], 'an unknown name must change nothing');
  nodes = [];

  // ---- what the client-key page can and cannot do ----
  // The honest answer to "where is the reveal button" is that there is none,
  // and why. Saying so is the fix; adding a button that cannot work is not.
  assert.match(html, /不提供明文查看或密钥重置/, 'the page must state the unsupported key operations');
  assert.match(html, /不提供明文查看/, 'plaintext revealing remains explicitly unavailable');
  assert.match(html, /不生成也不存储/, 'and that the gateway does not issue keys');
  assert.match(html, /源头轮换与重读/, 'rotation must be explained, not just refused');
  assert.match(html, /sha256 前缀/, 'the fingerprint column must say what it is');

  // ---- panel password ----
  assert.match(html, /id="panelPasswordForm"/, 'changing the password needs a form');
  assert.match(html, /单向验证器/, 'storing a verifier rather than the password must be stated');
  assert.match(html, /忘记即不可恢复/, 'the irreversibility must be stated before use');
  assert.match(html, /其他副本.*旧密码|只在启动时加载/,
    'the multi-replica caveat must explain that another replica may still accept the old password');
  assert.match(html, /是字符数，不是字节数|字符<\/strong>而不是字节/,
    'the characters-not-bytes trap must be stated');

  session = {authenticated: true, password_configured: true, password_source: 'env'};
  await context.refreshPanelSession();
  assert.match(element('panelPasswordSource').textContent, /环境变量/,
    'the page must say where the current password comes from');
  session = {authenticated: true, password_configured: true, password_source: 'store'};
  await context.refreshPanelSession();
  assert.match(element('panelPasswordSource').textContent, /存储/);

  // Mismatched confirmation never reaches the server.
  element('pwCurrent').value = 'current-password-value';
  element('pwNew').value = 'a-brand-new-password';
  element('pwRepeat').value = 'a-brand-new-passwerd';
  requests = [];
  await context.changePanelPassword();
  assert.equal(requests.length, 0, 'a typo in the confirmation must be caught locally');
  assert.match(element('panelFeedback').textContent, /不一致/);

  // The floor counts characters, matching the gateway. `kubectl describe
  // secret` reports bytes, which is how a short multi-byte password looks long.
  element('pwNew').value = '短密码';
  element('pwRepeat').value = '短密码';
  requests = [];
  await context.changePanelPassword();
  assert.equal(requests.length, 0);
  assert.match(element('panelFeedback').textContent, /12 个字符/);
  element('pwCurrent').value = 'current-password-value';
  element('pwNew').value = '密码密码密码密码密码密码';
  element('pwRepeat').value = '密码密码密码密码密码密码';
  requests = [];
  context.fetch = panelFetch({'/admin/panel-password': {ok: true, status: 200,
    json: async () => ({configured: true, password_source: 'store',
                        other_sessions_invalidated: true})}});
  await context.changePanelPassword();
  assert.equal(requests[0].url, '/admin/panel-password',
    'twelve multi-byte characters are long enough');
  assert.equal(requests[0].options.method, 'PUT');

  // A successful change says the other sessions are gone and that the Secret
  // still needs updating, because other replicas read it at startup.
  element('pwCurrent').value = 'current-password-value';
  element('pwNew').value = 'a-brand-new-password';
  element('pwRepeat').value = 'a-brand-new-password';
  requests = [];
  await context.changePanelPassword();
  const change = JSON.parse(requests[0].options.body);
  assert.equal(change.new_password, 'a-brand-new-password');
  assert.equal(change.current_password, 'current-password-value');
  assert.equal((requests[0].options.headers || {})['x-api-key'], undefined,
    'the cookie authorises this; no key is held in the page');
  assert.equal(element('pwNew').value, '', 'the new password is cleared after use');
  assert.equal(element('pwCurrent').value, '', 'and so is the current one');
  assert.equal(element('pwRepeat').value, '');
  assert.match(element('panelFeedback').textContent, /其他会话已失效/);
  assert.match(element('panelFeedback').textContent, /Secret/,
    'other replicas keep the old password until they restart, so say so');

  // A wrong current password and a lockout are different, actionable messages.
  for (const [status, code, pattern] of [
      [401, 'invalid_panel_password', /当前密码不正确/],
      [429, 'too_many_attempts', /退避期内正确密码也会被拒/],
      [503, 'policy_store_unavailable', /改环境变量/]]) {
    element('pwCurrent').value = 'current-password-value';
    element('pwNew').value = 'a-brand-new-password';
    element('pwRepeat').value = 'a-brand-new-password';
    context.fetch = panelFetch({'/admin/panel-password': {ok: false, status,
      json: async () => ({detail: {error: {code}}})}});
    await context.changePanelPassword();
    assert.match(element('panelFeedback').textContent, pattern, code + ' needs its own message');
  }

  // Clearing the overlay falls back to the environment, and when that is unset
  // the honest report is that password login is now unavailable.
  element('pwClearCurrent').value = 'current-password-value';
  requests = [];
  context.fetch = panelFetch({'/admin/panel-password': {ok: true, status: 200,
    json: async () => ({configured: true, password_source: 'env'})}});
  await context.clearPanelPassword();
  assert.equal(requests[0].options.method, 'DELETE');
  assert.match(element('panelFeedback').textContent, /环境变量/);
  assert.equal(element('pwClearCurrent').value, '');

  element('pwClearCurrent').value = 'current-password-value';
  context.fetch = panelFetch({'/admin/panel-password': {ok: true, status: 200,
    json: async () => ({configured: false, password_source: 'none'})}});
  await context.clearPanelPassword();
  assert.match(element('panelFeedback').textContent, /不可用/,
    'with no baseline left, password login is unavailable and must be reported');
  assert.equal(element('panelFeedback').dataset.state, 'warn');

  // Neither password may ever appear in a URL.
  assert.ok(!requests.some(r => r.url.includes('password-value')
                                || r.url.includes('a-brand-new-password')),
    'a password must never travel in a URL');

  session = {authenticated: true, password_configured: true};
  await context.refreshPanelSession();
  context.fetch = panelFetch();
  await context.loadAllPanels();

  // ---- GHCP accounts ----
  assert.match(html, /id="accountsContent"/, 'account area needs a container');
  assert.match(html, /capability 是向上游的时点发现/, 'capability must be labelled as a snapshot');

  context.renderAccounts({policy: {state: 'loaded', backend: 'mysql', tenants: 0},
    accounts: [{account: 'biz', enabled: true, token_source: 'file', requests: 7,
                capability: ['gpt-4o', 'kimi-k3'], capability_at: 1700000000,
                allowed_models: ['gpt-4o']}]});
  assert.match(element('accountsContent').innerHTML, /biz/);
  assert.match(element('accountsContent').innerHTML, /已启用/);
  context.openAccountDetail('biz');
  assert.match(element('accountDetail').innerHTML, /kimi-k3/);
  assert.doesNotMatch(element('accountsContent').innerHTML, /ghu_|Bearer/, 'no token material');

  context.renderAccounts({policy: {state: 'loaded', backend: 'json', tenants: 0},
    accounts: [{account: '<img onerror=bad>', enabled: false, token_source: null, requests: 0,
                capability: [], capability_at: null, allowed_models: null}]});
  assert.doesNotMatch(element('accountsContent').innerHTML, /<img onerror/, 'account names escaped');
  context.openAccountDetail('<img onerror=bad>');
  assert.doesNotMatch(element('accountDetail').innerHTML, /<img onerror/, 'detail account names escaped');
  assert.match(element('accountDetail').innerHTML, /尚未发现能力|没有已发现模型/);
  assert.match(element('accountsContent').innerHTML, /已停用/);

  // An account shipped inert reads differently from one an operator switched
  // off: both are out of service, but only the second was a decision.
  context.renderAccounts({policy: {state: 'loaded', backend: 'mysql', tenants: 0},
    accounts: [{account: 'fresh', enabled: false, enabled_in_config: false,
                enablement_source: 'config', token_source: 'file', requests: 0,
                configured_models: [], capability: [], capability_at: null, allowed_models: null}]});
  context.openAccountDetail('fresh');
  let inert = element('accountsContent').innerHTML + element('accountDetail').innerHTML;
  assert.match(inert, /配置声明不投入服务/, 'an inert account must not read as operator-disabled');
  assert.doesNotMatch(inert, /已停用/);
  assert.match(inert, /启用前(?:请)?先「刷新能力」/, 'enabling it blind would make it a candidate for everything');
  assert.match(inert, /data-needs-capability="1"/, 'the guard must be reachable from the control');

  // Once capability is known the warning goes away; discovery alone must not
  // flip the account to enabled.
  context.renderAccounts({policy: {state: 'loaded', backend: 'mysql', tenants: 0},
    accounts: [{account: 'fresh', enabled: false, enabled_in_config: false,
                enablement_source: 'config', token_source: 'file', requests: 0,
                configured_models: [], capability: ['gpt-4o'], capability_at: 1700000000,
                allowed_models: null}]});
  context.openAccountDetail('fresh');
  inert = element('accountsContent').innerHTML + element('accountDetail').innerHTML;
  assert.match(inert, /配置声明不投入服务/, 'discovery is not an enablement');
  assert.doesNotMatch(inert, /data-needs-capability/, 'capability is known now');
  assert.doesNotMatch(inert, /启用前(?:请)?先「刷新能力」/);

  // Unchecking every model must not reach the server — disabling is a separate control.
  nodes = [{selector: '#accountDetail [data-enable]', checked: true, dataset: {enable: 'biz'}},
           {selector: '#accountDetail [data-account]', checked: false, dataset: {account: 'biz', model: 'gpt-4o'}}];
  requests = [];
  await context.saveAccount('biz');
  assert.equal(requests.length, 0, 'an all-unchecked save must be refused locally');
  assert.match(element('accountsFeedback').textContent, /取消「启用」/);

  // Enabling an inert account with no capability must not reach the server:
  // it would be a wildcard candidate for every model until discovery runs.
  nodes = [{selector: '#accountDetail [data-enable]', checked: true,
            dataset: {enable: 'fresh', needsCapability: '1'}},
           {selector: '#accountDetail [data-account-mode]', checked: true, dataset: {accountMode: 'all'}}];
  requests = [];
  await context.saveAccount('fresh');
  assert.equal(requests.length, 0, 'enabling without capability must be refused locally');
  assert.match(element('accountsFeedback').textContent, /先点「刷新能力」/);

  // Disabling that same account is always allowed — the guard is about enabling.
  nodes = [{selector: '#accountDetail [data-enable]', checked: false,
            dataset: {enable: 'fresh', needsCapability: '1'}},
           {selector: '#accountDetail [data-account-mode]', checked: true, dataset: {accountMode: 'all'}}];
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
  assert.match(element('accountsFeedback').textContent, /非能力认证/);
  assert.ok(!requests.some(r => r.url.includes('lb-panel-key')));

  requests = [];
  context.fetch = panelFetch({'/admin/copilot/accounts/biz/discover': {ok: false, status: 503,
    json: async () => ({detail: {error: {code: 'policy_write_failed'}}})}});
  await context.discoverAccount('biz');
  assert.match(element('accountsFeedback').textContent, /未能确认能力发现结果/,
    'a failed discovery must report that the persisted snapshot needs verification');
  assert.doesNotMatch(element('accountsFeedback').textContent, /已有快照保持不变/,
    'a missing acknowledgement cannot prove the persisted snapshot stayed unchanged');

  // ---- routing surface: preserve the shared-panel safety boundary ----
  // Stateful routing regressions live in test_dashboard_routing_ui.cjs. They
  // replace the former route/global two-write tests with the stricter contract:
  // routing cannot edit global enablement, and uncertain model writes require
  // readback instead of claiming that the old policy stayed unchanged.
  for (const id of ['routingContent', 'ruleModel', 'ruleProvider', 'ruleAccounts',
                    'ruleAccountMode', 'routeReview']) {
    assert.match(html, new RegExp('id="' + id + '"'), id + ' must exist');
  }
  assert.match(html, /配置候选/, 'configuration eligibility must be named precisely');
  assert.match(html, /时点判断|时点发现/, 'the candidate evidence is point-in-time');
  assert.match(html, /不代表实时流量/, 'eligibility must not masquerade as live traffic');
  assert.match(html, /仍只走 ADB|仍然只走 ADB/, 'Messages stays Databricks-only');
  const routingJs = section('// ---------- Routing rules:', '// A verifier is stored');
  assert.doesNotMatch(routingJs, /data-global-enable|writeAccountEnablement\(/,
    'the model editor must never write a global account setting');
  assert.doesNotMatch(routingJs, /请求将落到|请求会落到/,
    'a missing GHCP candidate does not prove a usable fallback exists');

  // ---- upstream credentials ----
  assert.match(html, /id="credentialsContent"/, 'credential area needs a container');
  assert.match(html, /不会退化为存明文/, 'the no-plaintext-fallback rule must be stated');
  assert.match(html, /不显示也不返回明文/, 'the panel must state plaintext is never shown');
  assert.match(html, /需.*滚动重启|滚动重启.*生效/, 'activation semantics must be stated');
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
  // The upstream secret keeps its one-request contract while the browser
  // carries the panel session cookie; neither is retained by page scripts.
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
  assert.match(element('credentialsFeedback').textContent, /静默遮蔽/, 'a 409 must explain the shadowing rule');
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
  await context.lockPanels();
  element('credSecret').value = 'dapi-after-lock';
  requests = [];
  await context.submitCredential();
  assert.equal(requests.filter(r => r.url !== '/admin/session').length, 0,
    'a locked panel must not submit credentials');
  assert.equal(element('credSecret').value, '', 'the secret is cleared even on the refusal path');
  requests = [];
  await context.startDeviceFlow();
  assert.equal(requests.filter(r => r.url !== '/admin/session').length, 0,
    'a locked panel must not start a device flow');
  assert.match(element('deviceStatus').textContent, /请先解锁/);

  // ---- Foundry Claude surface ----
  // The 2026-10-17 release made azure_anthropic a /v1/messages candidate but
  // gave it no stats section, no metrics and no tab: the provider that serves
  // traffic when ADB and GHCP are both down was the only invisible one.
  assert.match(html, /id="nav-foundry"[^>]*data-tab="foundry"/,
    'Foundry Claude needs its own tab button');
  assert.match(html, /id="tab-foundry"[^>]*role="tabpanel"/,
    'Foundry Claude needs its own tab panel');
  assert.match(html, /id="nav-foundry"[^>]*aria-controls="tab-foundry"/,
    'the tab button must point at its panel like the other five');
  assert.match(html, /id="tab-foundry"[^>]*aria-labelledby="nav-foundry"/,
    'the panel must name its button, or it is unlabelled in the a11y tree');
  assert.match(html, /foundry: \['Foundry Claude'/,
    'switchTab refuses a tab missing from TAB_INFO, so the title must be declared');
  assert.match(html, /id="foundryContent"/, 'the panel needs a render host');
  // Foundry bills in CCU. Reusing the per-token cost KPI would price it at
  // Anthropic's public rate; a zero would roll up as «free».
  assert.match(html, /CCU/, 'the CCU basis must be stated, not left blank');
  assert.ok(!/renderKpiGrid\('foundryGlobalGrid', KPI_DEFS,/.test(html),
    'the cost KPI must be filtered out of the Foundry grid');
  // Each KPI grid builds element ids from its prefix. Reusing 'kpi-azure' makes
  // getElementById hand Foundry's numbers to the Azure panel's nodes, so this
  // tab renders «—» forever while silently overwriting the other one.
  const kpiPrefixes = [...html.matchAll(/renderKpiGrid\('(\w+)GlobalGrid',[^)]*'(kpi-[\w-]+)'\)/g)]
    .map(m => ({grid: m[1], prefix: m[2]}));
  assert.ok(kpiPrefixes.length >= 3, 'the KPI grids must stay discoverable for this check');
  const prefixes = kpiPrefixes.map(e => e.prefix);
  assert.equal(new Set(prefixes).size, prefixes.length,
    'two KPI grids sharing an id prefix collide in the DOM: ' + prefixes.join(', '));

  console.log('Dashboard UI: authenticated cleanup, honest errors, safe release identity, '
    + 'panel gate (content hidden, not blurred), sub-tabs with per-section feedback, '
    + 'panel session (password + cookie, zero-interaction restore, key fallback), '
    + 'client keys, accounts (enablement vs configuration candidates), routing scope '
    + 'safety (stateful regressions in test_dashboard_routing_ui.cjs), panel password, credentials passed.');
})().catch(error => {console.error(error); process.exitCode = 1;});
