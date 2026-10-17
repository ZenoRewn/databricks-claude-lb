// Routing editor contracts: offline DOM/HTTP fixtures only. Author: Zeno Ren.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../dashboard.html'), 'utf8');
const clone = value => JSON.parse(JSON.stringify(value));
const tick = () => new Promise(resolve => setImmediate(resolve));
const POLICY = {state: 'loaded', backend: 'mysql', tenants: 1};
const account = (name, overrides = {}) => ({account: name, enabled: true,
  enablement_source: 'operator', selected: false, serves_model: true, reason: null,
  serves_model_ignoring_selection: true, reason_ignoring_selection: null,
  capability_known: true, requests: 0, circuit_open: false, ...overrides});
function diagnostic(model, overrides = {}) {
  return {model, providers: [], effective_openai_order: ['copilot', 'azure_openai'],
    copilot_accounts: [], effective_copilot_accounts: ['acct-a', 'acct-b'],
    copilot_candidates_empty: false, opens_claude_on_openai_entry_points: false,
    effective_messages_order: ['databricks', 'copilot', 'azure_anthropic'],
    messages_paths: [
      {provider: 'databricks', label: 'ADB', configured: true,
       forwarded_model: 'databricks-claude-opus-5-5', endpoints_total: 9,
       endpoints_serving_model: 9, reason: null},
      {provider: 'copilot', label: 'GHCP', configured: true,
       forwarded_model: 'claude-opus-5.5', endpoints_total: 5,
       endpoints_serving_model: 0, reason: 'no_endpoint_permits_model'},
      {provider: 'azure_anthropic', label: 'Foundry Claude', configured: false,
       forwarded_model: 'claude-opus-5-5', endpoints_total: 0,
       endpoints_serving_model: 0, reason: 'provider_not_configured'},
    ],
    accounts: [account('acct-a'), account('acct-b'), account('acct-off', {
      enabled: false, serves_model: false, reason: 'account_disabled',
      serves_model_ignoring_selection: false, reason_ignoring_selection: 'account_disabled'}),
      account('acct-unknown', {capability_known: false})],
    policy: POLICY, counters_are_process_local: true, ...overrides};
}
const seedDiagnostics = {
  'gpt-4o': diagnostic('gpt-4o', {providers: ['azure_openai'],
    effective_openai_order: ['azure_openai', 'copilot'], copilot_accounts: ['acct-a']}),
  'gpt-5.4': diagnostic('gpt-5.4'),
  'claude-opus-5.5': diagnostic('claude-opus-5.5', {providers: ['copilot'],
    opens_claude_on_openai_entry_points: true}),
};
function section(from, to) {
  assert.ok(html.includes(from) && html.includes(to), `Missing script section ${from}`);
  return html.slice(html.indexOf(from), html.indexOf(to));
}
function harness(overrides = {}) {
  const elements = new Map();
  let routeNodes = [];
  const listeners = new Map();
  function element(id) {
    if (!elements.has(id)) {
      const node = {id, value: '', textContent: '', innerHTML: '', hidden: false,
        checked: false, disabled: false, dataset: {}, style: {}, tabIndex: 0,
        classList: {add() {}, remove() {}, toggle() {}, contains() {return false;}},
        setAttribute(key, value) {this[key] = value;},
        removeAttribute(key) {delete this[key];},
        addEventListener(type, listener) {listeners.set(`${id}:${type}`, listener);},
        focus() {}, scrollIntoView() {}, querySelector() {return null;},
        querySelectorAll() {return [];}};
      Object.defineProperty(node, 'options', {get() {
        return [...this.innerHTML.matchAll(/<option[^>]*value="([^"]*)"/g)]
          .map(match => ({value: match[1]}));
      }});
      elements.set(id, node);
    }
    return elements.get(id);
  }
  const requests = [];
  const snapshots = clone(seedDiagnostics);
  const routingPayload = () => ({default_openai_order: ['copilot', 'azure_openai'],
    provider_labels: {copilot: 'GHCP', azure_openai: 'Foundry', databricks: 'ADB'},
    known_models: [...Object.keys(snapshots), 'constructor', '__proto__'],
    copilot_accounts_available: [account('acct-a'), account('acct-b'),
      account('acct-off', {enabled: false}), account('acct-unknown')],
    preferences: Object.values(snapshots).filter(row => row.providers.length || row.copilot_accounts.length),
    policy: POLICY});
  const ok = body => ({ok: true, status: 200, json: async () => clone(body)});
  const failure = (status, code = 'policy_write_failed') => ({ok: false, status,
    json: async () => ({detail: {error: {code, message: code}}})});
  const context = {URL, AbortController, setTimeout, clearTimeout, console,
    historyLoaded: false, loadHistory: async () => {},
    confirm: () => true, alert: () => {throw Error('Use inline feedback');},
    document: {getElementById: element, querySelectorAll(selector) {
      if (selector.includes('[data-rule-account]')) return routeNodes;
      return [];
    }, querySelector(selector) {
      return this.querySelectorAll(selector)[0] || null;
    }}};
  let handler = overrides.handler;
  context.fetch = async (url, options = {}) => {
    const method = (options.method || 'GET').toUpperCase();
    const request = {url, options, method};
    requests.push(request);
    if (handler) {
      const result = await handler(request, {snapshots, ok, failure, routingPayload});
      if (result !== undefined) return result;
    }
    if (url === '/admin/session') return ok({authenticated: true, expires_in: 1800,
      password_configured: true, password_source: 'env'});
    if (url === '/admin/model-routing') return ok(routingPayload());
    if (url.startsWith('/admin/model-routing/')) {
      const model = decodeURIComponent(url.slice('/admin/model-routing/'.length));
      if (method === 'PUT') {
        const body = JSON.parse(options.body);
        const current = snapshots[model] || diagnostic(model);
        const accounts = body.copilot_accounts || [];
        const effective = current.accounts.filter(row => row.serves_model_ignoring_selection
          && (!accounts.length || accounts.includes(row.account))).map(row => row.account);
        snapshots[model] = {...current, copilot_accounts: accounts,
          effective_copilot_accounts: effective,
          copilot_candidates_empty: !!current.accounts.length && !effective.length,
          ...(body.providers ? {providers: body.providers,
            effective_openai_order: [...body.providers,
              ...['copilot', 'azure_openai'].filter(p => !body.providers.includes(p))]} : {}),
          opens_claude_on_openai_entry_points: /claude/i.test(model)
            && (body.providers || current.providers).some(p => ['copilot', 'azure_openai'].includes(p))};
        return ok(snapshots[model]);
      }
      if (method === 'DELETE') {
        snapshots[model] = {...(snapshots[model] || diagnostic(model)), providers: [],
          effective_openai_order: ['copilot', 'azure_openai'], copilot_accounts: [],
          opens_claude_on_openai_entry_points: false};
        return ok(snapshots[model]);
      }
      return ok(snapshots[model] || diagnostic(model));
    }
    if (url === '/admin/copilot/accounts') return ok({accounts: [], policy: POLICY});
    if (url === '/admin/api-keys') return ok({keys: [], policy: POLICY});
    if (url === '/admin/credentials') return ok({credentials: [], store: {key: 'present'}});
    return failure(404, 'no_synthetic_fixture');
  };
  vm.createContext(context);
  vm.runInContext(section('// ---------- Formatters', '// ---------- Number Ticker')
    + section('// ---------- Release identity', '// ---------- Refresh ----------'), context);
  vm.runInContext('panelSession = {authenticated:true, expiresAt:Date.now()+1800000, '
    + 'passwordConfigured:true,passwordSource:"env"}', context);
  context.renderRouting(routingPayload());
  function editorState() {
    return vm.runInContext('typeof routeEditorState === "undefined" ? null : routeEditorState', context);
  }
  async function select(model) {
    element('ruleModel').value = '';
    element('ruleModelCustom').value = model;
    await context.loadRuleIntoEditor(model);
    await tick();
    await tick();
    return editorState();
  }
  function edit({provider = '', mode = 'selected', accounts = ['acct-a']} = {}) {
    element('ruleProvider').value = provider;
    element('ruleAccountMode').value = mode;
    routeNodes = [...new Set(['acct-a', 'acct-b', 'acct-off', 'acct-unknown', ...accounts])]
      .map(name => ({checked: accounts.includes(name), disabled: false,
        dataset: {ruleAccount: name}, addEventListener() {}}));
    context.renderRulePreview();
  }
  async function save() {
    await context.saveRule();
    assert.equal(typeof context.confirmRouteSave, 'function', 'route saves require the inline review action');
    element('routeRiskAck').checked = true;
    await context.confirmRouteSave();
    await tick();
  }
  return {context, element, requests, snapshots, ok, failure, routingPayload,
    editorState, select, edit, save, handler(fn) {handler = fn;},
    mutations: () => requests.filter(r => ['PUT', 'POST', 'DELETE'].includes(r.method)),
    text: () => element('routingFeedback').textContent};
}
const cases = [];
function test(name, body) {cases.push({name, body});}

test('routing state and explicit scope are present; global enablement is read-only', async () => {
  const h = harness();
  assert.ok(h.editorState(), 'the editor needs one model-scoped state object');
  assert.match(html, /id="ruleAccountMode"/);
  await h.select('gpt-4o');
  assert.doesNotMatch(h.element('ruleAccounts').innerHTML, /data-global-enable/,
    'the route editor must not offer a global account write');
  assert.match(h.element('ruleAccounts').innerHTML, /只读|全局状态/);
  assert.match(h.element('ruleAccounts').innerHTML, /acct-off/);
  assert.doesNotMatch(h.element('ruleProvider').innerHTML, /databricks/,
    'ADB must not appear as an inert OpenAI-style provider preference');
});

test('holding provider order preserves an existing Foundry-first rule in preview and payload', async () => {
  const h = harness();
  await h.select('gpt-4o');
  h.edit({provider: '', accounts: ['acct-b']});
  h.context.renderOrderPreview();
  const chain = h.element('ruleOrderPreview').innerHTML;
  assert.ok(chain.indexOf('Foundry') < chain.indexOf('GHCP'), 'keep must preview the stored order');
  await h.save();
  const writes = h.mutations();
  assert.equal(writes.length, 1);
  assert.equal(writes[0].url, '/admin/model-routing/gpt-4o');
  const body = JSON.parse(writes[0].options.body);
  assert.equal(body.providers, undefined, 'keep means omit providers, never reset them');
  assert.deepEqual(body.copilot_accounts, ['acct-b']);
  assert.ok(!writes.some(r => r.url.startsWith('/admin/copilot/accounts/')));
});

test('an explicit provider selection preserves the other provider as a later candidate', async () => {
  const h = harness();
  await h.select('gpt-5.4');
  h.edit({provider: 'azure_openai', mode: 'all', accounts: []});
  h.context.renderOrderPreview();
  const chain = h.element('ruleOrderPreview').innerHTML;
  assert.ok(chain.indexOf('Foundry') < chain.indexOf('GHCP'));
  assert.match(chain, /保留|后续候选/);
  await h.save();
  assert.deepEqual(JSON.parse(h.mutations()[0].options.body).providers, ['azure_openai']);
});

test('selected mode with zero accounts is invalid and cannot issue a write', async () => {
  const h = harness();
  await h.select('gpt-4o');
  h.edit({mode: 'selected', accounts: []});
  assert.match(h.element('rulePreview').textContent, /至少|待选择|无效|请选择/);
  assert.equal(h.element('saveRule').disabled, true);
  await h.context.saveRule();
  if (h.context.confirmRouteSave) await h.context.confirmRouteSave();
  assert.equal(h.mutations().length, 0);
});

test('all mode explicitly clears the account limit, including when no checkbox is selected', async () => {
  const h = harness();
  await h.select('gpt-4o');
  h.edit({mode: 'all', accounts: []});
  await h.save();
  const body = JSON.parse(h.mutations()[0].options.body);
  assert.deepEqual(body.copilot_accounts, []);
  assert.equal(body.providers, undefined);
});

test('candidate preview intersects the server verdict and distinguishes unknown capability', async () => {
  const h = harness();
  h.snapshots['gpt-4o'] = diagnostic('gpt-4o', {accounts: [account('acct-a'),
    account('acct-b', {serves_model: false, reason: 'not_in_capability',
      serves_model_ignoring_selection: false, reason_ignoring_selection: 'not_in_capability'}),
    account('acct-unknown', {capability_known: false})]});
  await h.select('gpt-4o');
  h.edit({accounts: ['acct-b']});
  assert.match(h.element('rulePreview').textContent, /无候选|没有.*候选|候选.*(?:无|0)/);
  assert.doesNotMatch(h.element('rulePreview').textContent, /请求将落到|请求会落到/,
    'an empty GHCP set does not prove Foundry deployment or key permission');
  h.edit({accounts: ['acct-unknown']});
  assert.match(h.element('rulePreview').textContent, /acct-unknown/);
  assert.match(h.element('ruleAccounts').innerHTML, /尚未发现|能力未知/);
  assert.doesNotMatch(h.element('ruleAccounts').innerHTML, />接流量</,
    'configuration eligibility must not be labelled actual traffic');
});

test('a selected disabled account stays non-candidate and routing never enables it', async () => {
  const h = harness();
  h.snapshots['gpt-4o'].copilot_accounts = ['acct-off'];
  await h.select('gpt-4o');
  h.edit({provider: 'copilot', accounts: ['acct-off']});
  assert.match(h.element('rulePreview').textContent, /无候选|没有.*候选|候选.*(?:无|0)/);
  await h.save();
  assert.equal(h.mutations().length, 1);
  assert.ok(h.mutations().every(r => r.url.startsWith('/admin/model-routing/')));
  assert.match(h.text(), /GHCP.*(?:无.*候选|候选.*0)/);
  assert.equal(h.element('routingFeedback').dataset.state, 'warn',
    'an applied rule with zero GHCP candidates needs an explicit warning');
});

test('diagnostic fetch failure blocks writes rather than offering an unchecked save', async () => {
  const h = harness({handler: (r, api) => r.method === 'GET' && r.url.endsWith('/gpt-4o')
    ? api.failure(503, 'diagnostic_unavailable') : undefined});
  await h.select('gpt-4o');
  h.edit({accounts: ['acct-b']});
  await h.context.saveRule();
  if (h.context.confirmRouteSave) await h.context.confirmRouteSave();
  assert.equal(h.mutations().length, 0);
  assert.equal(h.element('saveRule').disabled, true);
  assert.match(h.text() + h.element('rulePreview').textContent, /诊断|读取|重试/);
});

test('a diagnostic for another model cannot authorise the current model', async () => {
  const h = harness({handler: (r, api) => r.method === 'GET' && r.url.endsWith('/gpt-4o')
    ? api.ok(diagnostic('gpt-5.4')) : undefined});
  await h.select('gpt-4o');
  h.edit({accounts: ['acct-b']});
  await h.context.saveRule();
  if (h.context.confirmRouteSave) await h.context.confirmRouteSave();
  assert.equal(h.mutations().length, 0);
  assert.equal(h.element('saveRule').disabled, true);
});

test('late diagnostics never overwrite the newly selected model', async () => {
  let resolveOld;
  const old = new Promise(resolve => {resolveOld = resolve;});
  const h = harness({handler: (r, api) => {
    if (r.method === 'GET' && r.url.endsWith('/gpt-4o')) return old;
    if (r.method === 'GET' && r.url.endsWith('/gpt-5.4')) return api.ok(diagnostic('gpt-5.4', {
      accounts: [account('new-model-account')]}));
  }});
  const first = h.context.loadRuleIntoEditor('gpt-4o');
  // Attach a rejection observer immediately; an early implementation failure
  // should be reported by this case, not abort the runner as an unhandled task.
  Promise.resolve(first).catch(() => {});
  await tick();
  const second = h.context.loadRuleIntoEditor('gpt-5.4');
  await second;
  resolveOld(h.ok(diagnostic('gpt-4o', {accounts: [account('old-model-account')]})));
  await first;
  await tick();
  assert.equal(h.editorState()?.model, 'gpt-5.4');
  assert.equal(h.editorState()?.diagnostic?.model, 'gpt-5.4');
  assert.doesNotMatch(h.element('ruleAccounts').innerHTML, /old-model-account/);
});

test('route edits require inline review and cancellation sends no write', async () => {
  const h = harness();
  await h.select('gpt-4o');
  h.edit({accounts: ['acct-b']});
  await h.context.saveRule();
  assert.equal(h.mutations().length, 0, 'opening review must not already issue PUT');
  assert.ok(h.editorState()?.review, 'the review is visible editor state');
  assert.equal(typeof h.context.cancelRouteReview, 'function');
  h.context.cancelRouteReview();
  assert.ok(!h.editorState()?.review);
  assert.equal(h.mutations().length, 0);
});

test('an acknowledged PUT followed by failed readback reports uncertainty without replay', async () => {
  let wrote = false;
  const h = harness({handler: (r, api) => {
    if (r.method === 'PUT') {wrote = true; return api.ok(diagnostic('gpt-4o'));}
    if (wrote && r.method === 'GET' && r.url.startsWith('/admin/model-routing')) return api.failure(503);
  }});
  await h.select('gpt-4o');
  h.edit({accounts: ['acct-b']});
  await h.save();
  assert.equal(h.mutations().length, 1);
  assert.match(h.text(), /已.*(?:写入|确认|保存|ACK)/);
  assert.match(h.text(), /读回|重新读取|核验|确认实际状态/);
  assert.doesNotMatch(h.text(), /规则保持不变|规则未生效|执行中的规则保持不变/);
});

test('an uncertain PUT is not replayed and readback can expose a partial application', async () => {
  let wrote = false;
  const h = harness({handler: (r, api) => {
    if (r.method === 'PUT') {wrote = true; throw new TypeError('Synthetic ACK lost');}
    if (wrote && r.method === 'GET' && r.url.endsWith('/gpt-4o')) {
      return api.ok(diagnostic('gpt-4o', {providers: ['copilot'],
        copilot_accounts: ['acct-a'], effective_openai_order: ['copilot', 'azure_openai']}));
    }
  }});
  await h.select('gpt-4o');
  h.edit({provider: 'copilot', accounts: ['acct-b']});
  await h.save();
  assert.equal(h.mutations().length, 1, 'an uncertain management write must not be replayed');
  const writeIndex = h.requests.findIndex(r => r.method === 'PUT');
  assert.ok(h.requests.slice(writeIndex + 1).some(r => r.method === 'GET' && r.url.endsWith('/gpt-4o')),
    'a pre-write diagnostic is not evidence of post-failure readback');
  assert.match(h.text(), /部分|不一致|未完全|实际状态|读回/);
  assert.doesNotMatch(h.text(), /规则保持不变|执行中的规则保持不变|规则未生效/);
});

test('an uncertain PUT and failed readback preserve an unknown result', async () => {
  let wrote = false;
  const h = harness({handler: (r, api) => {
    if (r.method === 'PUT') {wrote = true; throw new TypeError('Synthetic ACK lost');}
    if (wrote && r.method === 'GET' && r.url.startsWith('/admin/model-routing')) return api.failure(503);
  }});
  await h.select('gpt-4o');
  h.edit({accounts: ['acct-b']});
  await h.save();
  assert.equal(h.mutations().length, 1);
  assert.match(h.text(), /未知|无法确认|需确认|不确定/);
  assert.doesNotMatch(h.text(), /保持不变|规则未生效|保存成功/);
});

test('whole-rule reset is reviewed and clears provider and account scope together', async () => {
  const h = harness();
  await h.select('claude-opus-5.5');
  await h.context.clearRouting('claude-opus-5.5');
  assert.equal(h.mutations().length, 0, 'review precedes DELETE');
  const reviewMarkup = [...['routingReview', 'routeReview', 'routingContent', 'ruleReview']]
    .map(id => h.element(id).innerHTML).join('') + JSON.stringify(h.editorState()?.review);
  assert.match(reviewMarkup, /整条|reset|clear/);
  assert.equal(typeof h.context.confirmRouteSave, 'function');
  h.element('routeRiskAck').checked = true;
  await h.context.confirmRouteSave();
  assert.equal(h.mutations().length, 1);
  assert.equal(h.mutations()[0].method, 'DELETE');
  assert.equal(h.mutations()[0].url, '/admin/model-routing/claude-opus-5.5');
  assert.deepEqual(h.snapshots['claude-opus-5.5'].providers, []);
  assert.deepEqual(h.snapshots['claude-opus-5.5'].copilot_accounts, []);
  assert.equal(h.snapshots['claude-opus-5.5'].opens_claude_on_openai_entry_points, false);
});

test('a risky reset cannot execute until its visible impact acknowledgement is checked', async () => {
  const h = harness();
  await h.select('claude-opus-5.5');
  await h.context.clearRouting('claude-opus-5.5');
  assert.equal(typeof h.context.confirmRouteSave, 'function');
  h.element('routeRiskAck').checked = false;
  await h.context.confirmRouteSave();
  assert.equal(h.mutations().length, 0, 'unchecked risk acknowledgement must fence DELETE');
  assert.ok(h.editorState()?.review);
  h.element('routeRiskAck').checked = true;
  await h.context.confirmRouteSave();
  assert.equal(h.mutations().length, 1);
});

test('failed DELETE is never replayed and cannot claim the old rule is unchanged', async () => {
  let deleting = false;
  const h = harness({handler: (r, api) => {
    if (r.method === 'DELETE') {deleting = true; return api.failure(503);}
    if (deleting && r.method === 'GET' && r.url.startsWith('/admin/model-routing')) return api.failure(503);
  }});
  await h.select('gpt-4o');
  await h.context.clearRouting('gpt-4o');
  assert.equal(typeof h.context.confirmRouteSave, 'function');
  h.element('routeRiskAck').checked = true;
  await h.context.confirmRouteSave();
  assert.equal(h.mutations().length, 1);
  assert.match(h.text(), /未知|无法确认|需确认|不确定/);
  assert.doesNotMatch(h.text(), /规则保持不变|已回到默认|恢复成功/);
});

test('changing tabs with a dirty route requires the discard decision', async () => {
  const h = harness();
  h.context.switchSubtab('routing');
  await h.select('gpt-4o');
  h.edit({accounts: ['acct-b']});
  let prompts = 0;
  h.context.confirm = () => {prompts++; return false;};
  h.context.switchSubtab('accounts');
  assert.equal(prompts, 1);
  assert.equal(vm.runInContext('activeSubtab', h.context), 'routing');
  assert.equal(h.editorState()?.dirty, true);
  h.context.confirm = () => true;
  h.context.switchSubtab('accounts');
  assert.equal(vm.runInContext('activeSubtab', h.context), 'accounts');
  assert.equal(h.mutations().length, 0);
});

test('an account checkbox keeps keyboard focus when the candidate matrix is rerendered', async () => {
  const h = harness();
  await h.select('gpt-4o');
  h.edit({mode: 'selected', accounts: ['acct-a']});
  const node = h.context.document.querySelectorAll('#ruleMatrix [data-rule-account]')[0];
  let focused = 0;
  node.focus = () => {focused++;};
  node.checked = false;
  h.context.renderRulePreview({target: node});
  assert.equal(focused, 1,
    'changing a checkbox must restore focus to that account after replacing the matrix DOM');
  assert.equal(h.mutations().length, 0);
});

test('unlisted model names remain configurable without treating catalogue absence as rejection', async () => {
  const h = harness();
  await h.select('gpt-unlisted-synthetic');
  h.edit({provider: 'copilot', mode: 'all', accounts: []});
  await h.save();
  assert.equal(h.mutations()[0]?.url, '/admin/model-routing/gpt-unlisted-synthetic');
  assert.deepEqual(JSON.parse(h.mutations()[0].options.body).providers, ['copilot']);
});

test('a missing model is refused before review or any mutation', async () => {
  const h = harness();
  h.element('ruleModel').value = '';
  h.element('ruleModelCustom').value = '';
  await h.context.saveRule();
  assert.equal(h.mutations().length, 0);
  assert.ok(!h.editorState()?.review);
  assert.match(h.text(), /模型|选择/);
});

test('distinct exclusion reasons keep their distinct repair directions', async () => {
  const h = harness();
  const reasons = ['account_disabled', 'account_never_enabled', 'not_selected_for_model',
    'not_in_capability', 'excluded_by_allowed_models', 'not_in_configured_models',
    'api_type_unsupported'];
  const rows = reasons.map(reason => account(`row-${reason}`, {serves_model: false,
    enabled: !['account_disabled', 'account_never_enabled'].includes(reason),
    reason, serves_model_ignoring_selection: reason === 'not_selected_for_model',
    reason_ignoring_selection: reason === 'not_selected_for_model' ? null : reason,
    capability_known: reason !== 'account_never_enabled'}));
  h.snapshots['gpt-4o'] = diagnostic('gpt-4o', {accounts: rows,
    copilot_accounts: rows.filter(row => row.reason !== 'not_selected_for_model').map(row => row.account)});
  await h.select('gpt-4o');
  const matrix = h.element('ruleAccounts').innerHTML;
  for (const reason of reasons) assert.match(matrix, new RegExp(`row-${reason}`));
  assert.match(matrix, /已停用|运维.*停用/);
  assert.match(matrix, /从未启用|未启用/);
  assert.match(matrix, /本模型.*(?:未选|没选)|模型选择/);
  assert.match(matrix, /能力|上游/);
  assert.match(matrix, /允许模型/);
  assert.match(matrix, /配置|config/);
  assert.match(matrix, /入口|api_types/);
  assert.match(matrix, /重启|配置.*修改|修改.*配置/);
});

test('a rejected account name remains a single rejected write and keeps the validation reason', async () => {
  const h = harness({handler: (r, api) => r.method === 'PUT'
    ? api.failure(400, 'unknown_copilot_account') : undefined});
  await h.select('gpt-4o');
  h.edit({accounts: ['unknown-typo']});
  await h.save();
  assert.equal(h.mutations().length, 1);
  assert.match(h.text(), /unknown_copilot_account|400|拒绝|验证/);
  assert.doesNotMatch(h.text(), /保存成功|已.*全部生效/);
});

test('opening a Claude model is explicitly reviewed without implying Messages routing changed', async () => {
  const h = harness();
  h.snapshots['claude-new-synthetic'] = diagnostic('claude-new-synthetic');
  await h.select('claude-new-synthetic');
  h.edit({provider: 'copilot', mode: 'all', accounts: []});
  await h.context.saveRule();
  const review = [...['routingReview', 'routeReview', 'routingContent', 'ruleReview']]
    .map(id => h.element(id).innerHTML).join('');
  assert.match(review, /放行|开放/);
  assert.match(review, /Messages|messages/);
  assert.match(review, /ADB|Databricks/);
  assert.match(review, /GHCP.*(?:分档|参考)|计价/,
    'opening Claude must preserve the channel-specific pricing notice');
  assert.equal(h.mutations().length, 0);
});

test('a legacy ADB-only preference does not open Claude and still requires acknowledgement on first GHCP opt-in', async () => {
  const h = harness();
  const model = 'claude-adb-only-synthetic';
  h.snapshots[model] = diagnostic(model, {providers: ['databricks'],
    opens_claude_on_openai_entry_points: false});
  await h.select(model);
  assert.match(h.element('routeEntryImpact').textContent, /未.*(?:放行|开放)/,
    'a nonempty preference does not prove it names an OpenAI-style provider');
  assert.doesNotMatch(h.element('ruleProvider').innerHTML, /value="databricks"/);
  h.edit({provider: 'copilot', mode: 'all', accounts: []});
  await h.context.saveRule();
  assert.equal(h.editorState()?.review?.risk, true,
    'the first real OpenAI-style opt-in must not inherit permission from ADB');
  assert.match(h.element('routeReview').innerHTML, /开放|放行/);
  h.element('routeRiskAck').checked = false;
  await h.context.confirmRouteSave();
  assert.equal(h.mutations().length, 0);
  h.element('routeRiskAck').checked = true;
  await h.context.confirmRouteSave();
  assert.equal(h.mutations().length, 1);
  assert.equal(h.mutations()[0].method, 'PUT');
  assert.deepEqual(JSON.parse(h.mutations()[0].options.body).providers, ['copilot']);
  assert.equal(h.snapshots[model].opens_claude_on_openai_entry_points, true);
});

test('untrusted model and account names remain escaped in list, choices, and diagnostic', async () => {
  const h = harness();
  const model = '<img onerror=bad>';
  const d = diagnostic(model, {accounts: [account(model)], copilot_accounts: [model],
    effective_copilot_accounts: [model]});
  h.snapshots[model] = d;
  h.context.renderRouting(h.routingPayload());
  await h.select(model);
  for (const id of ['routingContent', 'ruleModel', 'ruleAccounts']) {
    assert.doesNotMatch(h.element(id).innerHTML, /<img onerror/, `${id} must escape external values`);
  }
});

test('locking fences a pending diagnostic and clears model-specific identities', async () => {
  let resolveRead;
  const deferred = new Promise(resolve => {resolveRead = resolve;});
  const h = harness({handler: r => r.method === 'GET' && r.url.endsWith('/gpt-4o')
    ? deferred : undefined});
  const reading = h.context.loadRuleIntoEditor('gpt-4o');
  await tick();
  await h.context.lockPanels();
  resolveRead(h.ok(diagnostic('gpt-4o', {accounts: [account('must-stay-cleared')]})));
  await reading;
  await tick();
  assert.equal(h.element('ruleAccounts').innerHTML, '');
  assert.equal(h.element('ruleModelCustom').value, '');
  assert.ok(!h.editorState()?.model);
  assert.ok(!h.editorState()?.diagnostic);
});

test('the Anthropic Messages surface lists all three paths, including blocked ones', async () => {
  const h = harness();
  await h.select('claude-opus-5.5');
  const chain = h.element('ruleMessagesPaths').innerHTML;
  // ADB used to appear nowhere at all, so an operator could not tell whether
  // that was deliberate or an oversight. Absence was the bug.
  for (const label of ['ADB', 'GHCP', 'Foundry Claude']) {
    assert.ok(chain.includes(label), `the messages surface must name ${label}`);
  }
  assert.match(chain, /data-state="blocked"/,
    'a provider that cannot serve the model must be marked, not hidden');
});

test('each messages path shows the model name that provider actually receives', async () => {
  const h = harness();
  await h.select('claude-opus-5.5');
  const chain = h.element('ruleMessagesPaths').innerHTML;
  // One model, three spellings. Configuring Foundry deployments with GHCP's
  // dotted spelling returns 404 DeploymentNotFound, so this must be visible.
  assert.ok(chain.includes('databricks-claude-opus-5-5'));
  assert.ok(chain.includes('claude-opus-5.5'));
  assert.ok(chain.includes('claude-opus-5-5'));
});

test('the messages note names which paths are unusable and why', async () => {
  const h = harness();
  await h.select('claude-opus-5.5');
  const note = h.element('ruleMessagesNote').textContent;
  assert.match(note, /2 条不可用/);
  assert.match(note, /GHCP/);
  assert.match(note, /Foundry Claude/);
  assert.match(note, /没有端点允许本模型|未配置该渠道/);
});

test('the OpenAI-style order is labelled as that surface, not as the only one', async () => {
  // The old copy said "只调整 GHCP / Foundry 顺序" with no mention of a second
  // surface, which is what made the omission of ADB unreadable.
  assert.match(html, /OpenAI 风格入口/);
  assert.match(html, /Anthropic Messages 面/);
  assert.match(html, /v1\/messages/);
});

async function run() {
  let failed = 0;
  for (const {name, body} of cases) {
    try {await body(); console.log(`PASS ${name}`);}
    catch (error) {failed++; console.error(`FAIL ${name}\n${error.stack}`);}
  }
  console.log(`Dashboard routing UI: ${cases.length - failed} passed, ${failed} failed (offline fixtures).`);
  if (failed) process.exitCode = 1;
}
if (require.main === module) run().catch(error => {console.error(error); process.exitCode = 1;});
module.exports = {harness, diagnostic, account, run};
