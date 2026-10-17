// Session revocation and stale-response isolation. Author: Zeno Ren.
// All network responses are local synthetic promises; no browser/provider is used.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '../dashboard.html'), 'utf8');
const start = html.indexOf('// ---------- API keys & provider allow-lists');
const end = html.indexOf('// ---------- Refresh ----------', start);
assert.ok(start >= 0 && end > start, 'the panel script must remain inspectable');

function fixture() {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      id, innerHTML: '', textContent: '', value: '', hidden: false, open: false,
      options: [], dataset: {}, parentElement: {hidden: false},
      firstElementChild: {hidden: false},
      classList: {add() {}, remove() {}, toggle() {}},
      setAttribute(key, value) {this[key] = value;},
      removeAttribute(key) {delete this[key];},
      addEventListener() {}, focus() {},
    });
    return elements.get(id);
  }
  const context = {
    document: {getElementById: element, querySelectorAll: () => []},
    console, URL, AbortController, setTimeout, clearTimeout,
    confirm: () => true,
    escapeHtml: value => String(value).replace(/[&<>"']/g, c =>
      ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c])),
    fetchJSON: async () => {throw new Error('unexpected synthetic request');},
  };
  vm.createContext(context);
  vm.runInContext(html.slice(start, end), context);
  const run = code => vm.runInContext(code, context);
  const renders = [];
  for (const [fn, id] of [['renderKeys', 'keysContent'], ['renderRouting', 'routingContent'],
                         ['renderAccounts', 'accountsContent'], ['renderCredentials', 'credentialsContent']]) {
    context[fn] = payload => {renders.push({fn, payload}); element(id).innerHTML = 'PRIVATE IDENTITY';};
  }
  function authenticate() {
    run("panelSession = {authenticated: true, expiresAt: Date.now() + 1800000, passwordConfigured: true, passwordSource: 'env'}");
    context.renderSessionBar();
  }
  function revoke() {
    context.clearPanelContents();
    run('panelSession = {...panelSession, authenticated: false, expiresAt: 0}');
    context.renderSessionBar();
  }
  authenticate();
  return {context, element, run, renders, authenticate, revoke};
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => {resolve = yes; reject = no;});
  return {promise, resolve, reject};
}

test('clearing a session removes identity caches and unsent secret/editor drafts', () => {
  const f = fixture();
  f.run("routingLast = {preferences: [{model: 'PRIVATE_MODEL'}]}; routingDiagnostic = {model: 'PRIVATE_MODEL'}; routingAccounts = [{account: 'PRIVATE_ACCOUNT'}]; keyProviders = ['PRIVATE_PROVIDER']; keyProviderLabels = {PRIVATE_PROVIDER: 'PRIVATE_LABEL'}; deviceHandle = 'PRIVATE_DEVICE_HANDLE'");
  const fields = ['panelPassword', 'keysAdminKey', 'pwCurrent', 'pwNew', 'pwRepeat',
                  'pwClearCurrent', 'credSecret', 'credAccount', 'credApiBase',
                  'credDeployments', 'deviceAccount', 'ruleModelCustom', 'ruleProvider'];
  for (const id of fields) f.element(id).value = 'UNSENT_PRIVATE_DRAFT';
  for (const id of ['keysContent', 'routingContent', 'accountsContent', 'credentialsContent',
                    'ruleAccounts', 'keysPolicyState', 'panelPasswordSource', 'deviceStatus']) {
    f.element(id).innerHTML = 'PRIVATE IDENTITY';
    f.element(id).textContent = 'PRIVATE IDENTITY';
  }
  f.context.clearPanelContents();
  assert.equal(f.run('routingLast'), null, 'saved routing identities must not survive revocation');
  assert.equal(f.run('routingDiagnostic'), null);
  assert.equal(f.run('routingAccounts.length'), 0);
  assert.equal(f.run('keyProviders.length'), 0);
  assert.equal(f.run('Object.keys(keyProviderLabels).length'), 0);
  assert.equal(f.run('deviceHandle'), null, 'a device-flow handle is session-bound');
  for (const id of fields) assert.equal(f.element(id).value, '', id + ' must be cleared');
  for (const id of ['keysContent', 'routingContent', 'accountsContent', 'credentialsContent',
                    'ruleAccounts', 'keysPolicyState']) {
    assert.equal(f.element(id).innerHTML, '', id + ' must discard identity markup');
  }
  assert.equal(f.element('panelPasswordSource').textContent, '');
  assert.equal(f.element('deviceStatus').textContent, '');
});

test('lock hides the panel synchronously while server invalidation is still pending', async () => {
  const f = fixture();
  const pending = deferred();
  f.context.fetchJSON = async (_, options = {}) => options.method === 'DELETE'
    ? pending.promise : {authenticated: false, password_configured: true};
  const locking = f.context.lockPanels();
  assert.equal(f.element('panelBody').hidden, true, 'identity must leave the accessibility tree before I/O');
  assert.equal(f.element('panelLock').hidden, false);
  pending.resolve({authenticated: false});
  await locking;
});

test('lost logout ACK stays locally locked and reports unconfirmed server revocation', async () => {
  const f = fixture();
  const calls = [];
  f.context.fetchJSON = async (url, options = {}) => {
    calls.push({url, method: options.method || 'GET'});
    if (options.method === 'DELETE') throw new TypeError('synthetic lost ACK');
    return {authenticated: true, expires_in: 1800, password_configured: true};
  };
  await f.context.lockPanels();
  assert.equal(f.run('panelSession.authenticated'), false);
  assert.equal(f.element('panelBody').hidden, true);
  assert.equal(calls.filter(call => call.method === 'GET').length, 0,
    'an unconfirmed logout must not immediately restore its possibly live cookie');
  assert.match(f.element('panelLockFeedback').textContent, /未.*确认|无法确认|待确认/);
  assert.doesNotMatch(f.element('panelLockFeedback').textContent, /已在服务端失效|会话已失效/);
});

test('initial parallel panel reads cannot restore identities after lock', async () => {
  const f = fixture();
  const pending = deferred();
  f.context.fetchJSON = () => pending.promise;
  const loading = f.context.loadAllPanels();
  f.revoke();
  pending.resolve({synthetic: true});
  await loading;
  assert.equal(f.renders.length, 0, 'old authorized reads must be ignored once their epoch ends');
  assert.equal(f.element('keysContent').innerHTML, '');
});

test('one parallel 401 revokes the panel and suppresses successful sibling reads', async () => {
  const f = fixture();
  f.context.fetchJSON = async url => {
    if (url === '/admin/credentials') throw Object.assign(new Error('expired session'), {status: 401});
    return {synthetic: true};
  };
  await f.context.loadAllPanels();
  assert.equal(f.run('panelSession.authenticated'), false);
  assert.equal(f.element('panelBody').hidden, true);
  assert.equal(f.renders.length, 0, 'siblings may not retain identity after authorization failed');
});

test('a late section reload is ignored even after a different session is unlocked', async () => {
  const f = fixture();
  const pending = deferred();
  f.context.fetchJSON = () => pending.promise;
  const loading = f.context.reloadPanel('/admin/api-keys');
  f.revoke();
  f.authenticate();
  pending.resolve({synthetic: 'old session identity'});
  await loading;
  assert.equal(f.renders.length, 0, 'authenticated=true alone cannot establish read ownership');
  assert.equal(f.element('keysContent').innerHTML, '');
});

test('a late failed reload cannot refill a locked panel with stale feedback', async () => {
  const f = fixture();
  const pending = deferred();
  f.context.fetchJSON = () => pending.promise;
  const loading = f.context.reloadPanel('/admin/api-keys');
  f.revoke();
  pending.reject(new Error('PRIVATE_ACCOUNT_READ_FAILURE'));
  await loading;
  assert.equal(f.element('keysContent').innerHTML, '');
});

test('a section reload 401 clears every identity and closes the gate', async () => {
  const f = fixture();
  f.element('accountsContent').innerHTML = 'PRIVATE_ACCOUNT';
  f.context.fetchJSON = async () => {throw Object.assign(new Error('expired'), {status: 401});};
  await f.context.reloadPanel('/admin/api-keys');
  assert.equal(f.run('panelSession.authenticated'), false);
  assert.equal(f.element('accountsContent').innerHTML, '');
  assert.equal(f.element('panelBody').hidden, true);
});

test('a session status response started before lock cannot reopen the gate', async () => {
  const f = fixture();
  const pending = deferred();
  f.context.fetchJSON = () => pending.promise;
  const reading = f.context.refreshPanelSession();
  f.revoke();
  pending.resolve({authenticated: true, expires_in: 1800, password_configured: true});
  await reading;
  assert.equal(f.run('panelSession.authenticated'), false);
  assert.equal(f.element('panelBody').hidden, true);
});

test('a stale session-status failure cannot revoke a newer unlocked epoch', async () => {
  const f = fixture();
  const pending = deferred();
  f.context.fetchJSON = () => pending.promise;
  const reading = f.context.refreshPanelSession();
  f.revoke();
  f.authenticate();
  pending.reject(new Error('synthetic old connection failure'));
  await reading;
  assert.equal(f.run('panelSession.authenticated'), true);
  assert.equal(f.element('panelBody').hidden, false);
});

test('a device-flow start response arriving after lock cannot retain a handle or code', async () => {
  const f = fixture();
  const pending = deferred();
  f.context.fetchJSON = () => pending.promise;
  const starting = f.context.startDeviceFlow();
  f.revoke();
  pending.resolve({handle: 'PRIVATE_DEVICE_HANDLE', user_code: 'PRIVATE_USER_CODE',
                   verification_uri: 'https://github.com/login/device', expires_in: 900});
  await starting;
  assert.equal(f.run('deviceHandle'), null);
  assert.equal(f.element('deviceStatus').textContent, '');
});

test('a password-change ACK arriving after lock cannot refresh and reopen the session', async () => {
  const f = fixture();
  const pending = deferred();
  f.element('pwCurrent').value = 'synthetic-current-password';
  f.element('pwNew').value = 'synthetic-next-password';
  f.element('pwRepeat').value = 'synthetic-next-password';
  f.context.fetchJSON = (url) => url === '/admin/panel-password' ? pending.promise
    : Promise.resolve({authenticated: true, expires_in: 1800, password_configured: true});
  const changing = f.context.changePanelPassword();
  f.revoke();
  pending.resolve({configured: true});
  await changing;
  assert.equal(f.run('panelSession.authenticated'), false);
  assert.equal(f.element('panelBody').hidden, true);
  assert.equal(f.element('panelFeedback').textContent, '');
});

for (const operation of ['changePanelPassword', 'clearPanelPassword']) {
  test(operation + ' revokes identity for an expired-session 401', async () => {
    const f = fixture();
    f.element('accountsContent').innerHTML = 'PRIVATE_ACCOUNT';
    f.element('pwCurrent').value = 'synthetic-current-password';
    f.element('pwClearCurrent').value = 'synthetic-current-password';
    f.element('pwNew').value = 'synthetic-next-password';
    f.element('pwRepeat').value = 'synthetic-next-password';
    // _require_admin_access runs before password proof and returns a 401 with
    // no invalid_panel_password code when the HttpOnly session has expired.
    f.context.fetchJSON = async () => {throw Object.assign(new Error('Invalid API key'), {status: 401});};
    await f.context[operation]();
    assert.equal(f.run('panelSession.authenticated'), false);
    assert.equal(f.element('accountsContent').innerHTML, '');
    assert.equal(f.element('panelBody').hidden, true);
    assert.doesNotMatch(f.element('panelFeedback').textContent, /当前密码不正确/);
  });

  test(operation + ' preserves the session for the explicit current-password-proof error', async () => {
    const f = fixture();
    f.element('pwCurrent').value = 'synthetic-wrong-password';
    f.element('pwClearCurrent').value = 'synthetic-wrong-password';
    f.element('pwNew').value = 'synthetic-next-password';
    f.element('pwRepeat').value = 'synthetic-next-password';
    f.context.fetchJSON = async () => {
      throw Object.assign(new Error('Incorrect password'), {status: 401, code: 'invalid_panel_password'});
    };
    await f.context[operation]();
    assert.equal(f.run('panelSession.authenticated'), true);
    assert.equal(f.element('panelBody').hidden, false);
    assert.match(f.element('panelFeedback').textContent, /当前密码不正确/);
    for (const id of operation === 'changePanelPassword' ? ['pwCurrent', 'pwNew', 'pwRepeat'] : ['pwClearCurrent']) {
      assert.equal(f.element(id).value, '', id + ' must clear after rejected proof');
    }
  });
}

test('an unresolved logout blocks login POSTs so its late cookie deletion cannot race a new session', async () => {
  const f = fixture();
  const pending = deferred();
  const calls = [];
  f.context.fetchJSON = async (url, options = {}) => {
    const method = options.method || 'GET';
    calls.push({url, method});
    if (method === 'DELETE') return pending.promise;
    if (url === '/admin/session') return {authenticated: true, expires_in: 1800, password_configured: true};
    return {};
  };
  const locking = f.context.lockPanels();
  try {
    f.element('panelPassword').value = 'synthetic-relogin-password';
    await f.context.unlockPanels(false);
    assert.equal(calls.filter(call => call.method === 'POST').length, 0,
      'a new cookie must not be minted while the previous DELETE can still delete it');
    assert.equal(f.element('panelBody').hidden, true);
  } finally {
    pending.resolve({authenticated: false});
    await locking;
  }
});

test('an unconfirmed logout has a visible retry control within the locked gate', async () => {
  const lockStart = html.indexOf('id="panelLock"');
  const bodyStart = html.indexOf('id="panelBody"');
  assert.ok(lockStart >= 0 && bodyStart > lockStart);
  assert.match(html.slice(lockStart, bodyStart), /id="retryLockPanels"/,
    'retry must be reachable in the locked gate, outside the hidden identity area');
  const f = fixture();
  let fail = true;
  const calls = [];
  f.context.fetchJSON = async (url, options = {}) => {
    calls.push({url, method: options.method || 'GET'});
    if (fail) throw new TypeError('synthetic logout ACK loss');
    return {authenticated: false};
  };
  await f.context.lockPanels();
  assert.equal(f.element('retryLockPanels').hidden, false, 'failed logout offers a retry');
  assert.equal(f.element('panelLock').hidden, false);
  assert.equal(f.element('panelBody').hidden, true);
  fail = false;
  await f.context.lockPanels();
  assert.equal(f.element('retryLockPanels').hidden, true, 'confirmed revocation removes the retry');
  assert.equal(calls.filter(call => call.method === 'DELETE').length, 2);
  assert.equal(calls.filter(call => call.method === 'GET' || call.method === 'POST').length, 0,
    'retrying revocation must not read or mint a session');
});
