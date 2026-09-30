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
  console.log('Dashboard UI: authenticated cleanup, honest errors, safe release identity passed.');
})().catch(error => {console.error(error); process.exitCode = 1;});
