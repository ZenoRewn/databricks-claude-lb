// Dashboard estimates preserve unknown coverage. Author: Zeno Ren.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../dashboard.html'), 'utf8');
for (const match of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) new vm.Script(match[1]);
const formatters = html.slice(html.indexOf('// ---------- Formatters'), html.indexOf('// ---------- Number Ticker'));
const tables = html.slice(html.indexOf('// ---------- Model tables'), html.indexOf('// ---------- History'));
const context = {};
vm.createContext(context);
vm.runInContext(formatters + tables, context);
const model = 'gpt-6-astra';
const summary = context.aggregateModels([
  {model_stats: {[model]: {requests: 1, input_tokens: 100, estimated_cost_usd: .5, priced_requests: 1, unpriced_requests: 0, known_cost_subtotal_usd: .5}}},
  {model_stats: {[model]: {requests: 1, input_tokens: 100, estimated_cost_usd: null, priced_requests: 0, unpriced_requests: 1, known_cost_subtotal_usd: 0}}}
])[model];
assert.equal(summary.estimated_cost_usd, null);
assert.equal(summary.known_cost_subtotal_usd, .5);
assert.equal(summary.pricing_status, 'partial');
assert.match(context.fmtModelCost(summary), /未计价/);
assert.match(context.fmtModelCost(summary, true), /50\.0000/);
const unknown = context.aggregateModels([{model_stats: {future: {requests: 1}}}]).future;
assert.equal(unknown.estimated_cost_usd, null);
assert.equal(context.fmtModelCost(unknown), '未知');
assert.equal(context.fmtModelCost({estimated_cost_usd: 0}), '$0');
for (const modelName of ['constructor', '__proto__', 'toString']) {
  const modelStats = JSON.parse(JSON.stringify({[modelName]: {requests: 1, input_tokens: 100, estimated_cost_usd: .5}}));
  const unusual = context.aggregateModels([{model_stats: modelStats}]);
  assert.deepEqual(Object.keys(unusual), [modelName]);
  assert.equal(unusual[modelName].requests, 1);
  assert.equal(unusual[modelName].estimated_cost_usd, .5);
  assert.equal(vm.runInContext('Object.prototype.input_tokens', context), undefined);
}
console.log('Copilot pricing UI: partial, unknown, credits and zero-cost invariants passed.');
