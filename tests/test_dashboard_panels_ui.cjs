// The four operations panels preserve scope and submit only relevant fields. Author: Zeno Ren.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../dashboard.html'), 'utf8');
const between = (from, to) => html.slice(html.indexOf(from), html.indexOf(to));
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {value: '', innerHTML: '', textContent: '', hidden: false,
    disabled: false, required: false, dataset: {}, addEventListener() {}, focus() {},
    setAttribute(name, value) { this[name] = value; }, removeAttribute(name) { delete this[name]; }});
  return elements.get(id);
}
let nodes = [];
const context = {document: {getElementById: element, querySelectorAll: selector =>
  nodes.filter(node => node.selector === selector).map(node => Object.assign({addEventListener() {}}, node))},
  URL, AbortController, setTimeout, clearTimeout, console, confirm: () => true,
  panelEpoch:0, panelRequestCurrent:() => true, expirePanelSession() {}};
vm.createContext(context);
vm.runInContext(between('// ---------- Formatters', '// ---------- Number Ticker') +
  between('// ---------- Release identity', '// ---------- Refresh'), context);

(async () => {
  // Provider switching must update accessibility/validation and must not send
  // hidden fields left over from another provider to an upstream validator.
  assert.equal(typeof context.syncCredentialFields, 'function', 'provider-specific fields need a controller');
  element('credProvider').value = 'copilot';
  context.syncCredentialFields();
  assert.equal(element('credApiBaseField').hidden, true);
  assert.equal(element('credDeploymentsField').hidden, true);
  assert.equal(element('credApiBase').required, false);
  assert.equal(element('credentialDeviceFlow').hidden, false);
  element('credProvider').value = 'azure_openai';
  context.syncCredentialFields();
  assert.equal(element('credApiBaseField').hidden, false);
  assert.equal(element('credDeploymentsField').hidden, false);
  assert.equal(element('credApiBase').required, true);
  assert.equal(element('credentialDeviceFlow').hidden, true);
  element('credProvider').value = 'databricks';
  context.syncCredentialFields();
  assert.equal(element('credDeploymentsField').hidden, true);

  vm.runInContext('panelSession = {authenticated:true,expiresAt:Date.now()+1800000};', context);
  context.reloadPanel = async () => {};
  const writes = [];
  context.fetchJSON = async (url, options) => { writes.push({url, options}); return {}; };
  element('credProvider').value = 'copilot';
  element('credAccount').value = 'synthetic-gh';
  element('credSecret').value = 'synthetic-token';
  element('credApiBase').value = 'https://stale.invalid';
  element('credDeployments').value = 'stale-deployment';
  await context.submitCredential();
  assert.deepEqual(JSON.parse(writes.at(-1).options.body), {secret:'synthetic-token',meta:{}},
    'GHCP must not receive stale Foundry metadata');
  assert.equal(element('credSecret').value, '', 'submitted secrets are cleared');
  element('credProvider').value = 'databricks';
  element('credSecret').value = 'synthetic-token';
  await context.submitCredential();
  assert.deepEqual(JSON.parse(writes.at(-1).options.body).meta,
    {api_base:'https://stale.invalid'}, 'ADB must not receive stale Foundry deployments');

  // Candidate coverage is configuration evidence, independent of process
  // request counts. An operator-disabled account differs from never-enabled.
  context.renderAccounts({policy:{state:'loaded'}, catalogue_models:3, accounts:[
    {account:'gh-idle',enabled:true,capability:['gpt-a'],serves_models:0,receiving_traffic:false,requests:0},
    {account:'gh-new',enabled:false,enabled_in_config:false,enablement_source:'config',capability:[],requests:0},
    {account:'gh-stopped',enabled:false,enabled_in_config:true,enablement_source:'operator',capability:['gpt-a'],requests:2},
  ]});
  const accounts = element('accountsContent').innerHTML;
  assert.match(accounts, /配置候选覆盖/);
  assert.match(accounts, /未启用/);
  assert.match(accounts, /已停用/);
  assert.doesNotMatch(accounts, /实际接流量|接流量<|已启用但闲置/);
  assert.match(accounts, /data-edit-account="gh-idle"/);
  assert.doesNotMatch(accounts, /data-save-account=/, 'editing controls stay out of the overview');
  context.openAccountDetail('gh-new');
  assert.match(element('accountDetail').innerHTML, /data-needs-capability="1"/);
  assert.match(element('accountDetail').innerHTML, /影响所有模型/);

  // Account allowed_models has a different empty-set contract from routing:
  // null removes a limit, while an explicit mode with no models is refused.
  context.renderAccounts({policy:{state:'loaded'},accounts:[
    {account:'gh-explicit',enabled:true,capability:['gpt-a'],allowed_models:[],serves_models:0},
  ]});
  context.openAccountDetail('gh-explicit');
  assert.match(element('accountDetail').innerHTML, /data-account-mode="selected" checked/);
  assert.doesNotMatch(element('accountDetail').innerHTML, /data-model="gpt-a" checked/,
    'an explicit empty snapshot must not be broadened to follow all capabilities');
  nodes = [
    {selector:'#accountDetail [data-enable]',checked:true,dataset:{enable:'gh-explicit'}},
    {selector:'#accountDetail [data-account-mode]',checked:true,dataset:{accountMode:'selected'}},
    {selector:'#accountDetail [data-account]',checked:false,dataset:{account:'gh-explicit',model:'gpt-a'}},
  ];
  const beforeEmpty = writes.length;
  await context.saveAccount('gh-explicit');
  assert.equal(writes.length,beforeEmpty,'an empty explicit model selection must not write');
  assert.match(element('accountsFeedback').textContent,/至少要勾选一个模型/);
  nodes[1].dataset.accountMode = 'all';
  context.refreshRoutingDiagnostic = async () => {};
  await context.saveAccount('gh-explicit');
  assert.equal(JSON.parse(writes.at(-1).options.body).allowed_models,null,
    'only explicit follow-capability mode removes the account restriction');
  nodes[1].dataset.accountMode = 'selected';
  nodes[2].checked = true;
  await context.saveAccount('gh-explicit');
  assert.deepEqual(JSON.parse(writes.at(-1).options.body).allowed_models,['gpt-a'],
    'explicit selections remain explicit even when every current model is selected');

  context.renderKeys({providers:['copilot'],policy:{state:'loaded'},keys:[
    {tenant:'team-a',masked:'lb-a…1111',fingerprint:'hash-a',providers:['copilot'],restricted:true,requests:0},
  ]});
  assert.match(element('keysContent').innerHTML, /data-edit-key="team-a"/);
  assert.doesNotMatch(element('keysContent').innerHTML, /data-save=/, 'overview is read-only');
  context.openKeyDetail('team-a');
  assert.match(element('keyDetail').innerHTML, /data-reset="team-a"/);
  assert.match(element('keyDetail').innerHTML, /恢复配置渠道/);

  context.renderCredentials({store:{key:'present'},credentials:[
    {provider:'copilot',account:'gh-a',secret_fingerprint:'hash-a',key_state:'ok',validation:{ok:true},active:false},
  ]});
  assert.match(element('credentialsContent').innerHTML, /待重启生效/);
  assert.match(element('credentialsContent').innerHTML, /校验结果/);
  assert.match(element('credentialsContent').innerHTML, /加载状态/);
  assert.match(element('credentialsContent').innerHTML, /通过/);
  // A configured store's write failure is not the same state as a deployment
  // that never configured storage, and a generic 503 says neither outcome.
  for (const operation of ['changePanelPassword','clearPanelPassword']) {
    for (const code of ['policy_store_unavailable','policy_write_failed','synthetic_unavailable']) {
      element('pwCurrent').value = element('pwClearCurrent').value = 'synthetic-current-password';
      element('pwNew').value = element('pwRepeat').value = 'synthetic-next-password';
      context.fetchJSON = async () => { throw Object.assign(new Error('synthetic store failure'),{status:503,code}); };
      await context[operation]();
      const message = element('panelFeedback').textContent;
      if (code === 'policy_store_unavailable') assert.match(message,/未配置策略存储/);
      else {
        assert.match(message,/未能确认/);
        assert.doesNotMatch(message,/未配置策略存储|仍然有效|保持不变/);
        if (code === 'policy_write_failed') assert.match(message,/存储写入失败/);
      }
    }
  }
  // A completed GitHub request must not recreate a device handle or show a
  // verification code after the operator has locked the operations panel.
  let resolveDevice;
  context.panelRequestCurrent = epoch => epoch === vm.runInContext('panelEpoch',context)
    && vm.runInContext('panelSession.authenticated',context);
  context.fetchJSON = () => new Promise(resolve => { resolveDevice=resolve; });
  const startingDevice = context.startDeviceFlow();
  vm.runInContext('panelEpoch += 1; panelSession.authenticated = false;',context);
  context.clearPanelEditors();
  element('deviceStatus').textContent = '';
  resolveDevice({handle:'late-handle',verification_uri:'https://github.com/login/device',user_code:'LATE-CODE',expires_in:900});
  await startingDevice;
  assert.equal(vm.runInContext('deviceHandle',context),null);
  assert.equal(element('deviceStatus').textContent,'','a late authorization response cannot repopulate the locked page');
  context.openKeyDetail('team-a');
  context.openAccountDetail('gh-explicit');
  assert.equal(element('keyDetail').innerHTML,'','locking discards the cached key identity');
  assert.equal(element('accountDetail').innerHTML,'','locking discards cached account policies');
  assert.equal(element('credentialEntry').open,false);
  // Refusal must clear secrets before field validation can return early. The
  // panel may already be locked and its non-secret fields may have been reset.
  context.panelReady = () => false;
  element('credAccount').value = '';
  element('credSecret').value = 'synthetic-after-lock';
  await context.submitCredential();
  assert.equal(element('credSecret').value,'','a locked incomplete credential form still clears its secret');
  element('pwCurrent').value = 'synthetic-current';
  element('pwNew').value = 'new';
  element('pwRepeat').value = 'mismatch';
  await context.changePanelPassword();
  for (const id of ['pwCurrent','pwNew','pwRepeat']) assert.equal(element(id).value,'','locked password drafts are cleared');
  element('pwClearCurrent').value = 'synthetic-current';
  await context.clearPanelPassword();
  assert.equal(element('pwClearCurrent').value,'','refused password deletion clears current password');
  context.panelReady = () => true;
  context.confirm = () => false;
  element('pwCurrent').value = 'synthetic-current';
  element('pwNew').value = element('pwRepeat').value = 'synthetic-new-password';
  await context.changePanelPassword();
  for (const id of ['pwCurrent','pwNew','pwRepeat']) assert.equal(element(id).value,'','cancelled password changes clear secrets');
  element('pwClearCurrent').value = 'synthetic-current';
  await context.clearPanelPassword();
  assert.equal(element('pwClearCurrent').value,'','cancelled password deletion clears current password');
  vm.runInContext('keysInFlight=true;',context);
  element('credSecret').value = 'synthetic-busy';
  await context.submitCredential();
  assert.equal(element('credSecret').value,'','a duplicate submission cannot leave another secret in the form');
  vm.runInContext('keysInFlight=false;',context);
  console.log('Dashboard operation panels: provider fields, candidate scope and details passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
