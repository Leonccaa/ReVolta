// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the ReVolta project
// CPU browser verification with a synthetic ASGI exporter, using Node's native CDP client.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {mkdir, mkdtemp, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {parseArgs} from 'node:util';

const {values} = parseArgs({options: {
  chrome: {type: 'string'}, python: {type: 'string', default: '.venv/bin/python'},
  output: {type: 'string', default: 'dashboard-browser-results'},
}});
if (!values.chrome) throw Error('Pass --chrome with the Chromium or Chrome executable path.');
const output = resolve(values.output);
await mkdir(output, {recursive: true});
const profile = await mkdtemp(resolve(tmpdir(), 'revolta-dashboard-'));
const fixture = spawn(values.python, [resolve(dirname(fileURLToPath(import.meta.url)), 'fixture_server.py')], {stdio: ['ignore', 'pipe', 'pipe']});
const chrome = spawn(values.chrome, [
  '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--remote-debugging-port=0',
  '--user-data-dir=' + profile, 'about:blank',
], {stdio: ['ignore', 'ignore', 'pipe']});
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const report = {viewports: [], checks: [], exceptions: []};
let ws, fixtureLog = '';
fixture.stderr.on('data', chunk => { fixtureLog += chunk; });

function endpoint(stream, pattern) {
  return new Promise((resolve, reject) => {
    let text = '';
    const timer = setTimeout(() => reject(Error('Startup timed out: ' + text + fixtureLog)), 10000);
    stream.on('data', chunk => {
      text += chunk;
      const match = text.match(pattern);
      if (match) { clearTimeout(timer); resolve(match[1]); }
    });
  });
}

try {
  const [fixturePort, chromePort] = await Promise.all([
    endpoint(fixture.stdout, /^(\d+)\s/m),
    endpoint(chrome.stderr, /DevTools listening on ws:\/\/127\.0\.0\.1:(\d+)/),
  ]);
  const base = `http://127.0.0.1:${fixturePort}`;
  const targets = await (await fetch(`http://127.0.0.1:${chromePort}/json/list`)).json();
  ws = new WebSocket(targets[0].webSocketDebuggerUrl);
  await once(ws, 'open');
  let seq = 0;
  const pending = new Map(), urls = [];
  ws.addEventListener('message', event => {
    const message = JSON.parse(event.data), promise = pending.get(message.id);
    if (promise) {
      pending.delete(message.id);
      message.error ? promise.reject(Error(message.error.message)) : promise.resolve(message.result);
    }
    if (message.method === 'Runtime.exceptionThrown') report.exceptions.push(message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text);
    if (message.method === 'Network.requestWillBeSent') urls.push(message.params.request.url);
  });
  const call = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++seq;
    const timer = setTimeout(() => { pending.delete(id); reject(Error('CDP timeout: ' + method)); }, 10000);
    pending.set(id, {resolve: result => { clearTimeout(timer); resolve(result); }, reject: error => { clearTimeout(timer); reject(error); }});
    ws.send(JSON.stringify({id, method, params}));
  });
  const evaluate = async expression => {
    const result = await call('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
    if (result.exceptionDetails) throw Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  };
  const until = async (fn, label, timeout = 10000) => {
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      if (await fn()) return;
      await wait(100);
    }
    throw Error(label + ' · ' + JSON.stringify(await probe()));
  };
  const probe = () => evaluate(`(${(() => ({
    health: document.querySelector('#health')?.textContent,
    message: document.querySelector('#connection')?.textContent,
    coverage: document.querySelector('#coverage')?.textContent,
    samples: document.querySelector('#observation')?.textContent,
    order: [...document.querySelectorAll('[data-card]')].map(e => e.dataset.card),
    heights: [...document.querySelectorAll('[data-card]')].map(e => Math.round(e.getBoundingClientRect().height)),
    scopes: [...document.querySelectorAll('.card-heading span')].map(e => e.textContent),
    prefill: document.querySelector('[data-card=prefill] .kpi strong')?.textContent,
    prefillCaption: document.querySelector('[data-card=prefill] .kpi p')?.textContent,
    decode: document.querySelector('[data-card=decode] .kpi strong')?.textContent,
    decodeAverage: document.querySelector('[data-card=decode] .kpi.window strong')?.textContent,
    hit: document.querySelector('[data-card=cache] .kpi strong')?.textContent,
    lifetime: document.querySelector('#lifetime')?.textContent,
    cache: document.querySelector('[data-card=cache]')?.textContent,
    cacheState: document.querySelector('#cache-state')?.textContent,
    open: document.querySelector('#details')?.open,
    dialog: document.querySelector('#detail-body')?.textContent,
    overflow: document.documentElement ? document.documentElement.scrollWidth > window.innerWidth : false,
    badText: /NaN|undefined|Infinity/.test(document.body?.textContent || ''),
    images: document.querySelectorAll('#model img').length,
    focused: document.activeElement?.dataset.details,
  })).toString()})()`);
  const control = async command => (await fetch(base + '/__fixture', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(command)})).json();
  const viewport = width => call('Emulation.setDeviceMetricsOverride', {width, height: 1040, deviceScaleFactor: 1, mobile: width <= 390});
  const click = selector => evaluate(`(() => {const element=document.querySelector(${JSON.stringify(selector)});element.focus();element.click();})()`);
  const screenshot = async name => {
    const result = await call('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
    await writeFile(resolve(output, name), Buffer.from(result.data, 'base64'));
  };
  await call('Page.enable'); await call('Runtime.enable'); await call('Network.enable');
  await viewport(1440);
  await call('Page.navigate', {url: base + '/dashboard/'});
  await until(async () => (await probe()).health === 'Online', 'Normal entrypoint');
  await until(async () => (await probe()).prefill !== '—', 'Completed prefill observed');
  let p = await probe();
  assert.deepEqual(p.order, ['requests', 'decode', 'prefill', 'cache', 'speculative', 'latency']);
  assert.equal(new Set(p.heights).size, 1);
  assert.equal(p.hit, '80%');
  assert.ok(Number(p.prefill.replaceAll(',', '')) > 1000 && !p.prefill.includes('K'));
  assert.ok(p.prefillCaption.includes('completed request'));
  assert.ok(Number(p.decode) > 0 && Number(p.decodeAverage) > 0);
  report.checks.push('Native metric parsing, six equal cards, Now/window values, cache formula and full numeric prefill');
  for (const seconds of [60, 300, 900]) {
    await click(`[data-window="${seconds}"]`);
    p = await probe();
    assert.ok(p.scopes.every(label => label === `Last ${seconds / 60} min`));
    assert.ok(p.coverage.includes('Partial window'));
    assert.ok(Number(p.decodeAverage) > 0);
  }
  await screenshot('desktop.png');
  const originalSampleCount = Number((await probe()).samples.match(/(\d+) retained/)[1]);
  await call('Page.reload');
  await until(async () => (await probe()).health === 'Online', 'Reload from browser cache');
  p = await probe();
  assert.ok(Number(p.samples.match(/(\d+) retained/)[1]) >= originalSampleCount);
  assert.ok(p.scopes.every(label => label === 'Last 15 min'));
  report.checks.push('Partial 1/5/15-minute windows and same-engine reload persistence');

  for (const width of [1180, 820, 736, 390, 320]) {
    await viewport(width);
    p = await probe();
    assert.equal(p.overflow, false, `Horizontal overflow at ${width}`);
    assert.equal(p.badText, false);
    assert.equal(new Set(p.heights).size, 1, `Unequal cards at ${width}`);
    await click('[data-details=cache]');
    await until(async () => (await probe()).open, 'Cache dialog');
    for (const tab of ['summary', 'gpu', 'ram', 'ssd']) {
      await click(`[data-tab=${tab}]`);
      p = await probe();
      assert.ok(p.dialog.includes('Since engine start'));
      assert.ok(p.dialog.includes(({summary: 'Total input cache hit rate', gpu: 'Groups requested', ram: 'Reused in RAM', ssd: 'SSD write bytes (committed)'})[tab]), `Cache tab ${tab} did not show its data: ${p.dialog}`);
      assert.equal(p.badText, false);
    }
    const dialogLayout = await evaluate(`(() => {const d=document.querySelector('#details'), b=document.querySelector('#detail-body'), r=d.getBoundingClientRect();return {width:r.width,left:r.left,right:r.right,top:r.top,bottom:r.bottom,scrollable:b.scrollHeight>b.clientHeight,bodyOverflow:b.scrollWidth>b.clientWidth};})()`);
    assert.ok(dialogLayout.left >= 0 && dialogLayout.right <= width);
    assert.ok(dialogLayout.top >= 0 && dialogLayout.bottom <= 1040);
    assert.equal(dialogLayout.bodyOverflow, false);
    if (width === 390) { await click('[data-tab=ram]'); await screenshot('mobile-cache-details.png'); }
    await call('Input.dispatchKeyEvent', {type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27});
    await call('Input.dispatchKeyEvent', {type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27});
    await until(async () => !(await probe()).open, 'Escape closes dialog');
    assert.equal((await probe()).focused, 'cache');
    report.viewports.push({width, heights: p.heights, dialogLayout});
  }
  for (const card of ['requests', 'decode', 'prefill', 'speculative', 'latency']) {
    await click(`[data-details=${card}]`);
    assert.ok((await probe()).dialog.includes('Since engine start'));
    await click('#close');
  }
  await evaluate("document.querySelector('[data-details=cache]').click();document.querySelector('#details').close();document.querySelector('[data-details=cache]').click()");
  await wait(80);
  await click('[data-tab=ram]');
  assert.ok((await probe()).dialog.includes('Reused in RAM'));
  await click('#close');
  await viewport(390); await screenshot('mobile.png');
  await call('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: 'dark'}]});
  await screenshot('mobile-dark.png');
  await call('Emulation.setEmulatedMedia', {features: []});
  report.checks.push('Desktop/mobile widths, all lifetime details, Cache tabs, dialog bounds, Escape focus and dark mode');

  const beforePause = (await control({})).scrapes;
  await evaluate("Object.defineProperty(document,'hidden',{configurable:true,value:true});document.dispatchEvent(new Event('visibilitychange'))");
  await wait(2300);
  assert.equal((await control({})).scrapes, beforePause);
  assert.equal((await probe()).health, 'Paused');
  await evaluate("Object.defineProperty(document,'hidden',{configurable:true,value:false});document.dispatchEvent(new Event('visibilitychange'))");
  await until(async () => (await probe()).health === 'Online', 'Resume visible polling');
  report.checks.push('Hidden-tab polling pause and resume');

  await control({mode: 'idle'});
  await wait(2300);
  const retainedPrefill = (await probe()).prefill;
  await wait(2300);
  assert.equal((await probe()).prefill, retainedPrefill);
  assert.notEqual(retainedPrefill, '—');
  await control({mode: 'normal'});
  report.checks.push('Latest prefill survives decode/idle');

  await control({mode: 'stale-store'});
  await until(async () => (await probe()).cache.includes('unavailable or stale'), 'Stale store occupancy');
  await control({mode: 'missing-optional'});
  await until(async () => (await probe()).hit === '—', 'Missing optional metric compatibility');
  assert.equal((await probe()).badText, false);
  await control({mode: 'offline'});
  await until(async () => (await probe()).health === 'Offline', 'Offline mode');
  assert.equal((await probe()).decode, '—');
  await screenshot('offline.png');
  await control({mode: 'missing-all'});
  await until(async () => (await probe()).message.includes('Inference token metrics'), 'Disabled/missing statistics');
  await control({mode: 'escaped-label'});
  await until(async () => (await probe()).health === 'Online', 'Escaped model label');
  assert.equal((await probe()).images, 0);
  report.checks.push('Stale store, absent metrics, offline and escaped labels');

  await control({mode: 'normal', auth: true});
  await until(async () => (await probe()).health === 'API key required', 'Authentication boundary');
  assert.ok(!(await probe()).lifetime.includes('1M'));
  await evaluate("document.querySelector('#key').value='wrong';document.querySelector('#auth').requestSubmit()");
  await until(async () => (await probe()).message.includes('rejected'), 'Wrong API key');
  await evaluate("document.querySelector('#key').value='cpu-test-key';document.querySelector('#auth').requestSubmit()");
  await until(async () => (await probe()).health === 'Online', 'Valid API key');
  assert.equal(await evaluate("Object.values(localStorage).some(value=>value.includes('cpu-test-key'))"), false);
  assert.equal(await evaluate("document.querySelector('#key').value"), '');
  assert.ok(urls.every(url => !url.includes('cpu-test-key')));
  await control({mode: 'slow'});
  const beforeSlow = (await control({})).scrapes;
  await until(async () => (await control({})).scrapes > beforeSlow, 'Slow scrape in flight');
  await click('#disconnect');
  await wait(600);
  assert.equal((await probe()).health, 'API key required');
  assert.ok((await probe()).samples.includes('No browser observations'));
  await control({mode: 'normal'});
  await evaluate("document.querySelector('#key').value='cpu-test-key';document.querySelector('#auth').requestSubmit()");
  await until(async () => (await probe()).health === 'Online', 'Reconnect after in-flight disconnect');
  await click('#remember');
  assert.equal(await evaluate("Object.keys(localStorage).some(key=>key.startsWith('revolta-dashboard:1:'))"), false);
  await call('Page.reload');
  await until(async () => (await probe()).health === 'API key required', 'Reload requires key again');
  assert.equal(await evaluate("document.querySelector('#remember').checked"), false);
  await evaluate("document.querySelector('#key').value='cpu-test-key';document.querySelector('#auth').requestSubmit()");
  await until(async () => (await probe()).health === 'Online', 'Reconnect without saved history');
  report.checks.push('Bearer auth, rejection, in-flight disconnect, no key in URL/storage, opt-out persistence and no unauthenticated cache display');

  await control({mode: 'restart', auth: false});
  await until(async () => evaluate(`document.querySelector('#health').textContent==='Online' && document.querySelector('#observation').textContent.includes('1 retained')`), 'Engine restart starts a new observation window');
  await click('#clear');
  assert.ok((await probe()).samples.includes('No browser observations'));
  report.checks.push('Engine restart invalidates old observations and clear-history control');
  assert.deepEqual(report.exceptions, []);
  await writeFile(resolve(output, 'acceptance.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({checks: report.checks, viewports: report.viewports.map(v => v.width), exceptions: report.exceptions}));
} catch (error) {
  report.error = error.stack;
  report.fixtureLog = fixtureLog;
  await writeFile(resolve(output, 'acceptance-failed.json'), JSON.stringify(report, null, 2));
  throw error;
} finally {
  ws?.close();
  const stop = async child => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    const stopped = once(child, 'exit');
    child.kill('SIGTERM');
    await stopped;
  };
  await Promise.all([stop(chrome), stop(fixture)]);
  await rm(profile, {recursive: true, force: true});
}
