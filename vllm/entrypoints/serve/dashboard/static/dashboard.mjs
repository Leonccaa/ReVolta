// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the ReVolta project
import {
  add, chartPoints, delta, finite, fraction, GAP_SECONDS, makeSnapshot,
  ratio, statistics, windowStatistics,
} from './metrics.mjs';
import {ObservationHistory} from './history.mjs';

const $ = selector => document.querySelector(selector);
const escape = value => String(value).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const number = (value, compact = true) => finite(value) ? new Intl.NumberFormat('en-US', {
  maximumFractionDigits: value < 10 ? 2 : 1,
  notation: compact && Math.abs(value) >= 10000 ? 'compact' : 'standard',
}).format(value) : '—';
const percent = value => finite(value) ? `${number(value * 100, false)}%` : '—';
const seconds = value => finite(value) ? (value < 1 ? `${number(value * 1000, false)} ms` : `${number(value, false)} s`) : '—';
function bytes(value) {
  if (!finite(value)) return '—';
  const index = Math.min(4, Math.max(0, Math.floor(Math.log2(Math.abs(value) || 1) / 10)));
  return `${number(value / (1024 ** index), false)} ${['B', 'KiB', 'MiB', 'GiB', 'TiB'][index]}`;
}
const signedBytes = value => finite(value) ? `${value > 0 ? '+' : ''}${bytes(value)}` : '—';
const byteRate = value => finite(value) ? `${signedBytes(value)}/s` : '—';
function duration(value) {
  if (!finite(value) || value < 0) return '—';
  if (value < 60) return `${Math.floor(value)}s`;
  if (value < 3600) return `${Math.floor(value / 60)}m ${Math.floor(value % 60)}s`;
  if (value < 86400) return `${Math.floor(value / 3600)}h ${Math.floor(value % 3600 / 60)}m`;
  return `${Math.floor(value / 86400)}d ${Math.floor(value % 86400 / 3600)}h`;
}
const rows = entries => `<dl class="rows">${entries.map(([label, value]) => `<div><dt>${escape(label)}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl>`;
const kpi = (label, value, unit, caption, window = false) => `<div class="kpi${window ? ' window' : ''}"><small>${escape(label)}</small><strong>${escape(value)}</strong><span class="unit">${escape(unit)}</span><p>${escape(caption)}</p></div>`;
const titles = {requests: 'Requests', decode: 'Decode', prefill: 'Prefill', cache: 'Cache', speculative: 'Speculative decoding', latency: 'Latency'};
const scope = location.pathname;
const prefKey = `revolta-dashboard:preferences:${scope}`;
let storage = null, preferences = {};
try { storage = localStorage; preferences = JSON.parse(storage.getItem(prefKey) || '{}') || {}; } catch { /* Private mode can block storage. */ }
let windowSeconds = [60, 300, 900].includes(preferences.window) ? preferences.window : 60;
const history = new ObservationHistory({storage, scope, remember: preferences.remember !== false});
let info = null, token = '', state = 'connecting', message = 'Connecting to this engine…';
let timer, inFlight = false, detailCard = null, cacheTab = 'summary', connectionGeneration = 0;
const windowLabel = () => `Last ${windowSeconds / 60} min`;
function savePreferences() {
  try { storage?.setItem(prefKey, JSON.stringify({window: windowSeconds, remember: history.remember})); } catch { /* Optional persistence. */ }
}

$('#cards').innerHTML = Object.entries(titles).map(([key, title]) => `<article class="card" data-card="${key}"><div class="card-body"></div><button class="details" data-details="${key}" aria-label="View ${title} details">View details<span aria-hidden="true">↗</span></button></article>`).join('');
$('#remember').checked = history.remember;
$('#detail-tabs').innerHTML = [['summary', 'Summary'], ['gpu', 'GPU Hot'], ['ram', 'RAM'], ['ssd', 'SSD']].map(([key, label]) => `<button data-tab="${key}" aria-pressed="${key === cacheTab}">${label}</button>`).join('');

function chart(points, label) {
  const valid = points.filter(point => finite(point.value));
  const peak = Math.max(0, ...valid.map(point => point.value));
  const maximum = Math.max(1, peak);
  const now = history.samples.at(-1)?.t || Date.now() / 1000;
  let path = '', penDown = false;
  const dots = [];
  for (const point of points) {
    if (!finite(point.value)) { penDown = false; continue; }
    const x = 27 + (1 - (now - point.t) / windowSeconds) * 313;
    const y = 54 - point.value / maximum * 43;
    path += `${penDown ? 'L' : 'M'}${x.toFixed(2)} ${y.toFixed(2)} `;
    dots.push(`<circle class="dot" cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="1.5"/>`);
    penDown = true;
  }
  return `<figure class="chart"><figcaption><span>${escape(label)}</span><span>${valid.length ? `Peak ${escape(number(peak))}` : 'No observed intervals'}</span></figcaption><svg viewBox="0 0 346 78" role="img" aria-label="${escape(label)} over ${escape(windowLabel())}"><path class="axis" d="M27 8H340M27 55H340"/><text x="0" y="12">${escape(number(maximum))}</text><text x="13" y="57">0</text><path class="line" d="${path}"/>${dots.join('')}<text x="27" y="73">−${windowSeconds / 60} min</text><text x="331" y="73">0</text></svg></figure>`;
}

function bar(label, used, capacity, note, kind = '') {
  const utilization = fraction(used, capacity);
  const value = capacity === null ? bytes(used) : percent(utilization);
  return `<div class="${kind}"><div class="bar-label"><b>${escape(label)}</b><span>${escape(value)}</span></div><div class="track${finite(utilization) ? '' : ' unknown'}" role="meter" aria-label="${escape(label)} occupancy" ${finite(utilization) ? `aria-valuenow="${utilization * 100}" aria-valuemin="0" aria-valuemax="100"` : 'aria-valuetext="Capacity unavailable"'}><div class="fill" style="width:${finite(utilization) ? utilization * 100 : 0}%"></div></div><p class="bar-note">${escape(note)}</p></div>`;
}

function cacheBars(end, stats, fresh, tab = 'summary') {
  const v = end.v;
  const storeFresh = fresh && finite(v.storeTimestamp) && end.t - v.storeTimestamp <= 180 && end.t >= v.storeTimestamp - 10;
  const value = key => storeFresh ? v[key] : null;
  const bars = [
    ['gpu', bar('GPU Hot', value('hotUsed'), value('hotCapacity'), `Group hit rate ${percent(stats.hotHit)}`)],
    ['ram', bar('RAM', value('ramUsed'), value('ramCapacity'), `${bytes(value('ramUsed'))} · Δ ${signedBytes(stats.ramChange)} · ${byteRate(stats.ramChangeRate)}`, 'ram')],
    ['ssd', bar('SSD', value('ssdUsed'), value('ssdCapacity'), `${v.ssdCapacity === null ? 'Capacity not exported' : bytes(value('ssdUsed'))} · Δ ${signedBytes(stats.ssdChange)} · ${byteRate(stats.ssdChangeRate)}`, 'ssd')],
  ].filter(([key]) => tab === 'summary' || tab === key).map(([, html]) => html).join('');
  return `<div class="occupancy">${bars}</div><p class="note">${storeFresh ? `Store snapshot ${duration(Math.max(0, end.t - v.storeTimestamp))} ago` : 'Store occupancy unavailable or stale'} · ${tab === 'gpu' ? 'GPU Hot hits count cache groups.' : 'RAM / SSD rates are net occupancy changes.'}</p>`;
}

function render() {
  const now = Date.now() / 1000, end = history.samples.at(-1) || makeSnapshot('', now);
  const w = windowStatistics(history.samples, windowSeconds) || {...statistics(end, end), end, start: null, complete: false};
  const previous = history.samples.at(-2);
  const interval = previous ? end.t - previous.t : 0;
  const fresh = state === 'online' && now - end.t <= GAP_SECONDS && !document.hidden;
  const instant = fresh && previous && interval <= GAP_SECONDS ? statistics(end, previous) : null;
  const life = statistics(end);
  const uptime = info ? Math.max(0, now - info.started_at) : null;
  const lifeRPM = info && fresh && finite(life.requests) ? ratio(life.requests * 60, uptime) : null;
  const lifeTPM = info && fresh && finite(life.total) ? ratio(life.total * 60, uptime) : null;
  $('#health').dataset.state = document.hidden ? 'connecting' : state;
  $('#health').textContent = document.hidden ? 'Paused' : ({online: 'Online', offline: 'Offline', locked: 'API key required', connecting: 'Connecting'}[state]);
  $('#connection').textContent = document.hidden ? 'Polling paused while this tab is hidden.' : message;
  $('#auth').hidden = state !== 'locked';
  if (state === 'locked') $('#model').textContent = '';
  $('#disconnect').hidden = !token;
  $('#lifetime').innerHTML = [
    ['Service uptime', duration(fresh ? uptime : null)], ['Total tokens', number(life.total)],
    ['Cached input tokens', number(life.cached)], ['Output tokens', number(life.output)],
    ['Completed requests', number(life.requests)], ['Cache hit rate', percent(life.hit)],
    ['Avg RPM', number(lifeRPM)], ['Avg TPM', number(lifeTPM)],
  ].map(([label, value]) => `<div><dt>${label}</dt><dd>${escape(value)}</dd></div>`).join('');
  const observed = w.start ? duration(w.seconds) : 'Waiting for next sample';
  const hasGaps = history.samples.some((sample, i) => i && sample.t >= end.t - windowSeconds && sample.t - history.samples[i - 1].t > GAP_SECONDS);
  $('#coverage').textContent = `${w.complete ? windowLabel() : `Partial window · ${observed}`}${hasGaps ? ' · Chart has observation gaps' : ''}`;
  for (const button of document.querySelectorAll('[data-window]')) button.setAttribute('aria-pressed', String(Number(button.dataset.window) === windowSeconds));
  const latest = fresh ? history.latestPrefill : null;
  const values = {
    requests: {
      kpis: kpi('Now', number(fresh ? end.v.running : null), '', 'Running requests') + kpi(windowLabel(), number(w.rpm), 'req/min', 'Completed requests · RPM', true),
      rows: [['Waiting now', number(fresh ? end.v.waiting : null)], ['TPM', number(w.tpm)], ['Completed requests', number(w.requests)]],
      chart: chart(chartPoints(history.samples, 'requests', windowSeconds).map(point => ({...point, value: finite(point.value) ? point.value * 60 : null})), 'Completed requests · req/min'),
    },
    decode: {
      kpis: kpi('Now', number(instant?.outputRate), 'tok/s', 'Output token throughput') + kpi(windowLabel(), number(w.outputRate), 'tok/s', 'Average output throughput', true),
      rows: [['Output tokens', number(w.output)], ['Mean inter-token latency', seconds(w.itlMean)], ['Preemptions', number(w.preemptions)]],
      chart: chart(chartPoints(history.samples, 'output', windowSeconds), 'Output · tok/s'),
    },
    prefill: {
      kpis: kpi('Latest reported prefill', number(latest?.rate, false), 'tok/s', latest ? `${latest.count} completed request${latest.count === 1 ? '' : 's'} · observed ${duration(now - latest.t)} ago` : 'Waiting for a completed prefill sample') + kpi(windowLabel(), number(w.computedRate, false), 'tok/s', 'Computed input · wall-time average', true),
      rows: [['Total input throughput (incl. cached)', `${number(w.inputRate, false)} tok/s`], ['Computed input tokens', number(w.computed)], ['Completed prefill samples', number(w.prefillCount)]],
      chart: chart(chartPoints(history.samples, 'computed', windowSeconds), 'Computed input · tok/s'),
    },
    cache: {
      kpis: kpi(windowLabel(), percent(w.hit), '', 'Total input cache hit rate') + kpi(windowLabel(), number(w.cached), 'tokens', 'Cached input', true),
      rows: [['RAM input share', percent(w.ramShare)], ['SSD input share', percent(w.ssdShare)]],
      chart: cacheBars(end, w, fresh),
    },
    speculative: {
      kpis: kpi(windowLabel(), percent(w.specAcceptance), '', 'Draft token acceptance rate') + kpi(windowLabel(), number(w.acceptanceLength), 'tokens', 'Mean acceptance length incl. bonus', true),
      rows: [['Accepted draft tokens', number(w.accepted)], ['Drafted tokens', number(w.drafted)], ['Draft rounds', number(w.drafts)]],
      chart: chart(chartPoints(history.samples, 'accepted', windowSeconds), 'Accepted draft tokens · tok/s'),
    },
    latency: {
      kpis: kpi(windowLabel(), seconds(w.ttftP50), '', 'TTFT P50 · bucket upper bound') + kpi(windowLabel(), seconds(w.ttftP95), '', 'TTFT P95 · ≥10 samples', true),
      rows: [['Mean time to first token', seconds(w.ttftMean)], ['Mean request latency (E2E)', seconds(w.e2eMean)], ['Mean queue time', seconds(w.queueMean)]],
      chart: chart(history.samples.slice(1).flatMap((sample, i) => sample.t < end.t - windowSeconds ? [] : [{t: sample.t, value: sample.t - history.samples[i].t > GAP_SECONDS ? null : ratio(delta(sample, history.samples[i], 'ttftSum'), delta(sample, history.samples[i], 'ttftCount'))}]), 'Mean TTFT · seconds'),
    },
  };
  for (const [key, title] of Object.entries(titles)) {
    const value = values[key];
    $(`[data-card=${key}] .card-body`).innerHTML = `<div class="card-heading"><h2>${title}</h2><span>${windowLabel()}</span></div><div class="kpis">${value.kpis}</div>${rows(value.rows)}${value.chart}`;
  }
  $('#engine').textContent = `Engine: ReVolta ${info?.version || '—'}`;
  $('#observation').textContent = history.samples.length ? `Last observed ${new Date(end.t * 1000).toLocaleTimeString()} · ${history.samples.length} retained samples` : 'No browser observations yet';
  $('#cache-state').textContent = `${history.cacheState} · Up to 15 minutes / 500 samples · Saves at most every 30 seconds and on page exit. API keys are never saved.`;
  if (detailCard) renderDetails(w, life, end);
}

function detailRows(card, stats) {
  const common = {
    requests: [['Completed requests', number(stats.requests)], ['Input tokens', number(stats.input)], ['Output tokens', number(stats.output)], ['Total tokens', number(stats.total)], ['Preemptions', number(stats.preemptions)]],
    decode: [['Output tokens', number(stats.output)], ['Mean inter-token latency', seconds(stats.itlMean)], ['Mean time per output token', seconds(stats.tpotMean)]],
    prefill: [['Input tokens (incl. cached)', number(stats.input)], ['Computed input tokens', number(stats.computed)], ['Cached input tokens', number(stats.cached)], ['Completed prefill samples', number(stats.prefillCount)], ['Prefill phase time', seconds(stats.prefillTime)], ['Computed tokens / prefill phase time', `${number(stats.prefillPhase?.rate, false)} tok/s`]],
    cache: [['Total input cache hit rate', percent(stats.hit)], ['Cached input tokens', number(stats.cached)], ['Input tokens', number(stats.input)], ['GPU Hot group hit rate', percent(stats.hotHit)], ['RAM input share', percent(stats.ramShare)], ['SSD input share', percent(stats.ssdShare)]],
    speculative: [['Draft token acceptance', percent(stats.specAcceptance)], ['Mean acceptance length incl. bonus', number(stats.acceptanceLength)], ['Accepted draft tokens', number(stats.accepted)], ['Drafted tokens', number(stats.drafted)], ['Draft rounds', number(stats.drafts)]],
    latency: [['Mean time to first token', seconds(stats.ttftMean)], ['Mean inter-token latency', seconds(stats.itlMean)], ['Mean time per output token', seconds(stats.tpotMean)], ['Mean request latency (E2E)', seconds(stats.e2eMean)], ['Mean queue time', seconds(stats.queueMean)], ['TTFT samples', number(stats.ttftCount)]],
  };
  if (card === 'cache' && cacheTab !== 'summary') {
    return ({
      gpu: [['GPU Hot group hit rate', percent(stats.hotHit)], ['Groups requested', number(stats.hotWanted)], ['Groups missed', number(stats.hotMissed)], ['GPU-cached input tokens', number(stats.gpuInput)]],
      ram: [['RAM input share', percent(stats.ramShare)], ['RAM-cached input tokens', number(stats.ramInput)], ['Reused in RAM', number(stats.ramReuseInput)], ['Loaded from RAM', number(stats.ramLoadInput)]],
      ssd: [['SSD input share', percent(stats.ssdShare)], ['SSD-restored input tokens', number(stats.ssdInput)], ['SSD read bytes (verified)', bytes(stats.readBytes)], ['SSD write bytes (committed)', bytes(stats.writeBytes)]],
    })[cacheTab];
  }
  return common[card];
}

function renderDetails(w, life, end) {
  const body = $('#detail-body'), scroll = body.scrollTop;
  $('#detail-title').textContent = `${titles[detailCard]} details`;
  $('#detail-scope').textContent = `${windowLabel()} · ${w.start ? `${duration(w.seconds)} observed` : 'Waiting for next sample'} · Cumulative counters since engine start`;
  $('#detail-tabs').hidden = detailCard !== 'cache';
  const since = detailRows(detailCard, life);
  const selected = detailRows(detailCard, w);
  if (detailCard === 'requests') {
    selected.unshift(['RPM', number(w.rpm)], ['TPM', number(w.tpm)]);
    const uptime = info ? Math.max(0, end.t - info.started_at) : null;
    since.unshift(['Avg RPM', number(finite(life.requests) ? ratio(life.requests * 60, uptime) : null)], ['Avg TPM', number(finite(life.total) ? ratio(life.total * 60, uptime) : null)]);
  }
  if (detailCard === 'decode') selected.unshift(['Output throughput', `${number(w.outputRate)} tok/s`]);
  if (detailCard === 'prefill') selected.unshift(['Computed input throughput', `${number(w.computedRate, false)} tok/s`], ['Total input throughput (incl. cached)', `${number(w.inputRate, false)} tok/s`]);
  if (detailCard === 'latency') selected.unshift(['TTFT P50 upper bound', seconds(w.ttftP50)], ['TTFT P95 upper bound (≥10 samples)', seconds(w.ttftP95)]);
  let explanation = ({
    requests: 'RPM = completed requests × 60 / elapsed seconds. TPM = (input + output tokens) × 60 / elapsed seconds. Completed requests include the engine’s stop, length and abort finish reasons.',
    decode: 'Now uses the latest two observations (normally 2 seconds). Window throughput uses output-token counter changes / actual elapsed time. Per-request ITL and TPOT are separate latency measurements.',
    prefill: 'Latest reported prefill = new KV tokens / prefill-phase seconds for matching newly completed requests. These histograms are reported at request completion. The value is retained during idle/decode; its age is shown. The window average divides newly computed input tokens by wall time. Total input throughput includes cached input.',
    cache: 'Total cache hit rate = cached input tokens / input tokens = cached input / (total tokens − output tokens). RAM and SSD input shares use all classified prompt tokens as their denominator; they are not conditional lookup hit rates. GPU Hot uses groups, not tokens. These counters have different publication times.',
    speculative: 'Acceptance rate = accepted draft tokens / drafted tokens. Mean acceptance length = 1 + accepted draft tokens / draft rounds, including the bonus token. Unexported metrics remain unavailable.',
    latency: 'Means use histogram sum / count for the same period. TTFT percentiles show bucket upper bounds, not interpolated precision. P95 requires at least 10 samples; observations beyond the largest finite bucket show —.',
  })[detailCard];
  let current = '';
  if (detailCard === 'cache') {
    const v = end.v;
    const occupancyRows = [
      ['gpu', 'Engine KV usage (max across engines)', percent(v.kv)],
      ['ram', 'RAM allocation / capacity', `${bytes(v.ramUsed)} / ${bytes(v.ramCapacity)}`],
      ['ssd', 'SSD allocation / capacity', `${bytes(v.ssdUsed)} / ${bytes(v.ssdCapacity)}`],
    ].filter(([key]) => cacheTab === 'summary' || cacheTab === key).map(([, label, value]) => [label, value]);
    current = `<h3>Current occupancy · last reported snapshot</h3>${cacheBars(end, w, state === 'online', cacheTab)}${rows(occupancyRows)}`;
    explanation += ' Occupancy is a current snapshot. Δ and bytes/s describe net occupancy change within the selected window. Startup occupancy and unobserved peaks cannot be recovered from counters. SSD utilization stays unavailable when this build does not export capacity.';
  }
  body.innerHTML = `${current}<div class="detail-grid"><section><h3>${windowLabel()}</h3>${rows(selected)}</section><section><h3>Since engine start</h3>${rows(since)}</section></div><p class="explanation">${escape(explanation)}</p>`;
  body.scrollTop = scroll;
  for (const button of document.querySelectorAll('[data-tab]')) button.setAttribute('aria-pressed', String(button.dataset.tab === cacheTab));
}

async function request(path, requestToken) {
  const response = await fetch(path, {
    headers: requestToken ? {Authorization: `Bearer ${requestToken}`} : {}, cache: 'no-store',
    signal: AbortSignal.timeout(6000), credentials: 'same-origin',
  });
  if (response.status === 401 || response.status === 403) {
    throw Object.assign(new Error(requestToken ? 'API key was rejected. Enter a valid key to connect.' : 'This engine requires an API key.'), {authentication: true});
  }
  if (!response.ok) throw new Error(`Engine endpoint returned HTTP ${response.status}.`);
  return response;
}

async function poll() {
  if (inFlight || document.hidden || state === 'locked') return;
  clearTimeout(timer);
  inFlight = true;
  const generation = connectionGeneration, requestToken = token;
  try {
    let metadata = info || await (await request('config', requestToken)).json();
    const response = await request('metrics', requestToken);
    const id = response.headers.get('X-ReVolta-Instance-Id');
    if (!id) throw new Error('Engine instance identity is unavailable.');
    if (id !== metadata.instance_id) metadata = await (await request('config', requestToken)).json();
    if (id !== metadata.instance_id) throw new Error('Engine restarted during collection. Retrying.');
    const now = Date.now() / 1000;
    const snapshot = makeSnapshot(await response.text(), now);
    // A response authorized before Disconnect must never restore that session.
    if (generation !== connectionGeneration) return;
    if (!finite(snapshot.v.input) || !finite(snapshot.v.output)) throw new Error('Inference token metrics are not available. Check that engine statistics are enabled.');
    info = metadata;
    history.connect(id, now);
    const reset = history.append(snapshot);
    state = 'online';
    message = `${reset ? 'Counters reset; observation history restarted. ' : ''}Metrics updated every 2 seconds while this tab is visible. Now, selected-window averages and cumulative totals have separate scopes.`;
    const models = snapshot.models;
    const model = $('#model');
    model.textContent = models.length ? `— ${models.join(' · ')}` : '';
    if (models.length === 1 && /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(models[0])) {
      const link = document.createElement('a');
      link.href = `https://huggingface.co/${models[0]}`;
      link.target = '_blank'; link.rel = 'noopener noreferrer'; link.textContent = models[0];
      model.replaceChildren(document.createTextNode('— '), link);
    }
  } catch (error) {
    if (generation !== connectionGeneration) return;
    state = error.authentication ? 'locked' : 'offline';
    message = `${error.message} ${history.samples.length ? 'Showing last observed counters; realtime values are unavailable.' : ''}`;
    if (state === 'locked') {
      // A revoked key must not keep exposing cached data through this page.
      history.samples = []; history.latestPrefill = null; info = null;
    }
  } finally {
    inFlight = false;
    if (generation === connectionGeneration) render();
    if (!document.hidden && state !== 'locked') timer = setTimeout(poll, generation === connectionGeneration ? 2000 : 0);
  }
}

document.addEventListener('click', event => {
  const windowButton = event.target.closest('[data-window]');
  if (windowButton) { windowSeconds = Number(windowButton.dataset.window); savePreferences(); render(); }
  const details = event.target.closest('[data-details]');
  if (details) { detailCard = details.dataset.details; cacheTab = 'summary'; render(); $('#details').showModal(); }
  const tab = event.target.closest('[data-tab]');
  if (tab) { cacheTab = tab.dataset.tab; render(); }
});
$('#close').addEventListener('click', () => $('#details').close());
$('#details').addEventListener('close', () => {
  // Native close events are queued. A rapid reopen can happen before the old
  // event arrives; it must not clear the newly opened card's state.
  if (!$('#details').open) detailCard = null;
});
$('#details').addEventListener('click', event => {
  if (event.target !== $('#details')) return;
  const box = event.target.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) event.target.close();
});
$('#auth').addEventListener('submit', event => {
  event.preventDefault(); token = $('#key').value.trim(); $('#key').value = '';
  connectionGeneration++; info = null; state = 'connecting'; render(); poll();
});
$('#disconnect').addEventListener('click', () => {
  connectionGeneration++; token = ''; info = null; history.samples = []; history.latestPrefill = null;
  state = 'locked'; message = 'Disconnected. Enter an API key to reconnect.'; clearTimeout(timer); render();
});
$('#remember').addEventListener('change', event => { history.setRemember(event.target.checked); savePreferences(); render(); });
$('#clear').addEventListener('click', () => { history.clear(); message = 'Browser history cleared. New observations will refill the selected window.'; render(); });
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { clearTimeout(timer); history.save(Date.now() / 1000, true); render(); }
  else poll();
});
window.addEventListener('pagehide', () => history.save(Date.now() / 1000, true));
render();
poll();
