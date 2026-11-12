// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the ReVolta project
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  chartPoints, fraction, makeSnapshot, parseMetrics, prefillObservation,
  sameEpoch, statistics, STORE_STALE_SECONDS, storeSnapshot, ttftQuantile, windowStatistics,
} from '../../../../vllm/entrypoints/serve/dashboard/static/metrics.mjs';
import {
  CACHE_BYTES, CACHE_SCHEMA, MAX_SAMPLES, ObservationHistory,
} from '../../../../vllm/entrypoints/serve/dashboard/static/history.mjs';

const baseTime = 1800000000;
function text(step = 0, {idle = false, cachedOnly = false, mismatch = false} = {}) {
  const n = idle ? 0 : step;
  const values = {
    prompt_tokens_total: 10000 + n * 3000, generation_tokens_total: 1000 + n * 144,
    prompt_tokens_cached_total: 8000 + n * (cachedOnly ? 3000 : 2400),
    request_success_total: 10 + n, num_preemptions_total: 0,
    num_requests_running: idle ? 0 : 2, num_requests_waiting: 1, kv_cache_usage_perc: 0.5,
    request_prefill_time_seconds_sum: 4 + n * 0.45,
    request_prefill_time_seconds_count: 10 + n,
    request_prefill_kv_computed_tokens_sum: 2000 + n * (cachedOnly ? 0 : 600),
    request_prefill_kv_computed_tokens_count: 10 + n + (mismatch ? 1 : 0),
    time_to_first_token_seconds_sum: 1 + n * 0.25,
    time_to_first_token_seconds_count: 10 + n,
    inter_token_latency_seconds_sum: 20 + n * 2,
    inter_token_latency_seconds_count: 1000 + n * 144,
    e2e_request_latency_seconds_sum: 30 + n * 3,
    e2e_request_latency_seconds_count: 10 + n,
    spec_decode_num_accepted_tokens_total: 20 + n * 8,
    spec_decode_num_draft_tokens_total: 100 + n * 10,
    spec_decode_num_drafts_total: 10 + n * 2,
  };
  const lines = Object.entries(values).map(([name, value]) => `vllm:${name}{model_name="example/model",engine="0"} ${value}`);
  for (const [source, value] of [['local_compute', 2000 + n * (cachedOnly ? 0 : 600)], ['local_cache_hit', 4000 + n * 1200], ['external_kv_transfer', 4000 + n * 1200]]) lines.push(`vllm:prompt_tokens_by_source_total{model_name="example/model",engine="0",source="${source}"} ${value}`);
  const store = {unified_region_bytes: 1000, unified_allocated_main_bytes: 300 + n * 10, unified_allocated_state_bytes: 100, unified_allocated_draft_bytes: 50, unified_ssd_bytes: 100 + n * 5, store_snapshot_unix_s: baseTime + step * 2};
  for (const [metric, value] of Object.entries(store)) lines.push(`vllm:store_state{metric="${metric}"} ${value}`);
  for (const [event, increment] of Object.entries({prompt_tokens_gpu: 600, prompt_tokens_host_reuse: 800, prompt_tokens_host_load: 200, prompt_tokens_ssd: 800, prompt_tokens_compute: 600, gpu_hot_wanted_groups_total: 10, gpu_hot_miss_groups_total: 2})) lines.push(`vllm:store_events_total{event="${event}"} ${increment * (10 + n)}`);
  for (const le of ['0.1', '0.5', '1.0', '+Inf']) lines.push(`vllm:time_to_first_token_seconds_bucket{le="${le}"} ${le === '0.1' ? 0 : 10 + n}`);
  return lines.join('\n');
}
const snapshot = (step, options) => makeSnapshot(text(step, options), baseTime + step * 2);
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-8, `${a} != ${b}`);
const memory = () => {
  const map = new Map();
  return {writes: 0, getItem(key) { return map.get(key) || null; }, setItem(key, value) { this.writes++; map.set(key, value); }, removeItem(key) { map.delete(key); }};
};

test('Prometheus labels support escaped quotes, newlines, backslashes and braces', () => {
  const parsed = parseMetrics('vllm:prompt_tokens_total{engine="0",model_name="a}b\\\"c\\nline\\\\end"} 1.2e3 123\n');
  assert.equal(parsed.get('vllm:prompt_tokens_total')[0].labels.model_name, 'a}b"c\nline\\end');
  assert.equal(parsed.get('vllm:prompt_tokens_total')[0].value, 1200);
});

test('Missing, malformed, NaN, infinite and negative values remain unavailable', () => {
  const sample = makeSnapshot('vllm:prompt_tokens_total NaN\nvllm:generation_tokens_total +Inf\nvllm:request_success_total -1\nvllm:num_requests_running{bad=x} 3', baseTime);
  const values = statistics(sample, sample);
  assert.equal(values.input, null); assert.equal(values.rpm, null); assert.equal(values.tpm, null);
  assert.equal(sample.v.running, null);
  assert.throws(() => parseMetrics('x'.repeat(8 * 1024 * 1024 + 1)));
});

test('Known zero counters are distinct from missing counters', () => {
  const a = makeSnapshot('vllm:prompt_tokens_total 0\nvllm:generation_tokens_total 0\nvllm:request_success_total 0', baseTime);
  const b = {...a, t: baseTime + 2};
  const stats = statistics(b, a);
  assert.equal(stats.rpm, 0); assert.equal(stats.tpm, 0); assert.equal(stats.outputRate, 0);
  assert.equal(stats.computedRate, null); assert.equal(stats.hit, null);
});

test('Cache hit denominator excludes output and includes all cached input', () => {
  const life = statistics(snapshot(1));
  close(life.hit, 10400 / (life.total - life.output));
  const window = statistics(snapshot(1), snapshot(0));
  close(window.hit, 0.8); close(window.inputRate, 1500); close(window.computedRate, 300);
  close(window.rpm, 30); close(window.tpm, (3000 + 144) * 30);
});

test('Older source counters can supply cached input when direct cached counter is absent', () => {
  const sample = makeSnapshot(text(1).split('\n').filter(line => !line.startsWith('vllm:prompt_tokens_cached_total')).join('\n'), baseTime);
  assert.equal(sample.v.cached, 10400);
  assert.equal(fraction(120, 100), null);
});

test('Multiple engines aggregate raw counts before dividing; KV utilization uses max', () => {
  const sample = makeSnapshot('vllm:prompt_tokens_total{engine="0"} 10\nvllm:prompt_tokens_total{engine="1"} 100\nvllm:prompt_tokens_cached_total{engine="0"} 8\nvllm:prompt_tokens_cached_total{engine="1"} 1\nvllm:kv_cache_usage_perc{engine="0"} 0.2\nvllm:kv_cache_usage_perc{engine="1"} 0.7', baseTime);
  close(statistics(sample).hit, 9 / 110); close(sample.v.kv, 0.7);
});

test('1/5/15-minute windows display an actual two-second partial window immediately', () => {
  for (const seconds of [60, 300, 900]) {
    const w = windowStatistics([snapshot(0), snapshot(1)], seconds);
    assert.equal(w.complete, false); assert.equal(w.seconds, 2);
    close(w.outputRate, 72); close(w.rpm, 30);
  }
});

test('Full windows roll over the last selected duration without lifetime leakage', () => {
  const samples = Array.from({length: 601}, (_, i) => snapshot(i));
  for (const seconds of [60, 300, 900]) {
    const w = windowStatistics(samples, seconds);
    assert.equal(w.complete, true); assert.equal(w.seconds, seconds);
    close(w.outputRate, 72); assert.equal(w.output, seconds * 72);
  }
});

test('One observation cannot invent a rate or window sample', () => {
  const w = windowStatistics([snapshot(0)], 900);
  assert.equal(w.start, null); assert.equal(w.outputRate, null); assert.equal(w.hit, null);
});

test('Counter resets cannot be hidden by an increasing aggregate across engines', () => {
  const a = makeSnapshot('vllm:prompt_tokens_total{engine="0"} 100\nvllm:prompt_tokens_total{engine="1"} 100', baseTime);
  const b = makeSnapshot('vllm:prompt_tokens_total{engine="0"} 50\nvllm:prompt_tokens_total{engine="1"} 300', baseTime + 2);
  assert.ok(b.v.input > a.v.input); assert.equal(sameEpoch(a, b), false);
});

test('Model/engine membership changes invalidate the old epoch', () => {
  const a = snapshot(0), b = snapshot(1);
  b.series['new engine'] = 0;
  assert.equal(sameEpoch(a, b), false);
});

test('Prefill divides matching completed-request token/time histograms', () => {
  const latest = prefillObservation(snapshot(1), snapshot(0));
  close(latest.rate, 600 / 0.45); assert.equal(latest.count, 1); assert.equal(latest.t, baseTime + 2);
  assert.equal(prefillObservation(snapshot(1, {mismatch: true}), snapshot(0)), null);
});

test('Prefill rejects per-engine mismatches even when aggregate counts agree', () => {
  const sample = snapshot(1);
  sample.prefill = {
    a: {prefillTokens: 100, prefillTime: 1, prefillCount: 2, prefillTokenCount: 1},
    b: {prefillTokens: 100, prefillTime: 1, prefillCount: 1, prefillTokenCount: 2},
  };
  assert.equal(prefillObservation(sample), null);
});

test('Latest prefill survives idle with its original age and multi-request weighting', () => {
  const history = new ObservationHistory(); history.connect('a', baseTime);
  history.append(snapshot(0)); history.append(snapshot(3));
  assert.equal(history.latestPrefill.count, 3); close(history.latestPrefill.rate, 600 / 0.45);
  const idle = structuredClone(snapshot(3)); idle.t += 2;
  history.append(idle); assert.equal(history.latestPrefill.t, baseTime + 6);
});

test('Cache-only prefill is a measured zero, not an unavailable value', () => {
  const result = prefillObservation(snapshot(1, {cachedOnly: true}), snapshot(0));
  assert.equal(result.rate, 0); assert.equal(result.count, 1);
});

test('Observation gaps retain true counter-window averages and break charts/latest events', () => {
  const history = new ObservationHistory(); history.connect('a', baseTime);
  history.append(snapshot(0)); history.append(snapshot(1)); history.append(snapshot(10));
  assert.equal(history.latestPrefill, null);
  close(windowStatistics(history.samples, 60).outputRate, 72);
  assert.equal(chartPoints(history.samples, 'output', 60).at(-1).value, null);
});

test('RAM/SSD shares use classified inputs and occupancy change can be negative', () => {
  const a = snapshot(1), b = snapshot(2);
  b.v.ramMain = 100; b.v.ramUsed = 250;
  const w = statistics(b, a);
  close(w.ramShare, 1 / 3); close(w.ssdShare, 800 / 3000); close(w.hotHit, 0.8);
  assert.ok(w.ramChange < 0); assert.ok(w.ramChangeRate < 0);
  const missing = makeSnapshot('', baseTime);
  assert.equal(statistics({...missing, t: baseTime + 2}, missing).ramChangeRate, null);
});

test('TTFT percentiles are finite bucket upper bounds and P95 needs ten samples', () => {
  const a = snapshot(0), b = snapshot(1);
  a.buckets = {'0.1': 0, '0.5': 0, '1.0': 0, '+Inf': 0}; a.v.ttftCount = 0;
  b.buckets = {'0.1': 3, '0.5': 8, '1.0': 10, '+Inf': 10}; b.v.ttftCount = 10;
  close(ttftQuantile(b, a, 0.5), 0.5); close(ttftQuantile(b, a, 0.95), 1);
  b.buckets = {'0.1': 3, '0.5': 8, '1.0': 9, '+Inf': 9}; b.v.ttftCount = 9;
  assert.equal(ttftQuantile(b, a, 0.95), null);
  b.buckets = {'0.1': 0, '0.5': 0, '1.0': 0, '+Inf': 10}; b.v.ttftCount = 10;
  assert.equal(ttftQuantile(b, a, 0.95), null); // Out of the largest finite bucket.
});

test('Histogram count and buckets must describe the same observations', () => {
  const a = snapshot(0), b = snapshot(1);
  b.buckets['+Inf'] += 10;
  assert.equal(ttftQuantile(b, a, 0.5), null);
});

test('Store occupancy freshness is one rule with a verifiable snapshot age', () => {
  const fresh = storeSnapshot(snapshot(1), true);
  assert.equal(fresh.current, true); assert.equal(fresh.age, 0);
  assert.equal(storeSnapshot(snapshot(1), false).current, false);

  const stale = storeSnapshot({...snapshot(1), t: baseTime + 2 + STORE_STALE_SECONDS + 1}, true);
  assert.equal(stale.current, false); assert.equal(stale.age, STORE_STALE_SECONDS + 1);

  // The last observed value keeps ageing while the service is unreachable.
  assert.equal(storeSnapshot(snapshot(1), false, baseTime + 122).age, 120);

  const future = storeSnapshot({...snapshot(1), t: baseTime - 60}, true);
  assert.equal(future.current, false); assert.equal(future.age, 0); // Engine clock ahead of the observation.

  const missing = storeSnapshot(makeSnapshot('', baseTime), true);
  assert.equal(missing.current, false); assert.equal(missing.age, null);
});

test('Browser cache restores same-instance history and is scoped by URL path', () => {
  const storage = memory();
  const a = new ObservationHistory({storage, scope: '/a/', remember: true}); a.connect('engine', baseTime);
  a.append(snapshot(0)); a.append(snapshot(1)); a.save(baseTime + 2, true);
  const b = new ObservationHistory({storage, scope: '/a/', remember: true}); b.connect('engine', baseTime + 3);
  assert.equal(b.samples.length, 2); close(b.latestPrefill.rate, 600 / 0.45);
  const c = new ObservationHistory({storage, scope: '/b/', remember: true}); c.connect('engine', baseTime + 3);
  assert.equal(c.samples.length, 0);
});

test('Remembering is opt-in and writes nothing before it is enabled', () => {
  const storage = memory();
  storage.setItem(`revolta-dashboard:${CACHE_SCHEMA}:/a/`, JSON.stringify({schema: CACHE_SCHEMA, instance: 'engine', samples: [snapshot(0)]}));
  const history = new ObservationHistory({storage, scope: '/a/'});
  assert.equal(history.remember, false); assert.equal(history.cacheState, 'Disabled');
  assert.equal(storage.getItem(history.key), null, 'A record from a remembering session is dropped when remembering is off');
  storage.writes = 0; // Do not count the seeded record or the drop as page writes.
  history.connect('engine', baseTime);
  for (const step of [0, 1, 2]) history.append(snapshot(step));
  history.save(baseTime + 6, true);
  assert.equal(storage.writes, 0); assert.equal(storage.getItem(history.key), null);
  history.setRemember(true);
  assert.equal(storage.writes, 1);
  assert.ok(storage.getItem(history.key).includes('example/model'));
  const restored = new ObservationHistory({storage, scope: '/a/', remember: true});
  restored.connect('engine', baseTime + 7);
  assert.equal(restored.samples.length, 3);
});

test('A new engine identity or a counter reset clears cached history', () => {
  const storage = memory(), history = new ObservationHistory({storage, remember: true});
  history.connect('old', baseTime); history.append(snapshot(0)); history.append(snapshot(1));
  history.connect('new', baseTime + 4);
  assert.equal(history.samples.length, 0); assert.equal(history.latestPrefill, null);
  history.append(snapshot(1)); const reset = snapshot(0); reset.t = baseTime + 4;
  assert.equal(history.append(reset), true); assert.equal(history.samples.length, 1);
});

test('Cache bounds, schema, corruption, future times and retention are enforced', () => {
  const storage = memory(), history = new ObservationHistory({storage, remember: true});
  history.connect('a', baseTime);
  for (let i = 0; i < 1000; i++) history.append(snapshot(i));
  assert.ok(history.samples.length <= MAX_SAMPLES);
  assert.ok(history.samples.at(-1).t - history.samples[0].t <= 900);
  history.save(baseTime + 2000, true); assert.ok(storage.getItem(history.key).length <= CACHE_BYTES);
  for (const value of ['{', JSON.stringify({schema: CACHE_SCHEMA + 1, instance: 'a', samples: [snapshot(0)]}), JSON.stringify({schema: CACHE_SCHEMA, instance: 'a', samples: [snapshot(0), snapshot(2), snapshot(1)]})]) {
    storage.setItem(history.key, value);
    const copy = new ObservationHistory({storage, remember: true}); copy.connect('a', baseTime + 10); assert.equal(copy.samples.length, 0);
  }
  storage.setItem(history.key, JSON.stringify({schema: CACHE_SCHEMA, instance: 'a', samples: [snapshot(0), snapshot(10000)]}));
  const copy = new ObservationHistory({storage, remember: true}); copy.connect('a', baseTime + 10000);
  assert.equal(copy.samples.length, 0);
});

test('Writes are throttled and storage denial degrades to in-memory operation', () => {
  const storage = memory(), history = new ObservationHistory({storage, remember: true});
  history.connect('a', baseTime);
  for (let i = 0; i < 10; i++) history.append(snapshot(i));
  assert.equal(storage.writes, 1);
  history.save(baseTime + 18, true); assert.equal(storage.writes, 2);
  history.setRemember(false); assert.equal(storage.getItem(history.key), null);
  history.append(snapshot(10)); assert.equal(storage.writes, 2);
  const blocked = new ObservationHistory({storage: {getItem() { throw Error('blocked'); }, setItem() { throw Error('quota'); }}, remember: true});
  blocked.connect('a', baseTime); blocked.append(snapshot(0)); assert.equal(blocked.samples.length, 1);
  assert.equal(blocked.cacheState, 'Memory only');
});

test('Client clock reversal resets observations rather than producing negative rates', () => {
  const history = new ObservationHistory(); history.connect('a', baseTime);
  history.append(snapshot(1)); assert.equal(history.append(snapshot(0)), true);
  assert.equal(history.samples.length, 1);
});
