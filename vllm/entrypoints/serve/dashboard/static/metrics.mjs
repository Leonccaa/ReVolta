// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the ReVolta project
// Browser and Node compatible: no DOM, network, or server-side calculations.

export const POLL_SECONDS = 2;
export const GAP_SECONDS = 10;
export const RETENTION_SECONDS = 900;
const P = 'vllm:';
const fields = {
  input: 'prompt_tokens_total', output: 'generation_tokens_total',
  cached: 'prompt_tokens_cached_total', requests: 'request_success_total',
  preemptions: 'num_preemptions_total',
  prefillTime: 'request_prefill_time_seconds_sum',
  prefillCount: 'request_prefill_time_seconds_count',
  prefillTokens: 'request_prefill_kv_computed_tokens_sum',
  prefillTokenCount: 'request_prefill_kv_computed_tokens_count',
  ttftSum: 'time_to_first_token_seconds_sum', ttftCount: 'time_to_first_token_seconds_count',
  itlSum: 'inter_token_latency_seconds_sum', itlCount: 'inter_token_latency_seconds_count',
  e2eSum: 'e2e_request_latency_seconds_sum', e2eCount: 'e2e_request_latency_seconds_count',
  queueSum: 'request_queue_time_seconds_sum', queueCount: 'request_queue_time_seconds_count',
  tpotSum: 'request_time_per_output_token_seconds_sum', tpotCount: 'request_time_per_output_token_seconds_count',
  accepted: 'spec_decode_num_accepted_tokens_total', drafted: 'spec_decode_num_draft_tokens_total',
  drafts: 'spec_decode_num_drafts_total',
};
const gauges = {
  running: 'num_requests_running', waiting: 'num_requests_waiting', kv: 'kv_cache_usage_perc',
};
const stateFields = {
  ramCapacity: 'unified_region_bytes', ramMain: 'unified_allocated_main_bytes',
  ramState: 'unified_allocated_state_bytes', ramDraft: 'unified_allocated_draft_bytes',
  ssdUsed: 'unified_ssd_bytes', ssdCapacity: 'unified_ssd_capacity_bytes',
  readBytes: 'unified_prefetch_verified_bytes', writeBytes: 'unified_committed_bytes',
  hotCapacity: 'gpu_hot_capacity_slots', hotUsed: 'gpu_hot_valid_slots',
  storeTimestamp: 'store_snapshot_unix_s',
};
const eventFields = {
  gpuInput: 'prompt_tokens_gpu', ramReuseInput: 'prompt_tokens_host_reuse',
  ramLoadInput: 'prompt_tokens_host_load', ssdInput: 'prompt_tokens_ssd',
  computedStoreInput: 'prompt_tokens_compute',
  hotWanted: 'gpu_hot_wanted_groups_total', hotMissed: 'gpu_hot_miss_groups_total',
};
export const COUNTERS = new Set([
  ...Object.keys(fields), ...Object.keys(eventFields),
  'computed', 'readBytes', 'writeBytes',
]);
const wanted = new Set([
  ...Object.values(fields), ...Object.values(gauges), 'prompt_tokens_by_source_total',
  'store_state', 'store_events_total', 'time_to_first_token_seconds_bucket',
].map(name => P + name));
const samplePattern = /^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{((?:[^"{}]|"(?:\\.|[^"\\])*")*)\})?\s+(\S+)(?:\s+.*)?$/;

export const finite = value => typeof value === 'number' && Number.isFinite(value);
export const add = (...values) => values.every(finite) ? values.reduce((a, b) => a + b, 0) : null;
export const ratio = (a, b) => finite(a) && finite(b) && b > 0 ? a / b : null;
export const fraction = (a, b) => finite(a) && finite(b) && a >= 0 && a <= b ? ratio(a, b) : null;
const scaled = (value, multiplier) => finite(value) ? value * multiplier : null;
const labelKey = labels => JSON.stringify(Object.keys(labels).sort().map(key => [key, labels[key]]));

function parseLabels(text = '') {
  const labels = Object.create(null);
  const pattern = /\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:\\.|[^"\\])*)"\s*(?:,|$)/gy;
  let offset = 0;
  while (offset < text.length) {
    pattern.lastIndex = offset;
    const match = pattern.exec(text);
    if (!match || Object.hasOwn(labels, match[1])) return null;
    let valid = true;
    labels[match[1]] = match[2].replace(/\\(.)/g, (_, c) => {
      if (!['\\', '"', 'n'].includes(c)) valid = false;
      return c === 'n' ? '\n' : c;
    });
    if (!valid) return null;
    offset = pattern.lastIndex;
  }
  return labels;
}

export function parseMetrics(text) {
  if (text.length > 8 * 1024 * 1024) throw new Error('Metrics response is too large');
  const result = new Map();
  for (const line of text.split('\n')) {
    if (!line.startsWith(P)) continue;
    const match = line.match(samplePattern);
    if (!match || !wanted.has(match[1])) continue;
    const labels = parseLabels(match[2]);
    const value = Number(match[3]);
    if (labels === null || !finite(value) || value < 0) continue;
    if (!result.has(match[1])) result.set(match[1], []);
    result.get(match[1]).push({labels, value});
  }
  return result;
}

function rows(metrics, name, selector = {}) {
  return (metrics.get(P + name) || []).filter(row =>
    Object.entries(selector).every(([key, value]) => row.labels[key] === value));
}

function sum(metrics, name, selector) {
  const entries = rows(metrics, name, selector);
  return entries.length ? entries.reduce((total, row) => total + row.value, 0) : null;
}

export function makeSnapshot(text, t) {
  const metrics = parseMetrics(text), v = {};
  for (const [key, name] of Object.entries(fields)) v[key] = sum(metrics, name);
  for (const [key, name] of Object.entries(gauges)) {
    const entries = rows(metrics, name);
    // Gauges are per engine. A sum of KV fractions can exceed 100%.
    v[key] = key === 'kv' && entries.length ? Math.max(...entries.map(row => row.value)) : sum(metrics, name);
  }
  for (const [key, metric] of Object.entries(stateFields)) {
    const entries = rows(metrics, 'store_state', {metric});
    v[key] = key === 'storeTimestamp' && entries.length ? Math.min(...entries.map(row => row.value)) : sum(metrics, 'store_state', {metric});
  }
  for (const [key, event] of Object.entries(eventFields)) v[key] = sum(metrics, 'store_events_total', {event});
  v.computed = sum(metrics, 'prompt_tokens_by_source_total', {source: 'local_compute'});
  if (v.cached === null) v.cached = add(
    sum(metrics, 'prompt_tokens_by_source_total', {source: 'local_cache_hit'}),
    sum(metrics, 'prompt_tokens_by_source_total', {source: 'external_kv_transfer'}));
  if (v.computed === null && finite(v.input) && finite(v.cached) && v.input >= v.cached) v.computed = v.input - v.cached;
  v.ramUsed = add(v.ramMain, v.ramState, v.ramDraft);

  const series = {};
  for (const name of ['prompt_tokens_total', 'generation_tokens_total', 'request_success_total']) {
    for (const row of rows(metrics, name)) series[name + labelKey(row.labels)] = row.value;
  }
  const models = [...new Set(rows(metrics, 'prompt_tokens_total').map(row => row.labels.model_name).filter(Boolean))].sort();
  const buckets = {};
  for (const row of rows(metrics, 'time_to_first_token_seconds_bucket')) {
    const bound = row.labels.le;
    if (bound === '+Inf' || finite(Number(bound))) buckets[bound] = (buckets[bound] || 0) + row.value;
  }
  const prefill = {};
  for (const key of ['prefillTime', 'prefillCount', 'prefillTokens', 'prefillTokenCount']) {
    for (const row of rows(metrics, fields[key])) {
      const id = labelKey(row.labels);
      (prefill[id] ||= {})[key] = row.value;
    }
  }
  return {t, v, series, models, buckets, prefill};
}

export function sameEpoch(a, b) {
  const keys = Object.keys(a.series).sort();
  if (JSON.stringify(keys) !== JSON.stringify(Object.keys(b.series).sort())) return false;
  if (keys.some(key => b.series[key] < a.series[key])) return false;
  return [...COUNTERS].every(key => !finite(a.v[key]) || !finite(b.v[key]) || b.v[key] >= a.v[key]);
}

export function delta(end, start, key) {
  if (!start || !finite(end.v[key]) || !finite(start.v[key])) return null;
  const value = end.v[key] - start.v[key];
  return value >= 0 ? value : null;
}

export function prefillObservation(end, start = null) {
  const keys = Object.keys(end.prefill);
  if (!keys.length || (start && JSON.stringify(keys.sort()) !== JSON.stringify(Object.keys(start.prefill).sort()))) return null;
  let tokens = 0, seconds = 0, count = 0;
  for (const key of keys) {
    const current = end.prefill[key], previous = start?.prefill[key];
    if (![current, ...(previous ? [previous] : [])].every(item =>
      Object.values(item).length === 4 && Object.values(item).every(finite) && item.prefillCount === item.prefillTokenCount)) return null;
    const values = {};
    for (const field of ['prefillTokens', 'prefillTime', 'prefillCount']) {
      values[field] = current[field] - (previous?.[field] || 0);
      if (values[field] < 0) return null;
    }
    tokens += values.prefillTokens; seconds += values.prefillTime; count += values.prefillCount;
  }
  const rate = ratio(tokens, seconds);
  return rate === null || count <= 0 ? null : {rate, tokens, seconds, count, t: end.t};
}

export function ttftQuantile(end, start, q) {
  if (!start) return null;
  const number = bound => bound === '+Inf' ? Infinity : Number(bound);
  const bounds = Object.keys(end.buckets).sort((a, b) => number(a) - number(b));
  if (!bounds.includes('+Inf')) return null;
  const counts = bounds.map(bound => finite(start.buckets[bound]) ? end.buckets[bound] - start.buckets[bound] : NaN);
  if (counts.some((count, i) => !finite(count) || count < 0 || (i && count < counts[i - 1]))) return null;
  const total = counts.at(-1);
  const observedCount = delta(end, start, 'ttftCount');
  if (finite(observedCount) && total !== observedCount) return null;
  if (total <= 0 || (q >= 0.95 && total < 10)) return null;
  const index = counts.findIndex(count => count >= total * q);
  const value = number(bounds[index]);
  return finite(value) ? value : null; // +Inf is unavailable, never printed as a number.
}

export function statistics(end, start = null) {
  const v = {};
  for (const key of COUNTERS) v[key] = start ? delta(end, start, key) : end.v[key];
  const seconds = start ? end.t - start.t : null;
  const ramInput = add(v.ramReuseInput, v.ramLoadInput);
  const classifiedInput = add(v.gpuInput, ramInput, v.ssdInput, v.computedStoreInput);
  return {
    ...v, seconds, ramInput, classifiedInput,
    total: add(v.input, v.output), hit: fraction(v.cached, v.input),
    outputRate: ratio(v.output, seconds), computedRate: ratio(v.computed, seconds),
    inputRate: ratio(v.input, seconds), rpm: ratio(scaled(v.requests, 60), seconds),
    tpm: ratio(scaled(add(v.input, v.output), 60), seconds),
    specAcceptance: fraction(v.accepted, v.drafted),
    acceptanceLength: finite(ratio(v.accepted, v.drafts)) ? 1 + ratio(v.accepted, v.drafts) : null,
    hotHit: finite(v.hotWanted) && finite(v.hotMissed) ? fraction(v.hotWanted - v.hotMissed, v.hotWanted) : null,
    ramShare: fraction(ramInput, classifiedInput), ssdShare: fraction(v.ssdInput, classifiedInput),
    ttftMean: ratio(v.ttftSum, v.ttftCount), itlMean: ratio(v.itlSum, v.itlCount),
    e2eMean: ratio(v.e2eSum, v.e2eCount), queueMean: ratio(v.queueSum, v.queueCount),
    tpotMean: ratio(v.tpotSum, v.tpotCount), prefillPhase: prefillObservation(end, start),
    ttftP50: ttftQuantile(end, start, 0.5), ttftP95: ttftQuantile(end, start, 0.95),
    ramChange: start && finite(end.v.ramUsed) && finite(start.v.ramUsed) ? end.v.ramUsed - start.v.ramUsed : null,
    ssdChange: start && finite(end.v.ssdUsed) && finite(start.v.ssdUsed) ? end.v.ssdUsed - start.v.ssdUsed : null,
    ramChangeRate: start && finite(end.v.ramUsed) && finite(start.v.ramUsed) ? ratio(end.v.ramUsed - start.v.ramUsed, seconds) : null,
    ssdChangeRate: start && finite(end.v.ssdUsed) && finite(start.v.ssdUsed) ? ratio(end.v.ssdUsed - start.v.ssdUsed, seconds) : null,
  };
}

export function windowStatistics(samples, windowSeconds) {
  const end = samples.at(-1);
  if (!end) return null;
  // Take a real observation inside the selected window, never extrapolate or
  // invent an exact cutoff point between scrapes. Partial windows work at once.
  const start = samples.find(sample => sample.t >= end.t - windowSeconds);
  if (!start || start === end) return {end, start: null, complete: false, ...statistics(end, end)};
  return {end, start, complete: end.t - start.t >= windowSeconds - POLL_SECONDS * 1.5, ...statistics(end, start)};
}

export function chartPoints(samples, key, windowSeconds) {
  const end = samples.at(-1);
  if (!end) return [];
  return samples.slice(1).flatMap((sample, i) => {
    const previous = samples[i];
    if (sample.t < end.t - windowSeconds) return [];
    const dt = sample.t - previous.t;
    return [{t: sample.t, value: dt > GAP_SECONDS ? null : ratio(delta(sample, previous, key), dt)}];
  });
}
