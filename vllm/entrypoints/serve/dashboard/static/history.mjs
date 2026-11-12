// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the ReVolta project
import {finite, GAP_SECONDS, RETENTION_SECONDS, sameEpoch, prefillObservation} from './metrics.mjs';

export const CACHE_SCHEMA = 1;
export const CACHE_BYTES = 1536 * 1024;
export const MAX_SAMPLES = 500;
const size = text => new TextEncoder().encode(text).byteLength;

function validSnapshot(sample) {
  return sample && finite(sample.t) && sample.t > 0 &&
    sample.v && Object.values(sample.v).every(value => value === null || finite(value)) &&
    sample.series && Object.values(sample.series).every(finite) &&
    sample.buckets && Object.values(sample.buckets).every(finite) &&
    sample.prefill && Object.values(sample.prefill).every(item => Object.values(item).every(finite)) &&
    Array.isArray(sample.models) && sample.models.every(model => typeof model === 'string');
}

export class ObservationHistory {
  // Remembering is opt-in: a history object stores nothing until a caller
  // enables it, so a page or a caller that forgets to decide stays in memory.
  constructor({storage = null, scope = '', remember = false} = {}) {
    this.storage = storage;
    this.key = `revolta-dashboard:${CACHE_SCHEMA}:${scope}`;
    this.remember = remember;
    this.instance = null;
    this.samples = [];
    this.latestPrefill = null;
    this.lastSaved = 0;
    this.cacheState = remember ? 'Memory only' : 'Disabled';
    if (!this.remember) this.clear();
    // A record left by an earlier session that remembered by default must not
    // keep model names and label keys on disk while remembering is off.
  }

  connect(instance, now) {
    if (this.instance === instance) return;
    this.instance = instance;
    this.samples = [];
    this.latestPrefill = null;
    this.lastSaved = 0;
    if (!this.remember || !this.storage) return;
    try {
      const text = this.storage.getItem(this.key);
      if (!text || text.length > CACHE_BYTES || size(text) > CACHE_BYTES) return;
      const cache = JSON.parse(text);
      if (cache.schema !== CACHE_SCHEMA || cache.instance !== instance || !Array.isArray(cache.samples) ||
          cache.samples.length > MAX_SAMPLES || !cache.samples.every(validSnapshot)) return;
      const samples = cache.samples.filter(sample => sample.t >= now - RETENTION_SECONDS && sample.t <= now);
      if (samples.some((sample, i) => i && (sample.t <= samples[i - 1].t || !sameEpoch(samples[i - 1], sample)))) return;
      this.samples = samples;
      // Rebuild the last observed prefill from verified pairs, not an unchecked
      // cached derived rate. Never carry a latest event over an observation gap.
      for (let i = 1; i < samples.length; i++) this.updatePrefill(samples[i - 1], samples[i]);
      this.cacheState = samples.length ? 'Restored local history' : 'Local history enabled';
    } catch {
      this.cacheState = 'Memory only';
    }
  }

  updatePrefill(previous, sample) {
    if (sample.t - previous.t > GAP_SECONDS) this.latestPrefill = null;
    else this.latestPrefill = prefillObservation(sample, previous) || this.latestPrefill;
  }

  append(sample) {
    if (!validSnapshot(sample)) throw new Error('Invalid metric snapshot');
    const previous = this.samples.at(-1);
    let reset = false;
    if (previous) {
      if (sample.t <= previous.t || !sameEpoch(previous, sample)) {
        this.samples = [];
        this.latestPrefill = null;
        reset = true;
      } else this.updatePrefill(previous, sample);
    }
    this.samples.push(sample);
    this.samples = this.samples.filter(item => item.t >= sample.t - RETENTION_SECONDS).slice(-MAX_SAMPLES);
    this.save(sample.t, reset);
    return reset;
  }

  save(now, force = false) {
    if (!this.remember || !this.storage || !this.instance || (!force && now - this.lastSaved < 30)) return;
    try {
      let samples = this.samples.slice();
      let text;
      do {
        text = JSON.stringify({schema: CACHE_SCHEMA, instance: this.instance, samples});
        if (size(text) <= CACHE_BYTES) break;
        samples = samples.slice(Math.max(1, Math.floor(samples.length / 5)));
      } while (samples.length);
      if (size(text) > CACHE_BYTES) throw new Error('Cache capacity exceeded');
      this.storage.setItem(this.key, text);
      this.cacheState = 'Local history enabled';
    } catch {
      this.cacheState = 'Memory only';
    }
    this.lastSaved = now;
  }

  clear() {
    try { this.storage?.removeItem(this.key); } catch { /* Storage may be blocked. */ }
    this.samples = [];
    this.latestPrefill = null;
    this.lastSaved = 0;
  }

  setRemember(enabled) {
    this.remember = enabled;
    if (!enabled) {
      try { this.storage?.removeItem(this.key); } catch { /* In-memory use still works. */ }
      this.cacheState = 'Disabled';
    } else this.save(Date.now() / 1000, true);
  }
}
