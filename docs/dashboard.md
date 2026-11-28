# Built-in engine dashboard

The dashboard is an optional web page that shows live aggregate metrics of one ReVolta API server: requests, decode and prefill speed, cache use, speculative decoding and latency. It reads the same Prometheus metrics as `/metrics` and does all calculations in the browser. It is off by default.

Prerequisites: an OpenAI-compatible server started with `vllm serve` or the Docker image, with engine statistics on (the default).

## Turn it on

<a id="enable-and-open"></a>

Add `--enable-dashboard` to your `vllm serve` command, or `enable-dashboard: true` to the YAML file you pass with `--config`. Then open `http://<host>:<port>/dashboard/` (port 8000 by default); `/dashboard` redirects there.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `--enable-dashboard` / `--no-enable-dashboard` | boolean | `false` | Serves the page under `/dashboard/`. When off, none of the dashboard routes exist. |

The page becomes available once the model has loaded and the API server is listening. It cannot load while the API server is down; a page that is already open shows an offline state when a request fails or takes longer than 6 seconds.

What the data needs:

- **Engine statistics.** They are on by default. With `--disable-log-stats` the token metrics are missing and the page says so.
- **Store metrics for the Cache card.** The RAM, SSD and GPU Hot views use the store metrics (`--enable-store-metrics`, on by default, needs engine statistics), which the engine publishes at most once a minute. Their occupancy values come from Host KV's store gauges; with GPU KV and CPU or disk offload only some event counters are published, so most of those values show `—`.

The page needs no JavaScript build step, chart library, CDN or external font. Its HTML, CSS and JavaScript files are part of the wheel and the Docker image. The page loads nothing from other origins.

Routes follow the server's `--root-path`, and the page uses relative URLs, so it works behind a reverse proxy that serves the API under a path prefix.

## Access control

The page shell (`/dashboard/` and its assets) contains no metrics and needs no key. The two data routes follow the server's API key:

| Route | Content |
| --- | --- |
| `/dashboard/config` | Service instance ID, start time and engine version |
| `/dashboard/metrics` | The same Prometheus registry as `/metrics` |

When the server has an API key (`--api-key`, or `VLLM_API_KEY` if `--api-key` is not given), the data routes need `Authorization: Bearer <key>`. The page asks for the key, keeps it in the tab's memory only and sends it in that header; it never puts the key in a URL or in browser storage. A missing or wrong key gets `401` with `WWW-Authenticate: Bearer`. If the key is rejected later, the page drops the observations it holds in memory.

`/metrics` itself keeps its existing access policy; the dashboard does not change it. Data responses carry `Cache-Control: no-store`. `/dashboard/metrics` answers `503` when the engine has failed. Assets use `Cache-Control: no-cache` with conditional requests.

## What the page shows

The page has six cards: Requests, Decode, Prefill, Cache, Speculative decoding and Latency. **View details** opens a dialog with more values, the window values next to the totals since engine start, and an explanation of each calculation. On a narrow screen the dialog becomes a scrollable bottom sheet.

### Time scopes

<a id="metric-scopes"></a>

Each value belongs to one of three scopes:

- **Since engine start.** Cumulative engine counters: total tokens, cached input tokens, output tokens, completed requests, cache hit rate, and average RPM and TPM. Service uptime counts from API server setup, including model loading. The version shown is the installed engine version.
- **Now.** Current running and waiting requests, and rates from the two most recent observations, normally 2 seconds apart. After a gap of more than 10 seconds there is no "now" rate.
- **Last 1, 5 or 15 minutes.** Counter changes divided by the actual elapsed time. The window starts at the oldest real observation inside it; nothing is interpolated. A partial window works from the second observation and shows how much time it covers. Ratios divide summed counters; they are not averages of per-sample percentages.

The page polls every 2 seconds while its tab is visible and stops while the tab is hidden. Hidden time leaves a gap in the charts. Window averages still cover a gap when the counters stayed valid, because the change and the elapsed time are known; the activity pattern, gauge history and individual prefill events inside the gap are lost. A first visit has no window history, but the totals since engine start are available at once.

### Definitions

| Display | Definition |
| --- | --- |
| RPM | Completed requests × 60 / elapsed seconds. Every finish reason counts: stop, length, abort, error and repetition. |
| TPM | (Input tokens + output tokens) × 60 / elapsed seconds |
| Decode throughput | Output token change / elapsed seconds |
| Computed input throughput | Newly computed input token change / elapsed seconds |
| Total input throughput | All input token change / elapsed seconds, including cached input |
| Total cache hit rate | Cached input tokens / input tokens |
| Latest reported prefill | Computed KV tokens of newly completed requests / their prefill-phase seconds |
| Draft acceptance | Accepted draft tokens / drafted tokens |
| Mean acceptance length | 1 + accepted draft tokens / draft rounds; includes the bonus token |
| Latency means | Histogram sum change / count change |
| TTFT P50 / P95 | Upper bound of the histogram bucket that holds the percentile, not an interpolated value. P95 needs at least 10 samples; a value beyond the largest finite bucket shows `—`. |

Missing metrics show `—`, not zero.

**Prefill.** The prefill histograms are reported when a request completes. The latest value stays on screen through idle and decode periods, with its age and sample count. Several completions in one poll are weighted by their prefill time. A gap of more than 10 seconds clears the latest value until a new completion is seen. The large prefill number is always written in full, never as `K`. The window's computed input throughput is a different measurement: it divides by wall time.

**Cache.** The Cache details have Summary, GPU Hot, RAM and SSD views.

- RAM and SSD input shares divide by all classified prompt tokens. They are not hit rates of a lookup in that tier.
- GPU Hot hit rate counts cache groups, not tokens.
- Occupancy bars show the latest store snapshot and its net change in the window. Peaks the browser did not observe cannot be recovered.
- A store snapshot older than 180 seconds, or one whose age cannot be checked, is not current. The bars and the `RAM allocation / capacity` and `SSD allocation / capacity` lines then show `Unavailable` with the age of the last observed value, and so do they while the service is offline.
- This release does not export SSD capacity, so the SSD view shows bytes used but no utilization.
- Engine KV usage shows the maximum across engines when there is more than one.

### Metrics it reads

All names carry the `vllm:` prefix in `/metrics`.

| Card | Metrics |
| --- | --- |
| Requests | `request_success_total`, `num_requests_running`, `num_requests_waiting`, `prompt_tokens_total`, `generation_tokens_total`, `num_preemptions_total` |
| Decode | `generation_tokens_total`, `inter_token_latency_seconds`, `request_time_per_output_token_seconds` |
| Prefill | `prompt_tokens_total`, `prompt_tokens_cached_total`, `prompt_tokens_by_source_total`, `request_prefill_time_seconds`, `request_prefill_kv_computed_tokens` |
| Cache | `prompt_tokens_cached_total`, `kv_cache_usage_perc`, `store_state`, `store_events_total` |
| Speculative decoding | `spec_decode_num_accepted_tokens_total`, `spec_decode_num_draft_tokens_total`, `spec_decode_num_drafts_total` |
| Latency | `time_to_first_token_seconds`, `e2e_request_latency_seconds`, `request_queue_time_seconds` |

## Browser history

<a id="browser-history-and-caching"></a>

**Remember history** is off by default. Until you turn it on, the page writes no observations to browser storage and keeps them in memory only.

When it is on, one local storage record holds the observations: at most 15 minutes of samples, 500 samples and 1.5 MiB of serialized data. The page writes it at most once every 30 seconds, and also when the tab is hidden or closed. The record contains:

- the schema version and the service instance ID;
- the browser-side timestamp of each observation;
- aggregate counter, gauge and store values (tokens, completed requests, cache and speculative decoding counters, running and waiting counts, engine KV usage, RAM and SSD byte totals and the store snapshot time);
- the model names from the metric labels;
- for input tokens, output tokens and completed requests, each series' value with its full label set, which includes the model name and engine index;
- TTFT histogram bucket counts and the prefill histogram values with their labels.

Label values are stored as they are, and a model name can contain a deployment path, so the record reveals which model the service served. API keys are never stored. The selected window and the remember setting are stored separately as page preferences; they contain no metric values.

The record does not expire. To remove it, use **Clear history** (deletes the observations and starts a new window), turn **Remember history** off (deletes the record at once; observation in memory continues), or clear the site's data in the browser. While remembering is off, opening the page also deletes a record that an earlier session left behind. If storage is blocked, full or corrupted, the page falls back to memory.

The record is tied to the browser origin, the dashboard path, the schema and the service instance. After a reload with remembering on, the page restores the history once the data routes accept it. A restart of the service, a change in the model or engine series, a counter that decreases, or the browser clock going backwards starts a new history. API workers of one service share one instance ID.

## Cost

The server keeps no history: there is no background task, timer, database or engine change. Each open, visible tab requests `/dashboard/metrics` every 2 seconds, which costs API server CPU time to collect and serialize the metrics. Close the page, or leave the option off, during benchmarks.

The dashboard was tested with CPU-only suites (Python, JavaScript and a headless browser); its cost while a model serves requests on GPUs has not been measured. The CPU test commands are in [CONTRIBUTING.md](../CONTRIBUTING.md#dashboard-checks).<a id="cpu-verification"></a><a id="testing-without-a-gpu"></a>

## See also

- [Configuration](configuration.md) for `--enable-dashboard` and the other server options.
- [KV offload modes](kv-offload.md) for the CPU and SSD tiers shown in the Cache card.
- [Performance and measurements](performance.md) for benchmark results.
