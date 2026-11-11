# Built-in engine dashboard

The optional dashboard shows aggregate metrics from one ReVolta service. It is
separate from any externally hosted status site. It does not contain deployment
announcements, a maintenance schedule, hardware claims or an operator changelog.

## Enable and open

Add `--enable-dashboard` to your existing `revolta serve` command, then open
`http://localhost:8000/dashboard/` using your service's host and port. The option
is a boolean, defaults to `false`, and also accepts `enable-dashboard: true` in
the usual server YAML configuration. Omit it, or use `--no-enable-dashboard`, to
disable the page and all its data and asset routes. This is an experimental
Python API server feature prepared for a future release; it does not change the
supported hardware/model matrix or release acceptance requirements.

No JavaScript build step, third-party chart library, CDN or external font is
needed. HTML, CSS and ES modules are included in both the source distribution and
wheel. Assets use conditional HTTP caching; metric and configuration responses
use `Cache-Control: no-store`.

The browser fetches `/dashboard/metrics`, a read-only view of the **same registry
as `/metrics`**, every two seconds while visible. The separate URL lets dashboard
data follow `--api-key` / `VLLM_API_KEY` access control without changing the
existing Prometheus scrape policy. It does not introduce a new metering path.
When a key is configured, enter it in the page. The key remains in that tab's
memory and is sent in an Authorization header; it is never placed in a URL or
saved to browser storage. The unauthenticated HTML shell contains no metrics.
Configuration follows the ordinary server's CLI-key-over-environment priority.

Routes use the server's reverse-proxy root path, and assets/data use relative
URLs. The page becomes available with the API server after model loading. An
already open page shows an offline state on failed or timed-out scrapes; it
cannot serve itself while the API server is down.

## Metric scopes

The six cards are Requests, Decode, Prefill, Cache, Speculative decoding and
Latency. Their details open in a dialog; on a narrow screen it becomes a scrollable
bottom sheet. Every detail view includes the engine's cumulative counters.

- **Since engine start** shows cumulative tokens, completed requests, cache hit
  rate and average RPM/TPM. Service uptime starts at API service setup, including
  model loading. The service instance ID and start time are initialized before
  API workers fork. Version is the installed engine's version, not hand-filled
  site content.
- **Now** uses current running/waiting gauges and the most recent pair of token
  counter observations, normally two seconds apart. A long observation gap does
  not produce a realtime rate.
- **Last 1/5/15 min** uses raw counter differences over actual elapsed time.
  Partial windows work after the second sample. Once enough history exists, the
  baseline rolls forward within the selected duration. It uses a real sample
  inside the cutoff, not an interpolated counter; the exact span is shown in
  details and partial-window labels. Ratios use summed numerators/denominators,
  not averages of per-sample percentages.

| Display | Definition |
| --- | --- |
| RPM | Completed requests × 60 / elapsed seconds; native stop, length and abort finish reasons are included |
| TPM | (Input tokens + output tokens) × 60 / elapsed seconds |
| Decode throughput | Output token change / elapsed seconds |
| Computed input throughput | Newly computed input token change / elapsed seconds |
| Total input throughput | All input token change / elapsed seconds, including cached input |
| Total cache hit rate | Cached input tokens / input tokens; equivalently cached input / (total tokens − output tokens) |
| Latest reported prefill | Newly completed requests' computed KV tokens / their prefill-phase seconds; counts and engine labels must match |
| Draft acceptance | Accepted draft tokens / drafted tokens |
| Mean acceptance length | 1 + accepted draft tokens / draft rounds, including the bonus token |
| Latency mean | Histogram sum change / count change |
| TTFT P50/P95 | Histogram bucket upper bounds; P95 needs at least ten samples; an infinite bound is unavailable |

Prefill's completed-request histograms arrive at request completion. The latest
rate is retained through idle/decode periods with its observation age and sample
count. Multiple completions in one scrape are weighted by their total phase
time. Cache-only samples can report a real zero. The large Prefill number uses
full numeric notation, never `K`. The window's computed-input throughput is a
different measurement with wall time as its denominator.

Cache details provide Summary, GPU Hot, RAM and SSD views. Occupancy uses
horizontal bars and net changes; it does not infer peak occupancy before the
browser observed it. RAM/SSD input shares use all classified prompt tokens as
their denominator. They are **not** conditional tier lookup hit rates. GPU Hot
hit rate uses groups, not tokens. Current store occupancy becomes unavailable
when its snapshot is older than 180 seconds or its age cannot be verified.
SSD bytes can be shown without a capacity, but SSD utilization remains
unavailable when this engine build does not export capacity. Native engine KV
utilization is available in Cache details; with multiple engines it shows the
maximum exported fraction. Missing metrics are shown as `—`, not zero.

## Browser history and caching

**Remember history** is enabled initially and can be turned off. It stores only
aggregate observations in local browser storage: at most 15 minutes, 500 samples
and 1.5 MiB of serialized UTF-8 data. Writes are throttled to once per 30 seconds
and flushed when the page is hidden or exited. **Clear history** deletes those
observations and starts a new window; disabling remembering removes the stored
history while allowing in-memory observation to continue. Storage denial, quota
errors and corrupted caches fall back to memory without breaking the page.

The cache is scoped to browser origin, dashboard path, schema and service
instance. Reloading the same service can restore observed history after successful
authentication. An engine restart, model/engine-series change or a decreasing
counter invalidates it. Client clock reversal also starts a new observation
segment. API workers in one service share an identity; independent service
restarts do not.

Hidden tabs stop polling. A gap has no instantaneous chart points. Counter-based
window averages can still cover the gap if the same service's counters remain
valid: their exact change and elapsed time are known. The browser cannot recover
the activity pattern, gauge history or individual prefill events during that
gap. A resumed observation gap longer than ten seconds clears the latest prefill
event until a fresh matching completion is observed. A first visit has no earlier
window history; cumulative engine counters are immediately available.

There is no server-side history task, timer, database or EngineCore change. Each
open visible tab does request metrics, so collection/serialization still consumes
API server CPU and multiprocess metric reads. This is not a zero-overhead or
GPU-performance claim. Keep the option off in unrelated acceptance runs.

## CPU verification

These checks need no model, GPU, Torch import or running inference service:

```sh
uv venv .venv
uv pip install --python .venv/bin/python 'fastapi>=0.141.1,<0.142.0' httpx prometheus-client prometheus-fastapi-instrumentator regex uvicorn
.venv/bin/python -m unittest discover -s tests/entrypoints/serve/dashboard -p 'test_*.py' -v
node --test tests/entrypoints/serve/dashboard/test_metrics.mjs
```

The Python suite exercises actual ASGI routes, authentication, disabled routes,
shared worker identity, proxy paths, assets and exporter responses. The JavaScript
suite exercises parsing, counter math, rolling/partial windows, prefill matching,
histogram bounds, reset detection and browser-cache behavior. Use Node 22 or
later. Browser verification uses a CPU fixture service, including mobile dialogs,
reload, offline/missing metrics and authentication; these results do not establish
GPU serving performance or production readiness.

To repeat the browser check with an installed Chromium or Chrome executable:

```sh
node tests/entrypoints/serve/dashboard/browser_check.mjs --chrome /usr/bin/chromium --output dashboard-browser-results
```

It starts and stops its own local CPU fixture service and browser, and writes
screenshots and an acceptance report to the selected output directory. The
fixture has no model or GPU dependency and is not part of the installed package.
