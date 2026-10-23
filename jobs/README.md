# TPU job protocol (reservation 2026-10-20 23:00Z → 2026-10-21 23:00Z)

Workers never touch the TPU slice. They prepare a job directory here and submit it to the
orchestrator, who runs it on the 32-chip slice and writes the results back into the same directory.

## Hardware

- 8 hosts `ct6e-standard-4t` (4 × v6e chips each), one 4x8 slice = 32 chips, zone us-east5-b.
- A full-model serving job (`kind: e2e`) needs all 8 hosts. Only one runs at a time.
- `kind: kernel` and `kind: compile` jobs run on one host (4 chips) each, up to 8 in parallel,
  and are batched into the gaps between e2e jobs (model restarts). Make them self-contained.
- The orchestrator decides the order. Small kernel jobs get turned around fastest.

## Job directory: `jobs/<job-id>/`

`<job-id>` = `<worker-name>-<short-slug>-<n>`, e.g. `dsv4-buckets-fine-1`.

```
jobs/<job-id>/
  job.json            # spec, below
  patches/
    tpu-inference/*.patch   # applied in name order AFTER patches/tpu-inference/0001-0006
    vllm/*.patch            # applied to vllm (optional)
    tokamax/*.patch         # applied after patches/tokamax (optional)
  run.sh              # kernel/compile jobs only: runs on one host, see below
  NOTES.md            # what it tests, what you expect, how to read the result
  result/             # written by the orchestrator; do not create
```

Patches must apply cleanly with `git apply` on top of the pinned submodule + the base series.
Check this on CPU before submitting (`../apply_patches.sh` in a scratch clone, then `git apply --check`).

### job.json

```json
{
  "id": "dsv4-buckets-fine-1",
  "owner": "dsv4-buckets",
  "kind": "e2e",                       // e2e | kernel | compile
  "base": "dw-tpu@<commit>",            // dw-tpu commit whose patches/ + env.sh are the base
  "depends_on": [],                     // job ids whose patches must be stacked first (in order)
  "env": {"TPU_TOKEN_PADDINGS_PER_DP": "1,2,3,...,32,64,128,256,512", "DSV4_DENSE_FP8_GEMM": "0"},
  "serve_args_extra": [],               // appended to the vllm serve command
  "acceptance": "reduction-order",      // exact | reduction-order | sampling-equivalent (spec decode)
  "bench": {"agents": [32], "repeats": 1},
  "profile": false,                     // true = 3 s xprof capture after the bench
  "est_minutes": 60,
  "abort_if": ["post-ready RESOURCE_EXHAUSTED", "launch-id mismatch", "startup > 45 min"],
  "success": "tok/s/user > 23.3 at 32 agents with acceptance passing",
  "fallback": "what to run instead if this fails, or null"
}
```

### run.sh (kernel / compile jobs)

Runs on one TPU host as `bash run.sh` with `cwd = jobs/<job-id>`, the dw-tpu venv active
(`/opt/dsv4/.venv`, jax 0.11.2, libtpu 0.0.48, tokamax pinned), `$OUT` set to an empty
directory that is copied back to `result/`, and `$DW_TPU` pointing at the deployed dw-tpu tree with the
base series applied. Hard time limit: `est_minutes` + 50 %. Write a machine-readable `$OUT/result.json`.

## Submitting

```bash
P=~/.claude/skills/agent-progress/scripts/progress.py
export AGENT_PROGRESS_FILE=/home/titan-6/doubleword/dw-tpu/AGENT_PROGRESS.md
python3 $P request <worker-name> 'JOB jobs/<job-id> kind=<kind> est=<min>m — <one-line purpose>'
python3 $P wait-response R<n> --interval 60 --timeout 7200    # run in background
```

The RESPONSE says QUEUED (with position), RUNNING, or DONE/FAILED with the headline numbers and
`result/` path. A QUEUED response is not the final answer: keep waiting on the same request id
(re-arm `wait-response` after it fires; the orchestrator updates the response in place) until it
says DONE or FAILED, then `close` it.

## Results the orchestrator returns (e2e)

`result/summary.json` + `result/SUMMARY.md`: startup time, compile time and bucket list, HBM/KV blocks,
acceptance verdict (tokens, logprob error vs GPU reference, suffix causality), 32-agent output
tok/s/user (median), aggregate output tok/s, TPM incl. cached tokens, prefix hit rate, preemptions,
and the raw logs (`serve.log`, bench CSVs, probes).

## Reference numbers (32 agents, 10 turns, ~5k prompt, 500 out + 500 env tokens/turn, APC on)

| Stack | tok/s/user | TPM incl. cached |
|---|---|---|
| 0001-0003 (bucket 4) | 16.43 | — |
| + 0004 | 18.52 | — |
| + 0005 async (single run) | 23.30 | 893k |
| + 0006 dense (projection) | ~27 | ~1.0M |
# Informational benchmarks after a numerical gate failure

An e2e job with `acceptance: "reduction-order"` may opt in with
`"benchmark_on_gate_failure": true` (default: false). Only GPU-reference
MAE/top1 threshold failures permit the configured benchmark and profile to
continue. Greedy token identity, repeat/cache identity, finite probe evidence,
and suffix causality must still pass; crashes, timeouts, and OOMs still abort.
Completed informational runs have status `gate_failed_benchmarked`,
`acceptance_passed: false`, and `informational_only: true` in `summary.json`,
with a prominent warning in `SUMMARY.md`. These numbers do not qualify the job.

## Isolated prefill/decode measurements

Set `"bench_mode": "pd"` (default `"agent"`) and supply `pd.prefill` and
`pd.decode`; see `pd-base0/job.json` and `pd-specdec/job.json` for full examples.
Acceptance gates run first, unchanged. `profile: false` avoids a separate profiler run.
The client writes `result/pd.json`; summaries contain a P/D section and a best-case
workload model. Run `python3 runner/pd_model.py jobs/pd-base0/result/pd.json`
to print the model and assumptions independently. Client dry-run:
`python3 runner/vendor/pd_bench.py jobs/pd-base0/job.json --out /tmp/pd.json --dry-run`.

Prefill cohorts are simultaneous, with an untimed prefix-warming pass for incremental
cases and a cache reset between cohorts. Reported new-token throughput counts the
requested new suffix; observed uncached throughput additionally includes APC block
alignment recomputation. TTFT includes admission, prefill, and the one output token.
Use `--enable-prompt-tokens-details` in serve arguments for cache-use verification.

Decode sources may be `["random", "agent-replay"]`. Random prompts use BatchBench's
uniform non-special base-vocabulary token policy, with a Python RNG and exact token-ID
completion requests (not Rust-identical bytes or chat re-tokenization). Agent replay
uses the32 captured chat payloads, cycles them for N>32, and measures actual server
`/tokenize` lengths. It is not relabeled as5k/10k/15k; repeated payloads can share APC.
No request body carries a seed: TPU rejects per-request sampling seeds.

The common window starts at the latest request's warmup token and ends at the first
finish event. Tokens in `(start,end]` divided by window duration are the primary rate;
ITL preserves zero gaps within speculative bursts. Default512 slack output tokens
allow overlap despite staggered warmup; each request must still contribute at least
`measure_tokens` inside the shared window, otherwise the case is flagged. Boundary
metrics scrapes retain timestamps/skew and draft/accepted/per-position counters.
Spec step time is `window_seconds*N/drafts`, tokens/step is `1+accepted/drafts`,
and each position's acceptance is `accepted_pos/drafts`.

Capacity checks use published hybrid-aware KV capacity and worker allocation block
size, with conservative busiest-rank occupancy for32 attention-DP ranks. Cases that
exceed capacity or max model length, or have unavailable capacity evidence, are
skipped and flagged before requests. Unknown/lagged prefill counters, observed
prefill/preemption in the enclosing metric window, or insufficient overlap exclude
a decode case from the model. Crashes, HTTP errors and incomplete streams fail the
job. Conservative flags can require follow-up; they never silently become clean data.

The model reports N=32 at5k/10k/15k and their equal-weight average, plus replay at its
actual mean context. Separate equal-size P/D pools cost two32-chip slices, versus
one decode slice with free prefill. It uses incremental mean TTFT + mean(500/rate),
counts logical prompt+output TPM, and reports per-slice normalization. KV transfer,
additional queueing and tool latency are not measured; estimates are best-case.

P/D prefill-only jobs may omit `pd.decode`. `pd.prefill.repeats` defaults to1;
each repeat resets the cache and generates fresh prompts. `verify_warm_cache:true`
adds an untimed prefix replay to verify block-aligned APC hits before timing the
extension. Both this check and the timed request must pass; failures remain flagged.
`cache_affinity:true` sends `vllm_xargs.pd_cache_rank=index%32`; it requires the
job-local scheduler patch and `PD_CACHE_AFFINITY=1` in `jobs/pd-final-prefill`.
Acceptance traffic has no hint and retains the default routing. Every prefill
row reports TTFT mean/p50/p90, requested-new and including-cached throughput,
per-chip throughput (divide32), alignment recomputation, and raw cache hits.

Combine separate measured pools with:
```
python3 runner/pd_model.py jobs/pd-final-decode/result/pd.json \
  --prefill jobs/pd-final-prefill/result/pd.json --out /tmp/pd-model-v2.json
```
Model v2 reports all valid decode N/context/source cases, each nearest-context
prefill batching option, P:D chip sizing, and throughput per32 total chips. It
pools repeats by total tokens/total time, not by selecting the fastest repeat.
Continuous decode occupancy needs a rotating agent population larger than N;
both that capacity projection and the fixed-N-agent result are explicit. Read
the printed assumptions: fractional linear chip scaling, free transfer, no
queueing, and the distinction between actual-context TPM and82%-APC TPM.

For decode scaling, `pd.decode.capacity_check` accepts `"strict"` (unchanged
conservative default) or `"advisory"`. DSV4 hybrid `kv_cache_size_tokens` is a
full-model-length concurrency equivalent, **not physical token capacity**.
Advisory mode records estimated-capacity warnings and runs the case despite
them, including when estimated capacity is unavailable. It always enforces
`context + warmup_tokens + measure_tokens + slack_tokens <= MAX_LEN` (slack
still defaults to512); exceeding model length skips with `model_length_exceeded`.
The live sampler records preemption, running and waiting counters at both
window boundaries and within the window. Any cohort preemption through window
end flags `preempted`; observed running<N or waiting>0 inside the window flags
`not_all_running`. Missing required telemetry invalidates the case. The end
scrape occurs after first finish, so its running gauge is retained for audit
but excluded from occupancy checks to avoid confusing natural completion with
starvation. Sampled metrics cannot establish occupancy between scrapes. OOMs,
crashes, acceptance policy and existing prefill-contamination checks are unchanged.
`jobs/pd-scale-base0-v2` is the non-spec MAX_LEN4096/random2500 advisory grid.
