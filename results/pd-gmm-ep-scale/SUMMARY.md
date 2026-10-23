# pd-gmm-ep-scale

Status: passed
Acceptance passed: True

- startup_seconds: 596.845725774765
- compile_wall_span_seconds: 558.0
- token_buckets_per_dp: [4, 8, 16, 512]
- kv_blocks: [56544, 56556]
- tok_s_user_median: None
- aggregate_tok_s: None
- tpm_including_cached: None
- apc_hit_rate: None
- preemptions: None
- spec_decode_acceptance_rate: None

P/D cases reported separately; no mixing of contexts, concurrency or spec counters across windows.
Startup compiler-message wall span when readiness is known; task durations include all phases and overlap across hosts. Raw worker logs are preserved.

All eight host archives and checksums: see copies.json.

## P/D benchmark

Best-case projections; see pd.json for raw timestamps, capacity checks and contamination flags.

| Source | N | Context | Valid | Decode tok/s/user (mean) | Flags |
|---|---:|---:|---|---:|---|
| agent-replay | 192 | 5083 | True | 22.557050520181537 |  |
| agent-replay | 256 | 5083 | True | 22.493464533552647 |  |
| agent-replay | 320 | 5083 | True | 18.40071089044164 |  |
| agent-replay | 384 | 5083 | True | 18.335662584819318 |  |

### Speculative decode counters

- agent-replay N=192 context=None: {"acceptance_per_position": {}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": null, "tokens_per_step": null}
- agent-replay N=256 context=None: {"acceptance_per_position": {}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": null, "tokens_per_step": null}
- agent-replay N=320 context=None: {"acceptance_per_position": {}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": null, "tokens_per_step": null}
- agent-replay N=384 context=None: {"acceptance_per_position": {}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": null, "tokens_per_step": null}

### Prefill

- cold5k N=32: valid=True; new tok/s=17388.086667277028; including cached tok/s=17388.086667277028; TTFT={'mean': 9.185383077062852, 'p50': 9.184939292994386, 'p90': 9.192686691996641}; new tok/s/chip=543.3777083524071; incl-cached tok/s/chip=543.3777083524071; repeat=0; flags=[]

### Agent-workload model

```json
{
  "assumptions": [
    "Best-case model, not a measured disaggregated deployment; valid uncontaminated N=32 cases only.",
    "Each turn emits500 tokens; logical TPM counts the full prompt including cached history plus500 output tokens, no extra double-counting of the1650 increment.",
    "Separate equal-size P and D slices: mean incremental-prefill TTFT at concurrency32 plus mean per-user500/rate; two32-chip slices. Per-slice TPM divides by2.",
    "Decode-only scenario: prefill and KV transfer are free; one32-chip slice.",
    "No network/KV-transfer, routing, tool latency or queueing beyond measured batched TTFT; fixed N=32 decode occupancy.",
    "Random contexts5k/10k/15k weighted equally for average; agent-replay reports actual lengths separately. No interpolation or substitution for missing/flagged cases.",
    "Spec num_drafts sums per-request steps over DP ranks: step_seconds=window_seconds*N/drafts; tokens_per_step=1+accepted/drafts; per-position acceptance=accepted_pos/drafts. Boundary scrape lag is recorded; these are counter-normalized estimates."
  ],
  "incremental_prefill_ttft_seconds": null,
  "contexts": [],
  "random_context_average": null
}
```
