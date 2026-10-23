# pd-gmm-ep

Status: passed
Acceptance passed: True

- startup_seconds: 1650.0675137043
- compile_wall_span_seconds: 1600.0
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
| agent-replay | 32 | 5083 | True | 30.940565198456454 |  |
| agent-replay | 128 | 5083 | True | 29.92482698517899 |  |
| agent-replay | 256 | 5083 | True | 22.479924615681348 |  |
| random | 32 | 5000 | True | 30.839911457098165 |  |
| random | 128 | 5000 | True | 29.747635890190335 |  |
| random | 256 | 5000 | True | 22.29442683203222 |  |

### Speculative decode counters

- agent-replay N=32 context=None: {"acceptance_per_position": {}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": null, "tokens_per_step": null}
- agent-replay N=128 context=None: {"acceptance_per_position": {}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": null, "tokens_per_step": null}
- agent-replay N=256 context=None: {"acceptance_per_position": {}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": null, "tokens_per_step": null}
- random N=32 context=5000: {"acceptance_per_position": {}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": null, "tokens_per_step": null}
- random N=128 context=5000: {"acceptance_per_position": {}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": null, "tokens_per_step": null}
- random N=256 context=5000: {"acceptance_per_position": {}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": null, "tokens_per_step": null}

### Prefill

- cold5k N=32: valid=True; new tok/s=17393.197468832277; including cached tok/s=17393.197468832277; TTFT={'mean': 9.181926312470296, 'p50': 9.180490662998636, 'p90': 9.191981877993385}; new tok/s/chip=543.5374209010087; incl-cached tok/s/chip=543.5374209010087; repeat=0; flags=[]
- incr1650_on8k N=32: valid=True; new tok/s=10030.676834149346; including cached tok/s=58664.261484570416; TTFT={'mean': 5.236572882687142, 'p50': 5.235993638001673, 'p90': 5.249378575992887}; new tok/s/chip=313.45865106716707; incl-cached tok/s/chip=1833.2581713928255; repeat=0; flags=[]

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
  "incremental_prefill_ttft_seconds": 5.236572882687142,
  "contexts": [
    {
      "context_source": "agent-replay",
      "context_tokens": 5083,
      "per_user_decode_tok_s": 30.940565198456454,
      "decode_seconds_for500": 16.160016277603646,
      "spec_statistics": {
        "step_seconds": null,
        "tokens_per_step": null,
        "acceptance_per_position": {},
        "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts."
      },
      "separate_equal_slices": {
        "slices": 2,
        "turn_seconds": 21.396589160290787,
        "tok_s_user": 23.36821052431727,
        "logical_tpm_including_cached": 500984.5223318912,
        "logical_tpm_per_slice": 250492.2611659456
      },
      "decode_only_prefill_free": {
        "slices": 1,
        "turn_seconds": 16.160016277603646,
        "tok_s_user": 30.940562893674546,
        "logical_tpm_including_cached": 663326.0645198784,
        "logical_tpm_per_slice": 663326.0645198784
      }
    },
    {
      "context_source": "random",
      "context_tokens": 5000,
      "per_user_decode_tok_s": 30.839911457098165,
      "decode_seconds_for500": 16.212758512885785,
      "spec_statistics": {
        "step_seconds": null,
        "tokens_per_step": null,
        "acceptance_per_position": {},
        "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts."
      },
      "separate_equal_slices": {
        "slices": 2,
        "turn_seconds": 21.449331395572926,
        "tok_s_user": 23.310749914712886,
        "logical_tpm_including_cached": 492323.03819873615,
        "logical_tpm_per_slice": 246161.51909936807
      },
      "decode_only_prefill_free": {
        "slices": 1,
        "turn_seconds": 16.212758512885785,
        "tok_s_user": 30.839909174160805,
        "logical_tpm_including_cached": 651338.8817582763,
        "logical_tpm_per_slice": 651338.8817582763
      }
    }
  ],
  "random_context_average": null
}
```
