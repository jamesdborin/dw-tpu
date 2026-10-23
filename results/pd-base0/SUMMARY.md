# pd-base0

Status: passed
Acceptance passed: True

- startup_seconds: 1674.6521742343903
- compile_wall_span_seconds: 1635.0
- token_buckets_per_dp: [4, 8, 16, 512]
- kv_blocks: [33376, 33382]
- tok_s_user_median: None
- aggregate_tok_s: None
- tpm_including_cached: None
- apc_hit_rate: None
- preemptions: None
- spec_decode_acceptance_rate: None

32-agent measurements; median across repeats. TPM includes cached history, not compute throughput.
Startup compiler-message wall span when readiness is known; task durations include all phases and overlap across hosts. Raw worker logs are preserved.

All eight host archives and checksums: see copies.json.

## P/D benchmark

Best-case projections; see pd.json for raw timestamps, capacity checks and contamination flags.

| Source | N | Context | Valid | Decode tok/s/user (mean) | Flags |
|---|---:|---:|---|---:|---|
| random | 1 | 5000 | True | 28.497036825255552 |  |
| random | 8 | 5000 | True | 28.221279968078235 |  |
| random | 16 | 5000 | True | 28.13621356725246 |  |
| random | 32 | 5000 | True | 31.287585904408463 |  |
| random | 64 | 5000 | True | 28.52476284279149 |  |
| random | 128 | 5000 | True | 25.719242700556634 |  |
| random | 1 | 10000 | True | 28.26477912594056 |  |
| random | 8 | 10000 | True | 28.681656109435714 |  |
| random | 16 | 10000 | True | 28.341495792774435 |  |
| random | 32 | 10000 | True | 32.451079492682126 |  |
| random | 64 | 10000 | True | 29.07054030738082 |  |
| random | 128 | 10000 | True | 26.49003607197616 |  |
| random | 1 | 15000 | True | 28.292648187684208 |  |
| random | 8 | 15000 | True | 28.411681326189697 |  |
| random | 16 | 15000 | True | 28.483274966271473 |  |
| random | 32 | 15000 | True | 32.76076830129321 |  |
| random | 64 | 15000 | True | 29.474097769177213 |  |
| random | 128 | 15000 | True | 26.96036083118358 |  |
| agent-replay | 1 | 5083 | True | 28.3624784322629 |  |
| agent-replay | 8 | 5083 | True | 36.606843189826314 |  |
| agent-replay | 16 | 5083 | True | 34.3954622655971 |  |
| agent-replay | 32 | 5083 | True | 31.754118606064047 |  |
| agent-replay | 64 | 5083 | True | 28.912312305381892 |  |
| agent-replay | 128 | 5083 | True | 26.41443125707432 |  |

### Prefill

- cold5k N=32: valid=True; new tok/s=18583.118829542014; including cached tok/s=18583.118829542014; TTFT={'mean': 8.59296350443833, 'p50': 8.591523590002907, 'p90': 8.601553374003561}; flags=[]
- cold5k N=64: valid=True; new tok/s=18784.330136832734; including cached tok/s=18784.330136832734; TTFT={'mean': 12.99955336787582, 'p50': 12.998349780998979, 'p90': 17.012444243000935}; flags=[]
- incr1650_on8k N=32: valid=True; new tok/s=10633.072826366602; including cached tok/s=62187.36531784103; TTFT={'mean': 4.937368010437012, 'p50': 4.938036500501767, 'p90': 4.94947697099924}; flags=[]
- incr1650_on8k N=64: valid=False; new tok/s=3300.091884355531; including cached tok/s=19300.537384261137; TTFT={'mean': 14.183331454530958, 'p50': 16.73860425499879, 'p90': 20.78964723969766}; flags=['prefix_cache_reuse_unverified_or_insufficient']

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
  "incremental_prefill_ttft_seconds": 4.937368010437012,
  "contexts": [
    {
      "context_source": "random",
      "context_tokens": 5000,
      "per_user_decode_tok_s": 31.287585904408463,
      "decode_seconds_for500": 15.980780317802067,
      "spec_statistics": {
        "step_seconds": null,
        "tokens_per_step": null,
        "acceptance_per_position": {},
        "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts."
      },
      "separate_equal_slices": {
        "slices": 2,
        "turn_seconds": 20.918148328239077,
        "tok_s_user": 23.902689289424824,
        "logical_tpm_including_cached": 504824.79779265233,
        "logical_tpm_per_slice": 252412.39889632617
      },
      "decode_only_prefill_free": {
        "slices": 1,
        "turn_seconds": 15.980780317802067,
        "tok_s_user": 31.287583588331813,
        "logical_tpm_including_cached": 660793.7653855679,
        "logical_tpm_per_slice": 660793.7653855679
      }
    },
    {
      "context_source": "random",
      "context_tokens": 10000,
      "per_user_decode_tok_s": 32.451079492682126,
      "decode_seconds_for500": 15.407809079450365,
      "spec_statistics": {
        "step_seconds": null,
        "tokens_per_step": null,
        "acceptance_per_position": {},
        "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts."
      },
      "separate_equal_slices": {
        "slices": 2,
        "turn_seconds": 20.34517708988738,
        "tok_s_user": 24.575848998066782,
        "logical_tpm_including_cached": 990898.2316020527,
        "logical_tpm_per_slice": 495449.11580102635
      },
      "decode_only_prefill_free": {
        "slices": 1,
        "turn_seconds": 15.407809079450365,
        "tok_s_user": 32.451077075381065,
        "logical_tpm_including_cached": 1308427.4276793646,
        "logical_tpm_per_slice": 1308427.4276793646
      }
    },
    {
      "context_source": "random",
      "context_tokens": 15000,
      "per_user_decode_tok_s": 32.76076830129321,
      "decode_seconds_for500": 15.262158457772644,
      "spec_statistics": {
        "step_seconds": null,
        "tokens_per_step": null,
        "acceptance_per_position": {},
        "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts."
      },
      "separate_equal_slices": {
        "slices": 2,
        "turn_seconds": 20.199526468209655,
        "tok_s_user": 24.753055512806608,
        "logical_tpm_including_cached": 1473301.8641222494,
        "logical_tpm_per_slice": 736650.9320611247
      },
      "decode_only_prefill_free": {
        "slices": 1,
        "turn_seconds": 15.262158457772644,
        "tok_s_user": 32.76076587616362,
        "logical_tpm_including_cached": 1949920.7849492587,
        "logical_tpm_per_slice": 1949920.7849492587
      }
    },
    {
      "context_source": "agent-replay",
      "context_tokens": 5083,
      "per_user_decode_tok_s": 31.754118606064047,
      "decode_seconds_for500": 15.745990101259771,
      "spec_statistics": {
        "step_seconds": null,
        "tokens_per_step": null,
        "acceptance_per_position": {},
        "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts."
      },
      "separate_equal_slices": {
        "slices": 2,
        "turn_seconds": 20.683358111696784,
        "tok_s_user": 24.174024222751413,
        "logical_tpm_including_cached": 518260.1365847852,
        "logical_tpm_per_slice": 259130.0682923926
      },
      "decode_only_prefill_free": {
        "slices": 1,
        "turn_seconds": 15.745990101259771,
        "tok_s_user": 31.754116240680037,
        "logical_tpm_including_cached": 680767.6069313919,
        "logical_tpm_per_slice": 680767.6069313919
      }
    }
  ],
  "random_context_average": {
    "per_user_decode_tok_s": 32.16647789946127,
    "separate_equal_slices": {
      "mean_turn_seconds": 20.48761729544537,
      "tok_s_user": 24.404985352355038,
      "logical_tpm_including_cached": 984009.0094069551,
      "logical_tpm_per_slice": 492004.50470347755
    },
    "decode_only_prefill_free": {
      "mean_turn_seconds": 15.55024928500836,
      "tok_s_user": 32.153825371921116,
      "logical_tpm_including_cached": 1296442.2389958594,
      "logical_tpm_per_slice": 1296442.2389958594
    }
  }
}
```
