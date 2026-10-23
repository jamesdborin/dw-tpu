# pd-specdec

Status: passed
Acceptance passed: True

- startup_seconds: 2050.9005918502808
- compile_wall_span_seconds: 2002.0
- token_buckets_per_dp: [4, 8, 16, 512]
- kv_blocks: [30624, 30637]
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
| random | 1 | 5000 | True | 30.155147769288114 |  |
| random | 8 | 5000 | True | 29.762881657601486 |  |
| random | 16 | 5000 | True | 28.599978050776087 |  |
| random | 32 | 5000 | True | 30.71129645903614 |  |
| random | 64 | 5000 | True | 23.962151311625227 |  |
| random | 128 | 5000 | True | 18.85279146602494 |  |
| random | 1 | 10000 | True | 24.554899016177483 |  |
| random | 8 | 10000 | True | 26.16809257783948 |  |
| random | 16 | 10000 | True | 25.920883107604702 |  |
| random | 32 | 10000 | True | 28.644772722561825 |  |
| random | 64 | 10000 | True | 22.322277329564372 |  |
| random | 128 | 10000 | True | 18.363351167085256 |  |
| random | 1 | 15000 | True | 24.20695466832828 |  |
| random | 8 | 15000 | True | 25.83629956739022 |  |
| random | 16 | 15000 | True | 25.027216676758638 |  |
| random | 32 | 15000 | True | 28.230836344238746 |  |
| random | 64 | 15000 | True | 21.39853298363556 |  |
| random | 128 | 15000 | True | 18.055701358912085 |  |
| agent-replay | 1 | 5083 | True | 50.420328548977864 |  |
| agent-replay | 8 | 5083 | True | 54.88463898961607 |  |
| agent-replay | 16 | 5083 | True | 50.18763560179427 |  |
| agent-replay | 32 | 5083 | True | 46.080492213156795 |  |
| agent-replay | 64 | 5083 | True | 37.53656876277779 |  |
| agent-replay | 128 | 5083 | True | 32.03349638352847 |  |

### Prefill

- cold5k N=32: valid=True; new tok/s=17133.135410213228; including cached tok/s=17133.135410213228; TTFT={'mean': 9.32079708781248, 'p50': 9.319922920003592, 'p90': 9.327385112001503}; flags=[]
- cold5k N=64: valid=True; new tok/s=17395.854287407154; including cached tok/s=17395.854287407154; TTFT={'mean': 14.175868964468918, 'p50': 13.3777659615007, 'p90': 18.363356510703305}; flags=[]
- incr1650_on8k N=32: valid=True; new tok/s=10336.542401932636; including cached tok/s=60453.11162342421; TTFT={'mean': 5.077278712563384, 'p50': 5.079356010999618, 'p90': 5.09373768499645}; flags=[]
- incr1650_on8k N=64: valid=False; new tok/s=3201.1215355319796; including cached tok/s=18721.710798717337; TTFT={'mean': 16.241830891858626, 'p50': 17.28054813000199, 'p90': 21.446458842997526}; flags=['prefix_cache_reuse_unverified_or_insufficient']

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
  "incremental_prefill_ttft_seconds": 5.077278712563384,
  "contexts": [
    {
      "context_source": "random",
      "context_tokens": 5000,
      "per_user_decode_tok_s": 30.71129645903614,
      "decode_seconds_for500": 17.799007686183973,
      "spec_statistics": {
        "step_seconds": 0.04127789609183496,
        "tokens_per_step": 1.2708864795918369,
        "acceptance_per_position": {
          "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.15146683673469388,
          "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.07940051020408163,
          "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.04001913265306122
        },
        "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts."
      },
      "separate_equal_slices": {
        "slices": 2,
        "turn_seconds": 22.876286398747357,
        "tok_s_user": 21.85669436396716,
        "logical_tpm_including_cached": 461613.3849669864,
        "logical_tpm_per_slice": 230806.6924834932
      },
      "decode_only_prefill_free": {
        "slices": 1,
        "turn_seconds": 17.799007686183973,
        "tok_s_user": 28.091453681887685,
        "logical_tpm_including_cached": 593291.5017614679,
        "logical_tpm_per_slice": 593291.5017614679
      }
    },
    {
      "context_source": "random",
      "context_tokens": 10000,
      "per_user_decode_tok_s": 28.644772722561825,
      "decode_seconds_for500": 18.135809361721886,
      "spec_statistics": {
        "step_seconds": 0.04110958383017939,
        "tokens_per_step": 1.180129716981132,
        "acceptance_per_position": {
          "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.11871069182389937,
          "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.048840408805031446,
          "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.012578616352201259
        },
        "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts."
      },
      "separate_equal_slices": {
        "slices": 2,
        "turn_seconds": 23.21308807428527,
        "tok_s_user": 21.539572778939494,
        "logical_tpm_including_cached": 868475.5744468403,
        "logical_tpm_per_slice": 434237.78722342016
      },
      "decode_only_prefill_free": {
        "slices": 1,
        "turn_seconds": 18.135809361721886,
        "tok_s_user": 27.5697648793838,
        "logical_tpm_including_cached": 1111612.9199367547,
        "logical_tpm_per_slice": 1111612.9199367547
      }
    },
    {
      "context_source": "random",
      "context_tokens": 15000,
      "per_user_decode_tok_s": 28.230836344238746,
      "decode_seconds_for500": 18.32657514409121,
      "spec_statistics": {
        "step_seconds": 0.040949822404526603,
        "tokens_per_step": 1.1588794498381878,
        "acceptance_per_position": {
          "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.10629045307443366,
          "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.04136326860841424,
          "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.011225728155339806
        },
        "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts."
      },
      "separate_equal_slices": {
        "slices": 2,
        "turn_seconds": 23.403853856654592,
        "tok_s_user": 21.36400282886877,
        "logical_tpm_including_cached": 1271585.4483742693,
        "logical_tpm_per_slice": 635792.7241871346
      },
      "decode_only_prefill_free": {
        "slices": 1,
        "turn_seconds": 18.32657514409121,
        "tok_s_user": 27.282784484760008,
        "logical_tpm_including_cached": 1623871.3325329155,
        "logical_tpm_per_slice": 1623871.3325329155
      }
    },
    {
      "context_source": "agent-replay",
      "context_tokens": 5083,
      "per_user_decode_tok_s": 46.080492213156795,
      "decode_seconds_for500": 10.877401084481097,
      "spec_statistics": {
        "step_seconds": 0.042777569811633304,
        "tokens_per_step": 1.9696061643835616,
        "acceptance_per_position": {
          "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.7440068493150684,
          "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.18418236301369864,
          "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.04141695205479452
        },
        "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts."
      },
      "separate_equal_slices": {
        "slices": 2,
        "turn_seconds": 15.95467979704448,
        "tok_s_user": 31.338767456343582,
        "logical_tpm_including_cached": 671863.0606416623,
        "logical_tpm_per_slice": 335931.53032083117
      },
      "decode_only_prefill_free": {
        "slices": 1,
        "turn_seconds": 10.877401084481097,
        "tok_s_user": 45.96686249929271,
        "logical_tpm_including_cached": 985470.6944008366,
        "logical_tpm_per_slice": 985470.6944008366
      }
    }
  ],
  "random_context_average": {
    "per_user_decode_tok_s": 29.195635175278905,
    "separate_equal_slices": {
      "mean_turn_seconds": 23.164409443229072,
      "tok_s_user": 21.5848369122205,
      "logical_tpm_including_cached": 870300.6243007305,
      "logical_tpm_per_slice": 435150.31215036527
    },
    "decode_only_prefill_free": {
      "mean_turn_seconds": 18.087130730665688,
      "tok_s_user": 27.643964509654303,
      "logical_tpm_including_cached": 1114604.6490292614,
      "logical_tpm_per_slice": 1114604.6490292614
    }
  }
}
```
