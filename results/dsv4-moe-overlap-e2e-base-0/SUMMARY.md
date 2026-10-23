# dsv4-moe-overlap-e2e-base-0

Status: passed
Acceptance passed: True

- startup_seconds: 928.4798500537872
- compile_wall_span_seconds: 886.0
- token_buckets_per_dp: [4, 16, 512]
- kv_blocks: [50624]
- tok_s_user_median: 25.304642810317965
- aggregate_tok_s: 780.2075226252434
- tpm_including_cached: 965159.2267974091
- apc_hit_rate: 0.8247298659492828
- preemptions: 0.0
- spec_decode_acceptance_rate: None

32-agent measurements; median across repeats. TPM includes cached history, not compute throughput.
Startup compiler-message wall span when readiness is known; task durations include all phases and overlap across hosts. Raw worker logs are preserved.

All eight host archives and checksums: see copies.json.
