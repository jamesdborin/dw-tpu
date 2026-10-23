# baseline-1

Status: passed
Acceptance passed: True

- startup_seconds: 748.7871208190918
- compile_wall_span_seconds: 707.0
- token_buckets_per_dp: [4, 16, 512]
- kv_blocks: [50624]
- tok_s_user_median: 24.04492866220378
- aggregate_tok_s: 745.4656231155527
- tpm_including_cached: 922039.0333699776
- apc_hit_rate: 0.8242113091729222
- preemptions: 0.0
- spec_decode_acceptance_rate: None

32-agent measurements; median across repeats. TPM includes cached history, not compute throughput.
Startup compiler-message wall span when readiness is known; task durations include all phases and overlap across hosts. Raw worker logs are preserved.

All eight host archives and checksums: see copies.json.
