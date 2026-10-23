# dsv4-specdec-mtp-func-6

Status: passed
Acceptance passed: True

- startup_seconds: 808.2185757160187
- compile_wall_span_seconds: 756.0
- token_buckets_per_dp: [4, 16, 512]
- kv_blocks: [17696, 17706]
- tok_s_user_median: 19.882382233948533
- aggregate_tok_s: 564.0092810675699
- tpm_including_cached: 697632.3859056148
- apc_hit_rate: 0.8241740174430386
- preemptions: 0.0
- spec_decode_acceptance_rate: 0.3488401847515961

32-agent measurements; median across repeats. TPM includes cached history, not compute throughput.
Startup compiler-message wall span when readiness is known; task durations include all phases and overlap across hosts. Raw worker logs are preserved.

All eight host archives and checksums: see copies.json.
