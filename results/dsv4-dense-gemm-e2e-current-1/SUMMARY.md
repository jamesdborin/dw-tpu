# dsv4-dense-gemm-e2e-current-1

Status: failed
Acceptance passed: False

- startup_seconds: 736.5741240978241
- compile_wall_span_seconds: 684.0
- token_buckets_per_dp: [4, 16, 512]
- kv_blocks: [50624]
- tok_s_user_median: 29.119529561192174
- aggregate_tok_s: 891.494313371054
- tpm_including_cached: 1102511.7974035067
- apc_hit_rate: 0.8243250435247615
- preemptions: 0.0
- spec_decode_acceptance_rate: None

32-agent measurements; median across repeats. TPM includes cached history, not compute throughput.
Startup compiler-message wall span when readiness is known; task durations include all phases and overlap across hosts. Raw worker logs are preserved.

All eight host archives and checksums: see copies.json.

Error: RuntimeError("CommandFailed('profile.log: exit 1')")
