# Baseline 1: repeat the accepted asynchronous stack

Use the exact pinned dw-tpu base in job.json, with dense FP8 GEMM disabled and
per-attention-DP buckets 4,16,512. All other base flags remain active, including
H17 exact-size CSA gather and asynchronous multihost prompt-logprob replication.
There are no job patches.

Run after the cold-start psum and checkpoint verification:

```bash
python3 runner/run_job.py jobs/baseline-1 --dry-run
python3 runner/run_job.py jobs/baseline-1
```

Success requires exact 32-token greedy outputs and selected generated logprobs
on 17-, 300-, and 2,600-token prompts against BOTH preserved B2 (h17-model) and
H4-r2 evidence. Teacher-forced prompt IDs and all returned prompt logprobs on
17/300 must match H4-r2. Repeated-cache tokens/logprobs must match; suffix changes
at 143/178/251 must have zero effect before the boundary. The original GPU
reference is retained and its MAE/top1 comparison is reported.

Only after acceptance passes, run the original agent benchmark: 32 agents,
10 turns, roughly 5k initial tokens, 500 output + 500 environment tokens per
turn, prefix cache on, zero tool latency, no retries. A separate 3-turn warmup
is excluded. The validator demands 320 successful unique requests, 500 output
tokens per request, growing histories, and observed running concurrency >=32.

Compare with the historical single run: 23.30 output tok/s/user and about 893k
TPM including cached history. This job establishes a matched reservation
baseline; a performance difference alone does not waive any correctness gate.
An acceptance failure, post-ready allocation event, launch mismatch or timeout
invalidates the result. Inspect result/summary.json, SUMMARY.md, and all-host
archives referenced in result/copies.json. Do not promote candidates until this
baseline has passed. Startup is capped at 45 minutes; total controller time is
90 minutes. Expected entire e2e time is approximately 45–60 minutes with a cold
compile cache, but download/startup variability can exceed that estimate.
