# Results

Measured on TPU v6e in October 2026 with vLLM `f43d7e1a0f`, tpu-inference `3f7931b63` and the patches in this repository. Each `results/<job>/` directory holds the job spec (`job.json`) and the runner's `SUMMARY.md`, or the raw `result.json` for kernel jobs. The full raw outputs (`pd.json`, server logs, profiles) are not included because of their size.

Workload terms used below:
- **Agent benchmark (colocated):** 32 agents × 10 turns. The first prompt is ~5,000 tokens; each turn generates 500 tokens and appends 500 tokens of tool output. Prefix caching is on.
- **Decode-only:** all N requests are prefilled first, untimed. The window opens when every request is decoding and closes at the first finish. Contexts are the 32 real mid-conversation agent-benchmark payloads (~5,083 tokens), reused cyclically for N > 32. Temperature 1.0.
- **Prefill-only:** requests generate one token; "new tok/s" counts only uncached prompt tokens.

## Decode on 32 chips, by concurrency

TP32 MoE, attention DP32, FP4 experts, FP8 KV cache. Per chip = total ÷ 32.

| N | MTP k=3 tok/s/user | total | per chip | non-spec tok/s/user | total | per chip | non-spec experts |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 50.4 | 50 | 1.6 | 28.4 | 28 | 0.9 | GMM_TP |
| 8 | 54.9 | 439 | 13.7 | 36.6 | 293 | 9.2 | GMM_TP |
| 16 | 50.2 | 803 | 25.1 | 34.4 | 550 | 17.2 | GMM_TP |
| 32 | 46.1 | 1,475 | 46.1 | 31.8 | 1,016 | 31.8 | GMM_TP |
| 64 | 37.5 | 2,402 | 75.1 | 28.9 | 1,850 | 57.8 | GMM_TP |
| 128 | 32.0 | 4,100 | 128.1 | 29.9 | 3,830 | 119.7 | GMM_EP |
| 192 | 22.8 | 4,383 | 137.0 | 22.6 | 4,331 | 135.3 | GMM_EP |
| 256 | 15.9 | 4,068 | 127.1 | 22.5 | 5,758 | 179.9 | GMM_EP |
| 320 | 15.7 | 5,016 | 156.8 | 18.4 | 5,888 | 184.0 | GMM_EP |
| 384 | 15.5 | 5,959 | 186.2 | 18.3 | 7,041 | 220.0 | GMM_EP |

- Speculative: MTP k=3 on GMM_TP + packed gather, `pd-specdec` (N ≤ 128) and `pd-scale-specdec` (N ≥ 192). Acceptance is 0.71–0.75 / 0.18–0.19 / 0.05 by draft position, about 2.0 tokens per verify step on agent contexts.
- Non-speculative: GMM_TP + packed gather (`pd-base0`) for N ≤ 64; GMM_EP + f32 reduce + one-shot reduce-scatter (`pd-gmm-ep`, `pd-gmm-ep-scale`) from 128. GMM_TP cannot hold 256 requests in KV; GMM_EP has 69% more KV blocks.

## Prefill: 32-chip slice vs one 8-chip host

New (uncached) tok/s per chip. 32-chip: `pd-final-prefill` (GMM_TP + packed gather). 8-chip: TP8, attention DP8, MNBT 512 per rank; c8 = GMM_TP + packed gather, bf16 reduce, dense FP8 GEMM and 0401–0405; c2 = GMM_EP + f32 reduce + one-shot RS and 0401–0403, the 8-chip config that passes the TP32-referenced accuracy gate.

| Case | N | 32-chip | 8-chip c8 | c8 ÷ 32-chip | 8-chip c2 |
|---|---:|---:|---:|---:|---:|
| Cold 5,000-token prompts | 32 | 604 | 2,421 | 4.0× | 1,448 |
| | 128 | 591 | 2,500 | 4.2× | 1,500 |
| 1,650 new on 5k cached | 32 | 382 | 1,583 | 4.1× | 963 |
| 1,650 new on 10k cached | 64 | 332 | 1,584 | 4.8× | 971 |
| 1,650 new on 14k cached | 32 | 315 | 1,608 | 5.1× | 987 |

One 8-chip host with c8 prefills about as many tokens per second as the whole 32-chip slice (~20k cold, ~13k agent-turn new tok/s).

## Kernel micro-benchmarks

| Kernel | Baseline | Result | Job |
|---|---|---|---|
| FP4 grouped matmul, w13, 1 / 64 / 1,024 tokens | bf16 `gmm_v2` 165.5 / 6,109 / 8,186 µs | 89.4 / 2,526 / 3,143 µs | tokamax 0001, one chip |
| Dense FP8 GEMM, `wq_b` 1024 × 32768, M = 4 | 70.9 µs | 29.4 µs (72% of HBM peak) | `dsv4-dense-gemm-kernel-1` |
| Dense FP8 GEMM, all projections, 43 layers | 9.90 ms/step (M = 4) | 4.62 ms/step | `dsv4-dense-gemm-kernel-1` |
| Fast MoE combine, 4,096 tokens | 1.90 ms/layer | 0.76 ms/layer | 8-chip prefill microcheck |
| Fast dispatch | 0.339 ms/layer | 0.192 ms/layer | 8-chip prefill microcheck |
| One-shot all-gather, 32 chips, tiny | 27.4 µs | 17.5 µs | `dsv4-moe-overlap-coll32-2/3` |
| One-shot reduce-scatter, 32 chips, 4 rows | 41.9 µs | 38.2 µs | `dsv4-moe-overlap-coll32-2/3` |
| FUSED_MOE FP4, 32 chips, 128 tokens | GMM_EP 438 µs/layer | 624 µs/layer | `dsv4-fusedmoe-fp4-micro32-1` |

## Colocated agent benchmark (32 agents, 32 chips)

| Stack | tok/s per user | Job |
|---|---:|---|
| Accepted stack before the packed gather | 24.04 | `baseline-1` |
| + packed MoE route gather | 25.30 | `dsv4-moe-overlap-e2e-base-0` |
| + dense FP8 GEMM (on the 24.04 stack) | 29.12 | `dsv4-dense-gemm-e2e-current-1` |
| MTP k=3 (colocated, with the DP-scheduler prefill fix) | 19.88 | `dsv4-specdec-mtp-func-6` |

## P/D sizing: one 8-chip prefill host + 64 decode chips

`sizing/pd_sizing.py` combines the measured 8-chip prefill rates with the measured 32-chip decode rates. 64 decode chips run as two 32-chip copies. Each agent turn is 9,807 prompt tokens (1,985 uncached, 7,822 cached) plus 500 output tokens. The binding side is held at 85% (prefill) or 95% (decode). Cost: 72 chips at $1.89/chip-hour plus $0.11/hour disk for each of 17 hosts, $137.95/hour. Breakeven prices use input : cached : output = 1 : 0.1 : 4.

| Prefill | Decode | Turns/s | Binds | Concurrent decoding | Output tok/s | TPM incl. / excl. cached | Decode tok/s/user | $/M input / cached / output | Blended $/M |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| c8 | MTP k=3, ≤64 per copy | 8.28 | both (D 87%, P 85%) | 111 | 4,139 | 5.12M / 1.23M | 37.4 | 0.97 / 0.097 / 3.88 | 0.45 |
| c8 | MTP k=3, ≤32 per copy | 5.59 | decode | 61 | 2,795 | 3.46M / 0.83M | 46.0 | 1.44 / 0.144 / 5.75 | 0.67 |
| c2 | MTP k=3, ≤32 per copy | 5.02 | both (D 85%, P 85%) | 55 | 2,508 | 3.10M / 0.75M | 46.0 | 1.60 / 0.160 / 6.41 | 0.74 |

One 8-chip host prefills up to 9.7 agent turns/s with c8, or 5.9 with c2. With MTP at up to 128 requests per copy, the 64 decode chips could serve about 16 turns/s, so prefill binds and decode would sit about half idle. The balanced point is up to 64 requests per 32-chip copy with c8, or up to 32 per copy with c2.
