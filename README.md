# dw-tpu

DeepSeek V4 Flash serving on TPU v6e (Trillium) with vLLM and tpu-inference.

vLLM's TPU backend only ran DeepSeek-V4-Flash on TPU v7x. This repository holds what it took to serve the model on v6e, on 8-chip hosts and on a 32-chip 4×8 slice: the Pallas kernels, the correctness fixes, a prefill/decode (P/D) disaggregation path and the runner used to measure it. The write-up is [Running DeepSeek V4 Flash on TPU v6e](https://mainlymatmul.com/blog/deepseek-v4-flash-tpu-v6e/).

Everything is a patch on top of pinned upstream code. Nothing in `vllm/` or `tpu-inference/` is edited in place.

| Path | What it is |
|---|---|
| `vllm/`, `tpu-inference/` | Upstream submodules, pinned to vLLM `f43d7e1a0f` and tpu-inference `3f7931b63`. |
| `patches/tokamax/` | The FP4 grouped-matmul patch for tokamax `0.0.16.dev20261009`. |
| `patches/tpu-inference/` | Base series 0001–0006: v6e bring-up, correctness fixes, the first kernels. |
| `patches/tpu-inference-ext/` | Everything after the base: MoE communication, expert parallelism, MTP speculative decoding, 8-chip memory and MoE kernels, P/D. |
| `patches/tpu-inference-experimental/` | Work that did not pay off on v6e (FUSED_MOE with FP4), kept for reference. |
| `kernels/` | Read-only copies of the new source files the patches add, so the kernels can be read without applying anything. |
| `results/` | Curated result summaries, the sizing model and the tables behind the post. |
| `runner/` | The job runner used on the reservation: staging, provisioning, serving, accuracy gates and benchmarks. |
| `jobs/` | Job specs for the runner (`jobs/README.md` describes the protocol). |
| `env.sh` | Runtime flags for the 32-chip configuration. |

## Setup

```bash
git clone --recursive https://github.com/jamesdborin/dw-tpu.git && cd dw-tpu
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-tpu.txt
VLLM_TARGET_DEVICE=tpu uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e vllm
uv pip install --python .venv/bin/python --no-deps -e tpu-inference
./apply_patches.sh --all --tokamax .venv/bin/python   # add --reset to discard local changes first
```

`apply_patches.sh` checks that `tpu-inference` is a clean checkout of the pinned commit, then applies `patches/tpu-inference/*.patch` and, with `--all`, `patches/tpu-inference-ext/*.patch`, in filename order with `git apply`. `--tokamax PY` patches the tokamax installed in that Python environment. Without `--all` you get the base series only, which is what the runner stages for every job.

The full series (base + ext) applies cleanly in order on `3f7931b63`. Apart from 0001–0003, every patch is inert unless its environment flag is set, so applying all of them does not change behaviour until you opt in.

## Running

On a 32-chip slice (8 × ct6e-standard-4t, 4×8), start Ray across the hosts with `env.sh` sourced on every host, then:

```bash
source env.sh
vllm serve $CKPT --served-model-name dsv4-flash --tensor-parallel-size 32 \
  --max-model-len $MAX_LEN --max-num-batched-tokens $MNBT --max-num-seqs $MAX_SEQS \
  --gpu-memory-utilization $GPU_MEM --kv-cache-dtype fp8 --tokenizer-mode deepseek_v4 \
  --enable-prefix-caching --distributed-executor-backend ray \
  --additional-config '{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}'
```

On one 8-chip host (ct6e-standard-8t), no Ray is needed:

```bash
source env.sh
export MOE_SCALE_BF16=1 DSV4_WO_A_FP8_STORE=1 MOE_SCALE_E8M0=1 MOE_FAST_COMBINE=2 MOE_FAST_DISPATCH=1
vllm serve $CKPT --tensor-parallel-size 8 --max-num-batched-tokens 512 --gpu-memory-utilization 0.89 \
  --kv-cache-dtype fp8 --tokenizer-mode deepseek_v4 --enable-prefix-caching \
  --additional-config '{"sharding": {"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}'
```

Notes for v6e:
- VMs created through GCE need `LIBTPU_INIT_ARGS=--enable_tpunetd_client=false` (set in `env.sh`).
- `JAX_PLATFORMS` must be unset, not empty; empty falls back to CPU.
- Startup takes 15–45 minutes: about 9–11 minutes of weight loading and FP4 re-encoding, then XLA compilation of every token bucket. Set `VLLM_XLA_CACHE_PATH` to a persistent disk.

## Kernel and patch index

Speedups are against the code each change replaced, measured on v6e. "Bit-identical" changes were gated on exact token and logprob equality. "Summation order" changes were gated on greedy tokens and a teacher-forced error limit against a GPU reference. The post explains each one.

| Change | Patch | Source | Flag | Effect |
|---|---|---|---|---|
| FP4 grouped matmul: e2m1 weights as uint4 codes, decoded to bf16 in VMEM | [tokamax/0001](patches/tokamax/0001-gmm_v2-fp4-emulation-v6e.patch) | tokamax `gmm_v2.py` | `MOE_REQUANTIZE_WEIGHT_DTYPE=fp4` | f4 does not compile on v6e; 1.65–2.6× faster than bf16 experts at decode sizes |
| v6e enablement, DP scheduler `hash_block_size` fix | [0001](patches/tpu-inference/0001-dsv4-v6e-enablement.patch) | | | required |
| CSA indexer causal mask | [0002](patches/tpu-inference/0002-csa-indexer-causal-mask.patch) | | | correctness fix |
| Per-rank token buckets | [0003](patches/tpu-inference/0003-per-dp-token-buckets.patch) | | `TPU_TOKEN_PADDINGS_PER_DP` | +12% decode |
| Exact-size CSA SparseCore gather | [0004](patches/tpu-inference/0004-csa-gather-exact-size.patch) | | `DSV4_H17_SMALL_GATHER=1` | gather 2.5–5.8×, +13% decode, bit-identical |
| Multi-host async scheduling | [0005](patches/tpu-inference/0005-async-multihost-prompt-logprobs.patch) | | `TPU_ASYNC_REPLICATE_OUTPUTS=1` | +26% decode |
| Dense FP8 block-quantized GEMM | [0006](patches/tpu-inference/0006-dense-fp8-gemm.patch), [0007](patches/tpu-inference-ext/0007-dense-fp8-tile-table.patch) | [dense_fp8.py](kernels/tpu_inference/kernels/experimental/dense_fp8.py), [dense_fp8_tiles.py](kernels/tpu_inference/kernels/experimental/dense_fp8_tiles.py) | `DSV4_DENSE_FP8_GEMM=1` | 1.6–2.4× per projection; +21% colocated tok/s/user |
| Packed MoE route + hidden all-gather | [0101](patches/tpu-inference-ext/0101-H15b-packed-route-hidden-all-gather-MOE_PACKED_ROUTE.patch) | | `MOE_PACKED_ROUTE_GATHER=1` | two collectives become one; +5.2%, bit-identical |
| One-shot Pallas all-gather / reduce-scatter | [0102](patches/tpu-inference-ext/0102-One-shot-Pallas-all-gather-reduce-scatter-for-MoE-de.patch) | [oneshot.py](kernels/tpu_inference/kernels/collectives/oneshot.py) | `MOE_ONESHOT_SCATTER=1` | 17.5 µs vs 27.4 µs floor; ordered f32 reduce |
| Expert split, shared-expert TP | [0103](patches/tpu-inference-ext/0103-MOE_TP_EXPERT_SPLIT-GMM_TP-expert-groups-so-each-sha.patch), [0104](patches/tpu-inference-ext/0104-MOE_SHARED_EXPERT_TP-shared-expert-tensor-parallel-i.patch), [0108](patches/tpu-inference-ext/0108-MOE_SHARED_EXPERT_TP-return-a-zero-shared-output-MoE.patch) | | `MOE_TP_EXPERT_SPLIT`, `MOE_SHARED_EXPERT_TP` | +0.4% end to end; not used |
| Expert parallelism with an explicit f32 reduce | [0105](patches/tpu-inference-ext/0105-MOE_F32_REDUCE-f32-per-shard-combine-and-f32-reduce-.patch), [0106](patches/tpu-inference-ext/0106-MOE_F32_REDUCE-keep-the-w2-GMM-output-in-f32-too.patch), [0107](patches/tpu-inference-ext/0107-GMM_EP-packed-route-gather-MOE_F32_REDUCE-and-one-sh.patch) | | `MOE_F32_REDUCE=1` with GMM_EP | +13% decode at 128 requests, 69% more KV |
| MTP speculative decoding | [0201](patches/tpu-inference-ext/0201-dsv4-mtp-spec-decode.patch) | [dsv4_mtp.py](kernels/tpu_inference/spec_decode/jax/dsv4_mtp.py), [deepseek_v4_mtp.py](kernels/tpu_inference/models/vllm/experimental/deepseek_v4_mtp.py) | `--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'` | 1.2–1.8× per user up to 128 requests |
| Cache-affinity DP-rank pinning (benchmarks) | [0301](patches/tpu-inference-ext/0301-pd-prefill-cache-affinity.patch) | | `PD_CACHE_AFFINITY=1` | keeps a request's turns on the DP rank holding its prefix |
| Expert scales as bf16 | [0401](patches/tpu-inference-ext/0401-moe-bf16-scales.patch) | | `MOE_SCALE_BF16=1` | −2.0 GiB/chip, bit-identical |
| `wo_a` stored as FP8 | [0402](patches/tpu-inference-ext/0402-dsv4-wo-a-fp8-store.patch) | | `DSV4_WO_A_FP8_STORE=1` | −1.3 GiB/chip, bit-identical |
| Expert scales as e8m0 exponents | [0403](patches/tpu-inference-ext/0403-moe-e8m0-scales.patch) | [e8m0_scale.py](kernels/tpu_inference/layers/common/quantization/e8m0_scale.py) | `MOE_SCALE_E8M0=1` | −3.0 GiB/chip vs f32, bit-identical |
| Fast MoE combine (Pallas DMA row gather + f32 sum) | [0404](patches/tpu-inference-ext/0404-moe-fast-combine.patch) | [moe_fast_combine.py](kernels/tpu_inference/layers/common/moe_fast_combine.py) | `MOE_FAST_COMBINE=2` | 2.5× per layer; +24% 8-chip prefill |
| Fast MoE dispatch | [0405](patches/tpu-inference-ext/0405-moe-fast-dispatch.patch) | [moe_fast_dispatch.py](kernels/tpu_inference/layers/common/moe_fast_dispatch.py) | `MOE_FAST_DISPATCH=1` | 1.8× per layer, bit-identical |
| P/D connector `TPUConnectorDSV4`, DP scheduler support, test kit | [0501](patches/tpu-inference-ext/0501-dsv4-pd-connector.patch) | [tpu_connector_dsv4.py](kernels/tpu_inference/distributed/tpu_connector_dsv4.py) | `--kv-transfer-config` | |
| P/D fix: insert pulled KV before the decode forward | [0502](patches/tpu-inference-ext/0502-pd-insert-pulled-kv-before-forward.patch) | | on by default (`DSV4_PD_SYNC_INSERT=0` for A/B) | 31/32 wrong → 28/29 greedy-equal |
| P/D router: buffer SSE events across chunks | [0503](patches/tpu-inference-ext/0503-pd-router-sse-buffering.patch) | | | fixes a dropped first token |
| FUSED_MOE with FP4 (negative result) | [0901](patches/tpu-inference-experimental/fused-moe-fp4/0901-fused-moe-fp4-codes-routing-oneshot-metadata.patch), [0902](patches/tpu-inference-experimental/fused-moe-fp4/0902-fused-moe-next-expert-prefetch.patch) | | `USE_MOE_EP_KERNEL=1` | 1.4× slower than GMM_EP; applies on the base series only |

The P/D patches 0501–0503 are the squashed form of a 23-commit series built on 0001–0403. 0501 includes the `pdmini/` harness (router, load generator, transfer micro-benchmarks and correctness checks) used to test P/D on a 5-layer model. 0101–0107 overlap with the copies in older `jobs/*/patches/` directories, which the runner applied per job.

## Results

`results/README.md` has the decode-by-concurrency table, 32-chip vs 8-chip prefill, the kernel micro-benchmarks and the P/D sizing, with the job each number comes from. `results/sizing/pd_sizing.py` reproduces the sizing and breakeven prices for one 8-chip prefill host and 64 decode chips.

## Licence

Apache-2.0, the same as vLLM and tpu-inference. The submodules carry their own licences. Files under `kernels/` are copies of files the patches add to tpu-inference and keep its Apache-2.0 header.
