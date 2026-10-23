# kernels/

Read-only copies of the new source files that the patch series adds to tpu-inference, taken from `3f7931b63` with `patches/tpu-inference` and `patches/tpu-inference-ext` applied. They are here so the kernels can be read and linked without applying anything. The patches are the source of truth; `apply_patches.sh --all` produces these files at the same paths.

| File | Patch | What it is |
|---|---|---|
| `tpu_inference/kernels/experimental/dense_fp8.py` | 0006, 0007 | Dense FP8 block-quantized GEMM: FP8 weights on the MXU per 128-row K block, scale applied to the f32 partial sum, pipelined N tiles. |
| `tpu_inference/kernels/experimental/dense_fp8_tiles.py` | 0007 | Validated (M, K, N) tile table; shapes not in it fall back to the old path. |
| `tpu_inference/kernels/collectives/oneshot.py` | 0102 | One-shot all-gather and reduce-scatter with remote DMA after one barrier; the reduce-scatter sums in f32 in fixed rank order. |
| `tpu_inference/layers/common/quantization/e8m0_scale.py` | 0403 | FP4 block scales stored as 8-bit exponents. |
| `tpu_inference/layers/common/moe_fast_combine.py` | 0404 | MoE combine: per-row DMA gather of the top-6 expert outputs into VMEM and an f32 weighted sum, with no HBM intermediate. |
| `tpu_inference/layers/common/moe_fast_dispatch.py` | 0405 | MoE dispatch permutation as `argsort // topk` instead of an int32 index gather. |
| `tpu_inference/spec_decode/jax/dsv4_mtp.py` | 0201 | MTP proposer for DeepSeek V4. |
| `tpu_inference/models/vllm/experimental/deepseek_v4_mtp.py` | 0201 | The checkpoint's `mtp.0` layer as a draft model. |
| `tpu_inference/distributed/tpu_connector_dsv4.py` | 0501 | P/D KV connector for DeepSeek V4's hybrid cache. |

The FP4 grouped-matmul change edits tokamax's `gmm_v2.py` rather than adding a file; see `patches/tokamax/0001-gmm_v2-fp4-emulation-v6e.patch`.
