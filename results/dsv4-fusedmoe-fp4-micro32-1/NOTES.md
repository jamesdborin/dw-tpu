# dsv4-fusedmoe-fp4-micro32-1 — FUSED_MOE fp4 on the 32-chip slice (8 hosts, all-to-all over ICI)

**What**: patch 0101 adds two flag-like keyword arguments to `tpu_inference/kernels/fused_moe/v1/kernel.py:fused_ep_moe`
(default behaviour unchanged; every new branch is behind `fp4_codes=True` or `topk_ids is not None`):
- `fp4_codes=True`: w1/w2 are fp4 (e2m1) values as tokamax gmm_v2 uint4 codes (`e2m1_to_codes`, as GMM_TP/GMM_EP store
  them on v6e), with f32 block-32 scales in the existing `(E,2,H/32,1,I)` / `(E,I/32,1,H)` layout. Each weight tile is
  decoded to exact e2m1 bf16 in VMEM and multiplied by its bf16-cast scale *before* the matmul (same ops/order as tokamax
  gmm_v2 for blocks < MXU width), f32 accumulation; the activation is rounded to bf16 before the down projection and the
  per-expert output is rounded to nearest (as GMM_EP's gmm1/gmm2 outputs; the stock packing truncates).
- `topk_weights/topk_ids`: precomputed routing (DSV4 sqrtsoftplus + correction bias / hash routing are computed in JAX
  exactly as fused_moe_func), replacing the kernel's in-kernel softmax/sigmoid top-k. Duplicate expert ids per token are
  merged (the kernel's all-to-all buffers hold one row per token per expert).
- `act_fn="silu_and_mul_with_clamp"` (DSV4 swiglu limit 10, same as tokamax).

**Bench** (`bench_fused.py`, one process, 4 chips): one DSV4-Flash routed-expert layer (H4096, I2048, E256, top-6,
MXFP4) in the attention-DP layout (each chip owns its tokens, output returned to the owner), tokens per chip 4/8/16/32.
Paths: fused fp4 for 3 block configs (bf,bd1,bd2,bfc,bd1c,bd2c), GMM_EP and GMM_TP via the real `fused_moe_func`
(scatter_results=True, base series, no moe-overlap patches). All paths get bit-identical logical weights (integer hash of
the logical index, generated in each path's own layout), identical tokens/logits. Timing = 8-layer chain in one jit,
median of 20 calls / 8. Numerics: one layer, each path vs an f32 HIGHEST reference recomputed from the hash for the
first 16 tokens; fused vs GMM_EP bitwise fraction; fused run twice (bitwise repeatability).

**CPU evidence before submission** (dw-tpu-work/dsv4-fusedmoe-fp4): interpret mode on 4 CPU devices (toy shapes,
eager DMA mode) — fused fp4 output bit-identical to a GMM_EP-numerics emulation for 4 local experts incl. hash/duplicate
rows, 99.997% bit-identical (max 1 bf16 ulp, chunked f32 accumulation) for multi-chunk tiles; error vs float64 equal to
GMM's. Deviceless v6e AOT (Mosaic) compile OK for every config here at 2x2 and 4x8 topologies, tpc 4..32. This
bench script itself was run in CPU toy mode (fused + GMM_TP vs reference: rms 2.5e-5 / 2.9e-5 of ref rms 7.8e-3).

**Read result.json**: `runs.tpcN.<path>.us_per_layer`, `runs.tpcN.numerics`.
- PASS: all tpc have a fused config with `us_per_layer`; `fused_finite`, `fused_repeat_bitwise` true;
  `fused_vs_ref.rms <= 1.10 * gmm_ep_vs_ref.rms` and `fused_vs_ref.maxabs <= 2 * gmm_ep_vs_ref.maxabs`.
- FAIL: Mosaic/compile error, non-finite, non-repeatable, or error budget exceeded (=> kernel numerics bug).
- Speed is informational: expected weakness — the stock kernel DMAs every local expert's weights each step even with
  zero routed tokens (no skip), and does not prefetch the next expert's first weight block. At 4 chips (64 local
  experts) vs GMM (only routed experts' weights) it may lose at small token counts; micro32 (8 local experts) is the
  representative comparison. Note: GMM_TP here has no per-chip padding (I/4 = 512).

## micro32 specifics
Same bench as dsv4-fusedmoe-fp4-kernel-1 on all 32 chips (jax.distributed, one process per host): EP32 = 8 local
experts per chip, the serving layout (attention DP32). GMM_TP here has the serving per-chip padding (I/32 = 64 real
columns padded to 128 for gate and for up). Token counts 4,32,8 per chip (in that order; 16 dropped for the 15-min cap), 2 fused block configs (bd1=4096 tiles OOM VMEM at tpc32 x 32 chips in AOT, so the second config uses bd1=bd2=1024).
This is the representative comparison for decode: fused_ep_moe replaces the all-gather + GMM + reduce-scatter of
GMM_EP/GMM_TP with point-to-point all-to-all inside one kernel. Expected risk: the kernel's metadata all-reduce is a
31-step ring with a barrier per step, plus one barrier per local expert; at 32 chips this latency may dominate.

**Update 16:45Z (before run)**: patch 0101 also accepts the GMM_EP weight layout directly (w13 `(E, H, 2I)` = [gate|up],
scales `(E, H/32, 1, 2I)`): FUSED_MOE can then share the GMM_EP weights (fused for decode, GMM_EP for prefill, one copy).
Bench path `fusedg_<cfg0>` times the kernel on the GMM_EP weights; `numerics.fusedg_eq_fused_bitwise` must be true
(CPU interpret: bitwise equal; AOT 2x2/4x8 OK).

**Update 16:46Z (before run)**: 0101 adds opt-in `oneshot_metadata=True` (each chip sends its per-expert counts to all
peers at once: one barrier instead of the (N-1)-step ring with a barrier per step; same integer sums, CPU-bitwise equal
at 4 and 8 devices). Bench path `fusedo_<cfg0>`; `numerics.fusedo_eq_fused_bitwise` must be true. Patch 0102 is the
flag-gated vLLM integration (FUSED_MOE_FP4_EMULATION=1 with USE_MOE_EP_KERNEL=1; not exercised by this bench, applies
cleanly, default off).

**Update 17:00Z (before run)**: 0101 adds opt-in `prefetch_next_expert=True` (next local expert's first w1/w3 block is
fetched during the current expert's last w2 block). Bench path `fusedpo_<cfg0>` (prefetch + oneshot) runs LAST, after
GMM_TP, so a schedule problem cannot cost the other results; `numerics.fusedpo_eq_fused_bitwise` must be true (CPU: bitwise).
