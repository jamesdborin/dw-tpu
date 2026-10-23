# dsv4-moe-overlap-coll32-1 (kind: micro32, ~10 min, all 8 hosts, no vLLM / checkpoint)

## Why
At decode every MoE layer runs 3 synchronous XLA collectives over the 32 chips: routing
all-gather (6 KB, ~45 us), hidden all-gather (1 MiB, ~30 us) and output all-reduce (1 MiB, ~37 us).
That is about 4.8 ms/step (B1 profile). They are latency-bound (8x off the ICI bandwidth floor),
which is consistent with XLA stepping round a 32-chip ring. A 4-chip host cannot show this, so
this job measures it on the real 4x8 slice before any e2e job uses it.

## Patches (also used by the later e2e jobs)
- `0101` H15b `MOE_PACKED_ROUTE_GATHER=1`: routing indices/weights ride in 128 extra u16 columns of
  the hidden-state all-gather (3 -> 2 collectives per layer). Pure data movement, exact.
- `0102` `kernels/collectives/oneshot.py` + flags `MOE_ONESHOT_GATHER=1` (needs the packed gather) and
  `MOE_ONESHOT_SCATTER=1`. Pallas kernels: one barrier, then one remote DMA from every chip to every
  peer (ICI routes multi-hop). The AG is exact. The RS sums the 32 peer blocks in f32 in fixed rank
  order (reduction-order change vs XLA). Only for <= 512 global MoE rows (decode buckets); prefill
  keeps XLA. Default off: with no flag the code path is unchanged.

- `0103` `MOE_TP_EXPERT_SPLIT=2` (load-time weight layout, default 1 = off). GMM_TP gives each chip a
  64-wide slice of every expert's intermediate dim; gmm_v2's fused gate/up needs 128-lane halves, so
  today each chip streams w13 with half zeros (134 MB FP4 + 33 MB scales per layer, half padding) and
  runs w2 with K=64 under a 128 tile. Split 2: chip c holds experts [128*(c//16), +128) x intermediate
  slice c%16 of width 128 (no padding). Same collectives (AG + RS over 32), same per-chip FLOPs, half
  the w13 bytes, ~3.4 GB/chip less HBM for weights. Changes partial-sum grouping (reduction order).
  CPU: relayout bit-exact (unit test), fused_moe_func on 32 virtual devices matches a float64 reference
  as well as the padded layout (MAE 0.00209 vs 0.00206), NaN rows from non-local experts are masked.

CPU evidence (32 virtual devices, interpret mode with race detection): AG bitwise == XLA AG; RS
bitwise == rank-ordered f32 sum; full fused_moe_func bitwise equal for packed and packed+oneshot-AG
across T=32/128/512 x plain/hash/num_valid_tokens. Deviceless v6e:4x8 AOT: kernels compile; with
both flags each MoE layer has 2 one-shot custom calls and 0 XLA collectives.

## What run.sh does (same command on every host)
Copies `$DW_TPU/tpu-inference`, applies the three patches (skips them if already applied), then runs
`micro32.py` three times, each as a fresh process with its own coordinator port (base+0/1/2) and
timeout (420/200/200 s):
1. `default` flags: per-op latency (64 dependent ops per call, median of 15 calls) of XLA AG (packed
   u16 and routing s32), XLA psum_scatter, XLA all-reduce+slice, one-shot AG and one-shot RS at 1 and
   4 rows per rank. Hardware exactness checks of the one-shot AG and RS. Then an 8-layer DSV4-shaped
   MoE block (real fused_moe_gmm.py, FP4 routed experts, bf16 shared expert, router, norm) at 32 and
   128 global rows for variants base / packed / packed+osAG / packed+osAG+osRS: us per layer, bitwise
   vs base, and one 3-call xprof trace per variant (process 0, `$OUT/prof/`).
   Plus `split2_osall_sef` (patch 0104 MOE_SHARED_EXPERT_TP: shared expert TP-sliced inside the routed
   reduce-scatter instead of the replicated per-chip shared expert; timing only), and `split2` and `split2_packed_osag_osrs` (random weights in the split layout, timing only), and
   `split_check`: one layer with E=32 hash-defined weights in both layouts (split arrays are an exact
   relayout of the base ones, verified on CPU), base vs split output difference on hardware.
2. `sc_ar`: SparseCore offload of all-reduce / reduce-scatter (H15a flags): the same collectives
   plus the MoE block (base, packed, split2), to see whether the async SC all-reduce hides behind the
   shared expert.
3. `sc_all`: SC offload for the all-gathers too.

Outputs on process 0: `$OUT/{default,sc_ar,sc_all}.json`, `.log`, `result.json` (merged), `prof/`.

## Reading the result
- Hardware exactness (must hold, or the one-shot path is not used further): `collectives.ag_oneshot_bitwise_eq_xla_r1/r4`
  true, `rs_oneshot_bitwise_eq_ordered_f32_r1/r4` true, `moe.r*_packed.bitwise_eq_base` and
  `moe.r*_packed_osag.bitwise_eq_base` true. `r*_packed_osag_osrs` is expected to differ slightly
  (`maxabs_vs_base` small, `finite` true).
- Speed: compare `ag_packed_u16_r*_oneshot_us` vs `_xla_us`, and `rs_bf16_r*_oneshot_us` vs
  `xla_rs`/`xla_ar_slice`. The MoE `us_per_layer` gives the per-layer saving: multiply by 43 for ms/step.
- Expert split: `moe.r*_split2.us_per_layer` vs `moe.r*_base.us_per_layer` (expect w13 ~half). On hardware,
  `split_check.r*_{xla,osall}` must be `finite` with `meanabs` ~1e-3 of `mean_out` or less (bf16 rounding
  of differently grouped partial sums); `bitwise` false is expected.
- A hang or error in one mode shows as `rc=124` (timeout) or an `*_error` key; the other modes still run.

## Next
The winner goes to an e2e job (`dsv4-moe-overlap-e2e-*`): exact acceptance for packed / packed+osAG;
reduction-order acceptance if osRS or SC offload is included. Profile on, to see the per-layer timeline.

## Rerun of coll32-1 (attempt 2 hung in part 2)
Cause: micro32.py ran the 3 profiled calls only on process 0, so the other hosts launched the next
variant's program while process 0 was still running the previous one (multi-controller program-order
mismatch → hang). Fixed: every process runs the same calls, only process 0 traces. Also: all hosts log to
their own $OUT/<mode>.log with timestamped progress lines; `timeout -k 10` (SIGKILL after TERM); the
hardware AG bitwise check reads the replicated result locally (attempt 2 hit
"non-fully-addressable ... tiled=True"); added a 1x128 u16 all-gather (latency floor).

Part 1 of attempt 2 completed (evidence: dw-tpu-work/dsv4-moe-overlap/evidence/coll32-1-attempt2-default-part1.json),
us/op on 32 chips: packed AG r1 XLA 31.0 / one-shot 25.6; r4 34.3 / 37.1; route AG 27.5; RS r1 27.8 / 26.3;
r4 41.9 / 38.2; one-shot RS bit-exact vs rank-ordered f32 on hardware. => every 32-chip collective costs
~25 us; removing one (packed gather) is worth more than making them one-shot. Part 2 (MoE layer variants,
split, shared-expert TP) and the SC-offload modes are what this rerun is for.
