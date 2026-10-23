# Fine-bucket dense FP8 qualification

Run on one host with four local v6e chips. No checkpoint or serving process.
Four threads in one JAX process pin independent streams to SingleDeviceSharding;
there are no collectives or distributed initialization calls. The orchestrator
must provide its normal isolated single-host kernel environment. The worker
never creates/accesses a VM. A CPU-only schema/dispatch/interpreter check and
deviceless v6e compilation preceded submission.

246 cases: six profiled FP8 projection shapes × M=1..32,33,64,65,128,129,256,257,
512,513. This covers the fine bucket list, k+1 verification rows (including the
power-of-two-plus-one boundaries), and prefill. The six K,N pairs are wq_b
1024×32768; wo_b 8192×4096; gate_up 4096×4096; down 2048×4096; fused_wqa_wkv
4096×1536; indexer_wq_b 1024×8192. Existing q_a/kv is already fused. wo_a is
BF16 grouped reverse-RoPE+projection, not this dispatch; compressors and routing
are separate BF16/fused operators. See SHAPES.md in the worker directory.

New entries are **candidates**, not claimed tuned results. Existing 0006 entries
are unchanged. Whole-M blocks handle odd row counts, so there is no scheduler
padding or per-row quantization. The known losing prefill fused-q_a/kv path
retains baseline (also conservatively at 64/128/256 pending measurement).

Two seeds: BF16 activations; original FP8 weights; FP32 scale factors (one
log-uniform case, one exact power-of-two case). Float64 reference evaluates
rounded inputs and original FP32 scales. Every case reports max abs/relative,
RMS, relative L2, finiteness, repeatability, and exactness vs unchanged current
Tokamax GMM. Pass requires RMS <=1.10× current and max abs <=2× current, per seed.
No FP8 activation quantization, scale approximation, or BF16 accumulation.

Timing: four alternating trials × seven samples ×32 dependent calls. Compilation
is excluded. All operations are pinned to one chip; device placement is asserted.
`baseline_us` is flag-off GMM; `prior_flag_us` is the existing 0006 dispatch;
`new_us` is the candidate. For inherited entries, old and new are bit-exact and
use the same timings; otherwise the old flag falls back to baseline. This avoids
redundant compilation while making both comparisons explicit. Effective GB/s
counts logical input/scale/output bytes once, not hardware counter traffic.
Four simultaneous host streams can add dispatch noise: require >5% improvements
before changing near-tied choices. Results are atomically checkpointed per case
in OUT/result.json. Errors remain visible and cause a nonzero final exit.

The hard gate before e2e: all cases complete and pass numerically; no newly
enabled case regresses materially (>5%). If a candidate is slower, change its
dispatch to baseline (or tune and re-run) before running the fine-bucket e2e job.
The JSON numerical `passed` field alone is insufficient to approve performance.
Expected duration 35 min, hard cap 52.5 min. Preserve partial JSON if interrupted.

Additional unchanged-path control: after dense qualification, `woa_check.py`
checks all41 M values for grouped BF16 wo_a (eight4096×1024 products with reverse
RoPE). It reports the float64 grouped reference error, repeatability, and timing
under `result.json.wo_a`. Old/new are the identical unchanged function, so their
times alias and bit-exactness is by construction; this is a new-row regression
control, not a new optimization. Relative L2 must remain below1% and outputs
finite/repeatable. A wo_a failure also holds fine e2e until diagnosed. Other
unquantized routing/compressor/lm_head paths remain outside this FP8 kernel task.
