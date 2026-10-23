# Runtime flags for DeepSeek V4 Flash on a 32-chip TPU v6e slice (4x8, 8 x ct6e-standard-4t).
# Source on every host (driver and Ray workers) before starting Ray and vLLM.
# This is the fastest measured configuration plus the dense FP8 GEMM kernel.

# libtpu on GCE-created ct6e VMs has no tpunetd daemon.
export LIBTPU_INIT_ARGS="--enable_tpunetd_client=false ${EXTRA_LIBTPU_ARGS:-}"
export NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm VLLM_MLA_DISABLE=0

# FP4 routed experts (W4A16 via the tokamax fp4-emulation patch).
export MOE_REQUANTIZE_WEIGHT_DTYPE=fp4 MOE_REQUANTIZE_BLOCK_SIZE=32

# Numerics-neutral layout knobs from the bring-up.
export SLICE_ROPE_CACHE=1 ROPE_CACHE_ROW_MAJOR=1 HASH_TABLE_ROW_MAJOR=1 MOE_ROUTE_PADDING_TO_EXPERT0=1

# Startup: precompile every bucket, cache compiled programs, allow slow first steps.
export SKIP_JAX_PRECOMPILE=0 NUM_PRECOMPILE_WORKERS=8
export VLLM_XLA_CACHE_PATH=${VLLM_XLA_CACHE_PATH:-/opt/dsv4/xla-cache}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export DP_SCHED_BATCH_PREFILL=1 DP_SCHED_BATCH_PREFILL_FLUSH_TIMEOUT_MS=30000
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# 0003: per-rank token buckets. 4 is the decode bucket (~1 real token/rank at 32 agents).
export TPU_TOKEN_PADDINGS_PER_DP="4,16,512"
# 0004: exact-size CSA SparseCore gather.
export DSV4_H17_SMALL_GATHER=1
# 0005: multi-host-safe async scheduling (serve WITHOUT --no-async-scheduling).
export TPU_ASYNC_REPLICATE_OUTPUTS=1
# 0006: dense FP8 GEMM kernel (+21% tok/s/user on the 32-agent benchmark).
export DSV4_DENSE_FP8_GEMM=1

# vLLM sizing used for the measured runs.
export MAX_SEQS=32 MAX_LEN=16384 MNBT=512 GPU_MEM=0.76
