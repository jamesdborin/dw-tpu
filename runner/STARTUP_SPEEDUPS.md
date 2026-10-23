# Startup speed-ups

## Explicit live-server reuse (implemented)

```
python3 runner/run_job.py jobs/pd-scale-specdec --keep-server
python3 runner/run_job.py jobs/pd-final-decode --reuse-server
```

`--keep-server` preserves serve/Ray after a successful fresh run; fresh runs
still stop services on failure. `--reuse-server` always leaves serve/Ray alone,
even if the benchmark fails; it stops only its benchmark controller. Never
assume a server is usable after failure: the next reuse must pass all checks.
An already-running older local launcher still executes its old stop-in-finally
code. The orchestrator must preserve that server separately; changing files
cannot change an already-running Python process. No automatic fallback restart.

Reuse compares the local originating stage manifest, every host's verified
staged patch bytes, and the requested job. Identity includes canonical base
commit, ordered patches for all three repositories (including profile patches),
base env.sh hash, effective stacked env, serving argv and profile capability.
All env entries count: no guessed prefix exclusion for unknown environment
variables. Top-level pd/bench configurations and job IDs are not server identity.
The serving recipe functions (environment, Ray launch, serving argv) must also
match the staged recipe. Renaming identical patch files/jobs is harmless if
application order and bytes remain identical.

All Ray units and the head serve unit must be active, /health must return200,
the stage symlink and serve unit must belong to the recorded server launch,
and no benchmark controller may be active. Units must not have restarted after
acceptance. The originating job must have finished successfully with acceptance
passed, under the same gate mode and job-local acceptance script. Informational
gate-failed runs cannot seed inherited acceptance. Different identities produce
a field-level diff; unhealthy/missing/stale evidence refuses reuse.

The remote effective-job.json is checked against the original manifest. Older
launchers wrote it only locally: after matching immutable local/remote manifests,
the reuse command reconstructs that missing remote file from the verified staged
spec and stacked env. It never uses a subsequently edited old job.json as truth.

An isolated client bundle is uploaded under runs/<new-launch>/client. There is
no server stop, source staging/symlink replacement, package install, Ray start,
serve start, model load, compile warmup, or acceptance rerun. The new result keeps
its own state and records `acceptance_inherited_from=<origin>` with the remote
evidence path; raw originating evidence remains in the collected host archives.
The current pointer records both benchmark launch and originating server launch,
so successive benchmark-only jobs continue to inherit the same accepted server.
Profiling requires a server already launched with the same profile capability;
its profiler output directory remains the original server's configured directory
and is preserved in the all-runs host archives.

Dry-run prints preflight and benchmark launch commands without SSH or mutations.
Local tests use toy manifests and mocked health/units/controllers, including
failure cleanup and refusal paths; real deployment validation is orchestrator-only.

## Next reservation: processed-weight cache and queue policy

The storage prototype is `weights_cache.py`; it is NOT wired into serving. Do
not introduce a weight-cache hook into the current serving runs. The cache must key
checkpoint identity, ordered processing patches, quantization/layout settings,
library ABI, topology, host index, shard ownership and dtypes; publish each entry
atomically only after all shard hashes and metadata are complete. Load only local
addressable shards and preserve original sharding/layout when placing on devices.
Measure safetensors reads, CPU/device postprocessing and actual synchronized
host-to-device transfer separately before claiming a load-time speed-up.

Queue policy: retain a successful server and choose explicit reuse only after
identity comparison. A differing identity is a planned cold start, never a
silent fallback after a reuse error. Collect before any replacement; cache and
acceptance provenance must survive across jobs. Final teardown stays explicit.

### What the existing logs actually measure

`load_timing.py` streams serve logs and pairs initialization, checkpoint-loader
and wrapper timings per worker PID. These are eight-worker medians, not sums:

| Source | Init (s) | Checkpoint loader (s) | Storage-to-TPU wrapper (s) | Unseparated post-load remainder (s) |
|---|---:|---:|---:|---:|
| jobs/pd-base0/result/serve.log, first load | 11.303 | 29.180 | 595.805 | 555.287 |
| jobs/pd-specdec/result/serve.log, target | 11.335 | 29.210 | 599.735 | 559.181 |
| jobs/pd-specdec/result/serve.log, draft | 0.264 | 2.360 | 17.375 | 14.759 |

Main-model wrapper ranges are591.74–615.93s (base) and594.02–614.01s
(spec target). For example, spec target worker PID329031 logs load start at
13:07:54, initialization11.273s, then checkpoint-loader29.01s at13:08:34.
The same worker's main wrapper completes roughly ten minutes after load start.
The first checkpoint progress bar traverses46 shards in~28s; this is NOT proof
that all disk I/O alone costs28s. The iterator includes parameter assignment and
may include incremental processing. Nor is the555–559s remainder an isolated
FP4 cost: it includes quant processing, sharding/H2D and wrapper work. These logs
do not separately time FP4 re-encoding and device transfer or explicitly wait
for asynchronous device work at every boundary. A precise three-way split is
unavailable; no estimated split is presented as a measurement.

Reproduce without model imports:
```
prlimit --as=4000000000 -- python3 runner/load_timing.py \
  jobs/pd-specdec/result/serve.log --out /tmp/spec-load-timing.json
```

Next-reservation timing points, inspected in the pinned loader path:

1. `models/vllm/vllm_model_loader.py`: instrument checkpoint iterator/read and
   assignment separately; preserve incremental loader behavior and record
   process CPU time plus wall time per module, host and target/draft role.
2. `layers/vllm/quantization/mxfp4.py::process_weights_after_loading`: put a
   synchronized phase around `process_mxfp4_moe_weights` (dequantize, requantize,
   reorder), separate from `shard_moe_weights`. Block resulting arrays before
   ending either span; include compile time separately on first invocation.
3. `layers/common/process_weights/moe_weights.py`: instrument
   `general_device_put`; synchronize destination shards to time transfer,
   avoiding double-counting work already charged to FP4 processing.
4. `models/vllm/vllm_model_wrapper.py`: instrument `shard_model_to_tpu` and final
   rope/hash layout conversion. Time a final tree-wide readiness barrier before
   publishing load completion. Collect per-host spans and maximum critical-path
   time; do not sum eight hosts as elapsed time.

`load_timing.timed_phase` provides an injectable synchronization/JSON event hook.
It does not import JAX or install monkey patches. Supply explicit JAX readiness
callbacks in the future job-local instrumentation patch. Instrumentation itself
can serialize asynchronous work, so compare total startup before/after it.

### Processed-shard storage prototype

`weights_cache.publish` takes a lazy stream of processed local shard metadata
and byte chunks. It writes at most8MiB per chunk, hashes incrementally, fsyncs,
and atomically publishes a per-host cache directory only when the complete
expected shard schema and byte counts match. Interrupted writes never expose a
complete cache. A per-key host-local lock serializes writers. `restore` verifies
EVERY shard before invoking any consumer, then maps one shard read-only at a
time. No pickle or full-model materialization is used. Raw bytes plus explicit
logical dtype/layout metadata preserve custom BF16/FP4 storage without assuming
that NumPy's opaque dtype serialization preserves its meaning.

The key includes checkpoint content hash, processing/source hash, quantization,
layout version, runtime versions/ABI, topology, host count/index, target/draft
role and the full expected shard-schema hash. The processing hash must include
all base/job patch bytes plus weight-affecting environment settings. The schema
includes tensor name, dtype, global/local shape, global slice, logical local
device slot, layout and storage byte count. Supply endianness/storage packing in
layout metadata, and architecture/runtime ABI in runtime_versions. Never key
only by checkpoint path, model name or timestamp.

Future JAX adapter, deliberately not enabled today:

- On a cold load, after all postprocessing/layout decisions, iterate each
  parameter's `addressable_shards`, one local shard at a time. Explicitly
  materialize that shard only; pass its byte-view to `local_array_chunks`.
  Do not call `device_get` on a global model tree. Preserve logical dtype and
  byte packing separately (custom dtype byte views may need `view(uint8)`).
- On a hit, first instantiate a **metadata-only processed model skeleton** and
  its static-forward context. Reconstruct local arrays from mapped storage using
  the stored dtype/shape, place them on the corresponding local device with the
  expected layout, block transfer before unmapping, and assemble global arrays
  with the original sharding. Do not rerun dequantization/requantization on hit.
  Skeleton construction/parameter attachment remains an integration task; the
  prototype does not claim to replace the existing vLLM loader yet.
- Reproduce draft/shared-parameter aliases instead of silently duplicating them;
  collectives and model graph metadata must match the uncached path. Include
  generated rope/hash buffers or regenerate them deterministically under a key
  that includes their shape/layout settings (including MAX_LEN).
- All hosts agree cache-hit/miss and schema/topology before taking a fast path.
  Mixed cached/uncached collective order is forbidden: on any miss use the cold
  path on ALL hosts. One incomplete host cache cannot establish a valid hit.
- Cold-load and cached-load exact token/logprob, repeat/cache identity, suffix
  causality and spec sampling gates must pass. Test the real request path on a
  toy multi-host fixture first, then measure startup/HBM on TPU. No real-shape
  CPU tests on titan-6. Compare per-host bytes, device-transfer time and actual
  ready time; cache save overhead occurs once and can exceed a short run's gain.

Current validation covers raw processed-shard roundtrip, byte-for-byte identity,
key invalidation, truncated writes, missing schema entries, corruption before
any consumer runs, and bounded chunk size. It is a storage/restore prototype,
not evidence of an end-to-end model startup speed-up.

### Queue prototype (implemented, opt into the new command)

```
python3 runner/queue_jobs.py --dry-run --fresh-first \
  jobs/pd-scale-specdec jobs/pd-final-decode
python3 runner/queue_jobs.py --fresh-first \
  jobs/pd-scale-specdec jobs/pd-final-decode
```

This plans a cold `--keep-server` start, then `--reuse-server` for identical
bench-only jobs. A changed identity is shown as an explicit cold-start step with
its diff; failed reuse/health/acceptance stops the queue without fallback. Without
`--fresh-first`, a matching current manifest selects reuse for the first job,
subject to all live checks. Stale/inactive current state refuses reuse; request
`--fresh-first` explicitly if a cold start is wanted. The final server remains
running, and reservation teardown remains the orchestrator's explicit action.
The new queue does not modify any already-running external orchestration script.

The cache prototype verifies checksums in a full streaming pass before mapping
shards, so a cold filesystem cache can read bytes twice. Include that cost in
measurements. Budget checkpoint + processed cache + temporary publication space
before enabling the hook; ENOSPC aborts publication, and no automatic eviction
of a live cache is implemented. Prototype API use is covered by
`runner/tests/test_weights_cache.py` with an8-byte processed tensor; that is the
only cache data generated during this CPU-only task.
