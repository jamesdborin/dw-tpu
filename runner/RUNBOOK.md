# Reservation runner: 20–21 October 2026

These commands are for the orchestrator. Worker agents prepare jobs and run CPU
checks only. This runner has been tested locally; OS Login/IAP, provisioning,
TPU startup, and end-to-end performance still require the reservation run.

Defaults: project `tech-426212`, zone `us-east5-b`, reservation
`dsv4-v6e32-east5b-20261020`, MIG `dsv4-day-32-20261020`, eight
`ct6e-standard-4t` VMs, one 4x8 / 32-chip slice, 1 TB boot disks. Every gcloud
invocation sets `CLOUDSDK_CONFIG=$HOME/.config/gcloud-tpu-ops`. The runner never
reads/copies the service-account key or switches to personal credentials.

## Cold start to first result

Run from the dw-tpu root. Local requirements are Python >=3.10, git, tar, patch,
gcloud, and >=20 GiB free disk. No local JAX or GPU is required. Stage building
uses existing local pinned git objects; it never edits the shared submodules.
The pristine tokamax 0.0.16.dev20261009 package is taken from uv's package cache
(or downloaded as a wheel into runner/.state/wheels using pip).

1. Review the commands without contacting GCP:

   ```bash
   runner/provision.sh create --dry-run
   python3 runner/cold_start.py --dry-run
   python3 runner/run_job.py jobs/baseline-1 --prepare-only
   python3 runner/run_job.py jobs/baseline-1 --dry-run
   ```

   `--prepare-only` builds a real CPU-validated archive. `--dry-run` performs no
   cloud calls or result writes. Without `--hosts`, host/IP placeholders are
   printed because discovery itself would require GCP. For exact instance/IP
   commands use `--hosts runner/.state/hosts.json` after discovery. An inventory
   is a JSON array of eight objects with `name`, `ip`, `status: RUNNING`, and
   optional `zone` (the old hosts.json networkInterfaces form also works).

2. Create the reservation-bound template/policy/MIG and wait for ACTIVE topology:

   ```bash
   runner/provision.sh create
   runner/provision.sh status
   ```

   Provisioning waits up to two hours for FULFILLED and ACTIVE. Typical VM and
   topology startup after capacity is available was about 5–10 minutes. Existing
   resources are verified and reused; permission errors never trigger a blind
   create/delete. The new template retains the proven default compute VM
   service account and cloud-platform scope, and enables OS Login. An existing
   approved template can be selected with `--template` if it matches the new
   reservation, shape, and service account.

3. Upload, install the pinned runtime, run the all-host psum, and start downloads:

   ```bash
   python3 runner/cold_start.py --download-in-background
   ```

   Setup is approximately 10–25 minutes, psum <5 minutes. The expected smoke
   result is four copies of **496** on each host, with 32 global / 4 local
   devices. Setup uses requirements-tpu.txt, editable vLLM/tpu-inference,
   apply_patches.sh and pristine tokamax; the old payload's source/runtime is
   not substituted for the declared dw-tpu commit. Persistent XLA cache is
   `/opt/dsv4/xla-cache`, outside staged source trees.

   `--payload` defaults to the preserved
   `../dsv4-tpu/bench-a1/payload.tar.gz` (384 MB). Only its exact `bench-a1/uv`
   and `bench/batchbench-agent` binaries are extracted. On another workstation,
   copy this archive or point `--payload` at it. All Python probes, checkpoint
   manifest, numerical references, profile replay payloads and validators are
   already vendored under runner/ with source paths/hashes in
   vendor-manifest.json. No GCS source was needed by the proven recipe.

   Checkpoint source is `deepseek-ai/DeepSeek-V4-Flash`, pinned revision
   `60d8d70770c6776ff598c94bb586a859a38244f1`. Downloads use the preserved
   checkpoint-manifest.json, verify every file length and every safetensors
   SHA256, and verify shard coverage against the index. Budget roughly
   15–45 minutes for transfer plus hashing; network speed is not guaranteed.
   A separate `.venv-download` and stable tools directory keep downloads
   independent of job source changes. Logs: `/opt/dsv4/logs/checkpoint.log`.

4. During those downloads, run independent kernel jobs or a micro32 job:

   ```bash
   python3 runner/run_kernel_jobs.py jobs/dsv4-buckets-shapes-1 jobs/dsv4-dense-gemm-kernel-1
   # Only when prepared/queued by the MoE worker:
   python3 runner/run_micro32.py jobs/dsv4-moe-overlap-coll32-1
   python3 runner/cold_start.py --wait-download
   ```

   Kernel/compile jobs run one per host, up to eight at once; extra jobs form
   another batch. Each gets 1.5 x est_minutes (plus a 20-second termination
   grace), a clean source + tokamax stack, `$DW_TPU` and empty `$OUT`.
   Multihost/Ray environment is cleared. Kernel result.json and logs are copied
   to the job's result/; runner.json records process exit/copy status separately
   from the job's own numerical/performance verdict.

   `micro32` uses all eight hosts exclusively, with `JAX_COORDINATOR_ADDRESS`,
   `JAX_PROCESS_ID`, `JAX_NUM_PROCESSES=8`, and the proven TPU process grid/address
   environment. Each run.sh must call jax.distributed.initialize itself and
   write result.json on rank 0. All rank logs are archived. Hard runtime cap is
   15 minutes, including JAX initialization, with termination grace afterward.
   `run_micro32.py --prepare-only` validates its staging without GCP.

   Kernel/micro32/e2e commands hold an exclusive local cluster lock and verify
   no VFIO holders before changes. They cannot overlap model loading/serving.
   Background downloads are the sole exception: they perform disk/network work
   and hold no device or source/runtime resources. Without
   `--download-in-background`, cold_start.py waits for checkpoint completion
   itself before returning; the single command remains usable for simple runs.

5. Run the first full-model job:

   ```bash
   python3 runner/run_job.py jobs/baseline-1
   python3 runner/status.py             # another terminal; current controller + logs
   ```

   Every host must have the pinned CHECKPOINT_VERIFIED marker. The runner
   copies all evidence before source changes, stops its systemd service groups
   and Ray, checks VFIO holders, uploads a hashed standalone git tree, applies
   the base series + dependency patches + job patches, reinstalls editables,
   launches Ray head/workers, verifies 8 live nodes / 32 TPU resources, then
   starts vLLM. Job env overrides are sourced on every Ray worker and driver.
   Dependency environments inherit in dependency order; the leaf job wins.
   serve_args_extra is an argv array applied only to the leaf job.

   Startup timeout defaults to 45 minutes; the job's supported startup abort
   rule can lower/change it. Total controller cap is est_minutes x 1.5. Jobs
   abort on launch mismatch, timestamped post-ready RESOURCE_EXHAUSTED, unknown
   allocation timestamps/unrecovered allocation errors, or failed acceptance.
   Recovered allocation retries whose event timestamps precede readiness are
   retained and classified even when Ray forwards them late. Worker-local logs
   are checked as well as forwarded head logs. No metrics are collected from a
   failed acceptance as a valid benchmark.

6. Review `jobs/baseline-1/result/{summary.json,SUMMARY.md}`. Full copies of all
   eight hosts, Ray logs, and SHA256s live in runner/.state/copies; copies.json
   links the exact archives. Repeated executions move the previous result into
   runner/.state/result-history before writing fresh output. Summary includes
   startup, compiler-message wall span, KV blocks/HBM, buckets, acceptance,
   median per-user output tok/s, aggregate output tok/s, logical TPM including
   cached history, APC hit rate, preemptions and spec acceptance when exposed.
   Missing measurements are null, never invented zeroes. Task compile durations
   overlap across workers and must not be mistaken for wall time.

7. Copy or release explicitly:

   ```bash
   runner/copy_off.sh
   python3 runner/teardown.py --stop-only
   runner/provision.sh delete   # delegates copy-before-delete to teardown.py
   ```

   Deletion targets only this MIG, retaining the future reservation, template
   and policy. Failed copy-off blocks deletion. Do not manually delete VM disks
   before preserving results. Staged sources/caches are retained on VM disks
   until MIG deletion; local archives stay under .state. All public commands
   above, including status, summarize, micro32 and teardown, accept --dry-run.

## Gates and job extensions

The jobs/README.md contract is supported, plus micro32 and optional
sampling-equivalent `acceptance.sh`. Dependencies must use the same immutable
base commit and cycles are rejected. Patch files sort by name within each
repository and are checked cumulatively on CPU; all three repositories
(tpu-inference, vllm, tokamax) are supported. Kernel result gates and promotion
conditions in NOTES.md remain the orchestrator's decision; depends_on stacks
sources but does not automatically assert an earlier job's success.

Exact: saved B2/h17-model and H4-r2 greedy tokens + selected logprobs on all
three lengths, exact H4-r2 teacher-forced 17/300 prompt logprobs, repeat cache
identity, and suffix causality. Reduction-order: preserved GPU reference with
MAE <=1.10 x original B0 and top1 matches >=B0-2, greedy 32-token identity on
17/300, repeat cache identity and suffix causality zero. Full prompt-logprob
paths are required by these gates; do not silently remove them to pass a
compile check. `post-ready model/runner JIT compilation` enables an additional
strict timestamp/trace gate; the worker must precompile the required probes.

Sampling-equivalent: all-length greedy identity, GPU-reference/repeat/causality
checks, plus job-local `acceptance.sh` with the venv active and OUT/DW_TPU set.
It must write `$OUT/sampling-equivalent.json` containing `passed: true` and
`distribution_preserved: true` with actual supporting evidence. Greedy
agreement alone cannot satisfy distribution preservation. Failed scripts or
missing evidence prevent benchmarks. A checked distribution proof/test should
be documented in the job NOTES.md and reviewed by the orchestrator.

Recognized e2e abort labels: post-ready RESOURCE_EXHAUSTED, launch-id mismatch,
startup > N min, post-ready model/runner JIT compilation, reduction-order gate
failure, acceptance gate failure. Unknown labels fail validation instead of
being ignored. Kernel scripts own numerical/performance aborts in their specs.

Profile=true replays preserved 32-agent decode payloads after the benchmark and
uses the proven native vLLM/JAX start/stop profile controller for 3 seconds.
Profiles and replay metadata are copied with results. Profile metrics are not
included in benchmark measurements.

## OS Login/IAP failure path

The service-account login path is untested on this new reservation. gcloud
chooses the OS Login POSIX username; no local username is hardcoded. SSH uses
IAP, batch authentication, a 30-second connection timeout and keepalives, with
no automatic mutation retries. On failure, inspect the emitted error. The ops
identity needs compute.osAdminLogin (sudo setup), iap.tunnelResourceAccessor,
compute metadata read permissions, and iam.serviceAccountUser on the attached
VM service account. IAP TCP/22 reachability must already be allowed. See
[OS Login setup](https://docs.cloud.google.com/compute/docs/oslogin/set-up-oslogin)
and [service-account SSH](https://docs.cloud.google.com/compute/docs/tutorials/service-account-ssh).

Fallback is an orchestrator-approved repair to IAM/OS Login/IAP or a separately
prevalidated template with the same reservation binding and VM identity. The
runner does not automatically disable OS Login, add metadata keys, open public
SSH, or use personal credentials. Do not recreate a mismatched live template
without first diagnosing it. Provisioning flags follow the
[instance-template CLI reference](https://docs.cloud.google.com/sdk/gcloud/reference/compute/instance-templates/create)
and the proven 16 October reservation recipe.

## CPU verification

```bash
python3 -m unittest discover -s runner/tests -v
python3 -m compileall -q runner
python3 runner/run_job.py jobs/baseline-1 --prepare-only
python3 runner/run_kernel_jobs.py jobs/<kernel-id> --prepare-only
```

Dry-run output is a command plan, not evidence of working SSH, TPU kernels,
correctness, or measured throughput. Before the window, the orchestrator should
verify the selected template/IAM and inspect the dry-run plan. During the
window, successful smoke, hash verification, live gates and benchmark
validation provide the actual hardware evidence.
