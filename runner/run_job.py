#!/usr/bin/env python3
"""One command for stage -> Ray -> acceptance -> benchmark -> evidence (orchestrator only)."""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import shlex
import time
import uuid
from common import (ROOT, STATE, PYTHON, Transport, arguments, cluster_lock, collect, dump,
                    effective_env, extract_evidence, free_devices, fresh_result, job_stack, q, stop_script, unit_command)
from staging import apply_script, build_archive, stage_remote
from summarize import summarize


def environment(stack):
    text = '''export PATH=/opt/dsv4/.venv/bin:/opt/dsv4/tools:/usr/local/bin:/usr/bin:/bin
export PYTHONUNBUFFERED=1 HF_HOME=/opt/dsv4/hf-cache TPU_MULTIHOST_BACKEND=ray JAX_PLATFORMS='' RAY_USAGE_STATS_ENABLED=0
export VLLM_SERVER_DEV_MODE=1 RAY_DEDUP_LOGS=0 TZ=UTC
source /opt/dsv4/dw-tpu/env.sh
'''
    for key, value in effective_env(stack).items():
        text += f'export {key}={q(value)}\n'
    return text


def launch_ray(tr, hosts, env, launch):
    def ray(h):
        out = '/opt/dsv4/runs/' + launch
        cmd = env + f'exec ray start --block --node-ip-address={q(h["ip"])} ' + '--resources=\'{"TPU":4}\' '
        cmd += '--head --port=6379 --include-dashboard=false' if h == hosts[0] else '--address=' + q(hosts[0]['ip'] + ':6379')
        return f'mkdir -p {q(out)}\n' + unit_command('dsv4-ray', cmd, out + '/ray.log')
    tr.ssh(hosts[0], ray(hosts[0]))
    probe = "import socket,time\nfor i in range(120):\n try:\n  s=socket.create_connection(('127.0.0.1',6379),1);s.close();break\n except OSError: time.sleep(1)\nelse: raise RuntimeError('Ray head readiness timeout')"
    tr.ssh(hosts[0], PYTHON + ' -c ' + q(probe), timeout=150)
    tr.all(hosts[1:], ray)
    probe = "import ray,time\nray.init(address='auto')\nfor i in range(180):\n if ray.cluster_resources().get('TPU')==32 and sum(n['Alive'] for n in ray.nodes())==8: break\n time.sleep(1)\nelse: raise RuntimeError('Ray did not expose 32 TPUs on 8 hosts')"
    tr.ssh(hosts[0], env + PYTHON + ' -c ' + q(probe), timeout=240)


def serve_script(spec, env, out):
    args = ['vllm', 'serve', '/opt/dsv4/ckpt/dsv4-flash-full', '--served-model-name', 'dsv4-flash',
            '--tensor-parallel-size', '32', '--kv-cache-dtype', 'fp8', '--tokenizer-mode', 'deepseek_v4',
            '--enable-prefix-caching', '--distributed-executor-backend', 'ray',
            '--additional-config', '{"sharding":{"sharding_strategy":{"enable_dp_attention":true}}}']
    command = shlex.join(args) + ' --max-model-len "$MAX_LEN" --max-num-batched-tokens "$MNBT" --max-num-seqs "$MAX_SEQS" --gpu-memory-utilization "$GPU_MEM"'
    if spec.get('profile'):
        command += ' --profiler-config ' + q(json.dumps({'profiler': 'torch', 'torch_profiler_dir': out + '/traces'}))
    command += ' ' + shlex.join(spec['serve_args_extra'])
    return env + 'exec ' + command


def verify_live_logs(tr, hosts, out, ready, strict=False):
    # Ray can delay or omit worker forwarding. Check worker-local logs too, timestamp based.
    script = f'''{PYTHON} /opt/dsv4/dw-tpu/runner/audit_logs.py --out {q(out)}'''
    if ready is not None:
        script += ' --ready ' + q(ready)
    if strict:
        script += ' --strict-compiles'
    return tr.all(hosts, script, timeout=120)


def main():
    p = arguments(argparse.ArgumentParser(description=__doc__))
    p.add_argument('job', type=Path)
    p.add_argument('--prepare-only', action='store_true', help='CPU validate patches and build archive, no cloud/SSH')
    p.add_argument('--reuse-server', action='store_true', help='Require identical healthy live server and inherit passing acceptance; benchmark only, never stop server')
    p.add_argument('--keep-server', action='store_true', help='Keep serve/Ray after successful fresh job for explicit reuse')
    a = p.parse_args()
    if a.reuse_server and a.prepare_only: p.error('--reuse-server cannot be combined with --prepare-only')
    stack = job_stack(a.job); spec = stack[-1][1]
    if spec['kind'] != 'e2e':
        p.error('use run_kernel_jobs.py for kind=kernel/compile')
    launch = spec['id'] + '-' + ('DRYRUN' if a.dry_run else datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6])
    if a.prepare_only:
        if a.dry_run:
            print('CPU staging:', spec['base'], [s['id'] for _, s in stack])
        else:
            print(build_archive(stack, launch))
        return
    tr = Transport(a)
    with cluster_lock(a.dry_run):
        if a.reuse_server:
            from server_reuse import reuse_server
            return reuse_server(a, stack, launch, tr)
        hosts = tr.discover(); out = '/opt/dsv4/runs/' + launch
        from cold_start import REVISION
        tr.all(hosts, f'test \"$(cat /opt/dsv4/ckpt/CHECKPOINT_VERIFIED)\" = {q(REVISION)}')
        archive = STATE / (launch + '.tar.gz') if a.dry_run else build_archive(stack, launch)
        # Validate and package before touching a live service.
        collect(tr, hosts, STATE / 'copies' / (launch + '-before'))
        tr.all(hosts, stop_script(), timeout=180)
        stage_remote(tr, hosts, archive, launch)
        env = environment(stack)
        result_dir = a.job.resolve() / 'result'
        if not a.dry_run:
            fresh_result(a.job.resolve(), launch)
            dump(STATE / 'current.json', {'job': str(a.job.resolve()), 'id': spec['id'], 'kind': 'e2e', 'launch': launch, 'hosts': hosts})
            dump(result_dir / 'effective-job.json', dict(spec, env=effective_env(stack)))
        finished = False
        failure = None
        strict = 'post-ready model/runner JIT compilation' in spec.get('abort_if', [])
        try:
            tr.all(hosts, apply_script(stack), timeout=1800)
            tr.all(hosts, f'mkdir -p {q(out)}\nprintf "%s\\n" {q(launch)} > {q(out + "/launch-id")}\n' + free_devices())
            tr.ssh(hosts[0], 'printf %s ' + q(json.dumps(dict(spec, env=effective_env(stack)))) + ' > ' + q(out + '/effective-job.json'))
            launch_ray(tr, hosts, env, launch)
            # A successful health check must not come from an old API process.
            probe = "import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(('127.0.0.1',8000)); s.close()"
            tr.ssh(hosts[0], PYTHON + ' -c ' + q(probe))
            tr.ssh(hosts[0], unit_command('dsv4-serve', serve_script(spec, env, out), out + '/serve.log'))
            controller = env + f'exec {PYTHON} /opt/dsv4/dw-tpu/runner/remote_job.py /opt/dsv4/dw-tpu/jobs/{spec["id"]} --out {q(out)} --launch-id {q(launch)}'
            tr.ssh(hosts[0], unit_command('dsv4-job', controller, out + '/controller.log'))
            if a.dry_run:
                tr.ssh(hosts[0], f'cat {q(out + "/state.json")}; tail -40 {q(out + "/controller.log")}')
                verify_live_logs(tr, hosts, out, '<ready_unix>', strict)
            else:
                deadline = time.monotonic() + spec['est_minutes']*90 + 180
                while True:
                    status = tr.ssh(hosts[0], f'if test -f {q(out + "/state.json")}; then cat {q(out + "/state.json")}; else echo \'{{"status":"starting"}}\'; fi')
                    state = json.loads(status)
                    print(json.dumps(state), flush=True)
                    if state.get('launch_id', launch) != launch:
                        raise RuntimeError('launch-id mismatch in controller state')
                    verify_live_logs(tr, hosts, out, state.get('ready_unix'), strict)
                    if state['status'] in ('passed', 'failed', 'gate_failed_benchmarked'):
                        if state['status'] == 'failed':
                            raise RuntimeError(state.get('error', 'job failed'))
                        finished = True; break
                    if time.monotonic() >= deadline:
                        raise TimeoutError('controller wall timeout')
                    time.sleep(15)
        except BaseException as exc:
            failure = repr(exc)
            raise
        finally:
            # Stop producers, then collect final logs; copy before any future source change.
            try:
                if not (a.keep_server and finished):
                    tr.all(hosts, stop_script(), timeout=180)
            except BaseException as exc:
                failure = (failure or '') + '\ncleanup: ' + repr(exc)
                finished = False
            try:
                copies = collect(tr, hosts, STATE / 'copies' / launch)
                if not a.dry_run:
                    extract_evidence(copies / (hosts[0]['name'] + '.tar.gz'), result_dir, 'runs/' + launch)
                    dump(result_dir / 'copies.json', {'directory': str(copies), 'hosts': hosts, 'launch': launch})
            except BaseException as exc:
                failure = (failure or '') + '\ncollection: ' + repr(exc)
                finished = False
            if not a.dry_run:
                if not finished:
                    path = result_dir / 'state.json'
                    state = json.loads(path.read_text()) if path.exists() else {}
                    state.update(status='failed', error=failure, launch_id=launch)
                    dump(path, state)
                summarize(result_dir, dict(spec, env=effective_env(stack)))
            else:
                print('# extract head runs/' + launch + ' into ' + str(result_dir) + '; summarize to summary.json + SUMMARY.md')
        if failure:
            raise RuntimeError(failure)
        print('Dry run complete.' if a.dry_run else 'Job complete: ' + str(result_dir))

if __name__ == '__main__':
    main()
