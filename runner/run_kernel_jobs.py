#!/usr/bin/env python3
"""Run kernel/compile jobs in batches of eight, one bounded job per host."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import uuid
from common import (STATE, PYTHON, Transport, arguments, cluster_lock, collect, dump, extract_evidence,
                    free_devices, fresh_result, job_stack, q, stop_script, unit_command)
from run_job import environment
from staging import apply_script, build_archive, stage_remote


def main():
    p = arguments(argparse.ArgumentParser(description=__doc__))
    p.add_argument('jobs', nargs='+', type=Path)
    p.add_argument('--prepare-only', action='store_true', help='CPU stage/patch checks only; no cloud access')
    a = p.parse_args(); stacks = [job_stack(path) for path in a.jobs]
    if any(s[-1][1]['kind'] not in ('kernel', 'compile') for s in stacks):
        p.error('all jobs must be kind=kernel or compile')
    if len({s[-1][1]['id'] for s in stacks}) != len(stacks):
        p.error('duplicate jobs')
    batch = 'kernels-' + ('DRYRUN' if a.dry_run else time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:6])
    archives = [(STATE / (batch + '-' + s[-1][1]['id'] + '.tar.gz') if a.dry_run else build_archive(s, batch + '-' + s[-1][1]['id'])) for s in stacks]
    if a.prepare_only:
        print('\n'.join(map(str, archives))); return
    tr = Transport(a); errors = []
    with cluster_lock(a.dry_run):
        hosts = tr.discover()
        collect(tr, hosts, STATE / 'copies' / (batch + '-before'))
        tr.all(hosts, stop_script(), timeout=180)
        def one(item):
            h, stack, archive = item
            job, spec = stack[-1]; launch = batch + '-' + spec['id']; out = '/opt/dsv4/runs/' + launch
            failure = None
            if not a.dry_run:
                fresh_result(job, launch)
            try:
                stage_remote(tr, [h], archive, launch)
                tr.ssh(h, apply_script(stack), timeout=1800)
                tr.ssh(h, f'mkdir -p {q(out)}\n' + free_devices())
                limit = int(spec['est_minutes'] * 90)
                env = environment(stack) + 'unset TPU_MULTIHOST_BACKEND TPU_PROCESS_BOUNDS TPU_CHIPS_PER_PROCESS_BOUNDS TPU_PROCESS_ADDRESSES TPU_PROCESS_PORT CLOUD_TPU_TASK_ID\nexport JAX_PLATFORMS=tpu\n' + (
                       # Single host of a multi-host slice: stop libtpu waiting for the other 7 hosts.
                       'export TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_ADDRESSES=localhost:8476 TPU_PROCESS_PORT=8476 CLOUD_TPU_TASK_ID=0 TPU_SKIP_MDS_QUERY=1 TPU_ACCELERATOR_TYPE=v6e-4 TPU_HOST_BOUNDS=1,1,1\n')
                command = env + f'export DW_TPU=/opt/dsv4/dw-tpu OUT={q(out)}\ncd /opt/dsv4/dw-tpu/jobs/{spec["id"]}\nexec timeout --signal=TERM --kill-after=20 {limit} bash run.sh'
                # --wait gives exit code; systemd kills the entire control group when timeout exits.
                unit = unit_command('dsv4-kernel', command, out + '/run.log')
                unit = unit.replace('systemd-run --unit=', f'systemd-run --wait --property=RuntimeMaxSec={limit} --unit=')
                tr.ssh(h, unit, timeout=limit + 180)
            except Exception as exc:
                failure = repr(exc)
            finally:
                try:
                    tr.ssh(h, stop_script(), timeout=180)
                except Exception as exc:
                    failure = (failure or '') + '\nstop: ' + repr(exc)
                try:
                    copies = collect(tr, [h], STATE / 'copies' / launch)
                    if not a.dry_run:
                        dest = job / 'result'; dest.mkdir(exist_ok=True)
                        extract_evidence(copies / (h['name'] + '.tar.gz'), dest, 'runs/' + launch)
                        if not (dest / 'result.json').exists():
                            failure = (failure or '') + '\njob did not write $OUT/result.json'
                        else:
                            json.loads((dest / 'result.json').read_text())
                        dump(dest / 'runner.json', {'host': h, 'launch': launch, 'error': failure, 'passed': failure is None, 'copies': str(copies)})
                except Exception as exc:
                    failure = (failure or '') + '\ncopy/validation: ' + repr(exc)
            return spec['id'], failure
        for start in range(0, len(stacks), 8):
            items = list(zip(hosts, stacks[start:start+8], archives[start:start+8]))
            if not a.dry_run:
                dump(STATE / 'current.json', {'kind': 'kernel', 'launch': batch,
                     'runs': [{'host': h, 'job': stack[-1][1]['id'],
                               'launch': batch + '-' + stack[-1][1]['id']} for h, stack, _ in items]})
            if a.dry_run:
                results = [one(item) for item in items]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as pool:
                    results = list(pool.map(one, items))
            errors.extend((job, error) for job, error in results if error)
            print(json.dumps(results), flush=True)
    if errors:
        raise RuntimeError('Kernel failures: ' + json.dumps(errors))

if __name__ == '__main__':
    main()
