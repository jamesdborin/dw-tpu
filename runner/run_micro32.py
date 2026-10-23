#!/usr/bin/env python3
"""Exclusive all-host microbenchmark; JAX distributed env; hard limit 15 minutes."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import uuid
from common import (STATE, Transport, arguments, cluster_lock, collect, dump, extract_evidence,
                    free_devices, fresh_result, job_stack, q, stop_script, unit_command)
from run_job import environment
from staging import apply_script, build_archive, stage_remote


def main():
    p = arguments(argparse.ArgumentParser(description=__doc__))
    p.add_argument('job', type=Path); p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--coordinator-port', type=int, default=12345)
    a = p.parse_args(); stack = job_stack(a.job); job, spec = stack[-1]
    if spec['kind'] != 'micro32':
        p.error('requires kind=micro32')
    if not 1024 <= a.coordinator_port <= 65535:
        p.error('coordinator port must be 1024..65535')
    launch = spec['id'] + '-' + ('DRYRUN' if a.dry_run else time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:6])
    archive = STATE / (launch + '.tar.gz') if a.dry_run else build_archive(stack, launch)
    if a.prepare_only:
        print(archive); return
    tr = Transport(a); errors = []
    with cluster_lock(a.dry_run):
        hosts = tr.discover(); out = '/opt/dsv4/runs/' + launch
        collect(tr, hosts, STATE / 'copies' / (launch + '-before'))
        tr.all(hosts, stop_script(), timeout=180)
        stage_remote(tr, hosts, archive, launch)
        if not tr.dry:
            fresh_result(job, launch)
            dump(STATE / 'current.json', {'kind': 'micro32', 'launch': launch, 'hosts': hosts,
                                         'job': str(job), 'id': spec['id']})
        try:
            tr.all(hosts, apply_script(stack), timeout=1800)
            tr.all(hosts, f'mkdir -p {q(out)}\n' + free_devices())
            addresses = ','.join(h['ip'] + ':8476' for h in hosts)
            def one(h):
                rank = hosts.index(h)
                env = environment(stack) + 'unset TPU_MULTIHOST_BACKEND\nexport JAX_PLATFORMS=tpu\n'
                values = {'JAX_COORDINATOR_ADDRESS': hosts[0]['ip'] + ':' + str(a.coordinator_port),
                          'JAX_PROCESS_ID': str(rank), 'JAX_NUM_PROCESSES': '8', 'TPU_PROCESS_BOUNDS': '2,4,1',
                          'TPU_CHIPS_PER_PROCESS_BOUNDS': '2,2,1', 'TPU_PROCESS_ADDRESSES': addresses,
                          'TPU_PROCESS_PORT': '8476', 'CLOUD_TPU_TASK_ID': str(rank),
                          'DW_TPU': '/opt/dsv4/dw-tpu', 'OUT': out}
                for key, value in values.items():
                    env += f'export {key}={q(value)}\n'
                # RuntimeMaxSec also protects against a killed SSH client.
                limit = min(900, int(spec['est_minutes'] * 90))
                command = env + f'cd /opt/dsv4/dw-tpu/jobs/{spec["id"]}\nexec timeout --signal=TERM --kill-after=15 {limit} bash run.sh'
                unit = unit_command('dsv4-kernel', command, out + '/run.log')
                unit = unit.replace('systemd-run --unit=', f'systemd-run --wait --property=RuntimeMaxSec={limit} --unit=')
                try:
                    tr.ssh(h, unit, timeout=limit + 180)
                    return None
                except Exception as exc:
                    return h['name'] + ': ' + repr(exc)
            if tr.dry:
                errors = [e for e in map(one, hosts) if e]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    errors = [e for e in pool.map(one, hosts) if e]
        finally:
            try:
                tr.all(hosts, stop_script(), timeout=180)
            finally:
                copies = collect(tr, hosts, STATE / 'copies' / launch)
                if not tr.dry:
                    dest = job / 'result'; dest.mkdir(exist_ok=True)
                    extract_evidence(copies / (hosts[0]['name'] + '.tar.gz'), dest, 'runs/' + launch)
                    if not (dest / 'result.json').exists():
                        errors.append('rank0 did not write result.json')
                    else:
                        json.loads((dest / 'result.json').read_text())
                    dump(dest / 'runner.json', {'launch': launch, 'errors': errors, 'passed': not errors, 'copies': str(copies)})
        if errors:
            raise RuntimeError('; '.join(errors))

if __name__ == '__main__':
    main()
