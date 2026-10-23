#!/usr/bin/env python3
"""Orchestrator queue: retain successful servers, reuse matching identities, never retry refusal as cold start."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from common import STATE, arguments, job_stack
from server_reuse import desired_manifest, identity_from_manifest, differences


def plan(identities, previous=None, fresh_first=False):
    result=[]
    for i, identity in enumerate(identities):
        diff=differences(previous, identity) if previous is not None else ['no prior server identity']
        reuse = previous is not None and not diff and not (i == 0 and fresh_first)
        result.append({'action':'reuse' if reuse else 'cold', 'identity_diff':diff})
        previous=identity
    return result


def main():
    p=arguments(argparse.ArgumentParser(description=__doc__))
    p.add_argument('jobs', nargs='+', type=Path)
    p.add_argument('--fresh-first', action='store_true', help='Explicitly cold-start first job even if local current identity matches')
    a=p.parse_args()
    stacks=[job_stack(path) for path in a.jobs]
    if any(stack[-1][1]['kind'] != 'e2e' for stack in stacks): p.error('queue accepts e2e jobs only')
    identities=[identity_from_manifest(desired_manifest(stack)) for stack in stacks]
    previous=None
    current=STATE/'current.json'
    if current.exists() and not a.fresh_first:
        state=json.loads(current.read_text()); launch=state.get('server_launch', state['launch'])
        from common import ID_RE
        if not ID_RE.fullmatch(launch): raise ValueError('unsafe current server launch')
        path=STATE/(launch+'.tar.manifest.json')
        if path.exists(): previous=identity_from_manifest(json.loads(path.read_text()))
    plans=plan(identities, previous, a.fresh_first)
    for path, step in zip(a.jobs, plans):
        command=[sys.executable, str(Path(__file__).with_name('run_job.py')), str(path),
                 '--reuse-server' if step['action']=='reuse' else '--keep-server',
                 '--project', a.project, '--zone', a.zone, '--group', a.group]
        if a.hosts: command += ['--hosts', str(a.hosts)]
        print(json.dumps({'job':str(path), **step}), flush=True)
        print(shlex.join(command), flush=True)
        if not a.dry_run:
            # Refused reuse/health/gates halt the queue. Never silently rerun cold.
            subprocess.run(command, check=True)
    print('Queue dry-run complete.' if a.dry_run else 'Queue complete; final server retained. Teardown remains explicit.')

if __name__=='__main__': main()
