#!/usr/bin/env python3
"""Recover measured loader spans; never label combined spans as pure I/O or H2D."""
import argparse
import json
from pathlib import Path
import re
import statistics


def analyze(lines):
    workers = {}; rows = []
    patterns = {'initialize': r'Initializing model took ([0-9.]+) seconds',
                'load_weights': r'Loading weights took ([0-9.]+) seconds',
                'storage_to_tpu': r'Total time to load model weights from storage to TPU: ([0-9.]+) seconds'}
    for number, line in enumerate(lines, 1):
        pid = re.search(r'\(Worker pid=(\d+)\)', line)
        if not pid: continue
        worker = workers.setdefault(pid[1], {'index': 0})
        for phase, pattern in patterns.items():
            match = re.search(pattern, line)
            if not match: continue
            worker[phase] = float(match[1]); worker[phase+'_line'] = number
            if phase == 'storage_to_tpu':
                row = dict(worker, pid=pid[1])
                row['unseparated_postload_seconds'] = (row['storage_to_tpu'] - row['initialize'] - row['load_weights']
                                                      if all(k in row for k in ('initialize', 'load_weights')) else None)
                rows.append(row); workers[pid[1]] = {'index': worker['index']+1}
    summary = []
    for index in sorted({r['index'] for r in rows}):
        selected = [r for r in rows if r['index'] == index]
        stats = {}
        for key in (*patterns, 'unseparated_postload_seconds'):
            values = [r[key] for r in selected if r.get(key) is not None]
            stats[key] = {'min': min(values), 'median': statistics.median(values), 'max': max(values)} if values else None
        summary.append({'load_index_per_worker': index, 'workers': len(selected), 'seconds': stats})
    return {'summary': summary, 'worker_loads': rows,
            'limitations': ['load_weights spans checkpoint iteration and parameter assignment, potentially incremental processing; not pure disk reads.',
                            'Remaining wrapper time combines FP4/dense processing, sharding/device transfer and wrapper setup. Logs do not separately time them.',
                            'Wrapper timer is not an explicit JAX block_until_ready timing boundary. Do not claim a precise H2D or FP4 cost from these logs.',
                            'Index0 is the first load per worker; later indices typically represent draft model loads. Verify against each job configuration.']}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('serve_log',type=Path); p.add_argument('--out',type=Path); a=p.parse_args()
    with a.serve_log.open(errors='replace') as f: result=analyze(f)
    result['source']=str(a.serve_log)
    text=json.dumps(result,indent=2)+'\n'
    if a.out: a.out.write_text(text)
    print(text)



# Optional next-reservation hook: callers supply a device synchronization barrier
# at phase boundaries; the helper itself imports no accelerator libraries.
from contextlib import contextmanager
import time


@contextmanager
def timed_phase(name, emit, synchronize, clock=time.monotonic):
    synchronize()
    start=clock()
    try:
        yield
    finally:
        synchronize()
        emit({'event':'processed_weight_phase', 'phase':name, 'seconds':clock()-start,
              'synchronized_boundaries':True})


if __name__=='__main__': main()
