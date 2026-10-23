#!/usr/bin/env python3
"""Summarize validated benchmark records; missing measurements remain null."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import statistics
from common import dump

ANSI = re.compile(r'\x1b\[[0-9;]*m')


def parse_log(text, ready_unix=None):
    text = ANSI.sub('', text)
    blocks = sorted({int(x) for x in re.findall(r'\bnum_blocks=(\d+)', text)})
    durations = [float(x) for x in re.findall(r'(?:[Cc]ompilation[^\n]*finished in|[Cc]ompilation[^\n]*took)\s+([\d.]+)', text)]
    # Wall span of compiler messages, not sum of parallel compilations.
    stamps = []
    ready = dt.datetime.fromtimestamp(ready_unix, dt.timezone.utc) if ready_unix is not None else None
    post_ready_messages = 0
    for line in text.splitlines():
        if 'compilation_manager.py' in line:
            m = re.search(r'\b(\d\d-\d\d \d\d:\d\d:\d\d)\b', line)
            if m:
                year = ready.year if ready else 2000
                stamp = dt.datetime.strptime(str(year) + '-' + m[1], '%Y-%m-%d %H:%M:%S').replace(tzinfo=dt.timezone.utc)
                if ready is None or stamp < ready:
                    stamps.append(stamp)
                else:
                    post_ready_messages += 1
    compile_seconds = (max(stamps)-min(stamps)).total_seconds() if len(stamps) > 1 else None
    return {'kv_blocks': blocks, 'compile_wall_span_seconds': compile_seconds,
            'compile_reported_task_seconds': sum(durations) if durations else None,
            'compile_reported_tasks': len(durations),
            'post_ready_compiler_messages': post_ready_messages if ready else None,
            'compile_note': 'Startup compiler-message wall span when readiness is known; task durations include all phases and overlap across hosts. Raw worker logs are preserved.',
            'hbm_gb': re.findall(r'hbm=([^\n]+?)Gb', text)[-1:] or None}


def counter_rates(deltas):
    def total(marker):
        xs = [v for k, v in deltas.items() if marker in k]
        return sum(xs) if xs else None
    hits, queries = total('prefix_cache_hits_total'), total('prefix_cache_queries_total')
    accepted = total('spec_decode_num_accepted_tokens_total')
    drafted = total('spec_decode_num_draft_tokens_total')
    return {'apc_hit_rate': hits / queries if hits is not None and queries else None,
            'preemptions': total('num_preemptions_total'),
            'spec_decode_acceptance_rate': accepted / drafted if accepted is not None and drafted else None}


def metrics_parse(text):
    values = {}
    for line in text.splitlines():
        m = re.match(r'(vllm:[\w:]+(?:\{[^}]*\})?)\s+([\d.eE+-]+)(?:\s|$)', line)
        if m and '_total' in m[1]:
            values[m[1]] = float(m[2])
    return values


def summarize(result, spec):
    result = Path(result)
    def read(name, default):
        p = result / name
        return json.loads(p.read_text()) if p.exists() else default
    state = read('state.json', {})
    out = {'job': spec['id'], 'base': spec['base'], 'status': state.get('status', 'incomplete'),
           'startup_seconds': state.get('startup_seconds'), 'acceptance': read('acceptance.json', {'passed': False, 'reason': 'no completed gate'}),
           'token_buckets_per_dp': [int(x) for x in spec.get('env', {}).get('TPU_TOKEN_PADDINGS_PER_DP', '4,16,512').split(',')],
           'runs': [], 'error': state.get('error')}
    log = result / 'serve.log'
    out.update(parse_log(log.read_text(errors='replace') if log.exists() else '', state.get('ready_unix')))
    out['acceptance']['evidence'] = {
        'greedy': read('correctness/comparison.json', None),
        'gpu_reference_comparison': read('probes/full-reference-comparison.json', None),
        'reduction_order': read('reference-acceptance.json', None),
        'suffix_causality': read('correctness/causality-summary.json', None),
        'prompt_exact': read('prompt-reference-comparison.json', None),
        'sampling_equivalent': read('sampling-equivalent.json', None),
    }
    for path in sorted((result / 'bench').glob('repeat-*/validated-summary.json')):
        rows = json.loads(path.read_text())
        for row in rows:
            n = row['agents']; before = path.parent / f'agents-{n}.metrics-before.txt'; after = path.parent / f'agents-{n}.metrics-after.txt'
            deltas = {}
            if before.exists() and after.exists():
                b, a = metrics_parse(before.read_text()), metrics_parse(after.read_text())
                deltas = {k: v-b.get(k, 0) for k, v in a.items()}
                if any(v < 0 for v in deltas.values()):
                    raise ValueError('metrics reset; summary invalid')
            row.update(counter_rates(deltas), repeat=path.parent.name)
            out['runs'].append(row)
    rows32 = [x for x in out['runs'] if x['agents'] == 32]
    fields = {'tok_s_user_median': 'output_tps_p50', 'aggregate_tok_s': 'aggregate_output_tps',
              'tpm_including_cached': 'logical_tpm', 'apc_hit_rate': 'apc_hit_rate',
              'preemptions': 'preemptions', 'spec_decode_acceptance_rate': 'spec_decode_acceptance_rate'}
    for name, key in fields.items():
        values = [x[key] for x in rows32 if x.get(key) is not None]
        out[name] = statistics.median(values) if values else None
    out['aggregation'] = '32-agent measurements; median across repeats. TPM includes cached history, not compute throughput.'
    if state.get('acceptance_inherited_from'):
        out['acceptance_inherited_from'] = state['acceptance_inherited_from']
        out['acceptance_evidence'] = state.get('acceptance_evidence')
        out['server_reused'] = True
    out['acceptance_passed'] = out['acceptance'].get('passed') is True
    out['informational_only'] = not out['acceptance_passed']
    if spec.get('bench_mode') == 'pd':
        from pd_model import model, row_spec_statistics
        out['bench_mode'] = 'pd'
        pd = read('pd.json', {'prefill': [], 'decode': []})
        out['pd_model'] = model(pd)
        out['pd_spec_statistics'] = [dict(context_source=r.get('context_source'), concurrency=r['concurrency'],
            context_tokens=r.get('context_tokens'), valid=r.get('valid'), statistics=row_spec_statistics(r))
            for r in pd['decode'] if not r.get('skipped')]
        out['aggregation'] = 'P/D cases reported separately; no mixing of contexts, concurrency or spec counters across windows.'
        out['pd_valid_decode_cases'] = sum(bool(r.get('valid')) for r in pd['decode'])
        out['pd_flagged_decode_cases'] = sum(not r.get('valid') for r in pd['decode'])
    dump(result / 'summary.json', out)
    lines = [f'# {spec["id"]}', '', f'Status: {out["status"]}', f'Acceptance passed: {out["acceptance"].get("passed", False)}', '']
    if out.get('acceptance_inherited_from'):
        lines += ['Acceptance status: acceptance_inherited_from=' + out['acceptance_inherited_from'],
                  'Acceptance evidence: [' + out['acceptance_inherited_from'] + '](' + out['acceptance_evidence'] + ')', '']
    if out['status'] == 'gate_failed_benchmarked':
        lines += ['**INFORMATIONAL ONLY: acceptance_passed=false. GPU-reference gate failed; these measurements do not qualify the job.**', '']
    for key in ('startup_seconds', 'compile_wall_span_seconds', 'token_buckets_per_dp', 'kv_blocks', *fields):
        lines.append(f'- {key}: {out[key]}')
    lines += ['', out['aggregation'], out['compile_note'], '', 'All eight host archives and checksums: see copies.json.']
    if out['error']:
        lines += ['', 'Error: ' + str(out['error'])]
    if spec.get('bench_mode') == 'pd':
        from pd_model import markdown
        lines += markdown(pd)
    (result / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    return out

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('job', type=Path); p.add_argument('--dry-run', action='store_true'); a = p.parse_args()
    if a.dry_run:
        print('Read', a.job / 'result', '; write summary.json and SUMMARY.md')
    else:
        effective = a.job / 'result/effective-job.json'
        spec = json.loads((effective if effective.exists() else a.job / 'job.json').read_text())
        print(json.dumps(summarize(a.job / 'result', spec), indent=2))
