#!/usr/bin/env python3
"""Head-side bounded controller: readiness, timestamp gate, acceptance, bench, profile.

Runs in a systemd control group so aborts terminate benchmark/probe descendants.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
from allocation_events import classify
from compile_events import classify_compiles

STRICT_COMPILES = False


class CommandFailed(RuntimeError):
    def __init__(self, log_name, code):
        super().__init__(f'{log_name}: exit {code}')
        self.code = code


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def audit(out, ready=None):
    log = (out / 'serve.log').read_text(errors='replace') if (out / 'serve.log').exists() else ''
    if re.search(r'launch[ _-]*id[^\n]*mismatch', log, re.I):
        raise RuntimeError('launch-id mismatch')
    for fatal in ('scheckne', 'Engine core initialization failed'):
        if fatal in log:
            raise RuntimeError(fatal)
    if ready is not None:
        result = classify(log, ready)
        write(out / 'allocation-timestamp-audit.json', result)
        if result['failures']:
            raise RuntimeError('post-ready/unrecovered/unclassified RESOURCE_EXHAUSTED')
        if STRICT_COMPILES:
            compilation = classify_compiles(log, ready)
            write(out / 'compile-audit.json', compilation)
            if compilation['failures']:
                raise RuntimeError('post-ready model/runner JIT compilation')
    return log


def checked(command, out, log_name, deadline, ready):
    print('RUN', json.dumps([str(x) for x in command]), flush=True)
    with (out / log_name).open('w') as log:
        process = subprocess.Popen([str(x) for x in command], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                audit(out, ready)
                code = process.poll()
                if code is not None:
                    if code:
                        raise CommandFailed(log_name, code)
                    return
                if time.time() >= deadline:
                    raise TimeoutError(f'{log_name}: job time limit exceeded')
                time.sleep(2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()


def acceptance(spec, out, deadline, ready, job):
    vendor = HERE / 'vendor'; refs = HERE / 'references'
    mode = spec['acceptance']
    passed = True
    threshold_errors = []
    py = sys.executable
    check = lambda cmd, name: checked(cmd, out, name, deadline, ready)
    check([py, vendor / 'probe.py', '--out', out / 'probes', '--reference', vendor / 'full-gpu-reference.json'], 'probe.log')
    cache = json.loads((out / 'probes/cache-summary.json').read_text())
    assert len(cache) == 3 and all(x['same_tokens'] and x['same_logprobs'] for x in cache.values())
    command = [py, vendor / 'correctness.py', '--out', out / 'correctness', '--prompts', vendor / 'prompts.json',
               '--baseline', refs / 'h4-r2/correctness/greedy.json', '--causality']
    if mode == 'exact':
        command.append('--exact-logprobs')
    elif mode == 'reduction-order':
        command.append('--reduction-order')
    else:
        command += ['--reduction-order', '--all-greedy-ids']
    check(command, 'correctness.log')
    if mode == 'exact':
        # Both accepted B2 and its async H4-r2 variant must agree on generated evidence.
        actual = json.loads((out / 'correctness/greedy.json').read_text())
        for label in ('h17-model', 'h4-r2'):
            reference = json.loads((refs / label / 'correctness/greedy.json').read_text())
            for name, response in actual.items():
                x, y = response['choices'][0], reference[name]['choices'][0]
                assert x['token_ids'] == y['token_ids'], (label, name, 'tokens')
                assert x['logprobs']['token_logprobs'] == y['logprobs']['token_logprobs'], (label, name, 'logprobs')
        check([py, vendor / 'compare_prompt_reference.py', '--experiment', out, '--baseline', refs / 'h4-r2/probes/full-http-probes.json'], 'prompt-exact.log')
    else:
        evidence_path = out / 'reference-acceptance.json'
        evidence_path.unlink(missing_ok=True)  # A crashed check cannot reuse stale evidence.
        try:
            check([py, vendor / 'reference_acceptance.py', '--experiment', out, '--baseline', refs / 'b0',
                   '--reference', vendor / 'full-gpu-reference.json', '--prompts', vendor / 'prompts.json'], 'reference.log')
        except CommandFailed as exc:
            if not (mode == 'reduction-order' and spec.get('benchmark_on_gate_failure') is True
                    and exc.code == 1 and evidence_path.exists()):
                raise
            evidence = json.loads(evidence_path.read_text())
            threshold_errors = evidence.get('threshold_errors')
            if not (evidence.get('passed') is False and isinstance(threshold_errors, list)
                    and threshold_errors and evidence.get('hard_errors') == []
                    and evidence.get('errors') == threshold_errors):
                raise
            audit(out, ready)
            passed = False
    if mode == 'sampling-equivalent':
        # A greedy test alone does not prove preserved sampling distribution.
        check(['bash', job / 'acceptance.sh'], 'sampling.log')
        evidence = json.loads((out / 'sampling-equivalent.json').read_text())
        assert evidence.get('passed') is True and evidence.get('distribution_preserved') is True, evidence
    write(out / 'acceptance.json', {'mode': mode, 'passed': passed, 'threshold_errors': threshold_errors,
                                  'informational_only': not passed, 'greedy_reference': 'h4-r2',
                                  'exact_also_checks': 'h17-model/B2' if mode == 'exact' else None,
                                  'gpu_anchor': 'b0', 'suffix_causality': 0})
    return passed


def benchmark(spec, out, deadline, ready):
    vendor = HERE / 'vendor'; py = sys.executable
    if spec.get('bench_mode', 'agent') == 'pd':
        config = out / 'pd-job.json'
        write(config, spec)
        checked([py, vendor / 'pd_bench.py', config, '--out', out / 'pd.json'], out, 'pd.log', deadline, ready)
        checked([py, HERE / 'pd_model.py', out / 'pd.json', '--out', out / 'pd-model.json'], out, 'pd-model.log', deadline, ready)
        return
    bb = '/opt/dsv4/tools/batchbench-agent'
    common = ['--host', 'http://127.0.0.1:8000', '--model', 'dsv4-flash', '--tokenizer-model', '/opt/dsv4/ckpt/dsv4-flash-full/tokenizer.json',
              '--input-tokens', '5000', '--output-tokens', '500', '--environment-tokens', '500', '--tool-call-latency-ms', '0',
              '--ignore-eos', '--max-retries', '0', '--request-timeout-secs', '3600', '--disable-user-tagging']
    check = lambda cmd, name: checked(cmd, out, name, deadline, ready)
    for repeat in range(spec['bench']['repeats']):
        root = out / 'bench' / f'repeat-{repeat + 1}'; root.mkdir(parents=True, exist_ok=True)
        # Preserve original workload and sampling defaults from the proven bench.sh binary.
        # Outer controller must always audit the serve.log under `out`, even for nested artifacts.
        cmd = [bb, *common, '--agents', '32', '--tool-invocations', '3', '--seed', str(999 + repeat),
               '--user-prefix', f'warm{repeat}', '--results-csv', root / 'warmup.csv']
        check(cmd, str(root.relative_to(out) / 'warmup.log'))
        check([py, vendor / 'validate_warmup.py', root], f'warmup-validation-{repeat}.log')
        for n in spec['bench']['agents']:
            time.sleep(6)
            metrics(root / f'agents-{n}.metrics-before.txt')
            check([py, vendor / 'run_measured.py', root, str(n), bb, *common, '--agents', str(n), '--tool-invocations', '10',
                   '--seed', str(n*1000 + 6), '--user-prefix', f'a{n}', '--results-csv', root / f'agents-{n}.csv'], f'measured-{repeat}-{n}.log')
            time.sleep(6)
            metrics(root / f'agents-{n}.metrics-after.txt')
        counts = list(map(str, spec['bench']['agents']))
        check([py, vendor / 'validate_bench.py', root, *counts], f'bench-validation-{repeat}.log')
        check([py, vendor / 'cache_metrics.py', root, *counts], f'cache-validation-{repeat}.log')
    print('BENCH DONE', flush=True)


def metrics(path):
    with urllib.request.urlopen('http://127.0.0.1:8000/metrics', timeout=30) as response:
        path.write_bytes(response.read())


def main():
    global STRICT_COMPILES
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job', type=Path); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--launch-id', required=True); p.add_argument('--dry-run', action='store_true')
    p.add_argument('--reuse-evidence', type=Path, help='Verified reuse provenance; skip acceptance and startup')
    a = p.parse_args(); spec = json.loads((a.job / 'job.json').read_text()); out = a.out
    STRICT_COMPILES = 'post-ready model/runner JIT compilation' in spec.get('abort_if', [])
    if a.dry_run:
        print('Controller: wait /health; audit RESOURCE_EXHAUSTED by event timestamps; run acceptance=' + spec['acceptance'] + '; bench_mode=' + spec.get('bench_mode', 'agent') + '; benchmark=' + json.dumps(spec.get('pd') if spec.get('bench_mode') == 'pd' else spec['bench']) + '; profile=' + str(spec.get('profile', False))); return
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(OUT=str(out), DW_TPU=str(HERE.parent))
    started = time.time(); deadline = started + spec['est_minutes']*90
    result = {'job': spec['id'], 'launch_id': a.launch_id, 'started_unix': started, 'status': 'starting'}
    def interrupted(signum, frame):
        raise RuntimeError(f'controller interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        assert (out / 'launch-id').read_text().strip() == a.launch_id, 'launch-id mismatch'
        inherited = json.loads(a.reuse_evidence.read_text()) if a.reuse_evidence else None
        if inherited and (inherited['acceptance'].get('passed') is not True or inherited['acceptance'].get('mode') != spec['acceptance']):
            raise RuntimeError('invalid inherited acceptance')
        limits = [int(re.search(r'\d+', x)[0])*60 for x in spec.get('abort_if', []) if x.startswith('startup >')]
        startup_limit = min(limits or [2700])
        write(out / 'state.json', result)
        while True:
            audit(out)
            if time.time() > min(started + startup_limit, deadline):
                raise TimeoutError('startup timeout')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                if inherited: raise RuntimeError('reuse health failed; no startup fallback')
            if inherited: raise RuntimeError('reuse health returned non-200; no startup fallback')
            time.sleep(5)
        ready = inherited['origin_ready_unix'] if inherited else time.time(); audit(out, ready)
        result.update(ready_unix=ready, startup_seconds=ready-started, status='acceptance')
        write(out / 'ready.json', {'ready_unix': ready}); write(out / 'state.json', result)
        if inherited:
            result.update(acceptance_passed=True, startup_seconds=0, server_reused=True,
                          acceptance_inherited_from=inherited['acceptance_inherited_from'],
                          acceptance_status='acceptance_inherited_from=' + inherited['acceptance_inherited_from'],
                          acceptance_evidence=inherited['acceptance_evidence'])
            write(out / 'acceptance.json', dict(inherited['acceptance'], inherited_from=inherited['acceptance_inherited_from'],
                  evidence=inherited['acceptance_evidence']))
        else:
            result['acceptance_passed'] = acceptance(spec, out, deadline, ready, a.job)
        result['status'] = 'benchmark'; write(out / 'state.json', result)
        benchmark(spec, out, deadline, ready)
        if spec.get('profile'):
            result['status'] = 'profile'; write(out / 'state.json', result)
            checked([sys.executable, HERE / 'vendor/replay_capture.py', '--payloads', HERE / 'vendor/profile-payloads.jsonl',
                     '--phase', 'decode', '--seconds', '3', '--output-tokens', '1000', '--label', a.launch_id,
                     '--output', out / 'profile'], out, 'profile.log', deadline, ready)
        audit(out, ready)
        result['status'] = 'passed' if result['acceptance_passed'] else 'gate_failed_benchmarked'
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        result['finished_unix'] = time.time()
        write(out / 'state.json', result)
        (out / 'DONE').write_text(result['status'] + '\n')

if __name__ == '__main__':
    main()
