#!/usr/bin/env python3
"""Isolated P/D experiments. Stdlib HTTP client; never imports model libraries."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import threading
import time
import urllib.request


def validate_config(pd):
    def positive(x):
        return type(x) is int and x > 0
    def grid(xs):
        return isinstance(xs, list) and xs and all(positive(x) for x in xs) and len(set(xs)) == len(xs)
    if not isinstance(pd, dict):
        raise ValueError('pd must be an object')
    p, d = pd.get('prefill', {}), pd.get('decode', {})
    if p.get('max_tokens') != 1 or not isinstance(p.get('cases'), list) or not p['cases']:
        raise ValueError('pd.prefill requires cases and max_tokens=1')
    names = set()
    for c in p['cases']:
        if not isinstance(c, dict) or not isinstance(c.get('name'), str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', c['name']) or c['name'] in names:
            raise ValueError('prefill case names must be unique safe names')
        names.add(c['name'])
        cold = positive(c.get('prompt_tokens')) and 'cached_prefix_tokens' not in c and 'new_tokens' not in c
        incr = positive(c.get('cached_prefix_tokens')) and positive(c.get('new_tokens')) and 'prompt_tokens' not in c
        if not (cold or incr) or not grid(c.get('concurrency')):
            raise ValueError('prefill case requires prompt_tokens OR cached_prefix_tokens + new_tokens, and concurrency')
    if not positive(p.get('repeats', 1)):
        raise ValueError('prefill repeats must be positive')
    for key in ('cache_affinity', 'verify_warm_cache'):
        if type(p.get(key, False)) is not bool:
            raise ValueError('prefill ' + key + ' must be boolean')
    if 'decode' not in pd: return
    if d.get('capacity_check', 'strict') not in ('strict', 'advisory'):
        raise ValueError('decode capacity_check must be strict or advisory')
    sources = d.get('context_source', ['random'])
    if not isinstance(sources, list) or not sources or len(set(sources)) != len(sources) or any(x not in ('random', 'agent-replay') for x in sources):
        raise ValueError('decode context_source must contain random and/or agent-replay')
    if not grid(d.get('concurrency')) or not grid(d.get('context_tokens')):
        raise ValueError('decode requires positive distinct concurrency/context grids')
    if not all(positive(d.get(k)) for k in ('measure_tokens', 'warmup_tokens')):
        raise ValueError('decode warmup_tokens and measure_tokens must be positive')
    if not positive(d.get('slack_tokens', 512)):
        raise ValueError('decode slack_tokens must be positive (default512)')
    if type(d.get('temperature')) not in (int, float) or not math.isfinite(d['temperature']) or d['temperature'] < 0:
        raise ValueError('decode temperature must be finite and nonnegative')


def quantile(xs, p):
    if not xs:
        return None
    xs = sorted(xs); pos = (len(xs) - 1) * p
    lo = int(pos); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def distribution(xs):
    return {'mean': statistics.mean(xs) if xs else None, 'p50': quantile(xs, .5), 'p90': quantile(xs, .9)}


def decode_window(records, warmup_tokens, measure_tokens):
    """Use token events, not SSE message count; a speculative burst has equal times."""
    if not records or any(len(r['token_times']) < warmup_tokens or r.get('finish_time') is None for r in records):
        return {'valid': False, 'flags': ['incomplete_warmup_or_finish'], 'requests': []}
    start = max(r['token_times'][warmup_tokens - 1] for r in records)
    end = min(r['finish_time'] for r in records)
    if end <= start:
        return {'valid': False, 'flags': ['no_shared_decode_window'], 'start': start, 'end': end, 'requests': []}
    requests = []
    all_itl = []
    for r in records:
        times = [t for t in r['token_times'] if start < t <= end]
        # No interval crossing the start boundary: it might include warmup/prefill.
        itl = [b - a for a, b in zip(times, times[1:])]
        all_itl.extend(itl)
        requests.append({'request': r['request'], 'tokens': len(times), 'decode_tok_s': len(times) / (end - start), 'itl_seconds': distribution(itl)})
    flags = []
    if any(r['tokens'] < measure_tokens for r in requests):
        flags.append('insufficient_measured_tokens')
    rates = [r['decode_tok_s'] for r in requests]
    return {'valid': not flags, 'flags': flags, 'start': start, 'end': end, 'seconds': end - start, 'concurrency': len(records),
            'requests': requests, 'per_user_decode_tok_s': distribution(rates),
            'aggregate_decode_tok_s': sum(rates), 'itl_seconds': distribution(all_itl)}


def parse_metrics(text):
    result = {}
    for line in text.splitlines():
        m = re.match(r'^(vllm:[\w:]+)(?:\{[^\n]*\})?\s+([-+\deE.]+)', line)
        if m and (m[1].startswith('vllm:spec_decode_') or m[1] in ('vllm:prompt_tokens_total', 'vllm:prompt_tokens_cached_total', 'vllm:prefix_cache_hits_total', 'vllm:prefix_cache_queries_total', 'vllm:num_preemptions_total', 'vllm:num_requests_running', 'vllm:num_requests_waiting')):
            value = float(m[2])
            if math.isfinite(value):
                result[m[1]] = result.get(m[1], 0) + value
                if m[1] == 'vllm:spec_decode_num_accepted_tokens_per_pos_total':
                    pos = re.search(r'position="([^"]+)"', line)
                    if pos:
                        key = m[1] + '{position="' + pos[1] + '"}'
                        result[key] = result.get(key, 0) + value
    return result


def spec_statistics(deltas, seconds, n):
    drafts = deltas.get('vllm:spec_decode_num_drafts_total', 0)
    accepted = deltas.get('vllm:spec_decode_num_accepted_tokens_total')
    positions = {k: v / drafts for k, v in deltas.items() if k.startswith('vllm:spec_decode_num_accepted_tokens_per_pos_total{') and drafts > 0}
    return {'step_seconds': seconds * n / drafts if drafts > 0 else None,
            'tokens_per_step': 1 + accepted / drafts if drafts > 0 and accepted is not None else None,
            'acceptance_per_position': positions,
            'step_note': 'num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.'}


def window_metrics(samples, window, initial, expected_prompt_tokens):
    """Conservative enclosing scrapes; absent/lagged counters never prove purity."""
    before = [s for s in samples if s['end'] <= window.get('start', -1) and 'metrics' in s]
    after = [s for s in samples if s['start'] >= window.get('end', math.inf) and 'metrics' in s]
    if not before or not after:
        return {'verified_no_prefill': False, 'flags': ['metrics_window_not_bracketed']}
    b, a = before[-1], after[-1]
    deltas = {k: a['metrics'][k] - v for k, v in b['metrics'].items() if k.split('{')[0].endswith('_total') and k in a['metrics']}
    key = 'vllm:prompt_tokens_total'
    flags = []
    if key not in deltas or key not in initial:
        flags.append('prefill_counter_unavailable')
    else:
        if b['metrics'][key] - initial[key] < expected_prompt_tokens:
            flags.append('prefill_counter_not_drained_at_window_start')
        if deltas[key] != 0:
            flags.append('prefill_tokens_in_enclosing_window')
    if any(v < 0 for v in deltas.values()): flags.append('metrics_counter_reset')
    if deltas.get('vllm:num_preemptions_total', 0) > 0: flags.append('preemption_in_enclosing_window')
    return {'verified_no_prefill': not flags, 'flags': flags, 'scrape_bounds': [b['start'], a['end']],
            'counter_deltas': deltas, 'note': 'Enclosing scrapes are used only for conservative prefill/preemption verification. Spec statistics use the dedicated boundary scrapes.'}


def occupancy_metrics(samples, window, n):
    """Verify observed occupancy inside the window; retain both boundary scrapes.

    The end RPC runs after first finish, so its running gauge includes natural
    completion. Never mistake that post-window tail for a scheduling shortage.
    """
    boundaries = {s['boundary']: s for s in samples if 'boundary' in s and 'metrics' in s}
    flags = []
    keys = ('vllm:num_preemptions_total', 'vllm:num_requests_running', 'vllm:num_requests_waiting')
    inside = [s for s in samples if 'metrics' in s and
              window.get('start', math.inf) <= s['start'] and s['end'] <= window.get('end', -1)]
    if set(boundaries) != {'start', 'end'} or any(k not in s['metrics'] for s in boundaries.values() for k in keys):
        flags.append('occupancy_boundary_metrics_unavailable')
    if not inside or any(k not in s['metrics'] for s in inside for k in keys):
        flags.append('occupancy_window_unverified')
    if any(s['metrics'].get(keys[1], n) < n or s['metrics'].get(keys[2], 0) > 0 for s in inside):
        flags.append('not_all_running')
    # Include admission: a case that preempted before all requests warmed is not
    # a clean capacity demonstration even if its eventual decode window is pure.
    end = boundaries.get('end')
    initial = samples[0].get('metrics', {}) if samples else {}
    delta = None
    if end and keys[0] in initial and keys[0] in end['metrics']:
        delta = end['metrics'][keys[0]] - initial[keys[0]]
        if delta > 0: flags.append('preempted')
        if delta < 0: flags.append('metrics_counter_reset')
    else:
        flags.append('preemption_counter_unavailable')
    return {'flags': flags, 'verified_all_running': not flags,
            'cohort_preemptions_through_window_end': delta,
            'boundary_metrics': {label: {k: snap['metrics'].get(k) for k in keys} for label, snap in boundaries.items()},
            'in_window_samples': [{k: s['metrics'].get(k) for k in keys} | {'start': s['start'], 'end': s['end']} for s in inside],
            'note': 'Polling validates observed samples, not every scheduler step. End-boundary gauge is after first completion and is recorded but excluded from occupancy checks.'}


def decode_capacity_check(lengths, output_tokens, capacity, mode, max_model_len):
    estimate = capacity_check(lengths, output_tokens, capacity)
    flags = list(estimate['flags'])
    if max(lengths) + output_tokens > max_model_len and 'model_length_exceeded' not in flags:
        flags.append('model_length_exceeded')
    hard = [f for f in flags if f == 'model_length_exceeded']
    return dict(estimate, mode=mode, allowed=not (hard if mode == 'advisory' else flags),
                flags=hard if mode == 'advisory' else flags,
                advisory_flags=[f for f in flags if f not in hard] if mode == 'advisory' else [],
                max_model_len=max_model_len,
                note='Hybrid published KV tokens are a full-model-length concurrency equivalent, not physical token capacity. Advisory mode defers capacity validation to live occupancy/preemption metrics; model length remains mandatory.')


def eligible_tokens(tokenizer_path):
    # Matches batchbench/rust/src/agent.rs synthetic_token_ids: base vocab, no special IDs.
    data = json.loads(Path(tokenizer_path).read_text())
    vocab = data['model']['vocab']
    special = {x['id'] for x in data.get('added_tokens', []) if x.get('special')}
    ids = [i for i in range(len(vocab)) if i not in special]
    if not ids: raise ValueError('empty token pool')
    return ids


class Client:
    def __init__(self, base, timeout=1800):
        self.base = base.rstrip('/'); self.timeout = timeout

    def fetch(self, route, body=None):
        req = urllib.request.Request(self.base + route, data=json.dumps(body).encode() if body is not None else None,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read().decode()

    def snapshot(self):
        start = time.monotonic()
        metrics = parse_metrics(self.fetch('/metrics'))
        return {'start': start, 'end': time.monotonic(), 'metrics': metrics}

    def reset(self):
        # Drain the preceding cohort before resetting or admitting new work.
        deadline = time.monotonic() + 60
        while True:
            m = self.snapshot()['metrics']
            keys = ('vllm:num_requests_running', 'vllm:num_requests_waiting')
            if all(k in m and m[k] == 0 for k in keys): break
            if time.monotonic() > deadline: raise RuntimeError('server did not become verifiably idle')
            time.sleep(.25)
        if json.loads(self.fetch('/reset_prefix_cache', {})).get('success') is not True:
            raise RuntimeError('prefix cache reset failed')

    def stream(self, index, prompt, max_tokens, temperature, barrier, event=None):
        body = {'model': 'dsv4-flash', 'prompt': prompt, 'max_tokens': max_tokens, 'min_tokens': max_tokens,
                'ignore_eos': True, 'temperature': temperature,
                'stream': True, 'stream_options': {'include_usage': True}, 'return_token_ids': True}
        if isinstance(prompt, dict):
            body.pop('prompt')
            body = dict(prompt, **body)
        else:
            body['add_special_tokens'] = False
        if getattr(self, 'cache_affinity', False):
            body['vllm_xargs'] = {'pd_cache_rank': index % 32}
        body.pop('seed', None)  # TPU sampling rejects per-request seeds, including replay seeds.
        route = '/v1/chat/completions' if isinstance(prompt, dict) else '/v1/completions'
        request = urllib.request.Request(self.base + route, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        barrier.wait(timeout=60)
        row = {'request': index, 'submit': time.monotonic(), 'token_times': [], 'finish_time': None,
               'prompt_sha256': hashlib.sha256(json.dumps(prompt).encode()).hexdigest()}
        done = False
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data:'): continue
                value = line[5:].strip()
                if value == '[DONE]': done = True; break
                obj = json.loads(value); now = time.monotonic()
                if obj.get('error'): raise RuntimeError(str(obj['error']))
                if obj.get('usage'): row['usage'] = obj['usage']
                for choice in obj.get('choices', []):
                    ids = choice.get('token_ids')
                    if ids is None and (choice.get('text') or choice.get('delta', {}).get('content') or choice.get('delta', {}).get('reasoning_content')):
                        raise RuntimeError('stream omitted token_ids; cannot measure speculative bursts')
                    row['token_times'].extend([now] * len(ids or []))
                    if event: event(index, len(row['token_times']), False)
                    if choice.get('finish_reason') is not None:
                        row['finish_time'] = now; row['finish_reason'] = choice['finish_reason']
                        if event: event(index, len(row['token_times']), True)
        row['done'] = time.monotonic()
        if not done or row['finish_time'] is None or len(row['token_times']) != max_tokens:
            raise RuntimeError('incomplete stream or unexpected output token count')
        if (not isinstance(prompt, dict) and row.get('usage', {}).get('prompt_tokens') != len(prompt)) or row.get('usage', {}).get('completion_tokens') != max_tokens:
            raise RuntimeError('stream usage differs from exact token counts')
        return row

    def cohort(self, prompts, max_tokens, temperature, monitor=False, warmup_tokens=16):
        samples = [self.snapshot()]; stop = threading.Event()
        counts = [0] * len(prompts); markers = set(); lock = threading.Lock()
        boundary_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        boundary_futures = []
        def boundary(label):
            snap = self.snapshot(); snap['boundary'] = label; return snap
        def event(i, count, finished):
            if not monitor: return
            with lock:
                counts[i] = count
                for label, trigger in [('start', min(counts) >= warmup_tokens), ('end', finished)]:
                    if trigger and label not in markers:
                        markers.add(label); boundary_futures.append(boundary_pool.submit(boundary, label))
        def sample():
            while not stop.wait(.25):
                try: samples.append(self.snapshot())
                except Exception as exc: samples.append({'start': time.monotonic(), 'end': time.monotonic(), 'error': repr(exc)})
        sampler = threading.Thread(target=sample, daemon=True)
        if monitor: sampler.start()
        try:
            barrier = threading.Barrier(len(prompts))
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
                futures = [pool.submit(self.stream, i, p, max_tokens, temperature, barrier, event) for i, p in enumerate(prompts)]
                records = [f.result() for f in futures]
            # Allow the frontend metrics to consume engine statistics before final scrape.
            time.sleep(2)
        finally:
            stop.set()
            if monitor: sampler.join(timeout=10)
            boundary_pool.shutdown(wait=True)
        samples.extend(f.result() for f in boundary_futures)
        samples.append(self.snapshot())
        return records, sorted(samples, key=lambda s: s['end'])


def prefill_result(records, case, n, samples, block_size=16):
    prefix = case.get('cached_prefix_tokens', 0)
    new = case.get('new_tokens', case.get('prompt_tokens'))
    duration = max(r['token_times'][0] for r in records) - min(r['submit'] for r in records)
    usage = [r['usage'] for r in records]
    cached = [(u.get('prompt_tokens_details') or {}).get('cached_tokens') for u in usage]
    flags = []
    minimum_cached = ((prefix - 1) // block_size) * block_size if prefix else 0
    if prefix and (minimum_cached <= 0 or any(x is None or x < minimum_cached for x in cached)):
        flags.append('prefix_cache_reuse_unverified_or_insufficient')
    if samples and samples[-1].get('metrics', {}).get('vllm:num_preemptions_total', 0) > samples[0].get('metrics', {}).get('vllm:num_preemptions_total', 0):
        flags.append('preemption_during_prefill')
    # APC reuses complete blocks; separately expose any recomputed alignment tail.
    return {'name': case['name'], 'concurrency': n, 'cached_prefix_tokens': prefix, 'new_tokens': new,
            'valid': not flags, 'flags': flags, 'seconds': duration,
            'new_prefill_tok_s_per_chip': n * new / duration / 32,
            'including_cached_tok_s_per_chip': n * (prefix + new) / duration / 32,
            'chips': 32, 'new_prefill_tok_s': n * new / duration, 'including_cached_tok_s': n * (prefix + new) / duration,
            'observed_cached_tokens': cached, 'minimum_expected_cached_per_request': minimum_cached,
            'observed_uncached_prefill_tok_s': sum(u['prompt_tokens'] - c for u, c in zip(usage, cached)) / duration if all(c is not None for c in cached) else None, 'ttft_seconds': distribution([r['token_times'][0] - r['submit'] for r in records]),
            'requests': records, 'metrics': samples}


def capacity_check(lengths, output_tokens, capacity):
    if not capacity:
        return {'allowed': False, 'flags': ['kv_capacity_unavailable']}
    block = capacity['block_size']
    # Legacy strict estimate, retained for compatibility. Hybrid published KV
    # tokens are a full-length concurrency equivalent, NOT physical capacity.
    # Advisory decode runs use live counters instead of trusting this bound.
    required_per_rank = math.ceil(len(lengths) / 32) * (math.ceil((max(lengths) + output_tokens) / block) + 1) * block
    available_per_rank = capacity['kv_cache_size_tokens'] // 32
    flags = []
    if max(lengths) + output_tokens > capacity['max_model_len']: flags.append('model_length_exceeded')
    if required_per_rank > available_per_rank: flags.append('kv_capacity_would_preempt')
    if capacity.get('max_concurrency') is not None and math.ceil(len(lengths) / 32) > math.floor(capacity['max_concurrency'] / 32):
        flags.append('conservative_hybrid_concurrency_limit')
    return {'allowed': not flags, 'flags': flags, 'required_prompt_kv_tokens': sum(lengths), 'required_per_rank_tokens': required_per_rank, 'available_per_rank_tokens': available_per_rank, 'capacity': capacity}


def read_capacity(client, serve_log):
    text = client.fetch('/metrics')
    infos = re.findall(r'^vllm:cache_config_info\{([^\n]+)\}', text, re.M)
    log = serve_log.read_text(errors='replace') if serve_log.exists() else ''
    blocks = [int(x) for x in re.findall(r'Using cache_config.block_size: (\d+)', log)]
    for info in infos:
        labels = dict(re.findall(r'(\w+)="([^"]*)"', info))
        if labels.get('kv_cache_size_tokens', '').isdigit() and blocks:
            return {'kv_cache_size_tokens': int(labels['kv_cache_size_tokens']), 'max_concurrency': float(labels['kv_cache_max_concurrency']) if labels.get('kv_cache_max_concurrency') not in (None, 'None') else None, 'block_size': max(blocks), 'max_model_len': int(os.environ.get('MAX_LEN', '16384')), 'note': 'Published hybrid full-model-length concurrency equivalent, NOT physical token capacity. Legacy strict estimates can wrongly reject short-context cohorts; advisory decode mode uses live occupancy/preemption checks. Master metric block_size can be stale16, while worker allocation is1024.'}

    return None


def run(spec, client, pool, out):
    pd = spec['pd']; validate_config(pd)
    result = {'schema_version': 1, 'bench_mode': 'pd', 'prefill': [], 'decode': [],
              'generator': 'BatchBench eligible base-vocabulary IDs excluding special tokens, uniform sampling; Python seeded RNG (not Rust byte-identical); exact token-ID completion prompts, no chat-template/re-tokenization overhead.',
              'timing': 'Client monotonic SSE arrival times; repeated timestamps preserve speculative bursts. Window (latest warmup token, first finish] excludes admission and tail. ITL includes zero gaps within bursts; not per-device execution latency.',
              'slack_tokens': pd.get('decode', {}).get('slack_tokens', 512)}
    def save():
        out.write_text(json.dumps(result, indent=2) + '\n')
    capacity = read_capacity(client, out.parent / 'serve.log'); result['capacity'] = capacity
    rng = random.Random(20261021)
    def prompts(n, c): return [[rng.choice(pool) for _ in range(c)] for _ in range(n)]
    save()
    client.cache_affinity = pd['prefill'].get('cache_affinity', False)
    for case in pd['prefill']['cases']:
        for n in case['concurrency']:
            for repeat in range(pd['prefill'].get('repeats', 1)):
                client.reset()
                prefix = case.get('cached_prefix_tokens', 0); new = case.get('new_tokens', case.get('prompt_tokens'))
                check = capacity_check([prefix + new] * n, 1, capacity)
                if not check['allowed']:
                    result['prefill'].append(dict(name=case['name'], concurrency=n, repeat=repeat, valid=False, skipped=True, flags=check['flags'], capacity_check=check)); save(); continue
                ids = prompts(n, prefix + new)
                warm_check = None
                if prefix:
                    warm = [p[:prefix] for p in ids]
                    client.cohort(warm, 1, 0)
                    if pd['prefill'].get('verify_warm_cache', False):
                        records, metrics = client.cohort(warm, 1, 0)
                        warm_check = prefill_result(records, dict(case, new_tokens=0), n, metrics, capacity['block_size'])
                rows, samples = client.cohort(ids, 1, 0)
                row = prefill_result(rows, case, n, samples, capacity['block_size'])
                row.update(capacity_check=check, repeat=repeat, warm_cache_verification=warm_check,
                           cache_affinity=client.cache_affinity)
                if warm_check is not None and not warm_check['valid']:
                    row['valid'] = False
                    row['flags'].append('untimed_warm_cache_verification_failed')
                result['prefill'].append(row); save()
    client.cache_affinity = False
    if 'decode' not in pd: return result
    d = pd['decode']; total = d['warmup_tokens'] + d['measure_tokens'] + d.get('slack_tokens', 512)
    for source in d.get('context_source', ['random']):
        replay, replay_lengths = [], []
        if source == 'agent-replay':
            path = Path(__file__).with_name('profile-payloads.jsonl')
            result['replay_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            replay = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(replay) != 32: raise ValueError('expected32 agent replay payloads')
            for body in replay:
                tok_body = {k: v for k, v in body.items() if k in ('messages', 'tools', 'tool_choice', 'chat_template', 'chat_template_kwargs', 'add_generation_prompt', 'continue_final_message')}
                tok_body['model'] = 'dsv4-flash'
                tokenized = json.loads(client.fetch('/tokenize', tok_body))
                replay_lengths.append(tokenized['count'])
        for context in d['context_tokens'] if source == 'random' else [None]:
            for n in d['concurrency']:
                client.reset()
                lengths = [context] * n if source == 'random' else [replay_lengths[i % 32] for i in range(n)]
                mode = d.get('capacity_check', 'strict')
                max_len = int(spec.get('env', {}).get('MAX_LEN', os.environ.get('MAX_LEN', (capacity or {}).get('max_model_len', 16384))))
                check = decode_capacity_check(lengths, total, capacity, mode, max_len)
                row = {'context_source': source, 'concurrency': n, 'context_tokens': context, 'actual_context_lengths': lengths,
                       'kv_prompt_tokens': sum(lengths), 'capacity_check': check}
                if not check['allowed']:
                    result['decode'].append(dict(row, valid=False, skipped=True, flags=check['flags'])); save(); continue
                payloads = prompts(n, context) if source == 'random' else [replay[i % 32] for i in range(n)]
                rows, samples = client.cohort(payloads, total, d['temperature'], monitor=True, warmup_tokens=d['warmup_tokens'])
                if [r['usage']['prompt_tokens'] for r in rows] != lengths:
                    raise RuntimeError('replay tokenizer counts differ from server usage')
                window = decode_window(rows, d['warmup_tokens'], d['measure_tokens'])
                verification = window_metrics(samples, window, samples[0]['metrics'], sum(lengths))
                boundaries = {s['boundary']: s for s in samples if 'boundary' in s}
                if set(boundaries) == {'start', 'end'}:
                    b, a = boundaries['start'], boundaries['end']
                    delta = {k: v - b['metrics'][k] for k, v in a['metrics'].items() if k in b['metrics'] and k.startswith('vllm:spec_decode_') and k.split('{')[0].endswith('_total')}
                    drafted = delta.get('vllm:spec_decode_num_draft_tokens_total', 0)
                    verification['spec_acceptance_rate'] = delta.get('vllm:spec_decode_num_accepted_tokens_total', 0) / drafted if drafted > 0 else None
                    verification['boundary_spec_counter_deltas'] = delta
                    verification['boundary_spec_statistics'] = spec_statistics(delta, window.get('seconds', 0), n)
                    verification['boundary_scrapes'] = boundaries
                    verification['boundary_scrape_lag_seconds'] = {label: snap['end'] - window.get(label, snap['end']) for label, snap in boundaries.items()}
                if mode == 'advisory':
                    verification['occupancy'] = occupancy_metrics(samples, window, n)
                    verification['flags'] += verification['occupancy']['flags']
                flags = window['flags'] + verification['flags']
                row.update(spec_statistics=verification.get('boundary_spec_statistics', spec_statistics({}, window.get('seconds', 0), n)),
                    valid=not flags, flags=flags, window=window, verification=verification, requests=rows, metrics=samples,
                    kv_tokens_at_window_start=sum(lengths) + sum(sum(t <= window.get('start', -1) for t in r['token_times']) for r in rows))
                result['decode'].append(row); save()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job', type=Path); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--tokenizer', default='/opt/dsv4/ckpt/dsv4-flash-full/tokenizer.json')
    p.add_argument('--dry-run', action='store_true'); a = p.parse_args()
    spec = json.loads(a.job.read_text()); validate_config(spec['pd'])
    if a.dry_run:
        print(json.dumps({'acceptance_runs_first': True, 'pd': spec['pd'], 'output': str(a.out)}, indent=2)); return
    run(spec, Client(a.base_url), eligible_tokens(a.tokenizer), a.out)


if __name__ == '__main__': main()
