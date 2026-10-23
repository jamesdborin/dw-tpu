#!/usr/bin/env python3
"""Best-case P/D model, with explicit slice cost and no KV-transfer cost."""
import argparse
import json
from pathlib import Path
import statistics


def model(pd, prefill_pd=None):
    if prefill_pd is not None: return model_v2(pd, prefill_pd)
    prefills = [r for r in pd.get('prefill', []) if r.get('valid') and r.get('concurrency') == 32 and r.get('cached_prefix_tokens', 0) > 0]
    prefill = prefills[0] if prefills else None
    ttft = prefill['ttft_seconds']['mean'] if prefill else None
    rows = []
    for row in pd.get('decode', []):
        if row.get('concurrency') != 32 or not row.get('valid'): continue
        rates = [r['decode_tok_s'] for r in row['window']['requests']]
        # Average per-user decode time, not 500 / arithmetic mean rate.
        decode_seconds = statistics.mean(500 / r for r in rates) if rates and min(rates) > 0 else None
        if not decode_seconds: continue
        context = statistics.mean(row['actual_context_lengths'])
        item = {'context_source': row['context_source'], 'context_tokens': context,
                'per_user_decode_tok_s': statistics.mean(rates), 'decode_seconds_for500': decode_seconds,
                'spec_statistics': row_spec_statistics(row)}
        for name, prefill_seconds, slices in [('separate_equal_slices', ttft, 2), ('decode_only_prefill_free', 0, 1)]:
            if prefill_seconds is None:
                item[name] = None; continue
            turn = prefill_seconds + decode_seconds
            tpm = 32 * (context + 500) * 60 / turn
            item[name] = {'slices': slices, 'turn_seconds': turn, 'tok_s_user': 500 / turn,
                          'logical_tpm_including_cached': tpm, 'logical_tpm_per_slice': tpm / slices}
        rows.append(item)
    random_rows = [r for r in rows if r['context_source'] == 'random' and r['context_tokens'] in (5000, 10000, 15000)]
    average = None
    if {r['context_tokens'] for r in random_rows} == {5000, 10000, 15000}:
        average = {'per_user_decode_tok_s': statistics.mean(r['per_user_decode_tok_s'] for r in random_rows)}
        for scenario in ('separate_equal_slices', 'decode_only_prefill_free'):
            if all(r[scenario] is not None for r in random_rows):
                turn = statistics.mean(r[scenario]['turn_seconds'] for r in random_rows)
                slices = random_rows[0][scenario]['slices']
                average[scenario] = {'mean_turn_seconds': turn, 'tok_s_user': 500 / turn,
                    'logical_tpm_including_cached': 32 * (10000 + 500) * 60 / turn,
                    'logical_tpm_per_slice': 32 * (10000 + 500) * 60 / turn / slices}
    return {'assumptions': [
        'Best-case model, not a measured disaggregated deployment; valid uncontaminated N=32 cases only.',
        'Each turn emits500 tokens; logical TPM counts the full prompt including cached history plus500 output tokens, no extra double-counting of the1650 increment.',
        'Separate equal-size P and D slices: mean incremental-prefill TTFT at concurrency32 plus mean per-user500/rate; two32-chip slices. Per-slice TPM divides by2.',
        'Decode-only scenario: prefill and KV transfer are free; one32-chip slice.',
        'No network/KV-transfer, routing, tool latency or queueing beyond measured batched TTFT; fixed N=32 decode occupancy.',
        'Random contexts5k/10k/15k weighted equally for average; agent-replay reports actual lengths separately. No interpolation or substitution for missing/flagged cases.',
        'Spec num_drafts sums per-request steps over DP ranks: step_seconds=window_seconds*N/drafts; tokens_per_step=1+accepted/drafts; per-position acceptance=accepted_pos/drafts. Boundary scrape lag is recorded; these are counter-normalized estimates.'
    ], 'incremental_prefill_ttft_seconds': ttft, 'contexts': rows, 'random_context_average': average}


def row_spec_statistics(row):
    from vendor.pd_bench import spec_statistics
    verification = row.get('verification', {})
    return (row.get('spec_statistics') or verification.get('boundary_spec_statistics') or
            spec_statistics(verification.get('boundary_spec_counter_deltas', {}),
                            row.get('window', {}).get('seconds', 0), row['concurrency']))


def model_v2(decode, prefill):
    """Capacity sizing at full decode occupancy; also show fixed-population latency."""
    groups = {}
    for r in prefill.get('prefill', []):
        if not r.get('valid') or r.get('new_tokens') != 1650 or not r.get('cached_prefix_tokens'): continue
        key = (r['cached_prefix_tokens'], r['concurrency'])
        groups.setdefault(key, []).append(r)
    pools = []
    for (context, n), rows in sorted(groups.items()):
        seconds = sum(r['seconds'] for r in rows)
        if seconds <= 0: continue
        ttfts = [q['token_times'][0] - q['submit'] for r in rows for q in r.get('requests', [])]
        from vendor.pd_bench import distribution
        pools.append({'cached_prefix_tokens': context, 'concurrency': n, 'valid_repeats': len(rows),
                      'new_tok_s': sum(r['concurrency'] * r['new_tokens'] for r in rows) / seconds,
                      'ttft_seconds': distribution(ttfts) if ttfts else {
                          k: statistics.mean(r['ttft_seconds'][k] for r in rows) for k in ('mean', 'p50', 'p90')}})
    results = []
    for row in decode.get('decode', []):
        if not row.get('valid'): continue
        rates = [r['decode_tok_s'] for r in row['window']['requests']]
        if not rates or min(rates) <= 0: continue
        n = row['concurrency']; context = statistics.mean(row['actual_context_lengths'])
        decode_seconds = statistics.mean(500 / r for r in rates)
        # Mean service time gives a consistent renewal-workload throughput.
        rate = 500 / decode_seconds
        item = {'context_source': row['context_source'], 'concurrency': n, 'context_tokens': context,
                'per_user_decode_tok_s': rate, 'arithmetic_mean_decode_tok_s': statistics.mean(rates),
                'spec_statistics': row_spec_statistics(row), 'prefill_options': [],
                'decode_only_prefill_free': {'tok_s_user': rate, 'aggregate_output_tok_s_per32chips': n * rate,
                    'logical_tpm_including_cached_per32chips': n * (context + 500) * 60 / decode_seconds}}
        if pools:
            nearest = min(abs(p['cached_prefix_tokens'] - context) for p in pools)
            for pool in pools:
                if abs(pool['cached_prefix_tokens'] - context) != nearest: continue
                demand = n / decode_seconds  # turns/s with N continuously decoding
                ratio = demand * 1650 / pool['new_tok_s']
                ttft = pool['ttft_seconds']['mean']; turn = ttft + decode_seconds
                total_chips = 32 * (1 + ratio)
                scale = 32 / total_chips
                item['prefill_options'].append({
                    'prefill_measurement': pool, 'cached_prefix_minus_decode_context_tokens': pool['cached_prefix_tokens'] - context,
                    'prefill_full_prompt_tokens': pool['cached_prefix_tokens'] + 1650,
                    'prefill_chips_per_decode_slice': 32 * ratio,
                    'p_to_d_ratio': ratio, 'total_chips': total_chips,
                    'turn_seconds': turn, 'ttft_seconds': pool['ttft_seconds'], 'tok_s_user': 500 / turn,
                    'agents_needed_for_full_decode_occupancy': demand * turn,
                    'aggregate_output_tok_s_per32chips': n * rate * scale,
                    'logical_tpm_including_cached_per32chips': demand * (context + 500) * 60 * scale,
                    'logical_tpm_at82pct_apc_per32chips': demand * (1650 / .18 + 500) * 60 * scale,
                    'fixed_N_agents_output_tok_s_per32chips': n * 500 / turn * scale,
                    'fixed_N_agents_logical_tpm_per32chips': n * (context + 500) * 60 / turn * scale})
        results.append(item)
    return {'schema_version': 2, 'assumptions': [
        'Projection, not a measured P/D deployment. Two separate result files; only valid cases enter sizing. No cache-failure waiver.',
        'Each slice has32 chips. Each turn has1650 new prompt tokens and500 output tokens. KV transfer, networking, queueing and tool time are free.',
        'Decode service time is mean(500 / per-request window rate); speculative bursts count tokens, not chunks. All measured N/context/source combinations are reported.',
        'Match decode context to nearest measured cached-prefix length; disclose that prefix and its1650-token extension. Nearest-prefix matching is a proxy, not an interpolation; its mismatch is reported. Ties are both reported.',
        'Every measured prefill concurrency is an alternative pool batching policy, not the decode concurrency. Repeats pool token counts / total wall time; TTFT pools request samples.',
        'P:D = (N / decode_seconds) *1650 / measured prefill new-token throughput. Fractional prefill slices assume linear scaling and identical chips; practical whole-slice allocations can cost more.',
        'Headline aggregate capacity assumes N continuously decoding and enough rotating agents to hide prefill. Required population is reported. Per-user turn latency includes TTFT; fixed-N-population aggregate numbers are separately reported.',
        'Per32-chip metrics normalize by32/(32+prefill_chips). Logical TPM counts context+500 once, including cached tokens. APC82% implies mean full prompt1650/0.18=9167; a separate82%-normalized TPM is reported, not imposed on every context.',
        'Warm block-alignment recomputation is included in measured prefill service. Unavailable spec counters are null, never zero acceptance.'
    ], 'prefill_pools': pools, 'contexts': results}


def markdown(pd):
    result = model(pd)
    lines = ['', '## P/D benchmark', '', 'Best-case projections; see pd.json for raw timestamps, capacity checks and contamination flags.', '',
             '| Source | N | Context | Valid | Decode tok/s/user (mean) | Flags |', '|---|---:|---:|---|---:|---|']
    for r in pd.get('decode', []):
        rate = r.get('window', {}).get('per_user_decode_tok_s', {}).get('mean')
        context = r.get('context_tokens') or (round(statistics.mean(r['actual_context_lengths'])) if r.get('actual_context_lengths') else '?')
        lines.append(f"| {r.get('context_source')} | {r['concurrency']} | {context} | {r.get('valid')} | {rate if rate is not None else 'n/a'} | {', '.join(r.get('flags', []))} |")
    lines += ['', '### Speculative decode counters', '']
    for r in pd.get('decode', []):
        if not r.get('skipped'):
            lines.append(f"- {r.get('context_source')} N={r['concurrency']} context={r.get('context_tokens')}: {json.dumps(row_spec_statistics(r), sort_keys=True)}")
    lines += ['', '### Prefill', '']
    for r in pd.get('prefill', []):
        lines.append(f"- {r['name']} N={r['concurrency']}: valid={r.get('valid')}; new tok/s={r.get('new_prefill_tok_s')}; including cached tok/s={r.get('including_cached_tok_s')}; TTFT={r.get('ttft_seconds')}; new tok/s/chip={r.get('new_prefill_tok_s_per_chip')}; incl-cached tok/s/chip={r.get('including_cached_tok_s_per_chip')}; repeat={r.get('repeat', 0)}; flags={r.get('flags')}")
    lines += ['', '### Agent-workload model', '', '```json', json.dumps(result, indent=2), '```']
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('pd_json', type=Path); p.add_argument('--out', type=Path); p.add_argument('--prefill', type=Path, help='Separate prefill pd.json: enables capacity model v2'); a = p.parse_args()
    result = json.dumps(model(json.loads(a.pd_json.read_text()), json.loads(a.prefill.read_text()) if a.prefill else None), indent=2) + '\n'
    if a.out: a.out.write_text(result)
    print(result)


if __name__ == '__main__': main()
