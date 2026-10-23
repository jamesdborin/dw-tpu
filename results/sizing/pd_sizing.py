#!/usr/bin/env python3
"""Sizing and breakeven pricing for disaggregated DSV4 Flash serving on TPU v6e.

Deployment: one 8-chip prefill host (TP8, attention DP8) + 64 decode chips run as two
32-chip copies (TP32, attention DP32). Workload: the 10-turn agent benchmark.

The inputs are measured prefill-only and decode-only throughputs (see ../README.md for the
jobs they come from). The method is the same as runner/pd_model.py: each side's capacity in
agent turns per second, the system runs at the smaller of the two with the binding side held
below saturation, and prefill TTFT is an M/G/1 queue per DP rank.

    python3 pd_sizing.py            # prints the table in ../README.md
"""
import math

# ---------------------------------------------------------------- workload
TURNS = 10                 # turns per agent
OUT_TOK = 500              # output tokens per turn
COLD_FIRST = 5000          # uncached prompt tokens on the first turn
FOLLOWUP = 1650            # uncached prompt tokens on each later turn (prefill never holds decode's KV)
MEAN_PROMPT = 3138302 / 320  # measured mean prompt per turn incl. chat template (9,807 tokens)
UNCACHED = (COLD_FIRST + (TURNS - 1) * FOLLOWUP) / TURNS  # 1,985 per turn
CACHED = MEAN_PROMPT - UNCACHED                            # 7,822 per turn

# ------------------------------------------------- measured prefill (8 chips)
# Computed (uncached) prompt tokens/s for the whole 8-chip instance, from observed cached_tokens.
PREFILL = {
    'c8': dict(cold=19843.9, incr=19165.2, dp=8,
               desc='GMM_TP + packed gather, bf16 reduce, dense FP8 GEMM, 0401-0405 (fastest)'),
    'c2': dict(cold=11905.0, incr=11648.9, dp=8,
               desc='GMM_EP + packed gather + f32 reduce + one-shot RS, 0401-0403 (passes the TP32 gate)'),
}

# --------------------------------------------- measured decode (32-chip copy)
# Per-user decode tok/s on real agent contexts at the top of each per-rank bucket regime:
# (name, max requests per copy, tok/s per user) = 500 / mean(500 / rate) over the window.
DECODE = [
    ('MTP k=3', 32, 45.967),
    ('MTP k=3', 64, 37.352),
    ('MTP k=3', 128, 31.402),
    ('GMM_EP, no spec', 128, 29.925),
    ('GMM_EP, no spec', 256, 22.493),
]
COPIES = 2

# -------------------------------------------------------------------- cost
CHIP_HOUR = 1.89     # v6e calendar rate, $/chip-hour
DISK_HOUR = 0.11     # boot/data disk per host, $/hour
HOSTS = 64 // 4 + 1  # 16 ct6e-standard-4t decode hosts + 1 ct6e-standard-8t prefill host
CHIPS = 64 + 8
RATIO = (1, 0.1, 4)  # input : cached input : output price ratio


def prefill_capacity(p):
    """Agent turns/s one instance can prefill."""
    sec = COLD_FIRST / p['cold'] + (TURNS - 1) * FOLLOWUP / p['incr']
    return TURNS / sec


def prefill_ttft(p, x):
    """Mean TTFT (s): M/G/1 per DP rank; service = 512-token chunk steps."""
    rate = (p['cold'] * COLD_FIRST + p['incr'] * (TURNS - 1) * FOLLOWUP) / (COLD_FIRST + (TURNS - 1) * FOLLOWUP)
    rate_rank, lam = rate / p['dp'], x / p['dp']
    step = 512 * p['dp'] / ((p['cold'] + p['incr']) / 2)
    sizes = [(1 / TURNS, COLD_FIRST), (1 - 1 / TURNS, FOLLOWUP)]
    rho = lam * sum(q * c / rate_rank for q, c in sizes)
    if rho >= 1:
        return math.inf
    es2 = sum(q * (c / rate_rank) ** 2 for q, c in sizes)
    return lam * es2 / (2 * (1 - rho)) + sum(q * math.ceil(c / 512) * step for q, c in sizes)


def point(pname, dec, prefill_util=0.85, decode_util=0.95):
    p = PREFILL[pname]
    name, n_max, rate = dec
    d_cap = COPIES * n_max * rate / OUT_TOK
    p_cap = prefill_capacity(p)
    x = min(decode_util * d_cap, prefill_util * p_cap)
    cost = CHIPS * CHIP_HOUR + HOSTS * DISK_HOUR
    unc, cac, out = x * UNCACHED, x * CACHED, x * OUT_TOK
    p_in = cost / (3600 * (RATIO[0] * unc + RATIO[1] * cac + RATIO[2] * out) / 1e6)
    ttft = prefill_ttft(p, x)
    return dict(prefill=pname, decode=f'{name}, <={n_max}/copy', turns_s=x,
                binds='prefill' if prefill_util * p_cap < decode_util * d_cap else 'decode',
                decode_util=x / d_cap, prefill_util=x / p_cap, running=x * OUT_TOK / rate,
                out_tok_s=out, tpm_incl=x * (MEAN_PROMPT + OUT_TOK) * 60, tpm_excl=x * (UNCACHED + OUT_TOK) * 60,
                tok_s_user=rate, ttft=ttft, turn_tok_s=OUT_TOK / (ttft + OUT_TOK / rate),
                cost=cost, price=(p_in * RATIO[0], p_in * RATIO[1], p_in * RATIO[2]),
                blended=cost / (3600 * (unc + cac + out) / 1e6))


def main():
    print(f'cost ${CHIPS * CHIP_HOUR + HOSTS * DISK_HOUR:.2f}/hour for {CHIPS} chips; '
          f'per turn {MEAN_PROMPT:.0f} prompt ({UNCACHED:.0f} uncached) + {OUT_TOK} output')
    for pname, p in PREFILL.items():
        print(f'\n{pname}: {p["desc"]}; prefill capacity {prefill_capacity(p):.2f} turns/s')
        print('decode | turns/s | binds | util D/P | running | out tok/s | TPM incl/excl cached | '
              'tok/s/user | TTFT s | $/M input / cached / output | blended')
        for dec in DECODE:
            r = point(pname, dec)
            if r['running'] > COPIES * dec[1]:
                continue
            print(f"{r['decode']} | {r['turns_s']:.2f} | {r['binds']} | {r['decode_util']:.0%}/{r['prefill_util']:.0%} | "
                  f"{r['running']:.0f} | {r['out_tok_s']:,.0f} | {r['tpm_incl'] / 1e6:.2f}M / {r['tpm_excl'] / 1e6:.2f}M | "
                  f"{r['tok_s_user']:.1f} | {r['ttft']:.1f} | {r['price'][0]:.2f} / {r['price'][1]:.3f} / {r['price'][2]:.2f} | "
                  f"{r['blended']:.2f}")


if __name__ == '__main__':
    main()
