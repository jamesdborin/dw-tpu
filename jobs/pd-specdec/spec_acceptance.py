#!/usr/bin/env python3
"""Sampling-equivalence evidence for speculative decoding (job-local gate).

Runs against the live server at 127.0.0.1:8000 after the runner's standard
probes (greedy tokens vs the non-speculative H4-r2 reference, GPU-reference
prompt logprobs, suffix causality). Writes $OUT/sampling-equivalent.json.

1. Distribution test at temperature 1 (top_p 1, the benchmark's sampling).
   For every generated token the server also returns the target model's
   top-20 log-probabilities at that position (computed from the verifier's
   logits). If tokens are sampled from that distribution, the randomized
   probability integral transform
       U = F(x-) + V * p(x),  V ~ U(0, 1),
   over the coarsened distribution (top-1 .. top-20, "rest") is i.i.d.
   Uniform(0, 1) across all positions and requests, whatever the drafts were.
   A sampler biased toward the draft (or any acceptance-rule bug) shifts mass
   to the top ranks. Checks: Kolmogorov-Smirnov on U, and z-tests of the
   observed top-1 and "rest" counts against their expected sums. Position 0
   (sampled from the prefill logits, no draft involved) is reported
   separately as a control.
2. Greedy repeat determinism over 256 tokens on the three probe prompts
   (identical tokens and logprobs on a second run).
3. Speculative-decoding counters from /metrics (drafts, accepted tokens per
   position, acceptance rate).
"""
import concurrent.futures as cf
import json
import math
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

HOST = os.environ.get("SPEC_HOST", "http://127.0.0.1:8000")
OUT = Path(os.environ.get("OUT", "."))
DW_TPU = Path(os.environ.get("DW_TPU", "/opt/dsv4/dw-tpu"))
TOPK = 20
N_REQUESTS = int(os.environ.get("SPEC_N_REQUESTS", "128"))
N_TOKENS = int(os.environ.get("SPEC_N_TOKENS", "256"))
CONCURRENCY = 32
KS_P_MIN = 1e-4
Z_MAX = 5.0


def post(path, data=None, timeout=1800):
    # Same request form as runner/vendor/correctness.py (empty POST body for
    # /reset_prefix_cache).
    req = urllib.request.Request(
        HOST + path,
        data=json.dumps(data).encode() if data is not None else b"",
        headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def spec_metrics():
    text = urllib.request.urlopen(HOST + "/metrics",
                                  timeout=30).read().decode()
    out = {}
    for line in text.splitlines():
        m = re.match(r"(vllm:spec_decode_\w+)(\{[^}]*\})?\s+([\d.eE+-]+)$",
                     line)
        if m:
            labels = m.group(2) or ""
            pos = re.search(r'position="(\d+)"', labels)
            key = m.group(1) + (f"[{pos.group(1)}]" if pos else "")
            out[key] = out.get(key, 0.0) + float(m.group(3))
    return out


def request_body(ids, max_tokens, temperature, logprobs=TOPK):
    # No per-request `seed`: the TPU platform rejects it at request
    # validation (SamplingType.RANDOM_SEED) once temperature > 0.
    return dict(model="dsv4-flash",
                prompt=ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                ignore_eos=True,
                logprobs=logprobs,
                return_tokens_as_token_ids=True,
                return_token_ids=True)


# Every request shape this script sends (checked offline against vLLM's
# request parsing and the TPU platform's validate_request).
REQUEST_SHAPES = {
    "sampling": dict(max_tokens=N_TOKENS, temperature=1.0, logprobs=TOPK),
    "greedy": dict(max_tokens=256, temperature=0.0, logprobs=1),
}


def complete(ids, max_tokens, temperature, seed=None, logprobs=TOPK):
    del seed
    return post("/v1/completions",
                request_body(ids, max_tokens, temperature, logprobs))


def token_id(key):
    return int(key.split(":", 1)[1]) if isinstance(key, str) else int(key)


def pit_values(choice, rng):
    """Randomized PIT per generated position: (position, U, rank, p1, rest)."""
    lp = choice["logprobs"]
    rows = []
    for pos, (tok, top) in enumerate(zip(choice["token_ids"],
                                         lp["top_logprobs"])):
        if top is None:
            raise ValueError(f"missing top_logprobs at position {pos}")
        items = sorted(((token_id(k), math.exp(v)) for k, v in top.items()),
                       key=lambda kv: -kv[1])[:TOPK]
        probs = [p for _, p in items]
        ids = [t for t, _ in items]
        s = sum(probs)
        if s > 1.0 + 1e-3:
            raise ValueError(f"top-{TOPK} probabilities sum to {s}")
        rest = max(0.0, 1.0 - s)
        v = rng.random()
        if tok in ids:
            r = ids.index(tok)
            u = sum(probs[:r]) + v * probs[r]
        else:
            r = TOPK
            u = s + v * rest
        rows.append((pos, min(max(u, 0.0), 1.0), r, probs[0], rest))
    return rows


def ks_uniform(values):
    xs = sorted(values)
    n = len(xs)
    d = max(max((i + 1) / n - x, x - i / n) for i, x in enumerate(xs))
    # Asymptotic Kolmogorov distribution with the Stephens correction.
    t = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2 * sum((-1)**(k - 1) * math.exp(-2 * k * k * t * t)
                for k in range(1, 101))
    return d, min(max(p, 0.0), 1.0)


def count_test(rows):
    n_top1 = sum(1 for r in rows if r[2] == 0)
    e_top1 = sum(r[3] for r in rows)
    v_top1 = sum(r[3] * (1 - r[3]) for r in rows)
    n_rest = sum(1 for r in rows if r[2] == TOPK)
    e_rest = sum(r[4] for r in rows)
    v_rest = sum(r[4] * (1 - r[4]) for r in rows)
    z1 = (n_top1 - e_top1) / math.sqrt(max(v_top1, 1e-9))
    zr = (n_rest - e_rest) / math.sqrt(max(v_rest, 1e-9))
    return dict(top1_observed=n_top1,
                top1_expected=e_top1,
                top1_z=z1,
                rest_observed=n_rest,
                rest_expected=e_rest,
                rest_z=zr)


def evaluate(rows):
    u = [r[1] for r in rows]
    d, p = ks_uniform(u)
    deciles = [0] * 10
    for x in u:
        deciles[min(int(x * 10), 9)] += 1
    out = dict(n=len(rows), ks_d=d, ks_p=p, pit_deciles=deciles)
    out.update(count_test(rows))
    out["passed"] = (p > KS_P_MIN and abs(out["top1_z"]) < Z_MAX
                     and abs(out["rest_z"]) < Z_MAX)
    return out


def synthetic_prompts(n, length, seed):
    rng = random.Random(seed)
    return [[rng.randrange(1000, 120000) for _ in range(length)]
            for _ in range(n)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {"passed": False, "distribution_preserved": False}
    path = OUT / "sampling-equivalent.json"
    try:
        prompts = json.loads(
            (DW_TPU / "runner/vendor/prompts.json").read_text())["base"]
        probe = [prompts[0], prompts[1], prompts[3]]
        before = spec_metrics()

        # 1. Distribution test.
        sample_prompts = synthetic_prompts(N_REQUESTS - len(probe), 512, 7)
        sample_prompts += [item["ids"] for item in probe]
        t0 = time.time()
        with cf.ThreadPoolExecutor(CONCURRENCY) as pool:
            responses = list(
                pool.map(
                    lambda a: complete(a[1], N_TOKENS, 1.0, 1000 + a[0]),
                    enumerate(sample_prompts)))
        sample_seconds = time.time() - t0
        (OUT / "sampling-responses.json").write_text(json.dumps(responses))
        rng = random.Random(12345)
        rows = []
        for response in responses:
            choice = response["choices"][0]
            assert len(choice["token_ids"]) == N_TOKENS, len(
                choice["token_ids"])
            rows += pit_values(choice, rng)
        after_sampling = spec_metrics()
        spec_rows = [r for r in rows if r[0] > 0]
        first_rows = [r for r in rows if r[0] == 0]
        dist = evaluate(spec_rows)
        control = evaluate(first_rows)

        # 2. Greedy repeat determinism (256 tokens).
        greedy = {}
        greedy_ok = True
        for item in probe:
            assert post("/reset_prefix_cache")["success"]
            a = complete(item["ids"], 256, 0.0, 0, logprobs=1)["choices"][0]
            b = complete(item["ids"], 256, 0.0, 0, logprobs=1)["choices"][0]
            same = (a["token_ids"] == b["token_ids"]
                    and a["logprobs"]["token_logprobs"]
                    == b["logprobs"]["token_logprobs"])
            greedy_ok &= same
            greedy[item["name"]] = dict(token_ids=a["token_ids"],
                                        token_logprobs=a["logprobs"]
                                        ["token_logprobs"],
                                        repeat_identical=same)
        (OUT / "greedy-256.json").write_text(json.dumps(greedy, indent=1))
        after = spec_metrics()

        # Diagnostic only (never gates): re-score the speculatively generated
        # greedy tokens teacher-forced through the prefill path (echo +
        # prompt logprobs, no drafts) and record the per-position logprob gap
        # to the verify-step logprobs.
        teacher_forced = {}
        try:
            for item in probe:
                g = greedy[item["name"]]
                assert post("/reset_prefix_cache")["success"]
                body = request_body(item["ids"] + g["token_ids"], 1, 0.0, 1)
                body["echo"] = True
                r = post("/v1/completions", body)["choices"][0]
                tf = r["logprobs"]["token_logprobs"][len(item["ids"]):][
                    :len(g["token_ids"])]
                gaps = [abs(a - b) for a, b in zip(tf, g["token_logprobs"])]
                teacher_forced[item["name"]] = dict(
                    max_abs_gap=max(gaps),
                    mean_abs_gap=sum(gaps) / len(gaps),
                    first_gap_over_1e3=next(
                        (i for i, x in enumerate(gaps) if x > 1e-3), None),
                    gaps=gaps)
        except Exception as exc:  # diagnostic must not fail the gate
            teacher_forced["error"] = repr(exc)
        (OUT / "teacher-forced-vs-spec.json").write_text(
            json.dumps(teacher_forced, indent=1))

        def delta(x, y):
            return {k: y[k] - x.get(k, 0.0) for k in y}

        sampling_counters = delta(before, after_sampling)
        drafted = sampling_counters.get(
            "vllm:spec_decode_num_draft_tokens_total", 0.0)
        accepted = sampling_counters.get(
            "vllm:spec_decode_num_accepted_tokens_total", 0.0)
        drafts = sampling_counters.get("vllm:spec_decode_num_drafts_total",
                                       0.0)
        # Speculation must actually have run, or the PIT test is vacuous.
        # Missing counters (metrics not exported) are reported, not failed.
        counters_present = (
            "vllm:spec_decode_num_draft_tokens_total" in after_sampling)
        speculation_ok = drafted > 0 or not counters_present
        evidence = {
            "passed": bool(dist["passed"] and greedy_ok and speculation_ok),
            "spec_counters_present": counters_present,
            "distribution_preserved": bool(dist["passed"]),
            "distribution_test": dist,
            "position0_control": control,
            "greedy_repeat_identical": greedy_ok,
            "teacher_forced_vs_spec_diagnostic": {
                k: {kk: vv for kk, vv in v.items() if kk != "gaps"}
                if isinstance(v, dict) else v
                for k, v in teacher_forced.items()
            },
            "sampling_requests": len(responses),
            "sampling_tokens_per_request": N_TOKENS,
            "sampling_wall_seconds": sample_seconds,
            "sampling_spec_counters": sampling_counters,
            "sampling_acceptance_rate":
            accepted / drafted if drafted else None,
            "sampling_mean_accepted_per_draft":
            accepted / drafts if drafts else None,
            "greedy_spec_counters": delta(after_sampling, after),
            "notes": ("PIT over top-20 + rest, i.i.d. U(0,1) under the "
                      "target distribution; thresholds KS p > 1e-4, "
                      "|z| < 5. Speculation must actually run "
                      "(drafted > 0 when the counters are exported)."),
        }
    except Exception as exc:  # recorded, then the gate fails
        evidence["error"] = repr(exc)
        path.write_text(json.dumps(evidence, indent=2) + "\n")
        raise
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: v for k, v in evidence.items()
                      if k != "sampling_spec_counters"}, indent=2))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
