"""Conservative strict model/runner compilation gate; preserve all signature lines."""
import datetime as dt
import re


def classify_compiles(text, ready_unix):
    ready = dt.datetime.fromtimestamp(ready_unix, dt.timezone.utc)
    failures, historical = [], []
    for line in text.splitlines():
        if 'no_tracing' in line and ('re-tracing' in line or 'recompil' in line.lower()):
            failures.append({'reason': 'no_tracing rejected model signature', 'line': line})
            continue
        lower = line.lower()
        if not any(x in lower for x in ('compiling jit(', 'compilation of worker', 'finished xla compilation of jit(', 're-tracing function')):
            continue
        # Restrict to serving model/runner paths, not unrelated scalar HTTP helpers.
        if not any(x in lower for x in ('model', 'decode', 'prefill', 'logits', 'logprobs', 'sample', 'select_from', 'compilation_manager.py', 'tpu_runner')):
            continue
        stamp = re.search(r'(\d{4}-\d\d-\d\d)[ T](\d\d:\d\d:\d\d(?:[.,]\d+)?)', line)
        short = re.search(r'\b(\d\d-\d\d) (\d\d:\d\d:\d\d(?:[.,]\d+)?)', line)
        try:
            if stamp:
                event = dt.datetime.fromisoformat(stamp[1] + 'T' + stamp[2].replace(',', '.')).replace(tzinfo=dt.timezone.utc)
            elif short:
                candidates = [dt.datetime.fromisoformat(str(y) + '-' + short[1] + 'T' + short[2].replace(',', '.')).replace(tzinfo=dt.timezone.utc) for y in (ready.year-1, ready.year, ready.year+1)]
                event = min(candidates, key=lambda x: abs((x-ready).total_seconds()))
            else:
                raise ValueError('missing timestamp')
            record = {'event_utc': event.isoformat(), 'line': line}
            (failures if event >= ready else historical).append(record)
        except ValueError:
            failures.append({'reason': 'unknown compile event timestamp', 'line': line})
    return {'historical_startup_lines': len(historical), 'failures': failures}
