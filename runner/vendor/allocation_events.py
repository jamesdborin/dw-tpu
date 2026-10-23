"""R24: classify allocation event time, never Ray log arrival time (UTC hosts)."""
import datetime as dt
import re

RECOVERED = ('final_attempt: false', 'ExecutePrepareWithOomRetries attempting to defragment and retry')
STAMP = re.compile(r'\b[IWFE](\d{2})(\d{2}) (\d{2}:\d{2}:\d{2}\.\d{1,6})\s')

def classify(text, ready_unix):
    ready = dt.datetime.fromtimestamp(ready_unix, dt.timezone.utc)
    historical, failures = [], []
    for line in text.splitlines():
        if 'RESOURCE_EXHAUSTED' not in line:
            continue
        m = STAMP.search(line)
        if not m:
            failures.append({'reason': 'unknown event timestamp', 'line': line})
            continue
        try:
            candidates = [dt.datetime.strptime(f'{year}-{m[1]}-{m[2]} {m[3]}', '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc) for year in (ready.year-1, ready.year, ready.year+1)]
            event = min(candidates, key=lambda v: abs((v-ready).total_seconds()))
        except ValueError:
            failures.append({'reason': 'invalid event timestamp', 'line': line})
            continue
        if event >= ready or not any(x in line for x in RECOVERED):
            failures.append({'reason': 'event at/after ready or recovery not explicit', 'event_utc': event.isoformat(), 'line': line})
        else:
            historical.append({'event_utc': event.isoformat(), 'line': line})
    return {'historical_startup_lines': len(historical), 'failures': failures, 'policy': 'R24 event timestamps; unknown or >=ready fails'}
