#!/usr/bin/env python3
"""Check raw per-host logs, including Ray events not forwarded by the head."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
from allocation_events import classify
from compile_events import classify_compiles
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True); p.add_argument('--ready', type=float)
p.add_argument('--strict-compiles', action='store_true')
p.add_argument('--dry-run', action='store_true'); a = p.parse_args()
if a.dry_run:
    print('Audit', a.out, 'and /tmp/ray/session_latest/logs at event timestamps >=', a.ready); sys.exit()
paths = list(a.out.glob('*.log'))
ray = Path('/tmp/ray/session_latest/logs')
if ray.exists():
    paths += [x for x in ray.iterdir() if x.is_file() and x.suffix in ('.out', '.err', '.log')]
failures = []; historical = 0
for path in paths:
    # Audit only serving/Ray logs, not audit JSON or controller error messages.
    if path.name == 'controller.log':
        continue
    text = path.read_text(errors='replace')
    result = classify(text, a.ready) if a.ready is not None else {'historical_startup_lines': 0, 'failures': []}
    historical += result['historical_startup_lines']
    failures.extend(dict(x, file=str(path)) for x in result['failures'])
    compilation = classify_compiles(text, a.ready) if a.ready is not None else {'failures': []}
    if a.strict_compiles:
        failures.extend(dict(x, file=str(path), kind='compile') for x in compilation['failures'])
    if re.search(r'launch[ _-]*id[^\n]*mismatch', text, re.I):
        failures.append({'file': str(path), 'reason': 'launch-id mismatch'})
result = {'failures': failures, 'historical_startup_lines': historical, 'files_checked': len(paths)}
(a.out / 'all-host-allocation-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
sys.exit(bool(failures))
