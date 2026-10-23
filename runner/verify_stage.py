"""Verify the transported files before activating a source tree (no dependencies)."""
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
m = json.loads((root / 'runner/stage-manifest.json').read_text())
for name, digest in m['files'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('stage hash mismatch: ' + name)
print('STAGE VERIFIED', m['base'], len(m['files']))
