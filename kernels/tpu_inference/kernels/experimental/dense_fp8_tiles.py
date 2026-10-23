"""Validated, immutable per-process dense FP8 tile overrides (no device work)."""
import functools
import json
import os

ROWS = frozenset(range(1, 34)) | {64, 65, 128, 129, 256, 257, 512, 513}
SHAPES = {(1024, 32768), (8192, 4096), (4096, 4096), (2048, 4096),
          (4096, 1536), (1024, 8192), (1024, 40960)}


def validate_table(data):
    if not isinstance(data, dict) or type(data.get('version')) is not int or data['version'] != 1:
        raise ValueError('dense FP8 tile table requires version=1')
    if data.get('device') != 'TPU v6 lite':
        raise ValueError('dense FP8 tile table requires TPU v6 lite')
    if not isinstance(data.get('entries'), list):
        raise ValueError('dense FP8 tile table requires entries list')
    table = {}
    for row in data['entries']:
        if not isinstance(row, dict):
            raise ValueError('tile entry must be an object')
        dims = tuple(row.get(key) for key in ('m', 'k', 'n'))
        if any(type(x) is not int for x in dims):
            raise ValueError('tile dimensions must be integers')
        m, k, n = dims
        if m not in ROWS or (k, n) not in SHAPES or dims in table:
            raise ValueError(f'unsupported or duplicate dense tile: {dims}')
        mode = row.get('mode')
        if mode == 'baseline':
            table[dims] = None
            continue
        if mode not in ('native', 'megablox'):
            raise ValueError('only native/megablox/baseline tiles are supported')
        tile = tuple(row.get(key) for key in ('tm', 'tk', 'tn'))
        if any(type(x) is not int or x <= 0 for x in tile):
            raise ValueError('tile sizes must be positive integers')
        tm, tk, tn = tile
        if m % tm or k % tk or n % tn or tk % 128 or tn % 128:
            raise ValueError('tile must divide dimensions; K/N tiles align to 128')
        if tm != m and tm % 8:
            raise ValueError('partial M tiles must align to 8')
        if (k, n) == (1024, 40960) and mode != 'native':
            raise ValueError('fusion search only qualifies native tiles')
        table[dims] = (mode, tm, tk, tn)
    return table


@functools.lru_cache(maxsize=None)
def _load(path):
    # Read once: changing a file after tracing cannot change compiled executables.
    with open(path, encoding='utf-8') as handle:
        return validate_table(json.load(handle))


def overrides():
    """Set DSV4_DENSE_FP8_TILES before startup; absent means unchanged defaults."""
    path = os.environ.get('DSV4_DENSE_FP8_TILES', '')
    return _load(path) if path else {}
