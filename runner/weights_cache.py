"""Prototype atomic per-host processed-shard cache; no model imports or live hook.

Call publish AFTER quantization/layout conversion, passing a lazy iterable of
(metadata, byte-chunk iterable). restore passes verified mapped shard buffers to
a synchronous consumer, which can copy/device_put before returning. No pickle,
whole-checkpoint gather, global arrays, or CPU re-quantization on cache hits.
"""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import tempfile

CHUNK_BYTES = 8 * 1024**2
REQUIRED_IDENTITY = {'checkpoint_sha256', 'processing_sha256', 'quantization', 'layout_version',
                     'runtime_versions', 'topology', 'host_index', 'host_count', 'model_role', 'shard_schema_sha256'}
REQUIRED_SHARD = {'tensor', 'dtype', 'global_shape', 'local_shape', 'index', 'device_slot', 'layout', 'storage_bytes'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def schema_digest(metadata):
    return hashlib.sha256(canonical(sorted(metadata, key=lambda m: (m['tensor'], m['device_slot'])))).hexdigest()


def cache_key(identity):
    missing = REQUIRED_IDENTITY - identity.keys()
    if missing: raise ValueError('incomplete cache identity: ' + ', '.join(sorted(missing)))
    if type(identity['host_index']) is not int or type(identity['host_count']) is not int or not 0 <= identity['host_index'] < identity['host_count']:
        raise ValueError('invalid host index/count')
    return hashlib.sha256(canonical(identity)).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''): digest.update(chunk)
    return digest.hexdigest()


def publish(root, identity, shards):
    """Bounded streaming publication, one host/key lock, atomic directory rename.

    Each shard is (metadata, chunks); chunks may be memoryview slices of a single
    local host array. Raw storage dtype/shape/layout must be supplied explicitly.
    Logical FP4 dtype is metadata, not an assumed NumPy void dtype roundtrip.
    """
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    key = cache_key(identity); destination = root/key
    with (root/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            verified_manifest(root, identity)
            return destination
        temp = Path(tempfile.mkdtemp(prefix='.'+key+'-', dir=root))
        try:
            entries=[]; seen=set()
            for number, (metadata, chunks) in enumerate(shards):
                if REQUIRED_SHARD - metadata.keys(): raise ValueError('incomplete shard metadata')
                unique=(metadata['tensor'], metadata['device_slot'])
                if unique in seen: raise ValueError('duplicate tensor/device shard')
                seen.add(unique)
                path = temp/f'{number:06d}.bin'; digest=hashlib.sha256(); size=0
                with path.open('wb') as stream:
                    for chunk in chunks:
                        view=memoryview(chunk)
                        if view.nbytes > CHUNK_BYTES: raise ValueError('cache writer chunk exceeds bounded size')
                        stream.write(view); digest.update(view); size += view.nbytes
                    stream.flush(); os.fsync(stream.fileno())
                if size != metadata['storage_bytes']: raise ValueError('shard storage size differs from expected schema')
                entries.append(dict(metadata, file=path.name, bytes=size, sha256=digest.hexdigest()))
            if not entries: raise ValueError('refuse empty cache publication')
            schema=[{k:v for k,v in e.items() if k not in ('file','bytes','sha256')} for e in entries]
            if schema_digest(schema) != identity['shard_schema_sha256']:
                raise ValueError('incomplete or mismatched expected shard schema')
            manifest={'schema_version':1, 'key':key, 'identity':identity, 'shards':entries}
            with (temp/'manifest.json').open('wb') as stream:
                stream.write(canonical(manifest)); stream.flush(); os.fsync(stream.fileno())
            with (temp/'COMPLETE').open('wb') as stream:
                stream.write(hashlib.sha256(canonical(manifest)).hexdigest().encode()); stream.flush(); os.fsync(stream.fileno())
            directory_fd=os.open(temp, os.O_RDONLY)
            try: os.fsync(directory_fd)
            finally: os.close(directory_fd)
            temp.rename(destination)
            directory_fd=os.open(root, os.O_RDONLY)
            try: os.fsync(directory_fd)
            finally: os.close(directory_fd)
            return destination
        finally:
            if temp.exists(): shutil.rmtree(temp)


def verified_manifest(root, identity):
    key=cache_key(identity); directory=Path(root)/key
    raw=(directory/'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != (directory/'COMPLETE').read_text():
        raise ValueError('incomplete or corrupted cache manifest')
    manifest=json.loads(raw)
    if manifest.get('schema_version') != 1 or manifest.get('key') != key or manifest.get('identity') != identity:
        raise ValueError('cache identity/schema mismatch')
    if not manifest.get('shards'): raise ValueError('empty shard manifest')
    schema=[{k:v for k,v in e.items() if k not in ('file','bytes','sha256')} for e in manifest['shards']]
    if schema_digest(schema) != identity['shard_schema_sha256']:
        raise ValueError('cached shard schema differs from expected identity')
    for shard in manifest['shards']:
        name=shard['file']
        if Path(name).name != name or name in ('.','..') or not name.endswith('.bin'):
            raise ValueError('unsafe shard path')
        path=directory/name
        if shard['bytes'] != shard['storage_bytes'] or path.is_symlink() or path.stat().st_size != shard['bytes'] or file_digest(path) != shard['sha256']:
            raise ValueError('corrupt cache shard: '+name)
    return manifest


def restore(root, identity, consume):
    """Verify ALL shards before transfer; map one shard at a time, read-only.

    Consumer must finish or own a copy before return; borrowed buffers cannot
    outlive the call. A JAX consumer must synchronize transfer before unmapping.
    """
    manifest=verified_manifest(root, identity); directory=Path(root)/manifest['key']
    for shard in manifest['shards']:
        with (directory/shard['file']).open('rb') as stream:
            if shard['bytes'] == 0:
                consume(shard, b'')
            else:
                with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
                    consume(shard, buffer)
    return manifest


def local_array_chunks(array):
    """Adapter for a contiguous NumPy/host shard, never a global JAX array.

    The caller explicitly materializes just one addressable shard. Standard
    buffer bytes preserve BF16/FP4 storage; logical dtype remains in metadata.
    """
    view=memoryview(array).cast('B')
    for offset in range(0, view.nbytes, CHUNK_BYTES): yield view[offset:offset+CHUNK_BYTES]
