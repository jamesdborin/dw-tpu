"""Build reproducible, standalone git checkouts without modifying local submodules."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import os
import sys
import zipfile
import csv
import base64
from common import ROOT, STATE, disk_guard, dump, output, patch_paths, q, run


def checkout(source, commit, destination):
    # --depth=1 via file:// copies reachable objects; no alternates to local paths.
    run(['git', 'init', '-q', destination])
    run(['git', '-C', destination, 'fetch', '-q', '--depth=1', Path(source).resolve().as_uri(), commit])
    run(['git', '-C', destination, 'checkout', '-q', '--detach', 'FETCH_HEAD'])


def validate_patches(tree, stack):
    for repo in ('tpu-inference', 'vllm'):
        target = tree / repo
        pin = output(['git', '-C', target, 'rev-parse', 'HEAD'])
        patches = sorted((tree / 'patches' / repo).glob('*.patch')) + patch_paths(stack, repo)
        if repo == 'tpu-inference' and stack and stack[-1][1].get('profile'):
            patches += sorted((tree / 'runner/patches/tpu-inference').glob('*.patch'))
        for patch in patches:
            run(['git', '-C', target, 'apply', '--check', patch])
            run(['git', '-C', target, 'apply', patch])
        run(['git', '-C', target, 'reset', '-q', '--hard', pin])
        run(['git', '-C', target, 'clean', '-qfd'])
    # Validate installed-package patches against an unmodified pinned distribution,
    # never against another experiment's site-packages.
    site = tree / 'runner' / 'pristine-tokamax'
    cached = Path(os.environ.get('UV_CACHE_DIR', str(Path.home() / '.cache/uv'))) / 'wheels-v5/pypi/tokamax/0.0.16.dev20261009-py3-none-any'
    if cached.exists():
        shutil.copytree(cached, site, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    else:
        wheels = STATE / 'wheels'
        wheels.mkdir(exist_ok=True)
        found = list(wheels.glob('tokamax-0.0.16.dev20261009-*.whl'))
        if not found:
            run([sys.executable, '-m', 'pip', '--isolated', 'download', '--no-deps',
                 '--python-version', '3.12', '--only-binary=:all:',
                 '--index-url', 'https://pypi.org/simple', '--dest', wheels, 'tokamax==0.0.16.dev20261009'])
            found = list(wheels.glob('tokamax-0.0.16.dev20261009-*.whl'))
        with zipfile.ZipFile(found[0]) as wheel:
            wheel.extractall(site)
    # Wheel RECORD protects against accidentally reusing a mutated package cache.
    record = site / 'tokamax-0.0.16.dev20261009.dist-info/RECORD'
    with record.open(newline='') as stream:
        for name, digest, size in csv.reader(stream):
            if not digest:
                continue
            algorithm, encoded = digest.split('=', 1)
            actual = base64.urlsafe_b64encode(hashlib.new(algorithm, (site / name).read_bytes()).digest()).decode().rstrip('=')
            if actual != encoded:
                raise ValueError('pristine tokamax wheel RECORD mismatch: ' + name)
    with tempfile.TemporaryDirectory(prefix='tokamax-check-', dir=STATE) as temp:
        target = Path(temp) / 'site'
        shutil.copytree(site, target)
        for patch in sorted((tree / 'patches/tokamax').glob('*.patch')) + patch_paths(stack, 'tokamax'):
            run(['git', '-C', target, 'apply', '--check', patch])
            run(['git', '-C', target, 'apply', patch])


def build_archive(stack, launch, base=None):
    disk_guard()
    STATE.mkdir(parents=True, exist_ok=True)
    base = base or stack[-1][1]['base'].split('@')[1]
    base = output(['git', '-C', ROOT, 'rev-parse', base + '^{commit}'])
    archive = STATE / (launch + '.tar.gz')
    with tempfile.TemporaryDirectory(prefix='stage-', dir=STATE) as temp:
        tree = Path(temp) / 'dw-tpu'
        checkout(ROOT, base, tree)
        # Exclude archived historical jobs/results; only selected jobs are deployed.
        for name in ('runner', 'jobs'):
            shutil.rmtree(tree / name, ignore_errors=True)
        shutil.copytree(ROOT / 'runner', tree / 'runner', ignore=shutil.ignore_patterns('.state', '__pycache__', '*.pyc'))
        for repo in ('tpu-inference', 'vllm'):
            pin = output(['git', '-C', tree, 'rev-parse', 'HEAD:' + repo])
            checkout(ROOT / repo, pin, tree / repo)
        for directory, spec in stack:
            shutil.copytree(directory, tree / 'jobs' / spec['id'], ignore=shutil.ignore_patterns('result', '.git', '__pycache__'))
        validate_patches(tree, stack)
        manifest = {'base': base, 'jobs': [s for _, s in stack], 'files': {}}
        for p in sorted(tree.rglob('*')):
            if p.is_file() and '.git' not in p.relative_to(tree).parts:
                manifest['files'][str(p.relative_to(tree))] = hashlib.sha256(p.read_bytes()).hexdigest()
        dump(tree / 'runner' / 'stage-manifest.json', manifest)
        with tarfile.open(archive, 'w:gz') as tf:
            tf.add(tree, arcname='.', recursive=True)
    dump(archive.with_suffix('.manifest.json'), manifest)
    return archive


def stage_remote(tr, hosts, archive, launch):
    destination = '/opt/dsv4/stages/' + launch
    tr.all(hosts, "sudo -n mkdir -p /opt/dsv4; sudo -n chown \"$(id -u):$(id -g)\" /opt/dsv4; mkdir -p /opt/dsv4/{stages,runs,logs,xla-cache}")
    tr.all(hosts, f'test ! -e {q(destination)}; test "$(df --output=avail -B1 /opt/dsv4 | tail -1)" -ge 21474836480')
    def upload(h):
        tr.upload(h, archive, destination)
        return tr.ssh(h, f'python3 {q(destination + "/runner/verify_stage.py")} {q(destination)}\n'
                         f'test ! -e /opt/dsv4/dw-tpu || test -L /opt/dsv4/dw-tpu\n'
                         f'ln -sfn {q(destination)} /opt/dsv4/dw-tpu')
    if tr.dry:
        for h in hosts:
            upload(h)
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
            list(pool.map(upload, hosts))
    return destination


def apply_script(stack):
    root = '/opt/dsv4/dw-tpu'
    # Restore the pristine pinned package before patches; no network install per job.
    text = f'''export PATH=/opt/dsv4/.venv/bin:/opt/dsv4/tools:$PATH
export JAX_PLATFORMS=cpu
cd {root}
SITE=$(/opt/dsv4/.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
test -f runner/pristine-tokamax/tokamax-0.0.16.dev20261009.dist-info/METADATA
rsync -a --delete runner/pristine-tokamax/tokamax/ \"$SITE/tokamax/\"
bash ./apply_patches.sh --reset --tokamax /opt/dsv4/.venv/bin/python
SITE=$(/opt/dsv4/.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
'''
    for repo in ('vllm',):
        text += f'for p in {root}/patches/{repo}/*.patch; do test ! -f "$p" || git -C {root}/{repo} apply "$p"; done\n'
    for directory, spec in stack:
        for repo in ('tpu-inference', 'vllm', 'tokamax'):
            for patch in sorted((directory / 'patches' / repo).glob('*.patch')):
                remote = root + '/jobs/' + spec['id'] + '/patches/' + repo + '/' + patch.name
                if repo == 'tokamax':
                    text += f'patch --batch -d "$SITE" -p1 --dry-run < {q(remote)}\npatch --batch -d "$SITE" -p1 < {q(remote)}\n'
                else:
                    text += f'git -C {q(root + "/" + repo)} apply --check {q(remote)}\ngit -C {q(root + "/" + repo)} apply {q(remote)}\n'
    if stack and stack[-1][1].get('profile'):
        text += f'for p in {root}/runner/patches/tpu-inference/*.patch; do git -C {root}/tpu-inference apply --check "$p"; git -C {root}/tpu-inference apply "$p"; done\n'
    text += '''VLLM_TARGET_DEVICE=tpu uv pip install --python /opt/dsv4/.venv/bin/python --no-deps --no-build-isolation -e vllm
uv pip install --python /opt/dsv4/.venv/bin/python --no-deps -e tpu-inference
/opt/dsv4/.venv/bin/python -c 'import importlib.metadata as m; import vllm,tpu_inference; assert m.version("jax")=="0.11.2"; assert m.version("libtpu")=="0.0.48"'
'''
    return text
