"""Strict identity/evidence checks and an isolated benchmark-only client bundle."""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
import tarfile
import tempfile
from common import ROOT, STATE, effective_env, output, q, dump

REPOS = ('tpu-inference', 'vllm', 'tokamax')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity_from_manifest(manifest):
    """Paths/job names are provenance; ordered patch bytes determine identity."""
    jobs, files = manifest['jobs'], manifest['files']
    env = {}
    for job in jobs: env.update(job.get('env', {}))
    patches = {}
    for repo in REPOS:
        paths = sorted(p for p in files if p.startswith('patches/' + repo + '/') and p.endswith('.patch'))
        for job in jobs:
            paths += sorted(p for p in files if p.startswith('jobs/' + job['id'] + '/patches/' + repo + '/') and p.endswith('.patch'))
        if repo == 'tpu-inference' and jobs[-1].get('profile'):
            paths += sorted(p for p in files if p.startswith('runner/patches/tpu-inference/') and p.endswith('.patch'))
        patches[repo] = [files[p] for p in paths]
    return {'base': manifest['base'], 'patches': patches, 'env': env,
            'base_env_sha256': files.get('env.sh'),
            'serve_args_extra': jobs[-1].get('serve_args_extra', []),
            'profile_enabled': jobs[-1].get('profile', False)}


def recipe_hash(source):
    import ast
    tree = ast.parse(source)
    functions = [ast.dump(n, include_attributes=False) for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name in ('environment', 'serve_script', 'launch_ray')]
    if len(functions) != 3: raise ValueError('server recipe functions missing')
    return digest('\n'.join(functions).encode())


def desired_manifest(stack):
    base = output(['git', '-C', ROOT, 'rev-parse', stack[-1][1]['base'].split('@')[1] + '^{commit}'])
    files = {}
    paths = output(['git', '-C', ROOT, 'ls-tree', '-r', '--name-only', base, 'patches', 'env.sh']).splitlines()
    import subprocess
    for path in paths:
        if path == 'env.sh' or path.endswith('.patch'):
            files[path] = digest(subprocess.check_output(['git', '-C', str(ROOT), 'show', base + ':' + path]))
    for directory, spec in stack:
        for repo in REPOS:
            for path in sorted((directory / 'patches' / repo).glob('*.patch')):
                files['jobs/' + spec['id'] + '/patches/' + repo + '/' + path.name] = digest(path.read_bytes())
        if (directory / 'acceptance.sh').exists():
            files['jobs/' + spec['id'] + '/acceptance.sh'] = digest((directory / 'acceptance.sh').read_bytes())
    if stack[-1][1].get('profile'):
        for path in sorted((ROOT / 'runner/patches/tpu-inference').glob('*.patch')):
            files['runner/patches/tpu-inference/' + path.name] = digest(path.read_bytes())
    return {'base': base, 'jobs': [s for _, s in stack], 'files': files,
            'server_recipe': recipe_hash((ROOT / 'runner/run_job.py').read_text())}


def differences(old, new, prefix=''):
    result = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(old.keys() | new.keys()):
            name = prefix + '.' + key if prefix else key
            if key not in old or key not in new:
                result.append(f'{name}: live={old.get(key)!r}; requested={new.get(key)!r}')
            else: result += differences(old[key], new[key], name)
    elif old != new:
        result.append(f'{prefix}: live={old!r}; requested={new!r}')
    return result


def assert_identical(old, new):
    diff = differences(old, new)
    if diff: raise RuntimeError('--reuse-server refused: server identity differs:\n' + '\n'.join(diff))


def acceptance_contract(manifest):
    job = manifest['jobs'][-1]
    return {'mode': job['acceptance'], 'script_sha256': manifest['files'].get('jobs/' + job['id'] + '/acceptance.sh')}


def remote_snapshot(launch, head):
    # Self-contained: sent read-only over SSH before uploading any client bundle.
    import ast, hashlib, json, subprocess
    from pathlib import Path
    root = Path('/opt/dsv4/dw-tpu').resolve()
    out = Path('/opt/dsv4/runs') / launch
    def active(unit):
        return subprocess.run(['sudo', '-n', 'systemctl', 'is-active', '--quiet', unit]).returncode == 0
    def started(unit):
        value = subprocess.check_output(['sudo', '-n', 'systemctl', 'show', unit, '-p', 'ActiveEnterTimestamp', '--value'], text=True).strip()
        return float(subprocess.check_output(['date', '-d', value, '+%s'], text=True))
    if not active('dsv4-ray'): raise RuntimeError('reuse refused: Ray unit inactive')
    if head and (not active('dsv4-serve') or active('dsv4-job')):
        raise RuntimeError('reuse refused: serve inactive or another controller active')
    manifest = json.loads((root / 'runner/stage-manifest.json').read_text())
    if root.name != launch: raise RuntimeError('reuse refused: live stage symlink does not match server launch')
    # Verify every staged patch, base env and acceptance script against its immutable manifest.
    for name, sha in manifest['files'].items():
        if name.endswith('.patch') or name == 'env.sh' or name.endswith('/acceptance.sh'):
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
                raise RuntimeError('reuse refused: mutated staged file ' + name)
    tree = ast.parse((root / 'runner/run_job.py').read_text())
    functions = [ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('environment', 'serve_script', 'launch_ray')]
    if len(functions) != 3: raise RuntimeError('reuse refused: server recipe functions missing')
    result = {'manifest': manifest, 'ray_started': started('dsv4-ray'),
              'server_recipe': hashlib.sha256('\n'.join(functions).encode()).hexdigest()}
    if head:
        import urllib.request
        with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=10) as response:
            if response.status != 200: raise RuntimeError('reuse refused: unhealthy server')
        command = subprocess.check_output(['sudo', '-n', 'systemctl', 'show', 'dsv4-serve', '-p', 'ExecStart', '--value'], text=True)
        if str(out) + '/' not in command: raise RuntimeError('reuse refused: serve unit belongs to another launch')
        if (out / 'launch-id').read_text().strip() != launch: raise RuntimeError('reuse refused: launch-id mismatch')
        result.update(state=json.loads((out / 'state.json').read_text()),
                      acceptance=json.loads((out / 'acceptance.json').read_text()),
                      acceptance_mtime=(out / 'acceptance.json').stat().st_mtime,
                      serve_started=started('dsv4-serve'))
        effective = out / 'effective-job.json'
        result['effective_job'] = json.loads(effective.read_text()) if effective.exists() else None
    # Avoid returning the enormous source manifest; keep only identity/evidence inputs.
    manifest['files'] = {p: h for p, h in manifest['files'].items() if p.endswith('.patch') or p == 'env.sh' or p.endswith('/acceptance.sh')}
    return result


def probe_script(launch, head):
    code = inspect.getsource(remote_snapshot) + '\nprint(json.dumps(remote_snapshot(' + repr(launch) + ', ' + repr(head) + ')))'
    return 'python3 -c ' + q('import json\n' + code)


def validate_snapshots(snapshots, desired, local_manifest, launch):
    wanted = identity_from_manifest(desired)
    local = identity_from_manifest(local_manifest)
    head = snapshots[0]
    for snap in snapshots:
        if snap['server_recipe'] != desired['server_recipe']:
            raise RuntimeError('reuse refused: serving recipe (environment/arguments/Ray) differs')
        actual = identity_from_manifest(snap['manifest'])
        assert_identical(local, actual)
        assert_identical(actual, wanted)
    effective = head.get('effective_job')
    if effective is not None:
        assert_identical({'env': local['env'], 'serve_args_extra': local['serve_args_extra'], 'base': local_manifest['jobs'][-1]['base']},
                         {k: effective.get(k) for k in ('env', 'serve_args_extra', 'base')})
    if head['state'].get('status') != 'passed' or head['state'].get('launch_id') != launch:
        raise RuntimeError('reuse refused: originating server launch did not finish successfully')
    if head['acceptance'].get('passed') is not True:
        raise RuntimeError('reuse refused: acceptance was not passed (informational runs cannot be inherited)')
    assert_identical(acceptance_contract(local_manifest), acceptance_contract(desired))
    if head['acceptance'].get('mode') != desired['jobs'][-1]['acceptance']:
        raise RuntimeError('reuse refused: acceptance evidence mode differs')
    if any(s['ray_started'] > head['acceptance_mtime'] for s in snapshots) or head['serve_started'] > head['acceptance_mtime']:
        raise RuntimeError('reuse refused: service restarted after acceptance evidence')
    return wanted


def client_bundle(spec, launch, metadata):
    """Ship only clients under runs/<new>/client; never touch the live stage."""
    STATE.mkdir(parents=True, exist_ok=True)
    archive = STATE / (launch + '.clients.tar.gz')
    with tarfile.open(archive, 'w:gz') as tf:
        def select(info):
            if any(p in ('.state', '__pycache__', 'pristine-tokamax') for p in Path(info.name).parts) or info.name.endswith('.pyc'): return None
            return info
        tf.add(ROOT / 'runner', arcname='runner', filter=select)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            dump(p / 'job.json', spec); dump(p / 'reuse.json', metadata)
            tf.add(p / 'job.json', arcname='job/job.json'); tf.add(p / 'reuse.json', arcname='reuse.json')
    return archive


def reuse_server(args, stack, launch, tr):
    from common import collect, extract_evidence, fresh_result, unit_command, PYTHON
    from run_job import environment, verify_live_logs
    from summarize import summarize
    import time
    spec = dict(stack[-1][1], env=effective_env(stack))
    current_path = STATE / 'current.json'
    if not current_path.exists(): raise RuntimeError('--reuse-server refused: no current launch state')
    current = json.loads(current_path.read_text())
    origin = current.get('server_launch', current['launch'])
    from common import ID_RE
    if not ID_RE.fullmatch(origin): raise RuntimeError('invalid server launch identifier')
    manifest_path = STATE / (origin + '.tar.manifest.json')
    if not manifest_path.exists(): raise RuntimeError('--reuse-server refused: original staging manifest missing: ' + str(manifest_path))
    local = json.loads(manifest_path.read_text()); desired = desired_manifest(stack)
    assert_identical(identity_from_manifest(local), identity_from_manifest(desired))
    hosts = tr.discover()
    if not args.dry_run and [(h['name'], h['ip']) for h in hosts] != [(h['name'], h['ip']) for h in current['hosts']]:
        raise RuntimeError('--reuse-server refused: current host inventory differs')
    scripts = [probe_script(origin, i == 0) for i in range(len(hosts))]
    if args.dry_run:
        for h, script in zip(hosts, scripts): tr.ssh(h, script)
        print('# Require remote identity, active units, /health, passing matching acceptance, no post-acceptance restart; no fallback.')
        out = '/opt/dsv4/runs/' + launch
        source = '/opt/dsv4/runs/' + origin
        tr.all(hosts, f'test ! -e {q(out)}; mkdir -p {q(out)}; printf "%s\\n" {q(launch)} > {q(out + "/launch-id")}')
        tr.upload(hosts[0], STATE / (launch + '.clients.tar.gz'), out + '/client')
        tr.ssh(hosts[0], f'ln {q(source + "/serve.log")} {q(out + "/serve.log")}')
        command = environment(stack) + f'exec {PYTHON} {q(out + "/client/runner/remote_job.py")} {q(out + "/client/job")} --out {q(out)} --launch-id {q(launch)} --reuse-evidence {q(out + "/client/reuse.json")}'
        tr.ssh(hosts[0], unit_command('dsv4-job', command, out + '/controller.log'))
        print('# Poll state.json, audit live logs, stop only dsv4-job; collect; leave serve/Ray running.')
        return
    snapshots = [json.loads(tr.ssh(h, script, timeout=120)) for h, script in zip(hosts, scripts)]
    identity = validate_snapshots(snapshots, desired, local, origin)
    source = '/opt/dsv4/runs/' + origin
    out = '/opt/dsv4/runs/' + launch
    metadata = {'server_launch': origin, 'server_identity': identity, 'acceptance_inherited_from': origin,
                'acceptance_evidence': source + '/acceptance.json', 'acceptance': snapshots[0]['acceptance'],
                'origin_ready_unix': snapshots[0]['state']['ready_unix'], 'requested_job': spec['id']}
    archive = client_bundle(spec, launch, metadata)
    # Backfill legacy effective-job evidence ONLY after matching immutable local
    # and remote manifests. Pre-feature launchers wrote this file locally only.
    if snapshots[0]['effective_job'] is None:
        effective = dict(local['jobs'][-1], env=identity['env'])
        tr.ssh(hosts[0], 'test ! -e ' + q(source + '/effective-job.json') + '\nprintf %s ' + q(json.dumps(effective)) + ' > ' + q(source + '/effective-job.json'))
    tr.all(hosts, f'test ! -e {q(out)}; mkdir -p {q(out)}\nprintf "%s\\n" {q(launch)} > {q(out + "/launch-id")}')
    tr.upload(hosts[0], archive, out + '/client')
    # Use a hard link while live, and copy as an ordinary file in collected evidence.
    # Symlinks are intentionally rejected by extract_evidence.
    tr.ssh(hosts[0], f'ln {q(source + "/serve.log")} {q(out + "/serve.log")}\n'
           f'cp {q(out + "/client/job/job.json")} {q(out + "/effective-job.json")}')
    result_dir = fresh_result(stack[-1][0], launch)
    dump(result_dir / 'effective-job.json', spec); dump(result_dir / 'reuse.json', metadata)
    dump(STATE / 'current.json', {'job': str(stack[-1][0]), 'id': spec['id'], 'kind': 'e2e', 'launch': launch,
                                 'server_launch': origin, 'hosts': hosts})
    failure = None
    strict = 'post-ready model/runner JIT compilation' in spec.get('abort_if', [])
    command = environment(stack) + f'exec {PYTHON} {q(out + "/client/runner/remote_job.py")} {q(out + "/client/job")} --out {q(out)} --launch-id {q(launch)} --reuse-evidence {q(out + "/client/reuse.json")}'
    try:
        tr.ssh(hosts[0], unit_command('dsv4-job', command, out + '/controller.log'))
        deadline = time.monotonic() + spec['est_minutes'] * 90 + 180
        while True:
            value = tr.ssh(hosts[0], f'if test -f {q(out + "/state.json")}; then cat {q(out + "/state.json")}; else echo \'{{"status":"starting"}}\'; fi')
            state = json.loads(value); print(json.dumps(state), flush=True)
            if state.get('launch_id', launch) != launch: raise RuntimeError('reuse controller launch-id mismatch')
            verify_live_logs(tr, hosts, out, metadata['origin_ready_unix'], strict)
            if state['status'] == 'passed': break
            if state['status'] in ('failed', 'gate_failed_benchmarked'): raise RuntimeError(state.get('error', 'reuse benchmark failed'))
            if time.monotonic() >= deadline: raise TimeoutError('reuse benchmark wall timeout')
            time.sleep(15)
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        # Only stop this benchmark controller; never touch serve/Ray on reuse,
        # including failure. Failed servers must pass fresh checks before reuse.
        try:
            tr.ssh(hosts[0], 'if sudo -n systemctl is-active --quiet dsv4-job; then sudo -n systemctl stop dsv4-job; fi')
            # Break the serve.log hardlink before tar (tar may otherwise emit a
            # hardlink entry referring to the originating run, not a regular file).
            tr.ssh(hosts[0], f'cp {q(out + "/serve.log")} {q(out + "/serve.log.snapshot")}\nmv {q(out + "/serve.log.snapshot")} {q(out + "/serve.log")}')
            copies = collect(tr, hosts, STATE / 'copies' / launch)
            extract_evidence(copies / (hosts[0]['name'] + '.tar.gz'), result_dir, 'runs/' + launch)
            dump(result_dir / 'copies.json', {'directory': str(copies), 'hosts': hosts, 'launch': launch, 'server_launch': origin})
        except BaseException as exc:
            failure = (failure or '') + '\ncollection: ' + repr(exc)
        if failure: dump(result_dir / 'state.json', {'status': 'failed', 'error': failure, 'launch_id': launch, **metadata})
        summarize(result_dir, spec)
    if failure: raise RuntimeError(failure)
    print('Reused server ' + origin + '; benchmark complete: ' + str(result_dir))
