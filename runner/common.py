"""Local-only helpers until a Transport method executes; Python standard library only."""
from __future__ import annotations
import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / 'runner' / '.state'
REMOTE = '/opt/dsv4'
PYTHON = REMOTE + '/.venv/bin/python'
# GCP placement; override with environment variables for your own project and reservation.
PROJECT = os.environ.get('DSV4_GCP_PROJECT', 'tech-426212')
ZONE = os.environ.get('DSV4_GCP_ZONE', 'us-east5-b')
GROUP = os.environ.get('DSV4_GCP_GROUP', 'dsv4-day-32-20261020')
RESERVATION = os.environ.get('DSV4_GCP_RESERVATION', 'dsv4-v6e32-east5b-20261020')
ID_RE = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}\Z')
ENV_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')

def q(value):
    return shlex.quote(str(value))

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def run(argv, **kwargs):
    return subprocess.run([str(x) for x in argv], check=True, **kwargs)

def output(argv):
    return run(argv, text=True, capture_output=True).stdout.strip()

def disk_guard(path=ROOT):
    if shutil.disk_usage(path).free < 20 * 1024**3:
        raise RuntimeError('DISK GUARD: less than 20 GiB local free space; copy/stage refused')

@contextlib.contextmanager
def cluster_lock(dry_run=False):
    if dry_run:
        yield
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'cluster.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another runner owns this cluster. Use status.py; do not overlap jobs.')
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def load_job(path):
    path = Path(path).resolve()
    spec = json.loads((path / 'job.json').read_text())
    if not isinstance(spec, dict):
        raise ValueError('job.json must be an object')
    for key in ('id', 'owner', 'kind', 'base', 'est_minutes'):
        if key not in spec:
            raise ValueError(f'missing job field: {key}')
    if not isinstance(spec['id'], str) or not ID_RE.fullmatch(spec['id']) or spec['id'] != path.name:
        raise ValueError('id must be a safe name matching the job directory')
    if spec['kind'] not in ('e2e', 'kernel', 'compile', 'micro32'):
        raise ValueError('kind must be e2e, kernel, compile or micro32')
    if not isinstance(spec['base'], str) or not re.fullmatch(r'dw-tpu@[0-9a-f]{7,40}', spec['base']):
        raise ValueError('base must be dw-tpu@<commit SHA>, never a moving branch')
    if type(spec['est_minutes']) not in (int, float) or not 0 < spec['est_minutes'] <= 1440:
        raise ValueError('est_minutes must be positive and <=1440')
    env = spec.setdefault('env', {})
    if not isinstance(env, dict) or any(not ENV_RE.fullmatch(k) or not isinstance(v, str) or '\x00' in v for k, v in env.items()):
        raise ValueError('env must map shell variable names to strings')
    deps = spec.setdefault('depends_on', [])
    if not isinstance(deps, list) or any(not isinstance(d, str) or not ID_RE.fullmatch(d) for d in deps):
        raise ValueError('depends_on must be a list of job IDs')
    extra = spec.setdefault('serve_args_extra', [])
    if not isinstance(extra, list) or any(not isinstance(s, str) or '\x00' in s for s in extra):
        raise ValueError('serve_args_extra must be an argv array')
    if spec['kind'] == 'e2e':
        if spec.setdefault('bench_mode', 'agent') not in ('agent', 'pd'):
            raise ValueError('bench_mode must be agent or pd')
        if spec['bench_mode'] == 'pd':
            from vendor.pd_bench import validate_config
            validate_config(spec.get('pd'))
        if type(spec.setdefault('benchmark_on_gate_failure', False)) is not bool:
            raise ValueError('benchmark_on_gate_failure must be boolean')
        if spec['benchmark_on_gate_failure'] and spec.get('acceptance') != 'reduction-order':
            raise ValueError('benchmark_on_gate_failure requires reduction-order acceptance')
        if spec.get('acceptance') not in ('exact', 'reduction-order', 'sampling-equivalent'):
            raise ValueError('e2e requires exact, reduction-order or sampling-equivalent acceptance')
        if spec['acceptance'] == 'sampling-equivalent' and not (path / 'acceptance.sh').is_file():
            raise ValueError('sampling-equivalent requires job-local acceptance.sh and $OUT/sampling-equivalent.json')
        bench = spec.setdefault('bench', {'agents': [32], 'repeats': 1})
        if not isinstance(bench, dict) or not isinstance(bench.get('agents'), list) or not bench['agents'] or any(type(n) is not int or not 1 <= n <= 32 for n in bench['agents']):
            raise ValueError('bench.agents must be a nonempty list in 1..32')
        if len(set(bench['agents'])) != len(bench['agents']) or type(bench.get('repeats')) is not int or not 1 <= bench['repeats'] <= 20:
            raise ValueError('bench requires distinct agents and repeats in 1..20')
        if type(spec.setdefault('profile', False)) is not bool:
            raise ValueError('profile must be boolean')
        rules = spec.setdefault('abort_if', ['post-ready RESOURCE_EXHAUSTED', 'launch-id mismatch', 'startup > 45 min'])
        if not isinstance(rules, list) or any(not isinstance(x, str) for x in rules):
            raise ValueError('abort_if must be a string array')
        for rule in rules:
            if rule not in ('post-ready RESOURCE_EXHAUSTED', 'launch-id mismatch',
                            'post-ready model/runner JIT compilation', 'reduction-order gate failure',
                            'acceptance gate failure') and not re.fullmatch(r'startup > [1-9][0-9]* min', rule):
                raise ValueError(f'unsupported abort_if rule (will not silently ignore): {rule}')
    elif not (path / 'run.sh').is_file():
        raise ValueError('kernel/compile job requires run.sh')
    return spec

def job_stack(path):
    path = Path(path).resolve()
    result, visiting, visited = [], set(), set()
    def visit(p):
        spec = load_job(p)
        key = spec['id']
        if key in visiting:
            raise ValueError('dependency cycle: ' + key)
        if key in visited:
            return
        visiting.add(key)
        for dep in spec['depends_on']:
            visit(path.parent / dep)
        visiting.remove(key)
        visited.add(key)
        result.append((p, spec))
    visit(path)
    commits = {output(['git', '-C', ROOT, 'rev-parse', s['base'].split('@')[1] + '^{commit}']) for _, s in result}
    if len(commits) != 1:
        raise ValueError('all dependencies must use the same base commit')
    return result

def patch_paths(stack, repo):
    return [p for directory, _ in stack for p in sorted((directory / 'patches' / repo).glob('*.patch'))]

def effective_env(stack):
    env = {}
    for _, spec in stack:
        env.update(spec['env'])
    return env

def arguments(parser):
    parser.add_argument('--dry-run', action='store_true', help='print commands; never contact cloud/hosts or write results')
    parser.add_argument('--hosts', type=Path, help='saved 8-host JSON inventory; otherwise discover MIG members')
    parser.add_argument('--project', default=PROJECT)
    parser.add_argument('--zone', default=ZONE)
    parser.add_argument('--group', default=GROUP)
    return parser

class Transport:
    def __init__(self, args):
        self.dry = args.dry_run
        self.project, self.zone, self.group = args.project, args.zone, args.group
        self.env = dict(os.environ, CLOUDSDK_CONFIG=str(Path.home() / '.config/gcloud-tpu-ops'))
        self.hosts_file = args.hosts
        self.identity_checked = False
        self.contacted_hosts = set()

    def identity(self):
        if self.identity_checked:
            return
        cmd = ['gcloud', 'auth', 'list', '--filter=status:ACTIVE', '--format=value(account)', '--quiet']
        print('CLOUDSDK_CONFIG=' + q(self.env['CLOUDSDK_CONFIG']) + ' ' + shlex.join(cmd), flush=True)
        if not self.dry:
            account = subprocess.run(cmd, env=self.env, check=True, capture_output=True, text=True).stdout.strip()
            if not account.endswith('.gserviceaccount.com') or '\n' in account:
                raise RuntimeError('Ops gcloud configuration must have exactly one active service account; personal credentials are refused.')
        self.identity_checked = True

    def cloud(self, argv, *, check=True):
        self.identity()
        cmd = ['gcloud', *argv, '--project=' + self.project, '--quiet']
        print('CLOUDSDK_CONFIG=' + q(self.env['CLOUDSDK_CONFIG']) + ' ' + shlex.join(cmd), flush=True)
        if self.dry:
            return ''
        p = subprocess.run(cmd, env=self.env, text=True, capture_output=True)
        if check and p.returncode:
            raise RuntimeError(p.stderr.strip())
        return p.stdout

    def discover(self):
        self.identity()
        if self.hosts_file:
            hosts = json.loads(self.hosts_file.read_text())
        else:
            members = self.cloud(['compute', 'instance-groups', 'managed', 'list-instances', self.group, '--zone=' + self.zone, '--format=json'])
            if self.dry:
                print('# Inventory placeholders below; supply --hosts for exact host names and IPs.')
                return [{'name': f'<host-{i}>', 'ip': f'<ip-{i}>', 'status': 'RUNNING'} for i in range(8)]
            names = [x['instance'].rsplit('/', 1)[-1] for x in json.loads(members)]
            if len(names) != 8:
                raise RuntimeError(f'expected 8 MIG members, found {len(names)}')
            hosts = [json.loads(self.cloud(['compute', 'instances', 'describe', name, '--zone=' + self.zone, '--format=json'])) for name in names]
        if len(hosts) != 8 or len({h['name'] for h in hosts}) != 8:
            raise ValueError('inventory must contain exactly 8 distinct hosts')
        for h in hosts:
            if h.get('status', 'RUNNING') != 'RUNNING' or h.get('zone', self.zone).rsplit('/', 1)[-1] != self.zone:
                raise ValueError('host not RUNNING in expected zone: ' + h['name'])
            if 'ip' not in h:
                h['ip'] = h['networkInterfaces'][0]['networkIP']
        hosts.sort(key=lambda h: h['name'])
        if not self.dry:
            dump(STATE / 'hosts.json', hosts)
        return hosts

    def command(self, host, script):
        return ['gcloud', 'compute', 'ssh', host['name'], '--project=' + self.project, '--zone=' + self.zone,
                '--tunnel-through-iap', '--quiet', '--ssh-flag=-oBatchMode=yes', '--ssh-flag=-oConnectTimeout=30',
                '--ssh-flag=-oServerAliveInterval=20', '--ssh-flag=-oServerAliveCountMax=3',
                '--command', 'bash -lc ' + q('set -euo pipefail\n' + script)]

    def ssh(self, host, script, *, stdin=None, stdout=None, timeout=3600):
        self.identity()
        cmd = self.command(host, script)
        print('CLOUDSDK_CONFIG=' + q(self.env['CLOUDSDK_CONFIG']) + ' ' + shlex.join(cmd), flush=True)
        if self.dry:
            return ''
        if host['name'] not in self.contacted_hosts:
            # Probe separately: rc255 can also be a remote program's exit code,
            # or a disconnect after a mutation. Never replay the payload/streams.
            probe = self.command(host, 'true')
            p = subprocess.CompletedProcess(probe, 255, b'', b'First-contact SSH deadline exceeded')
            deadline = time.monotonic() + 300
            delay = 5
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                print('CLOUDSDK_CONFIG=' + q(self.env['CLOUDSDK_CONFIG']) + ' ' + shlex.join(probe), flush=True)
                p = subprocess.run(probe, env=self.env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=min(timeout, 60, remaining))
                if p.returncode != 255:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                pause = min(delay, remaining)
                print(f'{host["name"]}: first-contact SSH exit 255; retry in {pause:g}s', flush=True)
                time.sleep(pause)
                delay = min(delay * 2, 30)
            if p.returncode == 0:
                self.contacted_hosts.add(host['name'])
        else:
            p = None
        if p is None or p.returncode == 0:
            p = subprocess.run(cmd, env=self.env, stdin=stdin, stdout=stdout or subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=timeout)
        if p.returncode:
            error = p.stderr.decode(errors='replace')[-6000:]
            body = (p.stdout or b'').decode(errors='replace')[-6000:]
            hint = ('\nOS Login/IAP fallback: verify the ops service account has compute.osAdminLogin, '
                    'iap.tunnelResourceAccessor and iam.serviceAccountUser on the VM service account; '
                    'allow IAP TCP/22. Have the orchestrator repair IAM/OS Login or supply an approved '
                    'prevalidated template. This tool never switches to personal credentials, metadata keys or public SSH.') if p.returncode == 255 or any(x in error for x in ('Permission denied', '403', 'IAP', 'OS Login')) else ''
            raise RuntimeError(f'{host["name"]}: remote exit {p.returncode}\n{body}\n{error}{hint}')
        return (p.stdout or b'').decode(errors='replace')

    def all(self, hosts, script, timeout=3600):
        def one(h):
            return self.ssh(h, script(h) if callable(script) else script, timeout=timeout)
        if self.dry:
            return [one(h) for h in hosts]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
            return list(pool.map(one, hosts))

    def upload(self, host, archive, destination):
        command = f'mkdir -p {q(destination)}; tar xzf - -C {q(destination)}'
        if self.dry:
            print(f'# stdin: {archive}')
            return self.ssh(host, command)
        with Path(archive).open('rb') as stream:
            return self.ssh(host, command, stdin=stream)


def free_devices():
    return """sudo -n true
command -v fuser >/dev/null
test -d /dev/vfio
if sudo -n fuser /dev/vfio/* 2>/dev/null; then
  echo 'VFIO holders remain; refusing source change or restart' >&2; exit 1
fi
"""

def stop_script():
    return """sudo -n true
for unit in dsv4-job dsv4-serve dsv4-ray dsv4-kernel; do
  if sudo -n systemctl is-active --quiet "$unit"; then sudo -n systemctl stop "$unit"; fi
done
if test -x /opt/dsv4/.venv/bin/ray; then /opt/dsv4/.venv/bin/ray stop --force; fi
""" + free_devices()

def unit_command(name, script, log):
    body = "exec > " + q(log) + " 2>&1\nset -euo pipefail\n" + script
    return (f'sudo -n systemctl reset-failed {q(name)} 2>/dev/null || true\n'
            f'sudo -n systemd-run --unit={q(name)} --uid="$(id -u)" --gid="$(id -g)" '
            '--collect --property=KillMode=control-group --property=TimeoutStopSec=30 '
            '--property=LimitMEMLOCK=infinity --property=LimitNOFILE=1048576 '
            f'/bin/bash -lc {q(body)}')


def collect(tr, hosts, destination):
    """Preserve all runs and Ray worker logs BEFORE stopping or changing source."""
    destination = Path(destination)
    if not tr.dry:
        disk_guard(destination.parent if destination.parent.exists() else ROOT)
        destination.mkdir(parents=True, exist_ok=True)
    errors = []
    for h in hosts:
        archive = destination / (h['name'] + '.tar.gz')
        cmd = """cd /opt/dsv4
test "$(df --output=avail -B1 . | tail -1)" -ge 21474836480
snapshot=$(mktemp -d /opt/dsv4/.copy.XXXXXX)
trap 'rm -rf "$snapshot"' EXIT
for directory in runs logs; do
  if test -d "$directory"; then cp -a --reflink=auto "$directory" "$snapshot/"; fi
done
if test -d /tmp/ray/session_latest/logs; then
  mkdir -p "$snapshot/logs/ray"
  cp -a --reflink=auto /tmp/ray/session_latest/logs/. "$snapshot/logs/ray/"
fi
tar czf - -C "$snapshot" .
"""
        if tr.dry:
            print(f'# stdout archive: {archive}')
            tr.ssh(h, cmd)
        else:
            try:
                disk_guard()
                partial = archive.with_suffix('.partial')
                with partial.open('wb') as out:
                    tr.ssh(h, cmd, stdout=out, timeout=1800)
                # Confirm the stream is a complete readable gzip/tar before accepting it.
                with tarfile.open(partial) as tf:
                    tf.getmembers()
                partial.replace(archive)
                digest = hashlib.sha256()
                with archive.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                        digest.update(chunk)
                (archive.with_suffix('.sha256')).write_text(digest.hexdigest() + '  ' + archive.name + '\n')
            except Exception as exc:
                errors.append(h['name'] + ': ' + repr(exc))
    if errors:
        dump(destination / 'copy-errors.json', errors)
        raise RuntimeError('copy-off incomplete (successful host archives retained): ' + '; '.join(errors))
    return destination


def fresh_result(job, launch):
    destination = Path(job) / 'result'
    if destination.exists():
        history = STATE / 'result-history' / launch
        history.parent.mkdir(parents=True, exist_ok=True)
        if history.exists():
            raise RuntimeError('result-history launch collision: ' + launch)
        shutil.move(destination, history)
    destination.mkdir(parents=True)
    return destination


def extract_evidence(archive, target, prefix):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            name = m.name.removeprefix('./')
            if not name.startswith(prefix.rstrip('/') + '/'):
                continue
            rel = Path(name).relative_to(prefix)
            if rel.is_absolute() or '..' in rel.parts or not (m.isfile() or m.isdir()):
                raise ValueError('unsafe result archive member: ' + m.name)
            dest = target / rel
            if m.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as src, dest.open('wb') as out:
                    shutil.copyfileobj(src, out)
