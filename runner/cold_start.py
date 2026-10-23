#!/usr/bin/env python3
"""Upload pinned sources/tools, create runtime, verify checkpoint, and psum all 32 chips."""
import argparse
import json
from pathlib import Path
import tarfile
import time
import hashlib
import io
from common import ROOT, STATE, PYTHON, Transport, arguments, cluster_lock, collect, disk_guard, dump, free_devices, output, q, unit_command
from staging import apply_script, build_archive, stage_remote

DEFAULT_PAYLOAD = ROOT.parent / 'dsv4-tpu/bench-a1/payload.tar.gz'
REVISION = json.loads((ROOT / 'runner/vendor/checkpoint-manifest.json').read_text())['revision']


def wait_download(tr, hosts):
    script = f'''deadline=$((SECONDS+7200))
until test -f /opt/dsv4/ckpt/CHECKPOINT_VERIFIED; do
  if ! sudo -n systemctl is-active --quiet dsv4-download; then
    tail -80 /opt/dsv4/logs/checkpoint.log; echo 'Checkpoint download stopped without verification' >&2; exit 1
  fi
  test \"$SECONDS\" -lt \"$deadline\" || {{ echo 'Checkpoint download timeout' >&2; exit 1; }}
  sleep 15
done
test \"$(cat /opt/dsv4/ckpt/CHECKPOINT_VERIFIED)\" = {q(REVISION)}
'''
    tr.all(hosts, script, timeout=7300)


def tools_archive(payload):
    disk_guard(); STATE.mkdir(parents=True, exist_ok=True)
    result = STATE / 'tools.tar.gz'
    with tarfile.open(payload) as src, tarfile.open(result, 'w:gz') as dst:
        checksums = []
        for name, target in [('bench-a1/uv', 'tools/uv'), ('bench/batchbench-agent', 'tools/batchbench-agent')]:
            member = src.getmember(name)
            if not member.isfile():
                raise ValueError('payload tool must be a regular file: ' + name)
            data = src.extractfile(member).read()
            checksums.append(hashlib.sha256(data).hexdigest() + '  ' + target)
            member.name = target; member.mode = 0o755
            dst.addfile(member, io.BytesIO(data))
        record = ('\n'.join(checksums) + '\n').encode()
        member = tarfile.TarInfo('tools/SHA256SUMS'); member.size = len(record)
        dst.addfile(member, io.BytesIO(record))
    (STATE / 'tools.sha256').write_bytes(record)
    return result


def main():
    p = arguments(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--base', default=output(['git', '-C', ROOT, 'rev-parse', 'HEAD']))
    p.add_argument('--payload', type=Path, default=DEFAULT_PAYLOAD, help='preserved payload.tar.gz; only uv + batchbench-agent are extracted')
    p.add_argument('--download-in-background', action='store_true', help='return after setup + psum + starting isolated download units; permit kernel jobs during download')
    p.add_argument('--wait-download', action='store_true', help='wait for already-started downloads only; no source/runtime changes or cluster lock')
    a = p.parse_args(); tr = Transport(a)
    if a.wait_download:
        wait_download(tr, tr.discover())
        return
    launch = 'cold-' + ('DRYRUN' if a.dry_run else time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    with cluster_lock(a.dry_run):
        hosts = tr.discover()
        tr.all(hosts, 'sudo -n true\nif ! command -v fuser >/dev/null; then sudo -n apt-get update -q; sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y psmisc; fi\n' + free_devices())
        # A rerun must preserve previous evidence before even changing an editable install.
        tr.all(hosts, 'sudo -n mkdir -p /opt/dsv4; sudo -n chown "$(id -u):$(id -g)" /opt/dsv4; mkdir -p /opt/dsv4/{runs,logs}')
        collect(tr, hosts, STATE / 'copies' / (launch + '-before'))
        archive = STATE / (launch + '.tar.gz') if a.dry_run else build_archive([], launch, a.base)
        tools = STATE / 'tools.tar.gz' if a.dry_run else tools_archive(a.payload)
        stage_remote(tr, hosts, archive, launch)
        try:
            for h in hosts:
                tr.upload(h, tools, '/opt/dsv4')
            tr.all(hosts, 'cd /opt/dsv4; sha256sum -c tools/SHA256SUMS')
            setup = '''sudo -n apt-get update -q
    sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -q build-essential cmake ninja-build rsync git patch psmisc curl
    export PATH=/opt/dsv4/tools:$PATH
    export JAX_PLATFORMS=cpu HF_HOME=/opt/dsv4/hf-cache
    if ! test -x /opt/dsv4/.venv/bin/python; then uv venv --python 3.12 /opt/dsv4/.venv; fi
    uv pip install --python /opt/dsv4/.venv/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -r /opt/dsv4/dw-tpu/requirements-tpu.txt
    if ! test -x /opt/dsv4/.venv-download/bin/python; then uv venv --python 3.12 /opt/dsv4/.venv-download; fi
    uv pip install --python /opt/dsv4/.venv-download/bin/python huggingface-hub==2.2.0
    cp /opt/dsv4/dw-tpu/runner/vendor/checkpoint{.py,-manifest.json} /opt/dsv4/tools/
    ''' + apply_script([])
            tr.all(hosts, '{\n' + setup + '\n} > /opt/dsv4/logs/setup.log 2>&1', timeout=3600)
            ips = ','.join(h['ip'] + ':8476' for h in hosts)
            def smoke(h):
                rank = hosts.index(h)
                env = {'LIBTPU_INIT_ARGS': '--enable_tpunetd_client=false', 'TPU_PROCESS_BOUNDS': '2,4,1',
                       'TPU_CHIPS_PER_PROCESS_BOUNDS': '2,2,1', 'TPU_PROCESS_ADDRESSES': ips,
                       'TPU_PROCESS_PORT': '8476', 'CLOUD_TPU_TASK_ID': str(rank)}
                return 'env ' + ' '.join(k + '=' + q(v) for k, v in env.items()) + f' timeout 300 {PYTHON} /opt/dsv4/dw-tpu/runner/vendor/smoke.py --coordinator {q(hosts[0]["ip"] + ":12345")} --rank {rank} --hosts 8 > /opt/dsv4/logs/smoke.json 2>&1\ncat /opt/dsv4/logs/smoke.json'
            results = tr.all(hosts, smoke, timeout=400)
            if not a.dry_run:
                for text in results:
                    lines = [json.loads(s) for s in text.splitlines() if s.startswith('{')]
                    assert any(x.get('passed') and x.get('devices') == 32 and x.get('psum') == [496]*4 for x in lines), text
                dump(STATE / 'cold-start.json', {'launch': launch, 'base': a.base, 'passed': True, 'hosts': hosts})
            # Separate interpreter + stable scripts let jobs replace source/venv while I/O runs.
            download = f'''test "$(df --output=avail -B1 /opt/dsv4 | tail -1)" -ge 200000000000
    export HF_HOME=/opt/dsv4/hf-cache
    unset HF_HUB_OFFLINE
    mkdir -p /opt/dsv4/ckpt
    rm -f /opt/dsv4/ckpt/CHECKPOINT_VERIFIED
    /opt/dsv4/.venv-download/bin/python /opt/dsv4/tools/checkpoint.py /opt/dsv4/ckpt/dsv4-flash-full --download
    printf '%s\\n' {q(REVISION)} > /opt/dsv4/ckpt/CHECKPOINT_VERIFIED'''
            tr.all(hosts, unit_command('dsv4-download', download, '/opt/dsv4/logs/checkpoint.log'))
            if not a.download_in_background:
                wait_download(tr, hosts)
        finally:
            collect(tr, hosts, STATE / 'copies' / launch)
        print('Cold start plan complete.' if a.dry_run else '32-device psum = 496 on all hosts; ' + ('downloads running independently.' if a.download_in_background else 'checkpoints verified.'))

if __name__ == '__main__':
    main()
