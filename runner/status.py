#!/usr/bin/env python3
"""Tail the current controller/server and show all-host systemd/VFIO status."""
import argparse
import json
from pathlib import Path
from common import STATE, Transport, arguments, q
p = arguments(argparse.ArgumentParser(description=__doc__))
p.add_argument('--launch', help='explicit launch directory name'); p.add_argument('--lines', type=int, default=40)
a = p.parse_args(); tr = Transport(a)
current = json.loads((STATE / 'current.json').read_text()) if (STATE / 'current.json').exists() else {}
launch = a.launch or current.get('launch')
if not launch and a.dry_run:
    launch = '<current-launch>'
if not launch:
    p.error('no current job; use --launch or provision.sh status')
hosts = tr.discover(); out = '/opt/dsv4/runs/' + launch
tr.all(hosts, 'sudo -n systemctl --no-pager status dsv4-ray dsv4-serve dsv4-job dsv4-kernel || true; sudo -n fuser /dev/vfio/* 2>/dev/null || true')
if not a.launch and current.get('kind') == 'kernel':
    for active in current['runs']:
        directory = '/opt/dsv4/runs/' + active['launch']
        print(tr.ssh(active['host'], f'tail -n {a.lines} {q(directory + "/run.log")}; test ! -f {q(directory + "/result.json")} || cat {q(directory + "/result.json")}'))
elif not a.launch and current.get('kind') == 'micro32':
    for host in hosts:
        print(tr.ssh(host, f'tail -n {a.lines} {q(out + "/run.log")}'))
else:
    print(tr.ssh(hosts[0], f'cat {q(out + "/state.json")} 2>/dev/null || true\ntail -n {a.lines} {q(out + "/controller.log")} {q(out + "/serve.log")} 2>/dev/null || true'))
