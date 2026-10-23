#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
from common import STATE, Transport, arguments, cluster_lock, collect
p = arguments(argparse.ArgumentParser(description='Copy all hosts runs and Ray logs; refuses <20GiB local free.'))
p.add_argument('--out', type=Path); a = p.parse_args()
with cluster_lock(a.dry_run):
    tr = Transport(a); hosts = tr.discover()
    collect(tr, hosts, a.out or STATE / 'copies' / time.strftime('manual-%Y%m%dT%H%M%SZ', time.gmtime()))
