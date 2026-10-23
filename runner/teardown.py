#!/usr/bin/env python3
"""Copy evidence, stop/free devices, copy final logs, then delete only the owned MIG."""
import argparse
import time
from common import STATE, Transport, arguments, cluster_lock, collect, stop_script

def main(argv=None):
    p = arguments(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--stop-only', action='store_true', help='keep VMs/reservation; stop service and Ray')
    a = p.parse_args(argv); tr = Transport(a)
    with cluster_lock(a.dry_run):
        hosts = tr.discover()
        label = 'release-' + ('DRYRUN' if a.dry_run else time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
        collect(tr, hosts, STATE / 'copies' / (label + '-before'))
        tr.all(hosts, stop_script(), timeout=180)
        collect(tr, hosts, STATE / 'copies' / label)
        if not a.stop_only:
            tr.cloud(['compute', 'instance-groups', 'managed', 'delete', a.group, '--zone=' + a.zone])
        print('Released service; reservation, template and workload policy retained.')

if __name__ == '__main__':
    main()
