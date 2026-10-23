#!/usr/bin/env python3
"""Adapted reservation-bound bulk MIG recipe from 2026-10-16/orch/provision.sh."""
import argparse
import json
import time
from common import RESERVATION, Transport, arguments


def main():
    p = arguments(argparse.ArgumentParser(description=__doc__))
    p.add_argument('action', choices=['create', 'status', 'delete'])
    p.add_argument('--reservation', default=RESERVATION)
    p.add_argument('--template', default='dsv4-resv-sa-1tb-20261020')
    p.add_argument('--policy', default='dsv4-v6e32-4x8-us-east5')
    p.add_argument('--subnet', default='tpu-vllm-test-us-east5')
    p.add_argument('--vm-service-account', default='481447246028-compute@developer.gserviceaccount.com')
    p.add_argument('--wait-seconds', type=int, default=7200)
    a = p.parse_args(); t = Transport(a)
    region = a.zone.rsplit('-', 1)[0]
    def cloud(cmd):
        return t.cloud(cmd)
    def status():
        cloud(['beta', 'compute', 'future-reservations', 'describe', a.reservation, '--zone=' + a.zone, '--format=json(status)'])
        cloud(['compute', 'reservations', 'describe', a.reservation, '--zone=' + a.zone, '--format=json'])
        cloud(['compute', 'instance-groups', 'managed', 'describe', a.group, '--zone=' + a.zone, '--format=json'])
    if a.action == 'status':
        # Print responses as well as commands.
        original = cloud
        def cloud(cmd):
            value = original(cmd); print(value); return value
        status(); return
    if a.action == 'delete':
        from teardown import main as teardown
        options = ['--project', a.project, '--zone', a.zone, '--group', a.group]
        if a.hosts:
            options += ['--hosts', str(a.hosts)]
        if a.dry_run:
            options.append('--dry-run')
        teardown(options)
        return
    deadline = time.monotonic() + a.wait_seconds
    while True:
        st = cloud(['beta', 'compute', 'future-reservations', 'describe', a.reservation, '--zone=' + a.zone, '--format=value(status.procurementStatus)']).strip()
        if a.dry_run or st == 'FULFILLED':
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Reservation not FULFILLED within wait deadline')
        print('reservation', st, '— retry in 30s', flush=True); time.sleep(30)
    # Describe before create; only NOT_FOUND triggers creation. Never replace an existing template.
    def ensure(describe, create, verify):
        if a.dry_run:
            cloud(describe); print('# create only if resource is NOT_FOUND'); cloud(create); return
        try:
            value = json.loads(cloud(describe))
        except RuntimeError as exc:
            if 'not found' not in str(exc).lower() and 'NOT_FOUND' not in str(exc):
                raise
            cloud(create)
            value = json.loads(cloud(describe))
        verify(value)
    def verify_template(v):
        prop = v['properties']
        assert prop['machineType'].endswith('ct6e-standard-4t')
        assert prop['scheduling']['provisioningModel'] == 'RESERVATION_BOUND'
        assert any(a.reservation in x for x in prop['reservationAffinity']['values'])
        assert any(s['email'] == a.vm_service_account for s in prop['serviceAccounts'])
        assert any(x['key']=='enable-oslogin' and x['value'].lower()=='true' for x in prop['metadata']['items'])
    ensure(['compute', 'instance-templates', 'describe', a.template, '--format=json'],
           ['compute', 'instance-templates', 'create', a.template, '--machine-type=ct6e-standard-4t',
            '--maintenance-policy=TERMINATE', '--instance-termination-action=DELETE', '--provisioning-model=RESERVATION_BOUND',
            '--reservation-affinity=specific', f'--reservation=projects/{a.project}/reservations/{a.reservation}', '--no-restart-on-failure',
            '--image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e', '--image-project=ubuntu-os-accelerator-images',
            f'--subnet=projects/{a.project}/regions/{region}/subnetworks/{a.subnet}', '--boot-disk-size=1000GB',
            '--boot-disk-type=hyperdisk-balanced', '--service-account=' + a.vm_service_account, '--scopes=cloud-platform',
            '--metadata=enable-oslogin=TRUE', '--labels=purpose=dsv4-reservation,owner=runner'], verify_template)
    def verify_policy(v):
        assert v['workloadPolicy']['acceleratorTopology'] == '4x8', v
    ensure(['compute', 'resource-policies', 'describe', a.policy, '--region=' + region, '--format=json'],
           ['compute', 'resource-policies', 'create', 'workload-policy', a.policy, '--region=' + region,
            '--type=high-throughput', '--accelerator-topology=4x8'], verify_policy)
    def verify_group(v):
        assert v['targetSize'] == 8 and v['instanceTemplate'].endswith('/' + a.template), v
    ensure(['compute', 'instance-groups', 'managed', 'describe', a.group, '--zone=' + a.zone, '--format=json'],
           ['beta', 'compute', 'instance-groups', 'managed', 'create', a.group, '--zone=' + a.zone, '--size=8',
            '--target-size-policy-mode=bulk', f'--template=projects/{a.project}/global/instanceTemplates/{a.template}',
            '--default-action-on-vm-failure=do-nothing', f'--workload-policy=projects/{a.project}/regions/{region}/resourcePolicies/{a.policy}'], verify_group)
    while True:
        value = cloud(['compute', 'instance-groups', 'managed', 'describe', a.group, '--zone=' + a.zone,
                       '--format=json(status.appliedAcceleratorTopologies)'])
        if a.dry_run or any(x.get('state') == 'ACTIVE' for x in json.loads(value).get('status', {}).get('appliedAcceleratorTopologies', [])):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('MIG topology not ACTIVE within wait deadline')
        time.sleep(30)
    print('Topology ACTIVE; run cold_start.py next.')

if __name__ == '__main__':
    main()
