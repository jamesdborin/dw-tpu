"""Toy manifests/controllers only: no cloud, TPU, model libraries or large arrays."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import remote_job
import server_reuse as reuse


def manifest(job='old'):
    return {'base': 'a'*40, 'server_recipe': 'recipe',
            'jobs': [{'id': job, 'base': 'dw-tpu@'+'a'*40, 'env': {'MNBT': '512'}, 'acceptance': 'exact',
                      'serve_args_extra': ['--x'], 'profile': False}],
            'files': {'env.sh': 'env', 'patches/tpu-inference/001.patch': 'base',
                      'jobs/'+job+'/patches/tpu-inference/010.patch': 'tpu',
                      'jobs/'+job+'/patches/vllm/010.patch': 'vllm',
                      'jobs/'+job+'/patches/tokamax/010.patch': 'tokamax'}}


def snapshot(m):
    return {'manifest': m, 'server_recipe': 'recipe', 'ray_started': 10, 'serve_started': 11,
            'state': {'status': 'passed', 'launch_id': 'origin', 'ready_unix': 12},
            'acceptance': {'passed': True, 'mode': 'exact'}, 'acceptance_mtime': 20,
            'effective_job': dict(m['jobs'][-1])}


class ReuseTests(unittest.TestCase):
    def test_identity_ignores_job_and_benchmark_fields_but_no_env_guessing(self):
        old, new = manifest(), manifest('new')
        new['jobs'][-1].update(pd={'decode': {'concurrency': [128]}}, bench={'agents': [32]}, est_minutes=90)
        reuse.assert_identical(reuse.identity_from_manifest(old), reuse.identity_from_manifest(new))
        for field, value in [('env', {'MNBT': '2048'}), ('serve_args_extra', ['--different']), ('profile', True)]:
            changed = copy.deepcopy(new); changed['jobs'][-1][field] = value
            with self.assertRaisesRegex(RuntimeError, 'identity differs'):
                reuse.assert_identical(reuse.identity_from_manifest(old), reuse.identity_from_manifest(changed))
        for repo in reuse.REPOS:
            changed = copy.deepcopy(new); changed['files']['jobs/new/patches/'+repo+'/010.patch'] = 'different'
            with self.assertRaisesRegex(RuntimeError, repo):
                reuse.assert_identical(reuse.identity_from_manifest(old), reuse.identity_from_manifest(changed))
        changed = copy.deepcopy(new); changed['base'] = 'b'*40
        with self.assertRaisesRegex(RuntimeError, 'base'):
            reuse.assert_identical(reuse.identity_from_manifest(old), reuse.identity_from_manifest(changed))

    def test_stack_order_and_effective_env(self):
        m = manifest(); m['jobs'].insert(0, dict(m['jobs'][0], id='dep', env={'OTHER': '1', 'MNBT': '256'}))
        m['files']['jobs/dep/patches/vllm/001.patch'] = 'dep'
        identity = reuse.identity_from_manifest(m)
        self.assertEqual(identity['env'], {'OTHER': '1', 'MNBT': '512'})
        self.assertEqual(identity['patches']['vllm'], ['dep', 'vllm'])
        changed = copy.deepcopy(m); changed['jobs'].reverse()
        self.assertTrue(reuse.differences(identity, reuse.identity_from_manifest(changed)))

    def test_refuse_failed_gates_restarts_mutated_manifest_and_gate_mode(self):
        old, desired = manifest(), manifest('new')
        good = snapshot(old)
        reuse.validate_snapshots([good], desired, old, 'origin')
        for key, value in [('acceptance', {'passed': False, 'mode': 'exact'}),
                           ('ray_started', 21), ('serve_started', 21),
                           ('state', {'status': 'failed', 'launch_id': 'origin'}),
                           ('server_recipe', 'changed')]:
            bad = copy.deepcopy(good); bad[key] = value
            with self.assertRaises(RuntimeError): reuse.validate_snapshots([bad], desired, old, 'origin')
        bad = copy.deepcopy(good); bad['effective_job']['env']['MNBT'] = '1024'
        with self.assertRaisesRegex(RuntimeError, 'env.MNBT'): reuse.validate_snapshots([bad], desired, old, 'origin')
        desired['jobs'][-1]['acceptance'] = 'sampling-equivalent'
        with self.assertRaises(RuntimeError): reuse.validate_snapshots([good], desired, old, 'origin')

    def test_probe_shell_compiles_and_requires_units_health_stage(self):
        script = reuse.probe_script('origin', True)
        subprocess.run(['bash', '-n'], input=script, text=True, check=True)
        for required in ('dsv4-ray', 'dsv4-serve', 'dsv4-job', '/health', 'launch-id', 'effective-job.json', 'stage-manifest.json'):
            self.assertIn(required, script)
        self.assertNotIn('systemctl stop', script)

    def test_remote_reuse_never_runs_acceptance_and_records_origin(self):
        for unhealthy in (False, True):
            with self.subTest(unhealthy=unhealthy), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); out = root/'out'; out.mkdir()
                common.dump(root/'job.json', {'id': 'new', 'acceptance': 'exact', 'est_minutes': 10})
                (out/'launch-id').write_text('new')
                common.dump(root/'reuse.json', {'acceptance': {'passed': True, 'mode': 'exact'},
                            'origin_ready_unix': 1, 'acceptance_inherited_from': 'origin', 'acceptance_evidence': '/runs/origin/acceptance.json'})
                response = MagicMock(); response.__enter__.return_value.status = 503 if unhealthy else 200
                with patch.object(sys, 'argv', ['remote_job.py', str(root), '--out', str(out), '--launch-id', 'new', '--reuse-evidence', str(root/'reuse.json')]), \
                     patch.object(remote_job.signal, 'signal'), patch.dict(remote_job.os.environ), \
                     patch.object(remote_job, 'audit'), patch.object(remote_job.urllib.request, 'urlopen', return_value=response), \
                     patch.object(remote_job, 'acceptance', side_effect=AssertionError('must not rerun gates')), \
                     patch.object(remote_job, 'benchmark') as bench:
                    if unhealthy:
                        with self.assertRaisesRegex(RuntimeError, 'no startup fallback'): remote_job.main()
                        bench.assert_not_called()
                    else:
                        remote_job.main(); bench.assert_called_once()
                        state = json.loads((out/'state.json').read_text())
                        self.assertEqual(state['acceptance_inherited_from'], 'origin')
                        self.assertEqual(state['startup_seconds'], 0)
                        self.assertEqual(state['status'], 'passed')

    def test_live_reuse_only_controls_benchmark_unit_and_collects(self):
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); state_dir=root/'state'; state_dir.mkdir(); job=root/'new'; job.mkdir()
                old=manifest(); desired=manifest('new'); spec=desired['jobs'][0]; spec.update(est_minutes=1, kind='e2e')
                hosts=[{'name':'head','ip':'10.0.0.1'}]
                common.dump(state_dir/'current.json', {'launch':'origin','hosts':hosts})
                common.dump(state_dir/'origin.tar.manifest.json', old)
                calls=[]
                class Transport:
                    dry=False
                    def discover(self): return hosts
                    def ssh(self, host, command, **kw):
                        calls.append(command)
                        if 'def remote_snapshot' in command: return json.dumps(snapshot(old))
                        if 'state.json' in command: return json.dumps({'status':'failed' if failed else 'passed','launch_id':'newlaunch','error':'bench failed'})
                        return ''
                    def all(self, hosts, command, **kw): calls.append(command); return []
                    def upload(self, *args): calls.append('client upload')
                def extract(*args):
                    common.dump(job/'result/state.json', {'status':'passed','acceptance_inherited_from':'origin','acceptance_evidence':'/runs/origin/acceptance.json'})
                    common.dump(job/'result/acceptance.json', {'passed':True})
                with patch.object(reuse, 'STATE', state_dir), patch.object(common, 'STATE', state_dir), \
                     patch.object(reuse, 'desired_manifest', return_value=desired), \
                     patch.object(reuse, 'client_bundle', return_value=root/'bundle'), \
                     patch.object(common, 'collect', return_value=root), patch.object(common, 'extract_evidence', side_effect=extract), \
                     patch('run_job.verify_live_logs'), contextlib.redirect_stdout(io.StringIO()):
                    if failed:
                        with self.assertRaisesRegex(RuntimeError, 'bench failed'):
                            reuse.reuse_server(type('Args', (), {'dry_run':False})(), [(job,spec)], 'newlaunch', Transport())
                    else: reuse.reuse_server(type('Args', (), {'dry_run':False})(), [(job,spec)], 'newlaunch', Transport())
                commands='\n'.join(calls)
                self.assertIn('systemctl stop dsv4-job', commands)
                for forbidden in ('systemctl stop dsv4-serve', 'systemctl stop dsv4-ray', 'ray stop', 'uv pip install', 'apply_patches.sh', 'ln -sfn'):
                    self.assertNotIn(forbidden, commands)
                self.assertTrue((job/'result/summary.json').exists())
