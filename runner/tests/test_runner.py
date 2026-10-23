import argparse
import contextlib
import datetime as dt
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

RUNNER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNNER))
sys.path.insert(0, str(RUNNER / 'vendor'))
import common
import staging
from allocation_events import classify
from compile_events import classify_compiles
from run_job import environment, serve_script
from summarize import counter_rates, parse_log, summarize


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.PIPE).strip()


class Jobs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def job(self, name='test', **overrides):
        p = self.root / name; p.mkdir(exist_ok=True)
        spec = {'id': name, 'owner': 'test', 'kind': 'e2e', 'base': 'dw-tpu@abcdef0',
                'est_minutes': 60, 'env': {}, 'acceptance': 'exact', 'bench': {'agents': [32], 'repeats': 1}}
        spec.update(overrides); common.dump(p / 'job.json', spec); return p
    def test_load_baseline(self):
        spec = common.load_job(RUNNER.parent / 'jobs/baseline-1')
        self.assertEqual(spec['env']['DSV4_DENSE_FP8_GEMM'], '0')
        self.assertEqual(spec['acceptance'], 'exact')
    def test_bad_values(self):
        for change in ({'id': '../bad'}, {'env': {'BAD KEY': '1'}}, {'env': {'OK': 2}},
                       {'est_minutes': -1}, {'base': 'dw-tpu@main'}, {'serve_args_extra': '--bad'},
                       {'bench': {'agents': [32], 'repeats': 0}}, {'depends_on': ['../bad']},
                       {'abort_if': ['ignore everything']}, {'acceptance': 'sampling-equivalent'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                common.load_job(self.job(**change))
    def test_dependencies_order_dedup_env(self):
        self.job('a', env={'X': 'a'}); self.job('b', depends_on=['a'], env={'X': 'b'})
        p = self.job('c', depends_on=['a', 'b'], env={'Y': 'c'})
        with patch.object(common, 'output', return_value='abcdef0'):
            stack = common.job_stack(p)
        self.assertEqual([s['id'] for _, s in stack], ['a', 'b', 'c'])
        self.assertEqual(common.effective_env(stack), {'X': 'b', 'Y': 'c'})
    def test_cycle(self):
        self.job('a', depends_on=['b']); p = self.job('b', depends_on=['a'])
        with self.assertRaisesRegex(ValueError, 'cycle'):
            common.job_stack(p)
    def test_micro32(self):
        p = self.job(kind='micro32'); (p / 'run.sh').write_text('echo ok')
        self.assertEqual(common.load_job(p)['kind'], 'micro32')
    def test_shell_env_and_args_are_literal(self):
        malicious = '$(touch /tmp/runner-should-not-exist); `echo bad` "words"'
        p = self.job(env={'EXAMPLE': malicious}, serve_args_extra=['--extra', malicious])
        spec = common.load_job(p); text = environment([(p, spec)])
        self.assertIn('export EXAMPLE=' + common.q(malicious), text)
        subprocess.run(['bash', '-n'], input=serve_script(spec, text, '/tmp/run'), text=True, check=True)
    def test_patch_order(self):
        a = self.job('a'); b = self.job('b')
        for p in (a, b):
            d = p / 'patches/tpu-inference'; d.mkdir(parents=True)
            (d / '002.patch').touch(); (d / '001.patch').touch()
        paths = common.patch_paths([(a, {}), (b, {})], 'tpu-inference')
        self.assertEqual([(p.parents[2].name, p.name) for p in paths], [('a', '001.patch'), ('a', '002.patch'), ('b', '001.patch'), ('b', '002.patch')])


class Staging(unittest.TestCase):
    def test_checkout_is_independent_and_patch_stacking(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source'; source.mkdir()
            git(source, 'init', '-q'); (source / 'value.txt').write_text('base\n')
            git(source, 'add', '.'); git(source, '-c', 'user.name=Test', '-c', 'user.email=test@invalid', 'commit', '-qm', 'base')
            pin = git(source, 'rev-parse', 'HEAD')
            staged = root / 'staged'; staging.checkout(source, pin, staged)
            first = root / 'first.patch'; first.write_text('diff --git a/value.txt b/value.txt\n--- a/value.txt\n+++ b/value.txt\n@@ -1 +1 @@\n-base\n+first\n')
            second = root / 'second.patch'; second.write_text(first.read_text().replace('-base', '-first').replace('+first', '+second'))
            for p in (first, second):
                git(staged, 'apply', '--check', str(p)); git(staged, 'apply', str(p))
            self.assertEqual((staged / 'value.txt').read_text(), 'second\n')
            self.assertEqual((source / 'value.txt').read_text(), 'base\n')
            self.assertFalse((staged / '.git/objects/info/alternates').exists())
    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); archive = root / 'bad.tar.gz'
            with tarfile.open(archive, 'w:gz') as tf:
                member = tarfile.TarInfo('runs/test/../../escape'); member.size = 1
                tf.addfile(member, io.BytesIO(b'x'))
            with self.assertRaises(ValueError):
                common.extract_evidence(archive, root / 'result', 'runs/test')


class Summary(unittest.TestCase):
    def test_log(self):
        log = '\x1b[32mINFO 10-17 01:25:28 [compilation_manager.py:1] Compilation of worker0 sample finished in 0.10 [secs].\nINFO 10-17 01:26:44 [compilation_manager.py:2] finished\nnum_blocks=50624 hbm=[(24.01, 31.24)]Gb'
        result = parse_log(log)
        self.assertEqual(result['kv_blocks'], [50624]); self.assertEqual(result['compile_wall_span_seconds'], 76)
        self.assertAlmostEqual(result['compile_reported_task_seconds'], .1)
    def test_missing_metrics_are_not_zero(self):
        self.assertEqual(counter_rates({}), {'apc_hit_rate': None, 'preemptions': None, 'spec_decode_acceptance_rate': None})
    def test_metric_rates(self):
        result = counter_rates({'vllm:prefix_cache_hits_total{x="a"}': 80, 'vllm:prefix_cache_queries_total{x="a"}': 100,
                                'vllm:num_preemptions_total': 0, 'vllm:spec_decode_num_accepted_tokens_total': 6,
                                'vllm:spec_decode_num_draft_tokens_total': 8})
        self.assertEqual(result, {'apc_hit_rate': .8, 'preemptions': 0, 'spec_decode_acceptance_rate': .75})
    def test_full_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); common.dump(root / 'state.json', {'status': 'passed', 'startup_seconds': 120})
            common.dump(root / 'acceptance.json', {'passed': True})
            common.dump(root / 'bench/repeat-1/validated-summary.json', [{'agents': 32, 'output_tps_p50': 23.3, 'logical_tpm': 893000, 'aggregate_output_tps': 700}])
            result = summarize(root, {'id': 'test', 'base': 'dw-tpu@abcdef0', 'env': {}})
            self.assertEqual(result['tok_s_user_median'], 23.3)
            self.assertEqual(result['tpm_including_cached'], 893000)
            self.assertIsNone(result['apc_hit_rate']); self.assertTrue((root / 'SUMMARY.md').exists())


class Audit(unittest.TestCase):
    def test_event_time_not_arrival_time(self):
        ready = dt.datetime(2026, 10, 17, 1, 0, tzinfo=dt.timezone.utc).timestamp()
        old = 'E1017 00:59:59.000123 123 RESOURCE_EXHAUSTED final_attempt: false'
        self.assertFalse(classify(old, ready)['failures'])
        self.assertTrue(classify(old.replace('00:59:59', '01:00:01'), ready)['failures'])
        self.assertTrue(classify('RESOURCE_EXHAUSTED final_attempt: false', ready)['failures'])
    def test_strict_compile(self):
        ready = dt.datetime(2026, 10, 17, 1, 0, tzinfo=dt.timezone.utc).timestamp()
        old = 'WARNING:2026-10-17 00:59:59,001: Compiling jit(model_fn)'
        self.assertFalse(classify_compiles(old, ready)['failures'])
        self.assertTrue(classify_compiles(old.replace('00:59:59', '01:00:01'), ready)['failures'])


class Transport(unittest.TestCase):
    def live_transport(self):
        tr = common.Transport(argparse.Namespace(dry_run=False, project='test', zone='us-east5-b', group='test', hosts=None))
        tr.identity_checked = True
        return tr

    def test_first_contact_retries_without_consuming_payload_stream(self):
        tr = self.live_transport(); host = {'name': 'fresh-host'}
        payload = io.BytesIO(b'archive')
        responses = [subprocess.CompletedProcess([], code, body, b'')
                     for code, body in [(255, b''), (255, b''), (0, b''), (0, b'ok'), (0, b'again')]]
        with patch('subprocess.run', side_effect=responses) as run, patch('common.time.sleep') as sleep, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(tr.ssh(host, 'tar xzf -', stdin=payload), 'ok')
            self.assertEqual(tr.ssh(host, 'echo again'), 'again')
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [5, 10])
        self.assertEqual(run.call_count, 5)
        for call in run.call_args_list[:3]:
            self.assertEqual(call.args[0], tr.command(host, 'true'))
            self.assertEqual(call.kwargs['stdin'], subprocess.DEVNULL)
        self.assertIs(run.call_args_list[3].kwargs['stdin'], payload)

    def test_remote_command_failures_are_not_retried(self):
        for code in (1, 255):
            tr = self.live_transport()
            responses = [subprocess.CompletedProcess([], 0, b'', b''),
                         subprocess.CompletedProcess([], code, b'', b'payload failed')]
            with self.subTest(code=code), patch('subprocess.run', side_effect=responses) as run, patch('common.time.sleep') as sleep, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, f'remote exit {code}'):
                    tr.ssh({'name': 'host'}, 'mutate')
                self.assertEqual(run.call_count, 2)
                sleep.assert_not_called()

    def test_first_contact_retry_deadline(self):
        tr = self.live_transport(); now = [0.0]
        def sleep(seconds):
            now[0] += seconds
        def failed(*args, **kwargs):
            now[0] += 1
            return subprocess.CompletedProcess([], 255, b'', b'connection refused')
        with patch('common.time.monotonic', side_effect=lambda: now[0]), patch('common.time.sleep', side_effect=sleep), patch('subprocess.run', side_effect=failed) as run, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'remote exit 255'):
                tr.ssh({'name': 'host'}, 'must not execute')
        self.assertEqual(now[0], 300)
        self.assertGreater(run.call_count, 1)
        self.assertTrue(all(c.args[0] == tr.command({'name': 'host'}, 'true') for c in run.call_args_list))
        self.assertNotIn('host', tr.contacted_hosts)

    def test_other_probe_failures_are_not_retried(self):
        tr = self.live_transport()
        with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 1, b'', b'failed')) as run, patch('common.time.sleep') as sleep, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'remote exit 1'):
                tr.ssh({'name': 'host'}, 'must not execute')
        self.assertEqual(run.call_count, 1)
        sleep.assert_not_called()

    def test_dry_run_never_executes(self):
        a = argparse.Namespace(dry_run=True, project='test', zone='us-east5-b', group='test', hosts=None)
        tr = common.Transport(a)
        with patch('subprocess.run', side_effect=AssertionError('dry-run ran subprocess')), contextlib.redirect_stdout(io.StringIO()):
            hosts = tr.discover(); tr.all(hosts, 'echo test')
        self.assertEqual(len(hosts), 8)
        self.assertTrue(tr.env['CLOUDSDK_CONFIG'].endswith('/.config/gcloud-tpu-ops'))
    def test_unit_shell(self):
        text = common.unit_command('dsv4-test', 'echo "$(id -u)"\nexec sleep 1', '/tmp/space path/log')
        subprocess.run(['bash', '-n'], input=text, text=True, check=True)
        self.assertIn('KillMode=control-group', text)
    def test_disk_guard(self):
        with patch('shutil.disk_usage', return_value=shutil._ntuple_diskusage(1, 1, 0)), self.assertRaises(RuntimeError):
            common.disk_guard()

if __name__ == '__main__':
    unittest.main()
