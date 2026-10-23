"""Exercise failure cleanup and command plans without cloud access."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import run_job
import cold_start
import run_micro32


class Plans(unittest.TestCase):
    def test_all_remote_shells_parse(self):
        scripts = []
        def ssh(self, host, script, **kw):
            scripts.append(script)
            return ''
        with patch.object(common.Transport, 'ssh', ssh), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(sys, 'argv', ['cold_start.py', '--dry-run', '--download-in-background']):
                cold_start.main()
            with patch.object(sys, 'argv', ['run_job.py', str(common.ROOT / 'jobs/baseline-1'), '--dry-run']):
                run_job.main()
            with tempfile.TemporaryDirectory() as temp:
                job = Path(temp) / 'micro-test'; job.mkdir()
                common.dump(job / 'job.json', {'id': job.name, 'owner': 'test', 'kind': 'micro32',
                    'base': json.loads((common.ROOT / 'jobs/baseline-1/job.json').read_text())['base'],
                    'est_minutes': 10, 'env': {}})
                (job / 'run.sh').write_text('true\n')
                with patch.object(sys, 'argv', ['run_micro32.py', str(job), '--dry-run']):
                    run_micro32.main()
        self.assertGreater(len(scripts), 100)
        for i, script in enumerate(scripts):
            with self.subTest(command=i):
                p = subprocess.run(['bash', '-n'], input='set -euo pipefail\n' + script, text=True, capture_output=True)
                self.assertEqual(p.returncode, 0, p.stderr + '\n' + script)
        joined = '\n'.join(scripts)
        self.assertIn('JAX_NUM_PROCESSES=8', joined)
        self.assertIn('RuntimeMaxSec=900', joined)
        self.assertIn('/opt/dsv4/.venv-download/bin/python', joined)
        self.assertIn('CHECKPOINT_VERIFIED', joined)

    def test_failure_still_stops_collects_and_reports(self):
        class FakeTransport:
            def __init__(self, args): self.dry = False
            def discover(self): return [{'name': 'host0', 'ip': '10.0.0.1'}]
            def all(self, hosts, script, **kw): return ['']
            def ssh(self, host, script, **kw): return ''
        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp) / 'failed-job'; job.mkdir()
            spec = json.loads((common.ROOT / 'jobs/baseline-1/job.json').read_text()); spec['id'] = job.name
            common.dump(job / 'job.json', spec)
            with patch.object(sys, 'argv', ['run_job.py', str(job)]), \
                 patch.object(run_job, 'Transport', FakeTransport), \
                 patch.object(run_job, 'cluster_lock', lambda x: contextlib.nullcontext()), \
                 patch.object(run_job, 'build_archive', return_value=Path(temp) / 'archive.tar.gz'), \
                 patch.object(run_job, 'stage_remote'), \
                 patch.object(run_job, 'launch_ray', side_effect=RuntimeError('injected launch failure')), \
                 patch.object(run_job, 'collect', return_value=Path(temp)) as collect, \
                 patch.object(run_job, 'extract_evidence'), \
                 patch.object(run_job, 'STATE', Path(temp) / 'state'):
                with self.assertRaisesRegex(RuntimeError, 'injected launch failure'):
                    run_job.main()
            self.assertEqual(collect.call_count, 2)
            summary = json.loads((job / 'result/summary.json').read_text())
            self.assertEqual(summary['status'], 'failed')
            self.assertFalse(summary['acceptance']['passed'])
            self.assertIn('injected launch failure', (job / 'result/SUMMARY.md').read_text())

    def test_fresh_result_preserves_previous(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); job = root / 'job'; (job / 'result').mkdir(parents=True)
            (job / 'result/result.json').write_text('{"passed":true}')
            with patch.object(common, 'STATE', root / 'state'):
                dest = common.fresh_result(job, 'new-launch')
            self.assertFalse((dest / 'result.json').exists())
            self.assertTrue((root / 'state/result-history/new-launch/result.json').exists())

if __name__ == '__main__':
    unittest.main()
