"""The informational benchmark exception must fail closed."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import remote_job as remote
import run_job
from summarize import summarize


class Gates(unittest.TestCase):
    def test_reference_checker_separates_threshold_and_hard_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ['base0_L17', 'base1_L300', 'base3_L2600']
            ref = {'rounds': {'plp': [{'name': n, 'prompt_logprobs': [{'0': -1., '1': -2.}]} for n in names]}}
            common.dump(root / 'ref.json', ref)
            common.dump(root / 'prompts.json', {'base': [{'name': n, 'ids': [0]} for n in names]})
            for variant, value in [('base', -1.1), ('test', -1.3)]:
                p = root / variant
                raw = {n: {'echo': {'choices': [{'logprobs': {'token_logprobs': [value], 'top_logprobs': [{'id:0': value, 'id:1': -2.}]}}]},
                    **{phase: {'usage': {'completion_tokens': 8}, 'choices': [{'logprobs': {'token_logprobs': [-1.] * 8}}]} for phase in ('first', 'repeat')}} for n in names}
                common.dump(p / 'probes/full-http-probes.json', raw)
                common.dump(p / 'probes/cache-summary.json', {n: {'same_tokens': True, 'same_logprobs': True} for n in names})
                common.dump(p / 'correctness/greedy.json', {n: {'choices': [{'token_ids': [0] * 32, 'logprobs': {'token_logprobs': [-1.] * 32}}]} for n in names[:2]})
                common.dump(p / 'correctness/causality-summary.json', [{'shared_prefix': n, 'max_logprob_abs_error': 0} for n in (143, 178, 251)])
            cmd = [sys.executable, str(remote.HERE / 'vendor/reference_acceptance.py'), '--experiment', str(root / 'test'), '--baseline', str(root / 'base'), '--reference', str(root / 'ref.json'), '--prompts', str(root / 'prompts.json')]
            for bad_causality in (False, True):
                if bad_causality:
                    common.dump(root / 'test/correctness/causality-summary.json', [])
                p = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(p.returncode, 1, p.stderr)
                evidence = json.loads((root / 'test/reference-acceptance.json').read_text())
                self.assertEqual(len(evidence['threshold_errors']), 3)
                self.assertEqual(bool(evidence['hard_errors']), bad_causality)

    def test_only_completed_threshold_rejection_is_permitted(self):
        cases = ['allowed', 'default', 'crash', 'no_evidence', 'hard', 'correctness', 'cache', 'oom', 'sampling']
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                common.dump(out / 'probes/cache-summary.json', {str(n): {'same_tokens': case != 'cache', 'same_logprobs': True} for n in range(3)})
                evidence = {'passed': False, 'threshold_errors': ['MAE threshold'], 'hard_errors': [], 'errors': ['MAE threshold']}
                # Stale success-looking threshold evidence must not cover a crash.
                common.dump(out / 'reference-acceptance.json', evidence)
                def checked(cmd, out, name, *args):
                    if name == 'correctness.log' and case == 'correctness':
                        raise remote.CommandFailed(name, 1)
                    if name == 'reference.log':
                        if case == 'hard':
                            evidence['hard_errors'] = ['greedy mismatch']
                            evidence['errors'].append('greedy mismatch')
                        if case != 'no_evidence':
                            common.dump(out / 'reference-acceptance.json', evidence)
                        raise remote.CommandFailed(name, -9 if case == 'crash' else 1)
                spec = {'acceptance': 'sampling-equivalent' if case == 'sampling' else 'reduction-order', 'benchmark_on_gate_failure': case != 'default'}
                with patch.object(remote, 'checked', side_effect=checked), patch.object(remote, 'audit', side_effect=RuntimeError('OOM') if case == 'oom' else None):
                    if case == 'allowed':
                        self.assertFalse(remote.acceptance(spec, out, 100, 1, out))
                        self.assertFalse(json.loads((out / 'acceptance.json').read_text())['passed'])
                    else:
                        with self.assertRaises((RuntimeError, AssertionError)):
                            remote.acceptance(spec, out, 100, 1, out)

    def test_controller_benchmarks_profiles_and_marks_failure(self):
        for bench_crash in (False, True):
            with self.subTest(bench_crash=bench_crash), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); out = root / 'out'; out.mkdir()
                common.dump(root / 'job.json', {'id': 'test', 'acceptance': 'reduction-order', 'benchmark_on_gate_failure': True, 'profile': True, 'est_minutes': 10})
                (out / 'launch-id').write_text('test')
                response = MagicMock(); response.__enter__.return_value.status = 200
                def accept(*args):
                    common.dump(out / 'acceptance.json', {'passed': False})
                    return False
                with patch.object(sys, 'argv', ['remote_job.py', str(root), '--out', str(out), '--launch-id', 'test']), patch.object(remote.signal, 'signal'), patch.dict(remote.os.environ), patch.object(remote, 'audit'), patch.object(remote.urllib.request, 'urlopen', return_value=response), patch.object(remote, 'acceptance', side_effect=accept), patch.object(remote, 'benchmark', side_effect=RuntimeError('benchmark crashed') if bench_crash else None) as bench, patch.object(remote, 'checked') as checked:
                    if bench_crash:
                        with self.assertRaisesRegex(RuntimeError, 'benchmark crashed'): remote.main()
                    else:
                        remote.main()
                    bench.assert_called_once()
                    self.assertEqual(checked.call_count, 0 if bench_crash else 1)
                state = json.loads((out / 'state.json').read_text())
                self.assertEqual(state['status'], 'failed' if bench_crash else 'gate_failed_benchmarked')
                summary = summarize(out, {'id': 'test', 'base': 'test'})
                self.assertFalse(summary['acceptance_passed'])
                if not bench_crash:
                    self.assertIn('INFORMATIONAL ONLY', (out / 'SUMMARY.md').read_text())

    def test_schema_defaults_and_restrictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / 'test'; job.mkdir()
            spec = json.loads((common.ROOT / 'jobs/baseline-1/job.json').read_text()); spec['id'] = 'test'
            spec.pop('benchmark_on_gate_failure', None)
            common.dump(job / 'job.json', spec)
            self.assertFalse(common.load_job(job)['benchmark_on_gate_failure'])
            for value, mode, valid in [(True, 'reduction-order', True), (True, 'exact', False), ('true', 'reduction-order', False)]:
                spec.update(benchmark_on_gate_failure=value, acceptance=mode)
                common.dump(job / 'job.json', spec)
                if valid: self.assertTrue(common.load_job(job)['benchmark_on_gate_failure'])
                else:
                    with self.assertRaises(ValueError): common.load_job(job)

    def test_local_launcher_preserves_informational_terminal_status(self):
        class Transport:
            def __init__(self, args): self.dry = False
            def discover(self): return [{'name': 'h', 'ip': '10.0.0.1'}]
            def all(self, *args, **kwargs): return []
            def ssh(self, host, command, **kwargs):
                return json.dumps({'status': 'gate_failed_benchmarked'})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); job = root / 'test'; job.mkdir()
            spec = json.loads((common.ROOT / 'jobs/baseline-1/job.json').read_text()); spec.update(id='test', acceptance='reduction-order', benchmark_on_gate_failure=True)
            common.dump(job / 'job.json', spec)
            def extract(*args):
                common.dump(job / 'result/state.json', {'status': 'gate_failed_benchmarked'})
                common.dump(job / 'result/acceptance.json', {'passed': False})
            with contextlib.ExitStack() as stack:
                for name, value in [('Transport', Transport), ('cluster_lock', lambda _: contextlib.nullcontext()), ('build_archive', lambda *a: root / 'archive'), ('stage_remote', lambda *a: None), ('launch_ray', lambda *a: None), ('collect', lambda *a: root), ('extract_evidence', extract), ('STATE', root / 'state')]:
                    stack.enter_context(patch.object(run_job, name, value))
                stack.enter_context(patch.object(sys, 'argv', ['run_job.py', str(job)]))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                run_job.main()
            result = json.loads((job / 'result/summary.json').read_text())
            self.assertEqual(result['status'], 'gate_failed_benchmarked')
            self.assertFalse(result['acceptance_passed'])
