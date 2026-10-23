"""Validate the real pinned worker patch without importing JAX/model libraries."""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import staging


class Profile(unittest.TestCase):
    def test_broadcast_rpc_only_profiles_configured_workers(self):
        base = common.load_job(common.ROOT / 'jobs/baseline-1')['base'].split('@')[1]
        pin = subprocess.check_output(['git', '-C', str(common.ROOT), 'rev-parse', base + ':tpu-inference'], text=True).strip()
        source = subprocess.check_output(['git', '-C', str(common.ROOT / 'tpu-inference'), 'show', pin + ':tpu_inference/worker/tpu_worker.py'], text=True)
        patch = common.ROOT / 'runner/patches/tpu-inference/0001-profile-configured-workers-only.patch'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'tpu_inference/worker/tpu_worker.py'
            target.parent.mkdir(parents=True); target.write_text(source)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            for extra in (['--check'], []):
                subprocess.run(['git', '-C', str(root), 'apply', *extra, str(patch)], check=True, capture_output=True)
            tree = ast.parse(target.read_text())
        worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TPUWorker')
        method = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == 'profile')
        profiler = MagicMock()
        scope = {'jax': SimpleNamespace(profiler=profiler),
                 'envs': SimpleNamespace(PYTHON_TRACER_LEVEL=1, PROFILE_SINGLE_DEVICE=False),
                 '_parse_profile_options': lambda prefix: ({}, {})}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<patched-profile>', 'exec'), scope)
        for start in (True, False):
            scope['profile'](SimpleNamespace(profile_dir=None), start)
        self.assertEqual(profiler.mock_calls, [])
        for directory in ('/tmp/rank0-traces', '/tmp/pprank1-traces'):
            scope['profile'](SimpleNamespace(profile_dir=directory), True)
            self.assertEqual(profiler.start_trace.call_args.args[0], directory)
            scope['profile'](SimpleNamespace(profile_dir=directory), False)
        self.assertEqual(profiler.start_trace.call_count, 2)
        self.assertEqual(profiler.stop_trace.call_count, 2)

    def test_patch_is_scoped_to_profile_jobs_and_after_job_patches(self):
        job = common.ROOT / 'jobs/baseline-1'
        spec = common.load_job(job)
        self.assertNotIn('runner/patches/tpu-inference', staging.apply_script([(job, spec)]))
        spec['profile'] = True
        script = staging.apply_script([(job, spec)])
        self.assertIn('runner/patches/tpu-inference', script)
        self.assertLess(script.index('runner/patches/tpu-inference'), script.index('uv pip install'))
        subprocess.run(['bash', '-n'], input=script, text=True, check=True)
