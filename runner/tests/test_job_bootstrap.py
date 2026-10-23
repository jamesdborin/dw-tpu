"""Source-only regression for a job consuming an already-applied patch stack."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class Micro32Bootstrap(unittest.TestCase):
    def test_overlapping_patches_are_not_reapplied_by_job(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            deployed = root / 'deployed/tpu-inference'; deployed.mkdir(parents=True)
            job = root / 'job'; patches = job / 'patches/tpu-inference'
            patches.mkdir(parents=True)
            (deployed / 'value.txt').write_text('base\n')
            def patch(name, old, new):
                path = patches / name
                path.write_text('diff --git a/value.txt b/value.txt\n--- a/value.txt\n+++ b/value.txt\n@@ -1 +1 @@\n-' + old + '\n+' + new + '\n')
                subprocess.run(['git', 'apply', str(path)], cwd=deployed, check=True, capture_output=True)
                return path
            first = patch('0101.patch', 'base', 'first')
            patch('0102.patch', 'first', 'final')
            # This is the failure of the old bootstrap, even on a valid final tree.
            for mode in ([], ['--reverse']):
                result = subprocess.run(['git', 'apply', *mode, '--check', str(first)], cwd=deployed, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
            for name in ('micro32.py', 'tpi_loader.py', 'splitw.py'):
                (job / name).write_text('# No model imports or device execution in this test.\n')
            script = (ROOT / 'jobs/dsv4-moe-overlap-coll32-1/run.sh').read_text()
            # Execute the real bootstrap, stopping before any TPU-related command.
            bootstrap = script.split('HOST=${JAX_COORDINATOR_ADDRESS', 1)[0]
            self.assertNotEqual(bootstrap, script)
            (job / 'bootstrap.sh').write_text(
                'trap \'test -z "${W:-}" || rm -rf "$W"\' EXIT\n' + bootstrap +
                '\ntest "$(cat "$W/ti/value.txt")" = final\n'
                'test -f "$W/micro32.py"\n')
            env = dict(os.environ, DW_TPU=str(deployed.parent), OUT=str(root / 'out'),
                       JAX_COORDINATOR_ADDRESS='127.0.0.1:12345', JAX_PROCESS_ID='0', JAX_NUM_PROCESSES='8')
            subprocess.run(['bash', str(job / 'bootstrap.sh')], env=env, check=True, capture_output=True)
            self.assertEqual((deployed / 'value.txt').read_text(), 'final\n')


if __name__ == '__main__':
    unittest.main()
