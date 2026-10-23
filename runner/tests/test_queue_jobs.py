from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from queue_jobs import plan


class QueueTests(unittest.TestCase):
    def test_keep_identical_jobs_reuse_and_explicit_changed_jobs_cold(self):
        a={'base':'a','env':{'MNBT':'512'}}; b={'base':'a','env':{'MNBT':'2048'}}
        result=plan([a,a,b,b])
        self.assertEqual([r['action'] for r in result],['cold','reuse','cold','reuse'])
        self.assertIn('env.MNBT',result[2]['identity_diff'][0])
        self.assertEqual(plan([a],previous=a)[0]['action'],'reuse')
        self.assertEqual(plan([a,a],previous=a,fresh_first=True)[0]['action'],'cold')
        self.assertEqual(plan([a,a],previous=a,fresh_first=True)[1]['action'],'reuse')
