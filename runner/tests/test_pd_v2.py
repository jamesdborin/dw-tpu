import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
from vendor import pd_bench as pd
from pd_model import model_v2, row_spec_statistics


class PDV2Tests(unittest.TestCase):
    def test_prefill_only_repeat_warm_verification(self):
        case = {'name': 'incr', 'cached_prefix_tokens': 32, 'new_tokens': 16, 'concurrency': [2]}
        spec = {'pd': {'prefill': {'cases': [case], 'max_tokens': 1, 'repeats': 2,
                                 'cache_affinity': True, 'verify_warm_cache': True}}}
        class Fake:
            calls = 0
            resets = 0
            def reset(self): self.resets += 1
            def cohort(self, prompts, *args):
                self.calls += 1
                assert self.cache_affinity
                return ([{'request': i, 'submit': 0, 'token_times': [2],
                          'usage': {'prompt_tokens': len(p), 'prompt_tokens_details': {'cached_tokens': 16}}}
                         for i, p in enumerate(prompts)], [])
        fake = Fake()
        cap = {'kv_cache_size_tokens': 100000, 'block_size': 16, 'max_model_len': 100}
        with tempfile.TemporaryDirectory() as t, patch.object(pd, 'read_capacity', return_value=cap):
            result = pd.run(spec, fake, [1, 2, 3], Path(t) / 'pd.json')
        self.assertEqual((fake.calls, fake.resets), (6, 2))
        self.assertFalse(fake.cache_affinity)
        self.assertEqual(len(result['decode']), 0)
        self.assertEqual([r['repeat'] for r in result['prefill']], [0, 1])
        for r in result['prefill']:
            self.assertTrue(r['valid']); self.assertTrue(r['warm_cache_verification']['valid'])
            self.assertEqual(r['new_prefill_tok_s_per_chip'], .5)
        for bad in (0, True, -1):
            spec['pd']['prefill']['repeats'] = bad
            with self.assertRaises(ValueError): pd.validate_config(spec['pd'])

    def test_affinity_request_has_no_seed(self):
        raw = ('data: ' + json.dumps({'choices': [{'token_ids': [1], 'finish_reason': 'length'}],
               'usage': {'prompt_tokens': 2, 'completion_tokens': 1}}) + '\n\ndata: [DONE]\n').encode()
        client = pd.Client('http://test'); client.cache_affinity = True
        with patch.object(pd.urllib.request, 'urlopen', return_value=contextlib.closing(io.BytesIO(raw))) as call:
            client.stream(65, [1, 2], 1, 0, threading.Barrier(1))
        body = json.loads(call.call_args.args[0].data)
        self.assertNotIn('seed', body)
        self.assertEqual(body['vllm_xargs'], {'pd_cache_rank': 1})

    def test_cache_failure_and_preemption_still_invalidate(self):
        records = [{'submit': 0, 'token_times': [1], 'usage': {'prompt_tokens': 48,
                    'prompt_tokens_details': {'cached_tokens': 0}}}]
        metrics = [{'metrics': {'vllm:num_preemptions_total': v}} for v in (0, 1)]
        row = pd.prefill_result(records, {'name': 'x', 'cached_prefix_tokens': 32, 'new_tokens': 16}, 1, metrics)
        self.assertFalse(row['valid']); self.assertEqual(len(row['flags']), 2)

    def test_capacity_model_and_legacy_spec(self):
        stats = {'boundary_spec_counter_deltas': {'vllm:spec_decode_num_drafts_total': 320,
                 'vllm:spec_decode_num_accepted_tokens_total': 480}}
        row = {'valid': True, 'concurrency': 32, 'context_source': 'random', 'actual_context_lengths': [10000]*32,
               'window': {'seconds': 2, 'requests': [{'decode_tok_s': 50}]*32}, 'verification': stats}
        self.assertEqual(row_spec_statistics(row)['tokens_per_step'], 2.5)
        self.assertEqual(row_spec_statistics(row)['step_seconds'], .2)
        prefill = {'valid': True, 'cached_prefix_tokens': 10000, 'new_tokens': 1650, 'concurrency': 32,
                   'seconds': 10, 'ttft_seconds': {'mean': 2, 'p50': 2, 'p90': 2}}
        result = model_v2({'decode': [row, dict(row, valid=False)]},
                          {'prefill': [prefill, dict(prefill, seconds=30), dict(prefill, valid=False, seconds=.01)]})
        self.assertEqual(len(result['contexts']), 1)
        pool = result['prefill_pools'][0]
        self.assertEqual(pool['new_tok_s'], 2640)
        option = result['contexts'][0]['prefill_options'][0]
        self.assertEqual(option['p_to_d_ratio'], 2)
        self.assertEqual(option['prefill_chips_per_decode_slice'], 64)
        self.assertAlmostEqual(option['tok_s_user'], 500/12)
        self.assertAlmostEqual(option['aggregate_output_tok_s_per32chips'], 1600/3)
        self.assertAlmostEqual(option['fixed_N_agents_output_tok_s_per32chips'], 32*500/12/3)
        self.assertAlmostEqual(option['logical_tpm_including_cached_per32chips'], 3.2*10500*60/3)
        self.assertEqual(model_v2({'decode': [row]}, {'prefill': []})['contexts'][0]['prefill_options'], [])

    def test_final_job_configuration(self):
        job = common.load_job(common.ROOT / 'jobs/pd-final-prefill')
        self.assertEqual(job['acceptance'], 'exact')
        self.assertEqual(job['env']['MNBT'], job['env']['TPU_TOKEN_PADDINGS_PER_DP'].split(',')[-1])
        self.assertGreater(int(job['env']['MAX_LEN']), 16650)
        self.assertNotIn('decode', job['pd'])
        self.assertEqual(job['pd']['prefill']['repeats'], 2)

    def test_job_patch_affinity_is_opt_in_and_validated(self):
        import subprocess
        base = common.load_job(common.ROOT / 'jobs/pd-final-prefill')['base'].split('@')[1]
        pin = subprocess.check_output(['git', '-C', str(common.ROOT), 'rev-parse', base + ':tpu-inference'], text=True).strip()
        source = subprocess.check_output(['git', '-C', str(common.ROOT / 'tpu-inference'), 'show', pin + ':tpu_inference/core/sched/dp_scheduler.py'], text=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); target = root / 'tpu_inference/core/sched/dp_scheduler.py'
            target.parent.mkdir(parents=True); target.write_text(source)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', '-C', str(root), 'apply', str(common.ROOT / 'jobs/pd-final-prefill/patches/tpu-inference/0301-pd-prefill-cache-affinity.patch')], check=True, capture_output=True)
            tree = ast.parse(target.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_pick_rank_for_request')
        scope = {'os': os, 'Request': object}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<routing>', 'exec'), scope)
        scheduler = types.SimpleNamespace(dp_size=32, _routing_policy='least', RoutingPolicy=types.SimpleNamespace(ROUND_ROBIN='rr'), _find_best_rank_for_request=lambda r: 9)
        request = types.SimpleNamespace(sampling_params=types.SimpleNamespace(extra_args={'pd_cache_rank': 1}))
        with patch.dict(os.environ, {'PD_CACHE_AFFINITY': '0'}):
            self.assertEqual(scope['_pick_rank_for_request'](scheduler, request), 9)
        with patch.dict(os.environ, {'PD_CACHE_AFFINITY': '1'}):
            self.assertEqual(scope['_pick_rank_for_request'](scheduler, request), 1)
            for bad in (32, -1, True, '1'):
                request.sampling_params.extra_args['pd_cache_rank'] = bad
                with self.assertRaises(ValueError): scope['_pick_rank_for_request'](scheduler, request)
            request.sampling_params.extra_args = {}
            self.assertEqual(scope['_pick_rank_for_request'](scheduler, request), 9)

    def test_run_exposes_boundary_spec_statistics(self):
        prompt = 'vllm:prompt_tokens_total'
        drafts = 'vllm:spec_decode_num_drafts_total'
        accepted = 'vllm:spec_decode_num_accepted_tokens_total'
        def snapshot(t, tokens, d=0, a=0, boundary=None):
            row = {'start': t, 'end': t+.01, 'metrics': {prompt: tokens, drafts: d, accepted: a,
                   'vllm:num_requests_running': 1 if t < 3 else 0,
                   'vllm:num_requests_waiting': 0, 'vllm:num_preemptions_total': 0}}
            if boundary: row['boundary'] = boundary
            return row
        class Fake:
            def reset(self): pass
            def cohort(self, prompts, total, *args, **kwargs):
                record = {'request': 0, 'submit': 0, 'token_times': list(range(1, total+1)),
                          'finish_time': total, 'usage': {'prompt_tokens': 16, 'completion_tokens': total}}
                samples = [snapshot(0, 0), snapshot(.8, 16), snapshot(1.01, 16, 10, 5, 'start'),
                           snapshot(3.01, 16, 12, 7, 'end'), snapshot(4, 16, 12, 7)]
                return [record], samples
        spec = {'pd': {'prefill': {'max_tokens': 1, 'cases': [{'name': 'cold', 'prompt_tokens': 16, 'concurrency': [1]}]},
                       'decode': {'concurrency': [1], 'context_tokens': [16], 'warmup_tokens': 1,
                                  'measure_tokens': 1, 'slack_tokens': 1, 'temperature': 1, 'capacity_check': 'advisory'}}}
        cap = {'kv_cache_size_tokens': 100000, 'block_size': 16, 'max_model_len': 100}
        with tempfile.TemporaryDirectory() as t, patch.object(pd, 'read_capacity', return_value=cap):
            result = pd.run(spec, Fake(), [1, 2], Path(t)/'pd.json')
        row = result['decode'][0]
        self.assertTrue(row['verification']['occupancy']['verified_all_running'])
        self.assertTrue(row['valid'])
        self.assertEqual(row['spec_statistics']['tokens_per_step'], 2)
        self.assertEqual(row['spec_statistics']['step_seconds'], 1)
        self.assertEqual(row['spec_statistics'], row['verification']['boundary_spec_statistics'])

    def test_advisory_capacity_keeps_hard_model_length(self):
        cap = {'kv_cache_size_tokens': 1000, 'max_concurrency': 32, 'block_size': 1024, 'max_model_len': 4096}
        self.assertFalse(pd.decode_capacity_check([2500]*256, 656, cap, 'strict', 4096)['allowed'])
        check = pd.decode_capacity_check([2500]*256, 656, cap, 'advisory', 4096)
        self.assertTrue(check['allowed']); self.assertIn('kv_capacity_would_preempt', check['advisory_flags'])
        self.assertFalse(pd.decode_capacity_check([3500]*256, 656, cap, 'advisory', 4096)['allowed'])
        self.assertTrue(pd.decode_capacity_check([2500], 656, None, 'advisory', 4096)['allowed'])
        self.assertFalse(pd.decode_capacity_check([3500], 656, None, 'advisory', 4096)['allowed'])
        self.assertTrue(pd.decode_capacity_check([3440], 656, cap, 'advisory', 4096)['allowed'])

    def test_advisory_live_occupancy_and_preemption(self):
        def sample(t, running, waiting=0, preempt=5, boundary=None):
            s = {'start': t, 'end': t+.01, 'metrics': {'vllm:num_requests_running': running,
                 'vllm:num_requests_waiting': waiting, 'vllm:num_preemptions_total': preempt}}
            if boundary: s['boundary'] = boundary
            return s
        samples = [sample(0, 0), sample(1.1, 256, boundary='start'), sample(2, 256), sample(3.1, 255, boundary='end')]
        window = {'start': 1, 'end': 3}
        check = pd.occupancy_metrics(samples, window, 256)
        self.assertTrue(check['verified_all_running'])  # natural end completion not a shortage
        self.assertEqual(check['boundary_metrics']['end']['vllm:num_requests_running'], 255)
        samples[2] = sample(2, 255, 1)
        self.assertIn('not_all_running', pd.occupancy_metrics(samples, window, 256)['flags'])
        samples[-1] = sample(3.1, 255, preempt=6, boundary='end')
        self.assertIn('preempted', pd.occupancy_metrics(samples, window, 256)['flags'])
        samples[-1] = sample(3.1, 255, preempt=4, boundary='end')
        self.assertIn('metrics_counter_reset', pd.occupancy_metrics(samples, window, 256)['flags'])
        self.assertFalse(pd.occupancy_metrics([], window, 256)['verified_all_running'])
        samples[1]['metrics'].pop('vllm:num_requests_running')
        self.assertIn('occupancy_boundary_metrics_unavailable', pd.occupancy_metrics(samples, window, 256)['flags'])

    def test_scaling_v2_config_and_capacity_mode_validation(self):
        spec = common.load_job(common.ROOT/'jobs/pd-scale-base0-v2')
        self.assertEqual(spec['pd']['decode']['capacity_check'], 'advisory')
        self.assertEqual(spec['pd']['decode']['concurrency'], [128,192,256,320,384,512])
        self.assertEqual(spec['env']['MAX_LEN'], '4096')
        spec['pd']['decode']['capacity_check'] = 'ignore'
        with self.assertRaisesRegex(ValueError, 'capacity_check'): pd.validate_config(spec['pd'])
