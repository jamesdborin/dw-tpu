import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import remote_job
from vendor import pd_bench as pd
from pd_model import model
from summarize import summarize


def record(i, times, end):
    return {'request': i, 'token_times': times, 'finish_time': end}


class PDTests(unittest.TestCase):
    def test_shared_window_counts_bursts_not_chunks(self):
        rows = [record(0, [0, 1, 2, 3, 3, 3, 4, 5], 5), record(1, [1, 2, 3, 4, 4, 4, 5, 6], 6)]
        w = pd.decode_window(rows, 2, 4)
        self.assertTrue(w['valid'])
        self.assertEqual((w['start'], w['end']), (2, 5))
        self.assertEqual([r['tokens'] for r in w['requests']], [5, 5])
        self.assertAlmostEqual(w['aggregate_decode_tok_s'], 10 / 3)
        self.assertEqual(w['itl_seconds']['p50'], .5)
        self.assertEqual(w['requests'][0]['itl_seconds']['mean'], .5)
        self.assertFalse(pd.decode_window(rows, 2, 6)['valid'])

    def test_no_overlap_and_missing_warmup(self):
        self.assertIn('no_shared_decode_window', pd.decode_window([record(0,[0,1,2],2), record(1,[3,4,5],5)],2,1)['flags'])
        self.assertFalse(pd.decode_window([record(0, [0], 1)], 2, 1)['valid'])

    def test_metrics_require_drained_prefill_and_enclosing_scrapes(self):
        prompt = 'vllm:prompt_tokens_total'; drafts = 'vllm:spec_decode_num_drafts_total'
        samples = [{'start':0,'end':1,'metrics':{prompt:100,drafts:2}}, {'start':4,'end':5,'metrics':{prompt:100,drafts:8}}]
        window = {'start':2,'end':3,'seconds':1,'concurrency':2}
        v = pd.window_metrics(samples, window, {prompt:0}, 100)
        self.assertTrue(v['verified_no_prefill']); self.assertEqual(v['counter_deltas'][drafts], 6)
        self.assertFalse(pd.window_metrics(samples, window, {prompt:0}, 101)['verified_no_prefill'])
        samples[-1]['metrics'][prompt] = 101
        self.assertIn('prefill_tokens_in_enclosing_window', pd.window_metrics(samples,window,{prompt:0},100)['flags'])
        self.assertFalse(pd.window_metrics([],window,{},100)['verified_no_prefill'])
        self.assertFalse(pd.window_metrics([dict(s,metrics={}) for s in samples],window,{},100)['verified_no_prefill'])

    def test_spec_position_metrics_preserved(self):
        m = pd.parse_metrics('vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 3\nvllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",position="0"} 4\nvllm:spec_decode_num_drafts_total 10\n')
        self.assertEqual(m['vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"}'], 7)
        self.assertEqual(m['vllm:spec_decode_num_drafts_total'], 10)

    def test_spec_step_normalizes_per_request_drafts(self):
        delta={'vllm:spec_decode_num_drafts_total':320, 'vllm:spec_decode_num_accepted_tokens_total':480,
               'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"}':240}
        stats=pd.spec_statistics(delta,2,32)
        self.assertEqual(stats['step_seconds'], .2)
        self.assertEqual(stats['tokens_per_step'], 2.5)
        self.assertEqual(stats['acceptance_per_position']['vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"}'], .75)
        self.assertIsNone(pd.spec_statistics({},2,32)['step_seconds'])

    def test_capacity_worst_rank_rounding_and_length(self):
        cap = {'kv_cache_size_tokens': 32*8192, 'block_size':1024,'max_model_len':10000}
        self.assertTrue(pd.capacity_check([5000]*32, 128, cap)['allowed'])
        self.assertFalse(pd.capacity_check([5000]*33, 128, cap)['allowed'])
        self.assertIn('model_length_exceeded', pd.capacity_check([10000],1,cap)['flags'])
        self.assertFalse(pd.capacity_check([10],1,None)['allowed'])
        self.assertIn('conservative_hybrid_concurrency_limit', pd.capacity_check([10]*33,1,dict(cap,max_concurrency=32))['flags'])

    def test_capacity_uses_worker_block_size(self):
        class Fake:
            def fetch(self, route):
                return 'vllm:cache_config_info{block_size="16",num_gpu_blocks="17706",kv_cache_size_tokens="1676850"} 1\n'
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'serve.log'; p.write_text('Using cache_config.block_size: 1024 instead of overriding\n')
            cap = pd.read_capacity(Fake(),p)
            self.assertEqual(cap['block_size'],1024)
            self.assertEqual(cap['kv_cache_size_tokens'],1676850)

    def test_incremental_cache_alignment_and_new_token_accounting(self):
        row = dict(record(0,[2],2),submit=0,usage={'prompt_tokens':9650,'prompt_tokens_details':{'cached_tokens':7168}})
        case = {'name':'incr','cached_prefix_tokens':8000,'new_tokens':1650}
        r = pd.prefill_result([row],case,1,[],1024)
        self.assertTrue(r['valid']); self.assertEqual(r['new_prefill_tok_s'],825)
        self.assertEqual(r['observed_uncached_prefill_tok_s'],1241)
        for detail in ({'cached_tokens':0}, None):
            row['usage']['prompt_tokens_details']=detail
            self.assertFalse(pd.prefill_result([row],case,1,[],1024)['valid'])

    def test_stream_preserves_token_bursts_and_rejects_missing_counts(self):
        for chat in (False,True):
            chunks=[{'choices':[{'token_ids':[2,3,4], 'finish_reason':None}]}, {'choices':[{'token_ids':[5], 'finish_reason':'length'}]}, {'choices':[],'usage':{'prompt_tokens':2,'completion_tokens':4}}]
            raw=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n'
            with patch.object(pd.urllib.request,'urlopen', return_value=contextlib.closing(io.BytesIO(raw.encode()))) as call:
                row=pd.Client('http://test').stream(0,{'messages':[], 'seed':123} if chat else [1,2],4,1,threading.Barrier(1))
            self.assertNotIn('seed', json.loads(call.call_args.args[0].data))
            self.assertEqual(len(row['token_times']),4)
            self.assertEqual(len(set(row['token_times'][:3])),1)
            self.assertEqual(call.call_args.args[0].full_url, 'http://test/v1/'+('chat/completions' if chat else 'completions'))
        bad='data: '+json.dumps({'choices':[{'text':'abc','finish_reason':'length'}]})+'\n\ndata: [DONE]\n\n'
        with patch.object(pd.urllib.request,'urlopen',return_value=contextlib.closing(io.BytesIO(bad.encode()))):
            with self.assertRaisesRegex(RuntimeError,'omitted token_ids'):
                pd.Client('http://test').stream(0,[1],4,1,threading.Barrier(1))

    def test_cohort_takes_both_boundary_snapshots(self):
        class Toy(pd.Client):
            def snapshot(self):
                now=pd.time.monotonic()
                return {'start':now,'end':now,'metrics':{}}
            def stream(self,i,p,total,temp,barrier,event):
                barrier.wait(timeout=5)
                event(i,2,False)
                event(i,4,True)
                return record(i,[0,1,2,3],3)
        with patch.object(pd.time,'sleep'):
            rows,samples=Toy('http://unused').cohort([[1],[2]],4,1,True,2)
        self.assertEqual(len(rows),2)
        self.assertEqual({s['boundary'] for s in samples if 'boundary' in s},{'start','end'})

    def test_model_cost_and_invalid_exclusion(self):
        data={'prefill':[{'valid':True,'concurrency':32,'cached_prefix_tokens':8000,'ttft_seconds':{'mean':2}}], 'decode':[]}
        for c in (5000,10000,15000):
            data['decode'].append({'valid':True,'concurrency':32,'context_source':'random','actual_context_lengths':[c]*32,'window':{'requests':[{'decode_tok_s':50}]*32}})
        data['decode'].append(dict(data['decode'][0],valid=False))
        m=model(data)
        self.assertEqual(len(m['contexts']),3)
        row=m['contexts'][1]
        self.assertEqual(row['separate_equal_slices']['turn_seconds'],12)
        self.assertEqual(row['decode_only_prefill_free']['tok_s_user'],50)
        self.assertEqual(row['separate_equal_slices']['logical_tpm_per_slice'],32*10500*60/12/2)
        self.assertEqual(m['random_context_average']['separate_equal_slices']['mean_turn_seconds'],12)

    def test_config_jobs_controller_and_summary(self):
        for job in ('pd-base0','pd-specdec'):
            spec=common.load_job(common.ROOT/'jobs'/job)
            self.assertEqual(spec['bench_mode'],'pd')
            self.assertEqual(spec['env']['GPU_MEM'],'0.68')
            with tempfile.TemporaryDirectory() as t:
                out=Path(t)
                with patch.object(remote_job,'checked') as check:
                    remote_job.benchmark(spec,out,100,1)
                self.assertEqual(check.call_count,2)
                self.assertTrue(any('pd_bench.py' in str(x) for x in check.call_args_list[0].args[0]))
                common.dump(out/'pd.json',{'prefill':[],'decode':[]})
                common.dump(out/'acceptance.json',{'passed':True})
                summarize(out,spec)
                self.assertIn('## P/D benchmark',(out/'SUMMARY.md').read_text())
            spec['pd']['decode']['context_source']=['bad']
            with self.assertRaises(ValueError):pd.validate_config(spec['pd'])
        self.assertEqual(common.load_job(common.ROOT/'jobs/baseline-1')['bench_mode'],'agent')

    def test_dry_run_never_reads_tokenizer_or_network(self):
        with patch.object(sys,'argv',['pd_bench.py',str(common.ROOT/'jobs/pd-base0/job.json'),'--out','unused','--dry-run']), patch.object(pd,'eligible_tokens',side_effect=AssertionError('tokenizer')), patch.object(pd.Client,'fetch',side_effect=AssertionError('network')), contextlib.redirect_stdout(io.StringIO()) as output:
            pd.main()
        self.assertIn('agent-replay',output.getvalue())

    def test_token_pool_excludes_special_ids(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'tokenizer.json';common.dump(path,{'model':{'vocab':{'a':0,'b':1,'c':2}},'added_tokens':[{'id':1,'special':True}]})
            self.assertEqual(pd.eligible_tokens(path),[0,2])
