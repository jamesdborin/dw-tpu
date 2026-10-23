"""Replay captured BatchBench payloads with prefill or all-first-token profile trigger.

Run only by serving owner, with ordinary benchmark traffic stopped. Uses native
vLLM profile API and records per-request SSE timestamps/usage and server metrics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from capture import capture


def fetch(base, route, post=False, timeout=30):
    req=urllib.request.Request(base.rstrip('/')+route,data=b'{}' if post else None,
                               headers={'Content-Type':'application/json'},method='POST' if post else 'GET')
    with urllib.request.urlopen(req,timeout=timeout) as r:return r.read().decode()

def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    raw=Path(args.payloads).read_bytes();payloads=[json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(payloads) not in (16,32):raise ValueError('exactly16 or32 captured requests required')
    payload_hash=hashlib.sha256(raw).hexdigest()
    for body in payloads:
        body.update(max_tokens=args.output_tokens,min_tokens=args.output_tokens,stream=True)
        body['stream_options']={'include_usage':True}
    (out/'replayed-payloads.jsonl').write_text(''.join(json.dumps(b,sort_keys=True)+'\n' for b in payloads))
    initial=fetch(args.base_url,'/metrics');(out/'metrics-before.txt').write_text(initial)
    running=re.findall(r'^vllm:num_requests_running(?:\{[^\n]*\})?\s+([\d.eE+-]+)',initial,re.M)
    waiting=re.findall(r'^vllm:num_requests_waiting(?:\{[^\n]*\})?\s+([\d.eE+-]+)',initial,re.M)
    if not running or not waiting or sum(map(float,running+waiting))!=0:
        raise RuntimeError('Expected verified idle server before profiling workload')
    reset=json.loads(fetch(args.base_url,'/reset_prefix_cache',post=True))
    (out/'cache-reset.json').write_text(json.dumps(reset))
    if reset.get('success') is not True:raise RuntimeError('cache reset failed')
    condition=threading.Condition();abort=threading.Event();records=[{'request':i,'first_token_unix':None,'done_unix':None} for i in range(len(payloads))]
    chunks=[[] for _ in payloads];metrics=[]
    def metrics_worker():
        while not abort.is_set():
            try:metrics.append({'unix':time.time(),'text':fetch(args.base_url,'/metrics',timeout=5)})
            except Exception as exc:metrics.append({'unix':time.time(),'error':repr(exc)})
            abort.wait(1)
    def request_worker(i):
        row=records[i];row['submit_unix']=time.time()
        try:
            req=urllib.request.Request(args.base_url.rstrip('/')+'/v1/chat/completions',data=json.dumps(payloads[i]).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=args.request_timeout) as response:
                for raw_line in response:
                    if abort.is_set():raise RuntimeError('workload aborted')
                    line=raw_line.decode().strip()
                    if not line.startswith('data:'):continue
                    data=line[5:].strip()
                    if data=='[DONE]':row['sse_done']=True;break
                    obj=json.loads(data);now=time.time();chunks[i].append({'unix':now,'chunk':obj})
                    if obj.get('error'):raise RuntimeError(str(obj['error']))
                    if obj.get('usage'):row['usage']=obj['usage']
                    for choice in obj.get('choices',[]):
                        delta=choice.get('delta',{})
                        if (delta.get('content') or delta.get('reasoning_content')) and row['first_token_unix'] is None:
                            with condition:row['first_token_unix']=now;condition.notify_all()
                        if choice.get('finish_reason'):row['finish_reason']=choice['finish_reason']
            if not row.get('sse_done'):raise RuntimeError('stream ended without DONE')
        except Exception as exc:row['error']=repr(exc)
        finally:
            with condition:row['done_unix']=time.time();condition.notify_all()
    workers=[threading.Thread(target=request_worker,args=(i,),daemon=True) for i in range(len(payloads))]
    sampler=threading.Thread(target=metrics_worker,daemon=True);sampler.start()
    manifest={'phase_target':args.phase,'agents':len(payloads),'source_payloads_sha256':payload_hash,
              'output_tokens_override':args.output_tokens,'streaming_override':True,'seconds':args.seconds,'start_unix':time.time(),
              'interpretation':'Capture-only first-turn synthetic BatchBench cohort, not ordinary measured benchmark. Phase target requires trace verification.'}
    started=False
    samples=[]
    def start_host_samples():
        for pid in getattr(args,'sample_pid',[]):
            command=[sys.executable,str(Path(__file__).with_name('host_sample.py')),'--pid',str(pid),
                     '--seconds',str(args.seconds),'--py-spy',args.py_spy,
                     '--output',str(out/f'host-{pid}.speedscope.json')]
            samples.append((pid,subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)))
    try:
        if args.phase=='prefill':
            def launch_workers():
                nonlocal started
                start_host_samples()
                for worker in workers:worker.start()
                started=True
            capture(args.base_url,args.seconds,out/'capture.json',args.label,len(payloads),on_started=launch_workers)
        else:
            for worker in workers:worker.start()
            started=True;deadline=time.monotonic()+args.barrier_timeout
            with condition:
                while not all(r['first_token_unix'] for r in records):
                    if any(r['done_unix'] for r in records):raise RuntimeError('Request finished/failed before all-first-token barrier')
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise TimeoutError('all-first-token barrier timed out')
                    condition.wait(min(1,remaining))
                if any(r['done_unix'] for r in records):raise RuntimeError('Cohort already draining at decode trigger')
            manifest['all_first_token_barrier_unix']=time.time()
            capture(args.base_url,args.seconds,out/'capture.json',args.label,len(payloads),on_started=start_host_samples)
        deadline=time.monotonic()+args.request_timeout
        for worker in workers:worker.join(max(0,deadline-time.monotonic()))
        if any(worker.is_alive() for worker in workers):raise TimeoutError('workload completion timed out')
        if any(r.get('error') for r in records):raise RuntimeError('request failures; see requests.json')
        if any(r.get('usage',{}).get('completion_tokens')!=args.output_tokens for r in records):
            raise RuntimeError('missing/wrong output token usage; see requests.json')
        capture_meta=json.loads((out/'capture.json').read_text())
        first=capture_meta['started_ack_unix'];last=capture_meta['stop_request_unix']
        manifest['context_token_bounds']=[{'request':r['request'],'prompt_tokens':r.get('usage',{}).get('prompt_tokens'),'end_context_upper_bound':r.get('usage',{}).get('prompt_tokens',0)+r.get('usage',{}).get('completion_tokens',0)} for r in records]
        manifest['requests_finished_before_profile_stop']=sum(r['done_unix']<last for r in records)
        manifest['requests_with_first_token_before_profile_start']=sum(r['first_token_unix']<first for r in records)
        observed=[]
        for sample in metrics:
            if first<=sample['unix']<=last and 'text' in sample:
                values=re.findall(r'^vllm:num_requests_running(?:\{[^\n]*\})?\s+([\d.eE+-]+)',sample['text'],re.M)
                if values:observed.append(sum(map(float,values)))
        manifest['running_during_capture']={'samples':len(observed),'min':min(observed) if observed else None,'max':max(observed) if observed else None}
        manifest['warnings']=[]
        if not observed or max(observed)<len(payloads):manifest['warnings'].append('Requested concurrency not observed in metrics; verify scheduler/trace before attributing agent-count performance.')
        if manifest['requests_finished_before_profile_stop']:manifest['warnings'].append('Cohort drained during capture; full-window concurrency is not constant.')
        manifest['ok']=True
    except BaseException as exc:
        manifest.update(ok=False,error=repr(exc))
        raise
    finally:
        abort.set();sampler.join(6)
        manifest['host_sample_exit_codes']={}
        for pid,process in samples:
            try:code=process.wait(timeout=args.seconds+25)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:code=process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();code=process.wait()
            manifest['host_sample_exit_codes'][str(pid)]=code
        if started:
            for worker in workers:worker.join(.1)
        manifest['end_unix']=time.time()
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (out/'requests.json').write_text(json.dumps(records,indent=2)+'\n')
        (out/'chunks.json').write_text(json.dumps(chunks)+'\n')
        (out/'metrics.json').write_text(json.dumps(metrics)+'\n')
        try:(out/'metrics-after.txt').write_text(fetch(args.base_url,'/metrics',timeout=5))
        except Exception:pass
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8000');p.add_argument('--payloads',required=True)
    p.add_argument('--phase',choices=('prefill','decode'),required=True);p.add_argument('--label',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=int,default=30);p.add_argument('--output-tokens',type=int,default=1000)
    p.add_argument('--barrier-timeout',type=int,default=180);p.add_argument('--request-timeout',type=int,default=300)
    p.add_argument('--sample-pid',type=int,action='append',default=[]);p.add_argument('--py-spy',default='py-spy')
    a=p.parse_args()
    if not 1<=a.seconds<=60 or not 500<=a.output_tokens<=2000:p.error('seconds1..60, output-tokens500..2000')
    def interrupt(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupt)
    print(json.dumps(run(a),indent=2))
