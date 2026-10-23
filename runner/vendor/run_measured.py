"""Run BatchBench with server-side active-request evidence, no synthetic load."""
import json,re,subprocess,sys,threading,time,urllib.request
from pathlib import Path
out=Path(sys.argv[1]);n=int(sys.argv[2]);cmd=sys.argv[3:]
# Equal initial cache state for all measured runs, including B0 repeat.
req=urllib.request.Request('http://127.0.0.1:8000/reset_prefix_cache',method='POST')
with urllib.request.urlopen(req,timeout=30) as response:
 result=json.load(response)
assert result.get('success') is True,result
(out/f'agents-{n}.cache-reset.json').write_text(json.dumps(result))
stop=threading.Event();samples=[]
def monitor():
 while not stop.is_set():
  record={'time':time.time()}
  try:
   with urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=5) as response:text=response.read().decode()
   for name in ['num_requests_running','num_requests_waiting']:
    values=re.findall(r'^vllm:'+name+r'(?:\{[^\n]*\})?\s+([\d.eE+-]+)',text,re.M)
    record[name]=sum(map(float,values)) if values else None
  except Exception as e:record['error']=str(e)
  samples.append(record);stop.wait(.5)
t=threading.Thread(target=monitor);t.start()
try:
 with (out/f'agents-{n}.log').open('w') as log:
  status=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
finally:
 stop.set();t.join()
 (out/f'agents-{n}.concurrency.json').write_text(json.dumps(samples,indent=2)+'\n')
assert status==0,status
peak=max((s.get('num_requests_running') or 0 for s in samples),default=0)
assert peak>=n,('Actual Running concurrency gate failed',n,peak)
print(json.dumps({'agents':n,'observed_running_peak':peak}))
