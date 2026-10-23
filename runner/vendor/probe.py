"""Full-model HTTP logprobs and repeat-cache evidence; stores raw responses."""
import argparse,json,math,urllib.request,time,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--host',default='http://127.0.0.1:8000');p.add_argument('--out',type=Path,required=True);p.add_argument('--reference',type=Path);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
prompts=json.loads(Path(__file__).with_name('prompts.json').read_text())['base']
def request(ids,echo=False):
 data=dict(model='dsv4-flash',prompt=ids,max_tokens=8,temperature=0,seed=0,ignore_eos=True,logprobs=10,echo=echo,return_tokens_as_token_ids=True,return_token_ids=True)
 req=urllib.request.Request(a.host+'/v1/completions',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(req,timeout=3600))
def metrics(name):
 text=urllib.request.urlopen(a.host+'/metrics',timeout=30).read().decode()
 (a.out/(name+'.metrics.txt')).write_text(text)
 result={}
 for line in text.splitlines():
  m=re.match(r'(vllm:(?:prefix_cache_\w+|prompt_tokens_by_source)_total)(\{[^}]*\})?\s+([\d.eE+-]+)',line)
  if m:result[m[1]+(m[2] or '')]=float(m[3])
 return result
def delta(before,after):
 result={k:v-before.get(k,0) for k,v in after.items()}
 assert all(v>=0 for v in result.values()), 'Counter reset during isolated probe'
 return result
raw={}; summary={}
for item in [prompts[0],prompts[1],prompts[3]]:
 name=item['name'];ids=item['ids']
 before=metrics(name+'-before')
 print(f'{time.time():.3f} {name} first start',flush=True)
 first=request(ids);raw[name]={'first':first}
 (a.out/'full-http-probes.json').write_text(json.dumps(raw,indent=2))
 time.sleep(1);after_first=metrics(name+'-after-first')
 print(f'{time.time():.3f} {name} first complete; repeat start',flush=True)
 repeat=request(ids);raw[name]['repeat']=repeat
 time.sleep(1);after_repeat=metrics(name+'-after-repeat')
 (a.out/'full-http-probes.json').write_text(json.dumps(raw,indent=2))
 ca,cb=first['choices'][0],repeat['choices'][0]
 for response in [first,repeat]:
  assert response['usage']['completion_tokens']==8
  choice=response['choices'][0]; assert len(choice['token_ids'])==8
  assert all(math.isfinite(v) and v != -9999 for v in choice['logprobs']['token_logprobs'])
  assert all(math.isfinite(v) for pos in choice['logprobs']['top_logprobs'] for v in pos.values())
 summary[name]={'same_tokens':ca['token_ids']==cb['token_ids'],'same_logprobs':ca['logprobs']==cb['logprobs'],'first_usage':first['usage'],'repeat_usage':repeat['usage'],'first_counter_deltas':delta(before,after_first),'repeat_counter_deltas':delta(after_first,after_repeat)}
 if name in ['base0_L17','base1_L300']:
  raw[name]['echo']=request(ids,True)
  (a.out/'full-http-probes.json').write_text(json.dumps(raw,indent=2))
  values=raw[name]['echo']['choices'][0]['logprobs']['token_logprobs']
  assert len(values)==len(ids)+8 and values[0] is None
  assert all(v is not None and math.isfinite(v) and v != -9999 for v in values[1:])
 (a.out/'full-http-probes.json').write_text(json.dumps(raw,indent=2))
 (a.out/'cache-summary.json').write_text(json.dumps(summary,indent=2))
if a.reference:
 ref=json.loads(a.reference.read_text());comparisons=[]
 for item in ref['rounds']['plp']:
  actual=raw[item['name']]['echo']['choices'][0]['logprobs']
  ids=next(x['ids'] for x in prompts if x['name']==item['name'])
  deltas=[];agree=[]
  for i,reference in enumerate(item['prompt_logprobs']):
   if reference is None:continue
   top=actual['top_logprobs'][i]
   assert top is not None
   current={int(k.split(':',1)[1]):v for k,v in top.items()}
   reference={int(k):v for k,v in reference.items()}
   value=actual['token_logprobs'][i]
   assert math.isfinite(value) and math.isfinite(reference[ids[i]])
   deltas.append(abs(value-reference[ids[i]]))
   agree.append(max(current,key=current.get)==max(reference,key=reference.get))
  assert deltas
  comparisons.append(dict(name=item['name'],positions=len(deltas),mean_abs_logprob_error=sum(deltas)/len(deltas),max_abs_logprob_error=max(deltas),top1_agreement=sum(agree)/len(agree)))
 (a.out/'full-reference-comparison.json').write_text(json.dumps(comparisons,indent=2))
print(json.dumps(summary,indent=2))
