"""Cold causal suffix invariance and greedy comparison with corrected B0."""
import argparse,json,math,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--prompts',type=Path,default=Path('/opt/dsv4/bench-a1/prompts.json'));p.add_argument('--baseline',type=Path);p.add_argument('--causality',action='store_true');p.add_argument('--exact-logprobs',action='store_true');p.add_argument('--reduction-order',action='store_true');p.add_argument('--all-greedy-ids',action='store_true');a=p.parse_args()
assert not (a.exact_logprobs and a.reduction_order);a.out.mkdir(parents=True,exist_ok=True)
def post(path,data=None):
 req=urllib.request.Request('http://127.0.0.1:8000'+path,data=json.dumps(data).encode() if data is not None else b'',headers={'Content-Type':'application/json'},method='POST')
 with urllib.request.urlopen(req,timeout=1800) as response:return json.load(response)
def query(ids,echo=False):
 assert post('/reset_prefix_cache')['success']
 return post('/v1/completions',dict(model='dsv4-flash',prompt=ids,max_tokens=32,temperature=0,seed=0,ignore_eos=True,logprobs=10,echo=echo,return_tokens_as_token_ids=True,return_token_ids=True))
prompts=json.loads(a.prompts.read_text())['base'];raw={};comparisons=[]
ref=json.loads(a.baseline.read_text()) if a.baseline else None
for item in [prompts[0],prompts[1],prompts[3]]:
 name=item['name'];response=query(item['ids']);raw[name]=response
 (a.out/'greedy.json').write_text(json.dumps(raw,indent=2)+'\n')
 choice=response['choices'][0];lp=choice['logprobs']['token_logprobs']
 assert response['usage']['completion_tokens']==32 and len(choice['token_ids'])==32
 assert all(v is not None and math.isfinite(v) for v in lp)
 if ref:
  baseline=ref[name]['choices'][0]
  exact=choice['token_ids']==baseline['token_ids']
  delta=max(abs(x-y) for x,y in zip(lp,baseline['logprobs']['token_logprobs']))
  comparisons.append(dict(name=name,exact_tokens=exact,max_logprob_abs_error=delta))
  (a.out/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
  if a.reduction_order:
   if a.all_greedy_ids or name in ['base0_L17','base1_L300']:assert exact,comparisons[-1]
  else:assert exact and delta<=(0.0 if a.exact_logprobs else .05),comparisons[-1]
if a.causality:
 ids=prompts[1]['ids'];assert len(ids)==300
 first=query(ids,True);(a.out/'causal-original.json').write_text(json.dumps(first))
 results=[]
 for boundary in (143,178,251):
  variant=ids[:boundary]+list(reversed(ids[boundary:]));second=query(variant,True)
  (a.out/f'causal-suffix-{boundary}.json').write_text(json.dumps(second))
  x=first['choices'][0]['logprobs']['token_logprobs'];y=second['choices'][0]['logprobs']['token_logprobs']
  assert len(x)==332 and len(y)==332
  delta=max(abs(x[i]-y[i]) for i in range(1,boundary))
  results.append(dict(shared_prefix=boundary,max_logprob_abs_error=delta))
  (a.out/'causality-summary.json').write_text(json.dumps(results,indent=2)+'\n')
  assert delta==0.0,results[-1]
print(json.dumps({'passed':True,'comparisons':comparisons,'causality':a.causality}))
