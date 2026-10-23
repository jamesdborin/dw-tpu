"""Apply orchestrator17:12 reduction-order rule to saved full-model evidence."""
import argparse,hashlib,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--prompts',type=Path,required=True);a=p.parse_args()
ref=json.loads(a.reference.read_text());prompts=json.loads(a.prompts.read_text())['base'];errors=[];summaries=[]
raw={k:json.loads((root/'probes/full-http-probes.json').read_text()) for k,root in [('baseline',a.baseline),('variant',a.experiment)]}
for item in ref['rounds']['plp']:
 name=item['name'];ids=next(x['ids'] for x in prompts if x['name']==name);summary={'name':name}
 for variant in ['baseline','variant']:
  actual=raw[variant][name]['echo']['choices'][0]['logprobs'];deltas=[];agreements=[]
  for i,gpu in enumerate(item['prompt_logprobs']):
   if gpu is None:continue
   gpu={int(k):v for k,v in gpu.items()};value=actual['token_logprobs'][i];assert math.isfinite(value)
   top={int(k.split(':',1)[1]):v for k,v in actual['top_logprobs'][i].items()};deltas.append(abs(value-gpu[ids[i]]));agreements.append(max(top,key=top.get)==max(gpu,key=gpu.get))
  summary[variant]={'mae':sum(deltas)/len(deltas),'max':max(deltas),'top1_matches':sum(agreements),'positions':len(deltas)}
 summary['mae_limit']=1.1*summary['baseline']['mae'];summary['minimum_top1_matches']=summary['baseline']['top1_matches']-2
 if summary['variant']['mae']>summary['mae_limit']:errors.append(name+': GPU-reference MAE exceeds110% baseline')
 if summary['variant']['top1_matches']<summary['minimum_top1_matches']:errors.append(name+': GPU-reference top1 loses more than2 positions')
 summaries.append(summary)
threshold_errors=list(errors)
base_greedy=json.loads((a.baseline/'correctness/greedy.json').read_text());greedy=json.loads((a.experiment/'correctness/greedy.json').read_text())
for name in ['base0_L17','base1_L300']:
 if name not in greedy:errors.append(name+': missing32-token greedy evidence');continue
 choice=greedy[name]['choices'][0]
 if len(choice['token_ids'])!=32 or choice['token_ids']!=base_greedy[name]['choices'][0]['token_ids']:errors.append(name+':32greedy token mismatch')
 if not all(v is not None and math.isfinite(v) for v in choice['logprobs']['token_logprobs']):errors.append(name+':nonfinite greedy logprob')
cache=json.loads((a.experiment/'probes/cache-summary.json').read_text())
for name in ['base0_L17','base1_L300','base3_L2600']:
 if name not in cache:errors.append(name+':missing cache/repeat evidence');continue
 if not(cache[name]['same_tokens'] and cache[name]['same_logprobs']):errors.append(name+':repeat differs')
 for phase in ['first','repeat']:
  response=raw['variant'][name][phase];choice=response['choices'][0]
  if response['usage']['completion_tokens']!=8 or not all(v is not None and math.isfinite(v) for v in choice['logprobs']['token_logprobs']):errors.append(name+':invalid finite probe')
path=a.experiment/'correctness/causality-summary.json'
if not path.exists():errors.append('Missing suffix-causality evidence')
else:
 causal=json.loads(path.read_text())
 if {x['shared_prefix'] for x in causal}!={143,178,251} or any(x['max_logprob_abs_error']!=0 for x in causal):errors.append('Suffix causality is not exact on all three prefixes')
result={'rule':'orchestrator2026-10-16T17:12Z','passed':not errors,'errors':errors,'teacher_forced':summaries,'reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),'max_logprob_error_is_reported_not_an_acceptance_threshold':True}
result.update(threshold_errors=threshold_errors, hard_errors=errors[len(threshold_errors):])
(a.experiment/'reference-acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(bool(errors))
