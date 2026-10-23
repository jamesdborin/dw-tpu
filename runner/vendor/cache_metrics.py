"""Preserve per-label counter deltas; do not confuse queried history with compute."""
import csv,json,re,sys
from pathlib import Path
out=Path(sys.argv[1]);summary={}
def parse(path):
 result={}
 for line in path.read_text().splitlines():
  if not line.startswith('vllm:'):continue
  match=re.match(r'(vllm:[\w:]+)(\{[^}]*\})?\s+([\d.eE+-]+)',line)
  if not match:continue
  name,labels,value=match.groups()
  if name.endswith('_total') and any(x in name for x in ['prefix_cache','prompt_tokens','num_preemptions']):result[name+(labels or '')]=float(value)
 return result
for n in map(int,sys.argv[2:] or ['16','32']):
 before=parse(out/f'agents-{n}.metrics-before.txt');after=parse(out/f'agents-{n}.metrics-after.txt')
 deltas={k:v-before.get(k,0) for k,v in after.items()}
 assert all(v>=0 for v in deltas.values()),'Counter reset detected'
 computed_keys=[k for k in deltas if k.startswith('vllm:prompt_tokens_by_source_total') and 'source="local_compute"' in k]
 computed=sum(deltas[k] for k in computed_keys) if computed_keys else None
 row=next(csv.DictReader((out/f'agents-{n}.csv').open()))
 summary[n]={'counter_deltas':deltas,'newly_computed_prompt_tokens':computed,'newly_computed_input_plus_output_tpm':60*(computed+int(row['total_output_tokens']))/float(row['total_duration_seconds']) if computed is not None else None,'note':'Server source=local_compute counters distinguish actual prefill from logical history. Six-second settle intervals are outside measured CSV duration.'}
(out/'cache-metric-deltas.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
