import csv,json,sys
from pathlib import Path
n=int(sys.argv[2]) if len(sys.argv)>2 else 32; expected=n*3; p=Path(sys.argv[1]);rows=list(csv.DictReader((p/'warmup.csv').open()));assert len(rows)==1
s=rows[0]
for k,v in [('agents',n),('completed_agents',n),('planned_tool_invocations',expected),('successful_requests',expected),('failed_requests',0),('total_requests',expected)]:assert int(s[k])==v,(k,s[k])
recs=[json.loads(x.split('REQUEST_METRIC',1)[1]) for x in (p/'warmup.log').read_text().splitlines() if 'REQUEST_METRIC' in x]
assert len(recs)==expected and len({(r['agent_id'],r['step']) for r in recs})==expected
assert all(r['output_tokens']==500 for r in recs)
print(f'Warmup valid:{expected} successful requests; excluded from measured summaries')
