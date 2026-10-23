import csv,json,statistics,sys
from pathlib import Path
out=Path(sys.argv[1]); summaries=[]
for n in map(int,sys.argv[2:] or ['16','32']):
 rows=list(csv.DictReader((out/f'agents-{n}.csv').open()));assert len(rows)==1
 s=rows[0]; recs=[json.loads(line.split('REQUEST_METRIC',1)[1]) for line in (out/f'agents-{n}.log').read_text().splitlines() if 'REQUEST_METRIC' in line]
 for k,v in [('planned_tool_invocations',n*10),('failed_requests',0),('total_requests',n*10),('successful_requests',n*10),('completed_agents',n)]:assert int(s[k])==v,(k,s[k],v)
 assert len(recs)==n*10
 assert len({(r['agent_id'],r['step']) for r in recs})==n*10
 assert all(r['output_tokens']==500 and r['latency_ms']>0 for r in recs)
 growth=[]; outliers=[]
 assert len({r['agent_id'] for r in recs})==n
 for agent in {r['agent_id'] for r in recs}:
  rs=sorted([r for r in recs if r['agent_id']==agent],key=lambda r:r['step'])
  assert [r['step'] for r in rs]==list(range(1,11))
  assert 4800<=rs[0]['prompt_tokens']<=5500
  gs=[b['prompt_tokens']-a['prompt_tokens'] for a,b in zip(rs,rs[1:])]
  outliers.extend(dict(agent_id=agent,step=i+2,growth=x) for i,x in enumerate(gs) if not 800<=x<=1200)
  growth+=gs
 assert 800<=statistics.median(growth)<=1200,statistics.median(growth)
 tin=sum(r['prompt_tokens'] for r in recs);tout=sum(r['output_tokens'] for r in recs)
 assert tin==int(s['total_input_tokens']) and tout==int(s['total_output_tokens'])
 dur=float(s['total_duration_seconds']);assert dur>0
 summaries.append(dict(agents=n,requests=len(recs),e2e_p50_s=statistics.median(r['latency_ms']/1000 for r in recs),output_tps_p50=statistics.median(r['output_tokens']*1000/r['latency_ms'] for r in recs),logical_tpm=60*(tin+tout)/dur,aggregate_output_tps=tout/dur,prompt_growth_min=min(growth),prompt_growth_max=max(growth),prompt_growth_median=statistics.median(growth),prompt_growth_outliers=outliers,duration_s=dur))
(out/'validated-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
print(json.dumps(summaries,indent=2))
