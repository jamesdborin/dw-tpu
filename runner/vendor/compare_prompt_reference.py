"""R40 exact teacher-forced prompt reference gate for the saved 17/300 sets."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args()
raw=json.loads((a.experiment/'probes/full-http-probes.json').read_text());ref=json.loads(a.baseline.read_text());rows=[]
for name,n in [('base0_L17',17),('base1_L300',300)]:
    x=raw[name]['echo']['choices'][0];y=ref[name]['echo']['choices'][0]
    assert len(x['prompt_token_ids'])==len(y['prompt_token_ids'])==n
    assert len(x['prompt_logprobs'])==len(y['prompt_logprobs'])==n
    xlp=x['logprobs']['token_logprobs'][:n];ylp=y['logprobs']['token_logprobs'][:n]
    assert len(xlp)==len(ylp)==n and xlp[0] is None and ylp[0] is None
    assert all(v is not None and math.isfinite(v) for v in xlp[1:])
    row={'name':name,'exact_prompt_ids':x['prompt_token_ids']==y['prompt_token_ids'],'exact_prompt_logprobs_all_returned':x['prompt_logprobs']==y['prompt_logprobs'],'max_selected_prompt_lp_delta':max(abs(u-v) for u,v in zip(xlp[1:],ylp[1:]))}
    rows.append(row)
    (a.experiment/'prompt-reference-comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert row['exact_prompt_ids'] and row['exact_prompt_logprobs_all_returned'] and row['max_selected_prompt_lp_delta']==0,row
print(json.dumps({'passed':True,'comparisons':rows}))
