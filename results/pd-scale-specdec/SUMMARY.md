# pd-scale-specdec

Status: passed
Acceptance passed: True

- startup_seconds: 2448.105124473572
- compile_wall_span_seconds: 2395.0
- token_buckets_per_dp: [4, 8, 16, 32, 64, 512]
- kv_blocks: [30720, 30739]
- tok_s_user_median: None
- aggregate_tok_s: None
- tpm_including_cached: None
- apc_hit_rate: None
- preemptions: None
- spec_decode_acceptance_rate: None

P/D cases reported separately; no mixing of contexts, concurrency or spec counters across windows.
Startup compiler-message wall span when readiness is known; task durations include all phases and overlap across hosts. Raw worker logs are preserved.

All eight host archives and checksums: see copies.json.

## P/D benchmark

Best-case projections; see pd.json for raw timestamps, capacity checks and contamination flags.

| Source | N | Context | Valid | Decode tok/s/user (mean) | Flags |
|---|---:|---:|---|---:|---|
| agent-replay | 128 | 5083 | True | 31.71392873143036 |  |
| agent-replay | 192 | 5083 | True | 22.828231900490362 |  |
| agent-replay | 256 | 5083 | True | 15.890436787794144 |  |
| agent-replay | 320 | 5083 | True | 15.675762137412818 |  |
| agent-replay | 384 | 5083 | True | 15.517015938333111 |  |
| agent-replay | 448 | 5083 | False | 15.343417447957174 | insufficient_measured_tokens |
| agent-replay | 512 | 5083 | False | 2.442573994359553 | insufficient_measured_tokens |
| random | 128 | 2500 | True | 19.87562821031386 |  |
| random | 192 | 2500 | True | 14.69694838045974 |  |
| random | 256 | 2500 | True | 14.68371577500617 |  |
| random | 320 | 2500 | True | 10.161043485532106 |  |
| random | 384 | 2500 | True | 10.34559013911152 |  |
| random | 448 | 2500 | True | 9.982432575511906 |  |
| random | 512 | 2500 | True | 10.098786030446055 |  |
| random | 128 | 3500 | True | 20.661115869331464 |  |
| random | 192 | 3500 | True | 14.216332661699616 |  |
| random | 256 | 3500 | True | 14.42306800218218 |  |
| random | 320 | 3500 | True | 9.834205445853788 |  |
| random | 384 | 3500 | True | 9.93727609111383 |  |
| random | 448 | 3500 | True | 9.761022374550569 |  |
| random | 512 | 3500 | True | 9.90557034951756 |  |

### Speculative decode counters

- agent-replay N=128 context=None: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.71475846069869, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.1851801310043668, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.048785480349344976}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.06133076701746383, "tokens_per_step": 1.9487240720524017}
- agent-replay N=192 context=None: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.7277908805031447, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.19452633647798742, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.0553999606918239}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.08665730688681954, "tokens_per_step": 1.977717177672956}
- agent-replay N=256 context=None: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.7502564709595959, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.17724116161616163, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.05093907828282828}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.12468240602020435, "tokens_per_step": 1.9784367108585859}
- agent-replay N=320 context=None: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.7368885869565217, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.19619565217391305, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.058423913043478264}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.12709434521736734, "tokens_per_step": 1.991508152173913}
- agent-replay N=384 context=None: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.739444179389313, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.18479643765903309, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.053256202290076333}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.1275622416717553, "tokens_per_step": 1.9774968193384224}
- agent-replay N=448 context=None: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.7432782061688312, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.18671367694805194, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.05357142857142857}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.1282544923635049, "tokens_per_step": 1.9835633116883118}
- agent-replay N=512 context=None: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.7546202956989247, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.18537886424731181, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.053448420698924734}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.8206913276665947, "tokens_per_step": 1.9934475806451613}
- random N=128 context=2500: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.14881150265957446, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.060339095744680854, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.01915724734042553}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.06152998677661092, "tokens_per_step": 1.2283078457446808}
- random N=192 context=2500: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.18317522321428573, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.07726469494047619, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.025576636904761904}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.08724440811160353, "tokens_per_step": 1.2860165550595237}
- random N=256 context=2500: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.1919732862903226, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.07730594758064516, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.02385752688172043}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.0880116946881503, "tokens_per_step": 1.2931367607526882}
- random N=320 context=2500: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.1877840909090909, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.07748997326203208, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.025033422459893047}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.126581145518719, "tokens_per_step": 1.290307486631016}
- random N=384 context=2500: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.20713569518716576, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.08808210784313726, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.030205548128342245}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.12743369540100652, "tokens_per_step": 1.3254233511586453}
- random N=448 context=2500: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.18605567226890757, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.07480961134453781, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.02414325105042017}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.1286168342574135, "tokens_per_step": 1.2850085346638656}
- random N=512 context=2500: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.19564800613496933, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.08249856211656442, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.027775115030674848}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.12902639638039745, "tokens_per_step": 1.3059216832822087}
- random N=128 context=3500: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.18747018129770993, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.07585877862595419, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.01854723282442748}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.06194479180537013, "tokens_per_step": 1.2818761927480917}
- random N=192 context=3500: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.15938487361769352, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.06309241706161138, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.020339652448657188}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.08710923165875217, "tokens_per_step": 1.242816943127962}
- random N=256 context=3500: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.17541026536312848, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.07563722067039107, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.026318086592178772}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.0883386254022984, "tokens_per_step": 1.2773655726256983}
- random N=320 context=3500: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.1614639945652174, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.06625339673913043, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.021161684782608697}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.1265167361956549, "tokens_per_step": 1.2488790760869566}
- random N=384 context=3500: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.17525875796178345, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.07346403927813164, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.024382961783439492}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.12747486410821954, "tokens_per_step": 1.2731057590233545}
- random N=448 context=3500: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.16930719656283566, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.06948174006444684, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.023513023630504833}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.128557260421103, "tokens_per_step": 1.2623019602577874}
- random N=512 context=3500: {"acceptance_per_position": {"vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"}": 0.18259828629032257, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"}": 0.0758820564516129, "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}": 0.02443296370967742}, "step_note": "num_drafts counts per-request draft steps summed over DP ranks; step_seconds=window*N/drafts.", "step_seconds": 0.12911489274195065, "tokens_per_step": 1.282913306451613}

### Prefill

- cold3k N=32: valid=True; new tok/s=16194.442361726182; including cached tok/s=16194.442361726182; TTFT={'mean': 5.9172107925605815, 'p50': 5.917254260995833, 'p90': 5.921828400988306}; new tok/s/chip=506.0763238039432; incl-cached tok/s/chip=506.0763238039432; repeat=0; flags=[]

### Agent-workload model

```json
{
  "assumptions": [
    "Best-case model, not a measured disaggregated deployment; valid uncontaminated N=32 cases only.",
    "Each turn emits500 tokens; logical TPM counts the full prompt including cached history plus500 output tokens, no extra double-counting of the1650 increment.",
    "Separate equal-size P and D slices: mean incremental-prefill TTFT at concurrency32 plus mean per-user500/rate; two32-chip slices. Per-slice TPM divides by2.",
    "Decode-only scenario: prefill and KV transfer are free; one32-chip slice.",
    "No network/KV-transfer, routing, tool latency or queueing beyond measured batched TTFT; fixed N=32 decode occupancy.",
    "Random contexts5k/10k/15k weighted equally for average; agent-replay reports actual lengths separately. No interpolation or substitution for missing/flagged cases.",
    "Spec num_drafts sums per-request steps over DP ranks: step_seconds=window_seconds*N/drafts; tokens_per_step=1+accepted/drafts; per-position acceptance=accepted_pos/drafts. Boundary scrape lag is recorded; these are counter-normalized estimates."
  ],
  "incremental_prefill_ttft_seconds": null,
  "contexts": [],
  "random_context_average": null
}
```
