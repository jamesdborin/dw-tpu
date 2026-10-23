# pd-final-prefill

Status: passed
Acceptance passed: True

- startup_seconds: 756.8411450386047
- compile_wall_span_seconds: 718.0
- token_buckets_per_dp: [4, 8, 16, 512]
- kv_blocks: [33376, 33382]
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

### Speculative decode counters


### Prefill

- cold5k N=32: valid=True; new tok/s=19317.198525993914; including cached tok/s=19317.198525993914; TTFT={'mean': 8.265807174564543, 'p50': 8.265157423003984, 'p90': 8.268943003994355}; new tok/s/chip=603.6624539373098; incl-cached tok/s/chip=603.6624539373098; repeat=0; flags=[]
- cold5k N=32: valid=True; new tok/s=19357.396985590058; including cached tok/s=19357.396985590058; TTFT={'mean': 8.250457948594885, 'p50': 8.250534263497684, 'p90': 8.251748466995195}; new tok/s/chip=604.9186557996893; incl-cached tok/s/chip=604.9186557996893; repeat=1; flags=[]
- cold5k N=64: valid=True; new tok/s=19570.097325224622; including cached tok/s=19570.097325224622; TTFT={'mean': 12.317225627720063, 'p50': 12.316033736002282, 'p90': 16.334921618002408}; new tok/s/chip=611.5655414132694; incl-cached tok/s/chip=611.5655414132694; repeat=0; flags=[]
- cold5k N=64: valid=True; new tok/s=19540.021103299103; including cached tok/s=19540.021103299103; TTFT={'mean': 12.344296236329228, 'p50': 12.345836505497573, 'p90': 16.354156736993172}; new tok/s/chip=610.625659478097; incl-cached tok/s/chip=610.625659478097; repeat=1; flags=[]
- cold5k N=128: valid=True; new tok/s=18902.96577683695; including cached tok/s=18902.96577683695; TTFT={'mean': 21.624404721327892, 'p50': 20.9562530395051, 'p90': 33.777569162998404}; new tok/s/chip=590.7176805261547; incl-cached tok/s/chip=590.7176805261547; repeat=0; flags=[]
- cold5k N=128: valid=True; new tok/s=18903.452009270615; including cached tok/s=18903.452009270615; TTFT={'mean': 21.73041531586796, 'p50': 20.94762822499615, 'p90': 33.793104559693894}; new tok/s/chip=590.7328752897067; incl-cached tok/s/chip=590.7328752897067; repeat=1; flags=[]
- incr1650_on5k N=32: valid=True; new tok/s=12205.951578556165; including cached tok/s=49193.68363478697; TTFT={'mean': 4.305009835469264, 'p50': 4.305569364500116, 'p90': 4.308239824994234}; new tok/s/chip=381.43598682988016; incl-cached tok/s/chip=1537.3026135870928; repeat=0; flags=[]
- incr1650_on5k N=32: valid=True; new tok/s=12234.954278139117; including cached tok/s=49310.57330280311; TTFT={'mean': 4.2959805021869215, 'p50': 4.295598785000038, 'p90': 4.303064334993541}; new tok/s/chip=382.3423211918474; incl-cached tok/s/chip=1540.9554157125972; repeat=1; flags=[]
- incr1650_on5k N=64: valid=True; new tok/s=10740.515068805305; including cached tok/s=43287.53042882138; TTFT={'mean': 7.737011001344399, 'p50': 6.93720815400593, 'p90': 9.806278731001658}; new tok/s/chip=335.6410959001658; incl-cached tok/s/chip=1352.7353259006682; repeat=0; flags=[]
- incr1650_on5k N=64: valid=True; new tok/s=10757.005996884453; including cached tok/s=43353.99386623128; TTFT={'mean': 7.723054098000603, 'p50': 6.926889143498556, 'p90': 9.792639580296235}; new tok/s/chip=336.15643740263914; incl-cached tok/s/chip=1354.8123083197274; repeat=1; flags=[]
- incr1650_on5k N=128: valid=True; new tok/s=11839.643045814948; including cached tok/s=47717.34924525419; TTFT={'mean': 11.704532614500977, 'p50': 10.919041252491297, 'p90': 17.75253557900287}; new tok/s/chip=369.9888451817171; incl-cached tok/s/chip=1491.1671639141935; repeat=0; flags=[]
- incr1650_on5k N=128: valid=True; new tok/s=11818.502940782131; including cached tok/s=47632.1482158795; TTFT={'mean': 11.678486999539473, 'p50': 10.941616586998862, 'p90': 17.782232600005226}; new tok/s/chip=369.3282168994416; incl-cached tok/s/chip=1488.5046317462343; repeat=1; flags=[]
- incr1650_on10k N=32: valid=True; new tok/s=11754.63061339679; including cached tok/s=82994.8161491349; TTFT={'mean': 4.456142665811967, 'p50': 4.455418128003657, 'p90': 4.466503132005164}; new tok/s/chip=367.3322066686497; incl-cached tok/s/chip=2593.5880046604657; repeat=0; flags=[]
- incr1650_on10k N=32: valid=True; new tok/s=11759.865424273436; including cached tok/s=83031.77708653668; TTFT={'mean': 4.456224583314452, 'p50': 4.456058278003184, 'p90': 4.461540680994221}; new tok/s/chip=367.49579450854486; incl-cached tok/s/chip=2594.743033954271; repeat=1; flags=[]
- incr1650_on10k N=64: valid=True; new tok/s=10623.260652452327; including cached tok/s=75006.6585461028; TTFT={'mean': 7.768048216937586, 'p50': 6.992800173495198, 'p90': 9.90093392599665}; new tok/s/chip=331.9768953891352; incl-cached tok/s/chip=2343.9580795657125; repeat=0; flags=[]
- incr1650_on10k N=64: valid=True; new tok/s=10652.5570062856; including cached tok/s=75213.50855953166; TTFT={'mean': 7.740671401171994, 'p50': 6.974895933002699, 'p90': 9.862342369001999}; new tok/s/chip=332.892406446425; incl-cached tok/s/chip=2350.4221424853645; repeat=1; flags=[]
- incr1650_on10k N=128: valid=True; new tok/s=11740.611866956322; including cached tok/s=82895.83530305524; TTFT={'mean': 11.808809261530541, 'p50': 10.996309666494199, 'p90': 17.83984125499701}; new tok/s/chip=366.89412084238506; incl-cached tok/s/chip=2590.494853220476; repeat=0; flags=[]
- incr1650_on10k N=128: valid=True; new tok/s=12253.097497879338; including cached tok/s=86514.29445472381; TTFT={'mean': 11.590670424804557, 'p50': 10.987646301509812, 'p90': 17.092128136000248}; new tok/s/chip=382.9092968087293; incl-cached tok/s/chip=2703.571701710119; repeat=1; flags=[]
- incr1650_on14k N=32: valid=True; new tok/s=8829.861310623019; including cached tok/s=83749.896673485; TTFT={'mean': 5.876869571249699, 'p50': 5.9333069604908815, 'p90': 5.944833817992185}; new tok/s/chip=275.93316595696933; incl-cached tok/s/chip=2617.184271046406; repeat=0; flags=[]
- incr1650_on14k N=32: valid=True; new tok/s=11341.810908079318; including cached tok/s=107575.35800693416; TTFT={'mean': 4.607929232593506, 'p50': 4.605514830996981, 'p90': 4.626303898105107}; new tok/s/chip=354.4315908774787; incl-cached tok/s/chip=3361.7299377166923; repeat=1; flags=[]
- incr1650_on14k N=64: valid=True; new tok/s=10550.95994489932; including cached tok/s=100074.25644707537; TTFT={'mean': 7.878344005172494, 'p50': 7.027256062996457, 'p90': 9.93464820800582}; new tok/s/chip=329.71749827810373; incl-cached tok/s/chip=3127.3205139711054; repeat=0; flags=[]
- incr1650_on14k N=64: valid=True; new tok/s=10528.215421862024; including cached tok/s=99858.52809220647; TTFT={'mean': 7.843051004828567, 'p50': 7.039378553992719, 'p90': 9.972264397998515}; new tok/s/chip=329.00673193318823; incl-cached tok/s/chip=3120.579002881452; repeat=1; flags=[]
- incr1650_on14k N=128: valid=True; new tok/s=12236.910556321915; including cached tok/s=116065.24254935634; TTFT={'mean': 11.446687294007688, 'p50': 10.65783795400057, 'p90': 17.08812752200756}; new tok/s/chip=382.40345488505983; incl-cached tok/s/chip=3627.0388296673855; repeat=0; flags=[]
- incr1650_on14k N=128: valid=True; new tok/s=12229.308516981995; including cached tok/s=115993.13835804135; TTFT={'mean': 11.296597291101989, 'p50': 10.617885925494193, 'p90': 17.106374805991074}; new tok/s/chip=382.16589115568735; incl-cached tok/s/chip=3624.7855736887923; repeat=1; flags=[]

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
  "incremental_prefill_ttft_seconds": 4.305009835469264,
  "contexts": [],
  "random_context_average": null
}
```
