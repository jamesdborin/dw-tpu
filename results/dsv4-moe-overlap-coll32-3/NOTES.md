# dsv4-moe-overlap-coll32-3 (micro32, ~8 min, default mode only)

Part 2/3 of coll32-2, which never ran: "building params" timed out because jax.random over the
8.6G-element expert tensors was generated unpartitioned. Weights are now an integer hash of the element
index (elementwise, partitioned to the local shard). Collectives are skipped (MICRO_SKIP_COLL=1; coll32-2
measured them) and the SC-offload modes are dropped (coll32-2: SC offload makes the RS slower).

Measures, 8 DSV4-shaped decode MoE layers, 32 and 128 global rows: us per layer for base, packed,
packed+one-shot AG, packed+one-shot AG+RS, split2, split2+one-shot, split2+one-shot+shared-expert TP;
bitwise checks vs base (packed, packed_osag must be bitwise equal); split_check: E=32 hash weights in
base and split layout (exact relayout), base vs split layer output on hardware (must be finite, meanabs
at bf16-rounding level). Result decides whether R14 keeps MOE_TP_EXPERT_SPLIT=2 / one-shot flags.
Outputs: $OUT/default.json, default.log (every host), prof/ (process 0).
