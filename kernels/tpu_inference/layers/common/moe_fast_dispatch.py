# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MOE_FAST_DISPATCH: the MoE dispatch permutation (tokens -> expert-sorted
rows) without the slow int32 index gather, and optionally as a Pallas row
gather. Both are pure data movement: bitwise identical to

    token_indices = jnp.arange(T).repeat(topk)
    x = hidden[token_indices[argsort_indices]]

Mode 1 (``sorted_token_indices``): token_indices[j] == j // topk, so the
index composition is ``argsort_indices // topk`` (elementwise; no gather).

Mode 2 (``dispatch_gather_pallas``): additionally a TensorCore Pallas row
gather. The source is copied once into VMEM (if it fits) and viewed as uint32
row pairs (a 1-row bf16 VMEM slice is not tile aligned); for each output row
pair the kernel DMAs the two source pairs and merges the selected 16-bit
halves into one 32-bit word, so the output is written in full tiles.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_VMEM_SRC_MAX_BYTES = 64 * 2**20


def sorted_token_indices(topk_argsort_indices: jax.Array,
                         topk: int) -> jax.Array:
    """== jnp.arange(T).repeat(topk)[topk_argsort_indices] (int32)."""
    return (topk_argsort_indices // topk).astype(jnp.int32)


def _dispatch_kernel(tok_smem, tok2_ref, src_hbm, out_ref, src_vmem, a_buf,
                     b_buf, sem, load_sem, probe, *, tm: int,
                     col_chunk: int, src_in_vmem: bool):
    i = pl.program_id(0)
    n = pl.num_programs(0)
    src_u32 = src_hbm.bitcast(jnp.uint32)  # [T / 2, H] row pairs
    src = src_vmem if src_in_vmem else src_u32

    def issue(tile, slot):
        base = tile * 2 * tm

        def body(p, carry):
            r0 = tok_smem[base + 2 * p] // 2
            r1 = tok_smem[base + 2 * p + 1] // 2
            pltpu.make_async_copy(src.at[pl.ds(r0, 1)],
                                  a_buf.at[slot, pl.ds(p, 1)],
                                  sem.at[slot]).start()
            pltpu.make_async_copy(src.at[pl.ds(r1, 1)],
                                  b_buf.at[slot, pl.ds(p, 1)],
                                  sem.at[slot]).start()
            return carry

        jax.lax.fori_loop(0, tm, body, 0, unroll=8)

    @pl.when(i == 0)
    def _():
        if src_in_vmem:
            cp = pltpu.make_async_copy(src_u32, src_vmem, load_sem.at[0])
            cp.start()
            cp.wait()
        issue(0, 0)

    @pl.when(i + 1 < n)
    def _():
        issue(i + 1, (i + 1) % 2)

    slot = i % 2
    # The DMA semaphore counts bytes: these two descriptors cover exactly the
    # 2 * tm same-sized row copies issued on this slot.
    pltpu.make_async_copy(src.at[pl.ds(0, tm)], a_buf.at[slot],
                          sem.at[slot]).wait()
    pltpu.make_async_copy(src.at[pl.ds(0, tm)], b_buf.at[slot],
                          sem.at[slot]).wait()

    # Which 16-bit half of a packed word holds the even row, as defined by the
    # compiler's own bf16 <-> 32-bit memref bitcast (used for both the source
    # view and the output block below).
    probe[...] = jax.lax.broadcasted_iota(jnp.int32, probe.shape,
                                          0).astype(jnp.bfloat16)
    word = probe.bitcast(jnp.uint32)[0:1, 0:1]
    even_is_low = (word & jnp.uint32(0xFFFF)) == jnp.uint32(0)  # [1, 1]
    tok2 = tok2_ref[...]  # [tm, 2]: source rows of output rows 2p, 2p + 1
    hi_a = ((tok2[:, 0:1] & 1) == 1) == even_is_low  # [tm, 1]
    hi_b = ((tok2[:, 1:2] & 1) == 1) == even_is_low
    out_u32 = out_ref.bitcast(jnp.uint32)  # [tm, H]
    lo16 = jnp.uint32(0xFFFF)
    sh = jnp.uint32(16)

    def col_body(c, carry):
        cols = pl.ds(pl.multiple_of(c * col_chunk, col_chunk), col_chunk)
        a = a_buf[slot, :, cols]
        b = b_buf[slot, :, cols]
        ha = jnp.where(hi_a, a >> sh, a & lo16)  # output row 2p
        hb = jnp.where(hi_b, b >> sh, b & lo16)  # output row 2p + 1
        out_u32[:, cols] = jnp.where(even_is_low, ha | (hb << sh),
                                     hb | (ha << sh))
        return carry

    jax.lax.fori_loop(0, out_ref.shape[1] // col_chunk, col_body, 0)


def _pick_tm(num_rows: int, src_rows: int, tm: int) -> int:
    pairs = num_rows // 2
    while tm >= 8:
        if pairs % tm == 0 and src_rows // 2 >= tm:
            return tm
        tm //= 2
    return 0


def pallas_supported(hidden: jax.Array, num_rows: int, tm: int = 128) -> bool:
    return (hidden.ndim == 2 and hidden.dtype == jnp.bfloat16
            and hidden.shape[0] % 2 == 0 and hidden.shape[1] % 128 == 0
            and num_rows % 2 == 0
            and _pick_tm(num_rows, hidden.shape[0], tm) > 0)


@functools.partial(jax.jit, static_argnames=("tm", "interpret"))
def dispatch_gather_pallas(hidden: jax.Array,
                           token_indices_sorted: jax.Array,
                           tm: int = 128,
                           interpret: bool = False) -> jax.Array:
    """hidden[token_indices_sorted] for bf16 hidden [T, H] (bitwise).

    Caller checks ``pallas_supported`` first.
    """
    num_tokens, hidden_size = hidden.shape
    num_rows = token_indices_sorted.shape[0]
    tm = _pick_tm(num_rows, num_tokens, tm)
    tok = token_indices_sorted.astype(jnp.int32)
    src_bytes = num_tokens * hidden_size * 2
    src_in_vmem = src_bytes <= _VMEM_SRC_MAX_BYTES
    col_chunk = 512 if hidden_size % 512 == 0 else (256 if hidden_size %
                                                      256 == 0 else 128)
    need = ((src_bytes if src_in_vmem else 0) + 2 * 2 * tm * hidden_size * 4 +
            2 * 2 * tm * hidden_size * 2)
    vmem_limit = min(110 * 2**20, need + 16 * 2**20)
    src_scratch = (num_tokens // 2, hidden_size) if src_in_vmem else (8, 128)
    kernel = functools.partial(_dispatch_kernel,
                               tm=tm,
                               col_chunk=col_chunk,
                               src_in_vmem=src_in_vmem)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_rows, hidden_size), hidden.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_rows // (2 * tm), ),
            in_specs=[
                pl.BlockSpec((tm, 2), lambda i, t: (i, 0)),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            out_specs=pl.BlockSpec((2 * tm, hidden_size), lambda i, t:
                                   (i, 0)),
            scratch_shapes=[
                pltpu.VMEM(src_scratch, jnp.uint32),
                pltpu.VMEM((2, tm, hidden_size), jnp.uint32),
                pltpu.VMEM((2, tm, hidden_size), jnp.uint32),
                pltpu.SemaphoreType.DMA((2, )),
                pltpu.SemaphoreType.DMA((1, )),
                pltpu.VMEM((16, 128), jnp.bfloat16),
            ]),
        compiler_params=pltpu.CompilerParams(
            # Cross-step prefetch and the step-0 source load: in order, one
            # core.
            dimension_semantics=("arbitrary", ),
            vmem_limit_bytes=vmem_limit),
        interpret=interpret,
    )(tok, tok.reshape(num_rows // 2, 2), hidden)


def fast_dispatch_gather(hidden: jax.Array, topk_argsort_indices: jax.Array,
                         topk: int, mode: int) -> jax.Array:
    """MOE_FAST_DISPATCH: hidden[arange(T).repeat(topk)[argsort]], bitwise."""
    tok = sorted_token_indices(topk_argsort_indices, topk)
    if mode >= 2 and pallas_supported(hidden, tok.shape[0]):
        return dispatch_gather_pallas(hidden, tok)
    return hidden[tok]
