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
"""MOE_FAST_COMBINE: weighted top-k gather-reduce ("combine") of the MoE
down-projection output without the f32 [tokens, topk, hidden] intermediate.

    out[t] = round(sum_k f32(x[idx[t * topk + k]]) * f32(w[t, k]))

This is the math of ``dense_gather_reduce``'s JAX fallback (f32 products, f32
sum over the topk experts, one rounding to ``out_dtype``). The topk sum is
added left to right (k = 0, 1, ..., topk - 1, starting from +0.0).

Mode 1 (``combine_xla``): one gather + convert + multiply-add per k, so XLA
never materializes the (sublane-padded) f32 [T, topk, H] tensor.

Mode 2 (``combine_pallas``): a TensorCore Pallas kernel. Per token tile it
DMAs the tile's topk * tm source rows from HBM into VMEM (double buffered
across grid steps), then accumulates the weighted rows in f32 and writes the
tile once. bf16 rows cannot be DMA'd one at a time (a 1-row slice of a packed
bf16 VMEM buffer is not tile aligned), so a bf16 source is viewed as uint32
row pairs ([N / 2, H]; each 32-bit word packs rows 2i and 2i + 1 of one
column) and the kernel DMAs the pair holding the row and picks the half. The
bf16 -> f32 conversion is exact (bf16 bits << 16).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def combine_xla(x: jax.Array, indices: jax.Array, topk_weights: jax.Array,
                topk: int, out_dtype=None) -> jax.Array:
    """Mode 1. x [N, H]; indices [T * topk]; topk_weights [T, topk]."""
    out_dtype = x.dtype if out_dtype is None else out_dtype
    num_tokens = topk_weights.shape[0]
    idx = indices.reshape(num_tokens, topk)
    w = topk_weights.astype(jnp.float32)
    acc = jnp.zeros((num_tokens, x.shape[-1]), jnp.float32)
    for k in range(topk):
        acc = acc + x[idx[:, k]].astype(jnp.float32) * w[:, k:k + 1]
    return acc.astype(out_dtype)


def _combine_kernel(idx_smem, idxv_ref, w_ref, x_hbm, out_ref, buf, sem,
                    probe, *, topk: int, tm: int, col_chunk: int,
                    pair: bool):
    i = pl.program_id(0)
    n = pl.num_programs(0)
    rows = topk * tm
    src = x_hbm.bitcast(jnp.uint32) if pair else x_hbm

    def issue(tile, slot):
        base = tile * rows

        def body(j, carry):
            # Token t = j // topk, expert slot k = j % topk lands at row
            # k * tm + t of the buffer, so each k is a contiguous [tm, H].
            r = idx_smem[base + j]
            if pair:
                r = r // 2
            dst = (j % topk) * tm + j // topk
            pltpu.make_async_copy(src.at[pl.ds(r, 1)],
                                  buf.at[slot, pl.ds(dst, 1)],
                                  sem.at[slot]).start()
            return carry

        jax.lax.fori_loop(0, rows, body, 0, unroll=8)

    @pl.when(i == 0)
    def _():
        issue(0, 0)

    @pl.when(i + 1 < n)
    def _():
        issue(i + 1, (i + 1) % 2)

    slot = i % 2
    # One wait for all `rows` same-sized row copies of this slot: the DMA
    # semaphore counts bytes, and this descriptor covers exactly their sum.
    pltpu.make_async_copy(src.at[pl.ds(0, rows)], buf.at[slot],
                          sem.at[slot]).wait()

    w = w_ref[...].astype(jnp.float32)  # [tm, topk]
    if pair:
        # Which 16-bit half of a packed word holds the even row, as defined by
        # this compiler's own bf16 <-> 32-bit memref bitcast (the same op that
        # views x as row pairs): write rows 0..15 as bf16(0..15) and look.
        probe[...] = jax.lax.broadcasted_iota(jnp.int32, probe.shape,
                                              0).astype(jnp.bfloat16)
        word = probe.bitcast(jnp.uint32)[0:1, 0:1]
        even_is_low = (word & jnp.uint32(0xFFFF)) == jnp.uint32(0)  # [1, 1]
        odd = idxv_ref[...] & 1  # [tm, topk]
        take_hi = (odd == 1) == even_is_low  # [tm, topk]

    def col_body(c, carry):
        cols = pl.ds(pl.multiple_of(c * col_chunk, col_chunk), col_chunk)
        acc = jnp.zeros((tm, col_chunk), jnp.float32)
        for k in range(topk):
            v = buf[slot, pl.ds(k * tm, tm), cols]
            if pair:
                bits = jnp.where(take_hi[:, k:k + 1],
                                 v & jnp.uint32(0xFFFF0000),
                                 v << jnp.uint32(16))
                v = jax.lax.bitcast_convert_type(bits, jnp.float32)
            acc = acc + v.astype(jnp.float32) * w[:, k:k + 1]
        out_ref[:, cols] = acc.astype(out_ref.dtype)
        return carry

    jax.lax.fori_loop(0, out_ref.shape[1] // col_chunk, col_body, 0)


def pallas_supported(x: jax.Array, num_tokens: int, topk: int,
                     tm: int = 64) -> bool:
    if x.ndim != 2 or x.dtype not in (jnp.bfloat16, jnp.float32):
        return False
    if x.shape[1] % 128:
        return False
    tm = min(tm, max(8, -(-num_tokens // 8) * 8))
    view_rows = x.shape[0] // 2 if x.dtype == jnp.bfloat16 else x.shape[0]
    if x.dtype == jnp.bfloat16 and x.shape[0] % 2:
        return False
    return view_rows >= topk * tm


@functools.partial(jax.jit,
                   static_argnames=("topk", "out_dtype", "tm", "interpret"))
def combine_pallas(x: jax.Array,
                   indices: jax.Array,
                   topk_weights: jax.Array,
                   topk: int,
                   out_dtype=None,
                   tm: int = 64,
                   interpret: bool = False) -> jax.Array:
    """Mode 2. x [N, H] bf16 or f32; indices [T * topk]; topk_weights [T, topk].

    Caller checks ``pallas_supported`` first.
    """
    out_dtype = x.dtype if out_dtype is None else jnp.dtype(out_dtype)
    num_tokens = topk_weights.shape[0]
    hidden = x.shape[1]
    tm = min(tm, max(8, -(-num_tokens // 8) * 8))
    padded = -(-num_tokens // tm) * tm
    idx2 = indices.reshape(num_tokens, topk).astype(jnp.int32)
    w2 = topk_weights.astype(jnp.float32)
    if padded != num_tokens:
        # Padding tokens gather row 0 with weight 0 and are sliced off.
        idx2 = jnp.pad(idx2, ((0, padded - num_tokens), (0, 0)))
        w2 = jnp.pad(w2, ((0, padded - num_tokens), (0, 0)))
    pair = x.dtype == jnp.bfloat16
    buf_dtype = jnp.uint32 if pair else x.dtype
    col_chunk = 512 if hidden % 512 == 0 else (256 if hidden %
                                                 256 == 0 else 128)
    buf_bytes = 2 * topk * tm * hidden * 4
    out_bytes = 2 * tm * hidden * jnp.dtype(out_dtype).itemsize
    vmem_limit = min(100 * 2**20, buf_bytes + out_bytes + 16 * 2**20)

    kernel = functools.partial(_combine_kernel,
                               topk=topk,
                               tm=tm,
                               col_chunk=col_chunk,
                               pair=pair)
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((padded, hidden), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(padded // tm, ),
            in_specs=[
                pl.BlockSpec((tm, topk), lambda i, idx: (i, 0)),
                pl.BlockSpec((tm, topk), lambda i, idx: (i, 0)),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            out_specs=pl.BlockSpec((tm, hidden), lambda i, idx: (i, 0)),
            scratch_shapes=[
                pltpu.VMEM((2, topk * tm, hidden), buf_dtype),
                pltpu.SemaphoreType.DMA((2, )),
                pltpu.VMEM((16, 128), jnp.bfloat16),
            ]),
        compiler_params=pltpu.CompilerParams(
            # Cross-step prefetch: steps must run in order on one core.
            dimension_semantics=("arbitrary", ),
            vmem_limit_bytes=vmem_limit),
        interpret=interpret,
    )(idx2.reshape(-1), idx2, w2, x)
    return out[:num_tokens] if padded != num_tokens else out


def fast_combine(x: jax.Array, indices: jax.Array, topk_weights: jax.Array,
                 topk: int, mode: int, out_dtype=None) -> jax.Array:
    """Dispatch on MOE_FAST_COMBINE mode (1: XLA, 2: Pallas with XLA
    fallback when the shapes do not fit the kernel)."""
    if mode >= 2 and pallas_supported(x, topk_weights.shape[0], topk):
        return combine_pallas(x, indices, topk_weights, topk, out_dtype)
    return combine_xla(x, indices, topk_weights, topk, out_dtype)
