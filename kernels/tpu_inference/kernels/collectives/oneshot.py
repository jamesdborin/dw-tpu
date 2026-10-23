# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One-shot all-gather and reduce-scatter for small (decode-sized) arrays.

XLA's TPU collectives step around a ring, so a 32-way all-gather of a few KB
costs ~31 hop latencies (~30-45 us measured on v6e 4x8) although the bytes
would cross ICI in a few us. Here every device writes its block straight into
every peer with one remote DMA each (ICI routes multi-hop), after one barrier.

Must be called inside a shard_map whose mesh has every axis manual; ``axes``
are the mesh axes the collective runs over (first axis major, like
jax.lax.all_gather over a tuple of axes). All other mesh axes keep their
own coordinate.

* all_gather_local: pure data movement, bit-identical to jax.lax.all_gather
  (tiled, axis 0).
* reduce_scatter_local: each device receives every peer's block for its rows
  and sums the n blocks in f32 in fixed rank order 0..n-1, then casts. Same
  result as psum_scatter up to summation order (and with f32 accumulation).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Interpret-mode params for CPU tests (None = compile for TPU).
INTERPRET = None


def _axes_info(axes: tuple[str, ...]):
    sizes = tuple(lax.axis_size(a) for a in axes)
    n = 1
    for s in sizes:
        n *= s
    return sizes, n


def _my_rank(axes, sizes):
    r = 0
    for a, s in zip(axes, sizes):
        r = r * s + lax.axis_index(a)
    return r


def _device_id(rank, axes, sizes, mesh_axes):
    """MESH device id of the peer at linear ``rank`` along ``axes``."""
    coord = {}
    for a, s in reversed(list(zip(axes, sizes))):
        coord[a] = rank % s
        rank = rank // s
    return tuple(coord[a] if a in coord else lax.axis_index(a)
                 for a in mesh_axes)


def _barrier(n, me, dev):
    sem = pltpu.get_barrier_semaphore()
    for k in range(n):
        pl.semaphore_signal(sem,
                            1,
                            device_id=dev(lax.rem(me + k, n)),
                            device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(sem, n)


def _push_all(n, me, dev, src_of, dst_of, local_sem, send_sem, recv_sem):
    """Copy src_of(t) into dst_of(me) on every device t (self by local DMA)."""
    local = pltpu.make_async_copy(src_of(me), dst_of(me), local_sem)
    local.start()
    copies = []
    # Rotate targets so devices do not all hit the same peer first.
    for k in range(1, n):
        t = lax.rem(me + k, n)
        c = pltpu.make_async_remote_copy(src_ref=src_of(t),
                                         dst_ref=dst_of(me),
                                         send_sem=send_sem,
                                         recv_sem=recv_sem,
                                         device_id=dev(t),
                                         device_id_type=pl.DeviceIdType.MESH)
        c.start()
        copies.append(c)
    local.wait()
    for c in copies:
        c.wait_send()
    # Every incoming copy has the same size as our outgoing ones.
    for c in copies:
        c.wait_recv()


def _ag_kernel(x_ref, o_ref, local_sem, send_sem, recv_sem, *, axes, sizes, n,
               rows, mesh_axes):
    me = _my_rank(axes, sizes)
    dev = functools.partial(_device_id,
                            axes=axes,
                            sizes=sizes,
                            mesh_axes=mesh_axes)
    _barrier(n, me, dev)
    _push_all(n, me, dev, lambda t: x_ref, lambda s: o_ref.at[s], local_sem,
              send_sem, recv_sem)


def _rs_kernel(x_ref, o_ref, local_sem, send_sem, recv_sem, *, axes, sizes, n,
               rows, mesh_axes):
    me = _my_rank(axes, sizes)
    dev = functools.partial(_device_id,
                            axes=axes,
                            sizes=sizes,
                            mesh_axes=mesh_axes)
    _barrier(n, me, dev)
    # Block t of our partial sums goes to device t, into its slot `me`.
    _push_all(n, me, dev, lambda t: x_ref.at[t], lambda s: o_ref.at[s],
              local_sem, send_sem, recv_sem)


def _call(kernel, x, out_shape, axes, mesh_axes, rows, n, sizes, name):
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    kw = {}
    if INTERPRET is not None:
        kw["interpret"] = INTERPRET
    return pl.pallas_call(
        functools.partial(kernel,
                          axes=axes,
                          sizes=sizes,
                          n=n,
                          rows=rows,
                          mesh_axes=mesh_axes),
        out_shape=out_shape,
        in_specs=[hbm],
        out_specs=hbm,
        scratch_shapes=[pltpu.SemaphoreType.DMA] * 3,
        compiler_params=pltpu.CompilerParams(collective_id=0,
                                             has_side_effects=True),
        name=name,
        **kw,
    )(x)


def all_gather_local(x: jax.Array, axes: tuple[str, ...],
                     mesh_axes: tuple[str, ...]) -> jax.Array:
    """[rows, ...] per device -> [n * rows, ...], block i from rank i."""
    sizes, n = _axes_info(axes)
    rows = x.shape[0]
    if n == 1:
        return x
    # Peer blocks live on a leading untiled axis: HBM row slices must be
    # tile-aligned, and rows is usually 1 or 4.
    out = jax.ShapeDtypeStruct((n, ) + x.shape, x.dtype)
    out = _call(_ag_kernel, x, out, axes, mesh_axes, rows, n, sizes,
                "oneshot_all_gather")
    return out.reshape((n * rows, ) + x.shape[1:])


def reduce_scatter_local(x: jax.Array,
                         axes: tuple[str, ...],
                         mesh_axes: tuple[str, ...],
                         out_dtype=None) -> jax.Array:
    """[n * rows, ...] partial sums -> [rows, ...] summed block of this rank."""
    sizes, n = _axes_info(axes)
    out_dtype = out_dtype or x.dtype
    if n == 1:
        return x.astype(out_dtype)
    assert x.shape[0] % n == 0, (x.shape, n)
    rows = x.shape[0] // n
    x = x.reshape((n, rows) + x.shape[1:])
    out = jax.ShapeDtypeStruct(x.shape, x.dtype)
    recv = _call(_rs_kernel, x, out, axes, mesh_axes, rows, n, sizes,
                 "oneshot_reduce_scatter")
    acc = recv[0].astype(jnp.float32)
    for i in range(1, n):
        acc = acc + recv[i].astype(jnp.float32)
    return acc.astype(out_dtype)
