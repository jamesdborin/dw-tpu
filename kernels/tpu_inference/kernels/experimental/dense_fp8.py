"""Dense FP8 W8A16 GEMM candidates with FP32 partials and accumulation.

`native` preserves FP8 RHS values as dot operands; `wide` explicitly converts
an entire tile to BF16. `pre` preserves baseline weight-scaling arithmetic.
`flipped` tests weight-as-LHS scheduling without changing input precision.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


@functools.partial(jax.jit, static_argnames=('tn', 'tk', 'tm', 'mode'))
def matmul(x, w, s, *, tn=1024, tk=1024, tm=16, mode='native'):
    m, k = x.shape
    n = w.shape[1]
    tm = min(tm, m)
    assert m % tm == 0 and k % tk == 0 and n % tn == 0 and tk % 128 == 0
    step_n = tn if mode in ('wide', 'flipped', 'native') else 256

    def kernel(x_ref, w_ref, s_ref, out_ref, acc_ref):
        k_index = pl.program_id(2)

        @pl.when(k_index == 0)
        def initialize():
            acc_ref[...] = jnp.zeros((tm, tn), jnp.float32)

        lhs = x_ref[...]
        rhs = w_ref[...]
        if mode not in ('native', 'flipped'):
            rhs = rhs.astype(jnp.bfloat16)
        scales = s_ref[...].reshape(tk // 128, tn)
        partials = []
        for start_n in range(0, tn, step_n):
            end_n = min(tn, start_n + step_n)
            acc = jnp.zeros((tm, end_n - start_n), jnp.float32)
            if mode == 'pre':
                block = rhs[:, start_n:end_n].reshape(tk // 128, 128, -1)
                block = block * scales[:, None, start_n:end_n].astype(jnp.bfloat16)
                acc = jnp.dot(lhs, block.reshape(tk, -1),
                              preferred_element_type=jnp.float32)
            else:
                for start_k in range(0, tk, 128):
                    a = lhs[:, start_k:start_k + 128]
                    b = rhs[start_k:start_k + 128, start_n:end_n]
                    if mode == 'flipped':
                        dot = jnp.dot(b.T, a.T, preferred_element_type=jnp.float32).T
                    else:
                        dot = jnp.dot(a, b, preferred_element_type=jnp.float32)
                    scale = scales[start_k // 128:start_k // 128 + 1, start_n:end_n]
                    acc = acc + dot * scale
            partials.append(acc)
        acc_ref[...] = acc_ref[...] + jnp.concatenate(partials, axis=1)

        @pl.when(k_index == k // tk - 1)
        def store():
            out_ref[...] = acc_ref[...].astype(jnp.bfloat16)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
        grid=(m // tm, n // tn, k // tk),
        in_specs=(
            pl.BlockSpec((tm, tk), lambda mi, ni, ki: (mi, ki)),
            pl.BlockSpec((tk, tn), lambda mi, ni, ki: (ki, ni)),
            pl.BlockSpec((tk // 128, 1, tn), lambda mi, ni, ki: (ki, 0, ni)),
        ),
        out_specs=pl.BlockSpec((tm, tn), lambda mi, ni, ki: (mi, ni)),
        scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=('parallel', 'parallel', 'arbitrary'),
            vmem_limit_bytes=96 * 1024 * 1024),
        name='dgemm_' + mode,
    )(x, w, s.reshape(k // 128, 1, n))


# Measured on one TPU v6e; unsupported shapes retain the original kernel.
_TUNED = {(4, 1024, 8192): ('native', 4, 1024, 2048),
 (4, 1024, 32768): ('megablox', 4, 1024, 4096),
 (4, 2048, 4096): ('native', 4, 1024, 4096),
 (4, 4096, 1536): ('native', 4, 4096, 512),
 (4, 4096, 4096): ('native', 4, 4096, 1024),
 (4, 8192, 4096): ('native', 4, 1024, 4096),
 (16, 1024, 8192): ('native', 16, 1024, 2048),
 (16, 1024, 32768): ('native', 16, 1024, 2048),
 (16, 2048, 4096): ('native', 16, 1024, 2048),
 (16, 4096, 1536): ('native', 16, 4096, 512),
 (16, 4096, 4096): ('native', 16, 1024, 4096),
 (16, 8192, 4096): ('native', 16, 8192, 512),
 (512, 1024, 8192): ('megablox', 128, 1024, 2048),
 (512, 1024, 32768): ('megablox', 128, 1024, 2048),
 (512, 2048, 4096): ('megablox', 128, 1024, 2048),
 (512, 4096, 4096): ('megablox', 128, 1024, 2048),
 (512, 8192, 4096): ('megablox', 128, 1024, 2048)}



# Full-row blocks are legal for odd M: no token-bucket rounding is required.
# Newly covered rows are candidates pending the reservation-wide kernel gate.
_DECODE_TILES = {
    (1024, 32768): (1024, 2048),
    (8192, 4096): (1024, 4096),
    (4096, 4096): (4096, 1024),
    (2048, 4096): (1024, 2048),
    (4096, 1536): (4096, 512),
    (1024, 8192): (1024, 2048),
}


_SUPPORTED_ROWS = frozenset(range(1, 33)) | {33, 64, 65, 128, 129, 256, 257, 512, 513}


def select_tile(m, k, n):
    """Exact old tiles first; extend to fine decode and speculative row counts.

    This dispatch changes no scheduler bucket, input precision, or output shape.
    The known losing M=512 fused-q_a/kv case retains the current kernel.
    """
    from tpu_inference.kernels.experimental.dense_fp8_tiles import overrides
    table = overrides()
    if (k, n) in _DECODE_TILES and (m, k, n) in table:
        return table[m, k, n]
    if (m, k, n) in _TUNED:
        return _TUNED[m, k, n]
    if (k, n) not in _DECODE_TILES or m not in _SUPPORTED_ROWS:
        return None
    if m <= 32:
        tk, tn = _DECODE_TILES[k, n]
        return 'native', m, tk, tn
    if m in (64, 128, 256, 512):
        if (k, n) == (4096, 1536):
            return None  # Preserve the efficient existing prefill path.
        return 'megablox', min(m, 128), 1024, 2048
    # Non-power-of-two verification batches (including k+1). Keep M whole
    # instead of selecting an illegal partial-row block or padding token rows.
    return 'native', m, 1024, 512

def try_dense_fp8(x, weight, scale):
    """Return a supported dense result, or None to retain the caller's fallback."""
    if x.dtype != jnp.bfloat16 or weight.dtype != jnp.float8_e4m3fn:
        return None
    if x.ndim != 2 or weight.ndim != 2 or scale.dtype != jnp.float32:
        return None
    m, k = x.shape
    n = weight.shape[1]
    entry = select_tile(m, k, n)
    if entry is None or weight.shape[0] != k:
        return None
    if scale.shape not in ((k // 128, 1, n), (1, k // 128, 1, n)):
        return None
    # The qualification is specific to v6e; do not alter other TPU paths.
    if jax.local_devices()[0].device_kind != "TPU v6 lite":
        return None
    mode, tm, tk, tn = entry
    scale = scale.reshape(k // 128, 1, n)
    if mode == "megablox":
        from tpu_inference.kernels.megablox.gmm import gmm
        return gmm(x, weight[None], jnp.array([m], jnp.int32),
                   preferred_element_type=jnp.bfloat16,
                   rhs_scale=scale[None], tiling=(tm, tk, tn))
    return matmul(x, weight, scale, tm=tm, tk=tk, tn=tn, mode=mode)
