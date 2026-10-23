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
"""uint8 exponent storage for power-of-two f32 scales (MOE_SCALE_E8M0).

A code is the f32 biased-exponent field: code c in 1..254 stands for the f32
2**(c - 127) and code 0 for +0.0 (e.g. zero-padded blocks). Decoding is a
shift and bitcast, so it reproduces the stored f32 bit pattern exactly for
every scale ``encode`` accepts (positive normal powers of two, or +0.0).
"""

import jax
import jax.numpy as jnp

_NON_E8M0_BITS = 0x807FFFFF  # sign + mantissa bits must be clear


def e8m0_encode(scale: jax.Array) -> tuple[jax.Array, jax.Array]:
    """f32 scale -> (uint8 codes, lossless flag as a 0-d bool array)."""
    bits = jax.lax.bitcast_convert_type(scale.astype(jnp.float32), jnp.uint32)
    exp = bits >> 23
    lossless = jnp.all(((bits & jnp.uint32(_NON_E8M0_BITS)) == 0)
                       & (exp != 255))
    return exp.astype(jnp.uint8), lossless


def e8m0_decode(codes: jax.Array) -> jax.Array:
    """uint8 codes -> the exact f32 scales they encode."""
    return jax.lax.bitcast_convert_type(codes.astype(jnp.uint32) << 23,
                                        jnp.float32)
