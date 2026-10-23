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
"""MTP proposer for DeepSeek V4 (vLLM/torchax model implementation).

Reuses the Eagle3Proposer algorithm (method ``mtp``) with three DSV4-specific
differences:

* DSV4 attention reads per-layer metadata from a ``GroupedAttentionMetadata``
  (one entry per KV-cache group). The MTP layer's sliding-window cache lives in
  whichever group vLLM put it in, not necessarily the last one, so the draft
  metadata is derived from that group's entry of the target metadata. That
  entry's block table is already in the attention-DP rank order the kernels
  expect (the input-batch CPU table is not).
* The draft conditions on the target's pre-hc_head residual
  ``[T, hc_mult * hidden]``, which the DSV4 target returns as its aux hidden
  state.
* The draft has no embedding / LM head of its own; both are shared with the
  target.
"""
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from tpu_inference.layers.common.attention_metadata import (
    AttentionMetadata, GroupedAttentionMetadata)
from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import get_model
from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer

logger = init_logger(__name__)

TARGET_EMBED = "vllm_model.model.embed_tokens.weight"
TARGET_LM_HEAD = "vllm_model.lm_head.weight"


def mtp_layer_prefixes(hf_config) -> tuple[str, ...]:
    start = hf_config.num_hidden_layers
    n = getattr(hf_config, "num_nextn_predict_layers", 1) or 1
    return tuple(f"model.layers.{start + i}." for i in range(n))


def find_draft_kv_cache_group(kv_cache_groups,
                              hf_config) -> tuple[int, tuple[str, ...]]:
    """Returns (group id, MTP layer names) of the KV-cache group holding the
    MTP layers. All MTP caches must share one group (they are all SWA)."""
    prefixes = mtp_layer_prefixes(hf_config)
    found: dict[int, list[str]] = {}
    for gid, group in enumerate(kv_cache_groups):
        for name in group.layer_names:
            if name.startswith(prefixes):
                found.setdefault(gid, []).append(name)
    if len(found) != 1:
        raise ValueError(
            "DeepSeek V4 MTP: expected the MTP layers' KV caches in exactly "
            f"one KV-cache group, found {found} (prefixes {prefixes})")
    gid, names = next(iter(found.items()))
    return gid, tuple(names)


class DeepseekV4MtpProposer(Eagle3Proposer):

    # SpeculativeDecodingManager passes the full GroupedAttentionMetadata.
    uses_grouped_attention_metadata = True

    def __init__(self, vllm_config, runner: Any):
        super().__init__(vllm_config, runner)
        assert self.method == "mtp"
        self._draft_gid: int | None = None
        self._draft_layer_names: tuple[str, ...] | None = None

    # ------------------------------------------------------------------ load
    def load_model(self, target_model: Any) -> None:
        state = self.runner.state
        missing = [k for k in (TARGET_EMBED, TARGET_LM_HEAD) if k not in state]
        if missing:
            raise ValueError(
                f"DeepSeek V4 MTP: target params {missing} not found")
        shared_params = {"vllm_model.model.embed_tokens.weight": state[TARGET_EMBED]}
        for prefix in mtp_layer_prefixes(
                self.draft_model_config.hf_config):
            shared_params[
                f"vllm_model.{prefix}shared_head.head.weight"] = state[
                    TARGET_LM_HEAD]

        model = get_model(
            self.vllm_config,
            self.rng_key,
            self.mesh,
            is_draft_model=True,
            shared_params=shared_params,
        )
        self.model_fn = model.model_fn
        self.compute_logits_fn = model.compute_logits_fn
        self.pooler_fn = model.pooler_fn
        self.combine_hidden_states_fn = model.combine_hidden_states_fn
        self.state = model.state
        self.state_leaves = model.state_leaves
        self.model = model.model
        if isinstance(self.state, nnx.State):
            self.state_leaves = tuple(jax.tree_util.tree_leaves(self.state))
        else:
            self.state_leaves = self.state

    # ------------------------------------------------------------ metadata
    def _resolve_draft_group(self) -> None:
        if self._draft_gid is None:
            self._draft_gid, self._draft_layer_names = (
                find_draft_kv_cache_group(
                    self.runner.kv_cache_config.kv_cache_groups,
                    self.draft_model_config.hf_config))
            logger.info("DeepSeek V4 MTP: draft KV-cache group %d, layers %s",
                        self._draft_gid, self._draft_layer_names)

    def _group(self, attn_metadata: AttentionMetadata
               ) -> GroupedAttentionMetadata:
        return GroupedAttentionMetadata(
            groups=(attn_metadata, ),
            layer_names_per_group=(self._draft_layer_names, ))

    def prepare_inputs(
        self,
        attn_metadata: GroupedAttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        last_sampled_token_id: jax.Array,
        next_prompt_token_id: jax.Array,
        is_in_prefill: jax.Array,
        num_rejected_tokens: jax.Array,
        num_reqs_dp: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, AttentionMetadata]:
        assert aux_hidden_states is not None and len(aux_hidden_states) > 0, (
            "DeepSeek V4 MTP needs the target's pre-hc_head hidden states")
        self._resolve_draft_group()
        if not isinstance(attn_metadata, GroupedAttentionMetadata):
            raise TypeError(
                "DeepSeek V4 MTP needs the per-group target metadata, got "
                f"{type(attn_metadata).__name__}")
        base = attn_metadata.groups[self._draft_gid]
        return self._prepare_inputs(
            state_leaves=self.state_leaves,
            num_reqs=num_reqs_dp,
            block_tables=base.block_tables,
            attn_metadata=base,
            input_ids=input_ids,
            aux_hidden_states=tuple(aux_hidden_states),
            last_sampled_token_id=last_sampled_token_id,
            next_prompt_token_id=next_prompt_token_id,
            is_in_prefill=is_in_prefill,
            num_rejected_tokens=num_rejected_tokens)

    # -------------------------------------------------------------- propose
    def propose(
        self,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,
        attn_metadata: AttentionMetadata,
        last_token_indices,
        target_hidden_states,
    ) -> tuple[list[jax.Array], jnp.ndarray]:
        self._resolve_draft_group()
        return self._propose_dsv4(
            state_leaves=self.state_leaves,
            kv_caches=kv_caches,
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            last_token_indices=last_token_indices,
            target_hidden_states=target_hidden_states,
            num_speculative_tokens=self.num_speculative_tokens,
            layer_name_to_kvcache_index=tuple(
                self.runner.layer_name_to_kvcache_index.items()))

    @jax.jit(
        donate_argnames=("kv_caches", ),
        out_shardings=(None, None),
        compiler_options={
            "xla_tpu_all_gather_collective_matmul_mode":
            "post_spmd_conservative",
            "xla_tpu_reduce_scatter_collective_matmul_mode":
            "post_spmd_conservative"
        },
        static_argnums=(0, 7, 8),
    )
    def _propose_dsv4(
        self,
        state_leaves: Any,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,
        attn_metadata: AttentionMetadata,
        last_token_indices,
        target_hidden_states,
        num_speculative_tokens: int,
        layer_name_to_kvcache_index: tuple,
    ) -> tuple[list[jax.Array], jnp.ndarray]:
        """Eagle3Proposer._propose with DSV4 grouped draft metadata.

        Step 0 runs the MTP layer over the accepted target tokens (writing its
        SWA cache at their positions) and drafts from the last one; steps
        1..k-1 feed back one draft token per request at position + 1.
        """
        kv_caches, hidden_states, residual, _ = self.model_fn(
            state_leaves,
            kv_caches,
            input_ids,
            target_hidden_states,
            self._group(attn_metadata),
            layer_name_to_kvcache_index,
            spec_step_idx=0,
        )
        if num_speculative_tokens == 1:
            return kv_caches, self._select_draft_token_ids(
                state_leaves, hidden_states, last_token_indices)

        positions, hidden_states, draft_token_ids = (
            self._select_inputs_for_loop_speculation(
                state_leaves, attn_metadata.input_positions, residual[0],
                hidden_states, last_token_indices))
        draft_token_ids_list = [draft_token_ids]
        for i in range(num_speculative_tokens - 1):
            (positions, clamped_positions, new_seq_lens, query_start_loc,
             new_block_tables) = self._update_inputs_for_loop_speculation(
                 positions, attn_metadata.seq_lens, attn_metadata.block_tables)
            attn_metadata = replace(
                attn_metadata,
                input_positions=clamped_positions,
                seq_lens=new_seq_lens,
                query_start_loc=query_start_loc,
                block_tables=new_block_tables,
            )
            kv_caches, new_hidden_states, residual, _ = self.model_fn(
                state_leaves,
                kv_caches,
                draft_token_ids_list[-1],
                hidden_states,
                self._group(attn_metadata),
                layer_name_to_kvcache_index,
                spec_step_idx=i + 1,
            )
            hidden_states = residual[0]
            draft_token_ids_list.append(
                self._get_draft_token_ids(state_leaves, new_hidden_states))
        return kv_caches, self._stack_draft_token_ids(draft_token_ids_list)
