# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU (torchax) MTP draft model for DeepSeek V4.

Mirrors ``vllm.models.deepseek_v4.amd.mtp`` but builds the decoder block from
the TPU ``DeepseekV4DecoderLayer`` (TPU MLA attention, TPU MoE, Pallas mHC)
and replaces the triton fused MTP-input norm with its eager definition:

    inputs_embeds = where(positions == 0, 0, inputs_embeds)
    e = enorm(inputs_embeds)                      # [T, H]
    h = hnorm(previous_hidden_states)             # [T, hc_mult, H], per stream
    x = h_proj(h) + e_proj(e)[:, None]            # [T, hc_mult, H]
    x = mtp_block(x)                              # one V4 decoder layer
    return x.flatten(1)                           # pre-hc_head residual

``compute_logits`` applies the MTP layer's own ``hc_head`` + ``norm`` and the
LM head (shared with the target). The MTP layer index is
``num_hidden_layers`` (43), whose ``compress_ratios`` entry is 0, so its
attention is sliding-window only (no compressor / indexer).
"""
import typing
from collections.abc import Callable, Iterable

import regex as re
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.fused_moe import \
    fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import HCHeadOp, MHCPostOp
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix

from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.experimental.deepseek_v4 import \
    DeepseekV4DecoderLayer

logger = init_logger(__name__)

# fp4 experts (Mxfp4MoEMethod) register ``w{1,2,3}_weight_scale``; every other
# FP8 block scale registers ``weight_scale_inv`` (see
# ``_make_deepseek_v4_weights_mapper`` in deepseek_v4.py).
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")

# Checkpoint names that belong to the MTP wrapper rather than to its decoder
# block (``mtp_block``).
_MTP_WRAPPER_WEIGHTS = ("enorm.", "hnorm.", "h_proj.", "e_proj.",
                        "shared_head.", "hc_head_fn", "hc_head_base",
                        "hc_head_scale")


class SharedHead(nn.Module):

    def __init__(self, config, prefix: str) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(config.vocab_size,
                                   config.hidden_size,
                                   prefix=maybe_prefix(prefix, "head"))


class DeepSeekV4MultiTokenPredictorLayer(nn.Module):

    def __init__(self, vllm_config: VllmConfig,
                 topk_indices_buffer: torch.Tensor, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.e_proj = ReplicatedLinear(config.hidden_size,
                                       config.hidden_size,
                                       bias=False,
                                       return_bias=False,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.e_proj")
        self.h_proj = ReplicatedLinear(config.hidden_size,
                                       config.hidden_size,
                                       bias=False,
                                       return_bias=False,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.h_proj")
        self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult,
                                                   self.hc_dim,
                                                   dtype=torch.float32),
                                       requires_grad=False)
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult,
                                                     dtype=torch.float32),
                                         requires_grad=False)
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32),
                                          requires_grad=False)
        self.shared_head = SharedHead(config, prefix)
        # Same prefix as the GPU MTP: the block's attention registers its
        # caches as ``model.layers.{43}.attn.*`` (no ``mtp_block`` segment).
        self.mtp_block = DeepseekV4DecoderLayer(
            vllm_config,
            prefix,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=None,
        )
        self.hc_head_op = HCHeadOp()
        self.mhc_post = MHCPostOp()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_size
        # Position 0 has no previous token for the MTP to condition on; the
        # GPU kernel zeroes that embedding row before enorm.
        inputs_embeds = torch.where(
            positions.unsqueeze(-1) == 0,
            torch.zeros_like(inputs_embeds),
            inputs_embeds,
        )
        inputs_embeds = self.enorm(inputs_embeds)
        # hnorm normalizes each of the hc_mult residual streams separately.
        previous_hidden_states = self.hnorm(
            previous_hidden_states.reshape(-1, hidden))
        h = self.h_proj(previous_hidden_states).reshape(
            -1, self.hc_mult, hidden)
        e = self.e_proj(inputs_embeds)
        hidden_states = h + e.unsqueeze(-2)

        x, residual, post_mix, res_mix = self.mtp_block(
            hidden_states, positions, input_ids)
        if post_mix is not None:
            x = self.mhc_post(x, residual, post_mix, res_mix)
        # Pre-hc_head residual, flat [T, hc_mult * H]: the next spec step's
        # ``previous_hidden_states``. hc_head is applied in compute_logits.
        return x.flatten(1)

    def pre_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.reshape(-1, self.hc_mult,
                                              self.hidden_size)
        hidden_states = self.hc_head_op(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        return self.shared_head.norm(hidden_states)


class DeepSeekV4MultiTokenPredictor(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )
        self.layers = nn.ModuleDict({
            str(idx):
            DeepSeekV4MultiTokenPredictorLayer(vllm_config,
                                               self.topk_indices_buffer,
                                               f"{prefix}.layers.{idx}")
            for idx in range(self.mtp_start_layer_idx,
                             self.mtp_start_layer_idx + self.num_mtp_layers)
        })
        # Shared with the target model at load time (see the DSV4 MTP
        # proposer); the checkpoint has no MTP-specific embedding.
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"))
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def _layer(self,
               spec_step_idx: int) -> DeepSeekV4MultiTokenPredictorLayer:
        idx = self.mtp_start_layer_idx + spec_step_idx % self.num_mtp_layers
        return self.layers[str(idx)]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return self._layer(spec_step_idx)(input_ids, positions,
                                          previous_hidden_states,
                                          inputs_embeds)

    def compute_logits(self,
                       hidden_states: torch.Tensor,
                       spec_step_idx: int = 0) -> torch.Tensor:
        layer = self._layer(spec_step_idx)
        return self.logits_processor(layer.shared_head.head,
                                     layer.pre_head(hidden_states))


class DeepSeekV4MTP(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = DeepSeekV4MultiTokenPredictor(vllm_config=vllm_config,
                                                   prefix=maybe_prefix(
                                                       prefix, "model"))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, hidden_states, inputs_embeds,
                          spec_step_idx)

    def compute_logits(self,
                       hidden_states: torch.Tensor,
                       spec_step_idx: int = 0) -> torch.Tensor:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def checkpoint_name_to_param_name(self, name: str) -> str | None:
        """Maps a checkpoint key to this module's parameter name, or None.

        Only ``mtp.{i}.*`` keys belong to the draft. The embedding and LM head
        are shared with the target and are not read from the checkpoint.
        """
        m = re.match(r"mtp\.(\d+)\.(.*)$", name)
        if m is None:
            return None
        mtp_idx, rest = int(m.group(1)), m.group(2)
        if mtp_idx >= self.model.num_mtp_layers:
            return None
        layer = self.model.mtp_start_layer_idx + mtp_idx
        if rest.startswith(("norm.", "head.")):
            rest = "shared_head." + rest
        if not rest.startswith(_MTP_WRAPPER_WEIGHTS):
            rest = "mtp_block." + rest
        name = f"model.layers.{layer}.{rest}"
        if name.endswith(".scale"):
            if _EXPERT_SCALE_RE.search(name) and getattr(
                    self.config, "expert_dtype", "fp4") == "fp4":
                name = name[:-len(".scale")] + ".weight_scale"
            else:
                name = name[:-len(".scale")] + ".weight_scale_inv"
        name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
        if name.endswith(".ffn.gate.bias"):
            name = name[:-len(".bias")] + ".e_score_correction_bias"
        return name

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )

        for ckpt_name, loaded_weight in weights:
            name = self.checkpoint_name_to_param_name(ckpt_name)
            if name is None:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if ".experts." in name or weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales: keep the raw exponent bytes (see the
                    # target loader).
                    if ("weight_scale" in name
                            and loaded_weight.dtype == torch.float8_e8m0fnu):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for (param_name, weight_name, expert_id,
                         expert_shard_id) in expert_mapping:
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(Callable[..., bool],
                                                    param.weight_loader)
                        if weight_loader(param,
                                         loaded_weight,
                                         name_mapped,
                                         shard_id=expert_shard_id,
                                         expert_id=expert_id,
                                         return_success=True):
                            loaded_params.add(name_mapped)
                            break
                    else:
                        raise KeyError(
                            f"MTP expert weight {ckpt_name} -> {name} has no "
                            "matching parameter")
                    continue
                param = params_dict[name]
                if "attn_sink" in name:
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    param[:narrow_weight.shape[0]].copy_(narrow_weight)
                else:
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                loaded_params.add(name)

        if not loaded_params:
            raise ValueError(
                "DeepSeek V4 MTP: no mtp.* weights found in the checkpoint")
        logger.info("DeepSeek V4 MTP draft loaded %d params",
                    len(loaded_params))
        return loaded_params
