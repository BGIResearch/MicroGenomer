""" PyTorch GLM model. """

import math
import copy
import warnings
import re
import sys
import os
import pathlib
import time
import argparse
import random
import numpy as np
from tqdm.auto import tqdm, trange
from functools import partial

import torch, deepspeed
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm, MSELoss, BCEWithLogitsLoss
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
from copy import deepcopy
from collections import namedtuple

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput

from .configuration_glm import GLMConfig
from .flash_attn_func import _flash_attention_block
from .positional_embeddings import *
from .utils import Pooling, ConvLayer


def get_checkpoint_fn():
    if deepspeed.checkpointing.is_configured():
        # checkpoint = deepspeed.checkpointing.non_reentrant_checkpoint
        checkpoint = deepspeed.checkpointing.checkpoint
    else:
        checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
        # checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=True)
    return checkpoint

# flags required to enable jit fusion kernels

if sys.platform != 'darwin':
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "StrainDNA"
_CONFIG_FOR_DOC = "GLMConfig"

GLM_3B_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "StrainDNA-3b",
]


def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)

DeepNormCoefficients = namedtuple("DeepNormCoefficients", ["alpha", "beta"])

def get_deepnorm_coefficients(config: GLMConfig):
    """
        DeepNorm coefficients from : https://kexue.fm/archives/8978
    """
    num_layers = config.num_layers
    return DeepNormCoefficients(alpha=(2 * num_layers) ** 0.5, beta=(2 * num_layers) ** -0.5)


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config: GLMConfig):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            kv_size = config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            self.embedding = torch.nn.Embedding(config.pre_seq_len, kv_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(kv_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, kv_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len,
                                                config.num_layers * config.kv_channels * config.multi_query_group_num * 2)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values

def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm
    class LayerNorm(FusedLayerNorm):
        def __init__(self, *args, pb_relax=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.pb_relax = pb_relax

        def forward(self, x):
            if not self.pb_relax:
                return super().forward(x)
            return super().forward(x / (x.abs().max().detach() / 8))
except ModuleNotFoundError:
    print('Please install apex to use fused_layer_norm, fall back to torch.nn.LayerNorm')
    from  torch.nn import LayerNorm


class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, config: GLMConfig, layer_number):
        super(FlashSelfAttention, self).__init__()

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads

        self.dropout_p = config.attention_dropout
        self.output_attentions = config.output_attentions
        
        self.causal = config.is_causal
        self.softmax_scale = 1 / math.sqrt(self.hidden_size_per_attention_head)

        self.num_attention_heads = config.num_attention_heads

    def forward(self, q, k, v, attn_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda
        """
        batch_size, seqlen = q.shape[0], q.shape[1]
        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        max_s = seqlen
        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                  device=q.device)
        output = flash_attn_unpadded_func(
            q, k, v, cu_seqlens, cu_seqlens, max_s, max_s,
            self.dropout_p if self.training else 0.0,
            softmax_scale=self.softmax_scale, causal=self.causal
        )
        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output
        """
        q = q.permute(1, 0, 2, 3)
        k = k.permute(1, 0, 2, 3)
        v = v.permute(1, 0, 2, 3)

        output, attention_probs = _flash_attention_block(q, k, v, attn_mask, 
                                        softmax_scale=self.softmax_scale, 
                                        causal=self.causal, 
                                        dropout_prob=self.dropout_p)

        output = output.permute(1, 0, 2, 3)
        attention_probs = attention_probs.permute(1, 0, 2)
        
        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = output.size()[:-2] + (self.hidden_size_per_partition,)
        output = output.view(*new_context_layer_shape)
        attention_probs = None
        if not self.output_attentions:
            attention_probs = None
        else:
            attention_probs = (
                torch.mean(attention_probs, dim=0, keepdim=False),
                torch.max(attention_probs, dim=0, keepdim=False)[0]
            )

        return output, attention_probs


class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: GLMConfig, layer_number, device=None):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)

        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size,
                                         bias=config.add_bias_linear or config.add_qkv_bias,
                                         device=device, **_config_to_kwargs(config)
                                         )
        self.use_flash_attn = config.use_flash_attn
        self.core_attention = FlashSelfAttention(config, self.layer_number)

        # Output.
        self.dense = nn.Linear(self.projection_size, config.hidden_size, bias=config.add_bias_linear, device=device, **_config_to_kwargs(config))
        
        self.rotary_embedding_2d = config.rotary_embedding_2d
        self.rotary_emb = RotaryEmbedding(self.hidden_size_per_attention_head // 2 if self.rotary_embedding_2d else self.hidden_size_per_attention_head, 
                                          base=config.rotary_freq_base, precision=config.torch_dtype, learnable=False)

        ##### LoRA
        self.lora = config.lora
        if config.lora:
            self.lora_linear = torch.nn.ModuleDict()
            self.lora_dropout = torch.nn.Dropout(config.lora_dropout)
            self.lora_alpha = config.lora_alpha
            self.lora_r = config.lora_r
            for name in ('Q', 'K', 'V', 'O'):
                self.lora_linear[f'{name}_A'] = torch.nn.Linear(config.hidden_size, config.lora_r, bias=False)
                self.lora_linear[f'{name}_B'] = torch.nn.Linear(config.lora_r, config.hidden_size, bias=False)
                torch.nn.init.kaiming_uniform_(self.lora_linear[f"{name}_A"].weight, a=math.sqrt(5))
                torch.nn.init.zeros_(self.lora_linear[f'{name}_B'].weight)

    def forward(
            self, hidden_states, attention_mask, position_ids, kv_cache=None, use_cache=True
    ):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads_per_partition, 3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # apply relative positional encoding (rotary embedding)
        if position_ids is not None: # [seq_len, 2, batch_size, 32, 2]
            if self.rotary_embedding_2d:
                q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1)) # 32
                k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
                # import pdb; pdb.set_trace();
                cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1) # 32
                position_ids, block_position_ids = \
                    position_ids[:, 0, :].transpose(0, 1).contiguous(), \
                    position_ids[:, 1, :].transpose(0, 1).contiguous()
                q1, k1 = apply_rotary_pos_emb_index_torch(q1, k1, cos, sin, position_ids)
                q2, k2 = apply_rotary_pos_emb_index_torch(q2, k2, cos, sin, block_position_ids)
                query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1))
                key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1))
            else:
                # [b, sq] -> [sq, b]
                position_ids = position_ids.transpose(0, 1)
                cos, sin = self.rotary_emb(value_layer, seq_len=position_ids.max() + 1)
                query_layer, key_layer = apply_rotary_pos_emb_index_torch(query_layer, key_layer, cos, sin, position_ids)

        if self.lora:
            # query_layer = query_layer +  lora_layer["Q_B"](lora_layer["Q_A"](self.lora_dropout(hidden_states)))* self.scaling
            scaling = self.lora_alpha / self.lora_r
            query_layer = query_layer + ( self.lora_linear['Q_B'](self.lora_linear['Q_A'](self.lora_dropout(hidden_states))) * scaling ).reshape(query_layer.shape)
            key_layer   = key_layer   + ( self.lora_linear['K_B'](self.lora_linear['K_A'](self.lora_dropout(hidden_states))) * scaling ).reshape(key_layer.shape)
            value_layer = value_layer + ( self.lora_linear['V_B'](self.lora_linear['V_A'](self.lora_dropout(hidden_states))) * scaling ).reshape(value_layer.shape)
            
        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=0)
            value_layer = torch.cat((cache_v, value_layer), dim=0)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.expand(-1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1)
            key_layer = key_layer.contiguous().view(key_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(-1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1)
            value_layer = value_layer.contiguous().view(value_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))

        # ==================================
        # core attention computation
        # ==================================

        context_layer, attention_probs = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
        output = self.dense(context_layer)
        if self.lora:
            scaling = self.lora_alpha / self.lora_r
            output = output + self.lora_linear['O_B'](self.lora_linear['O_A'](self.lora_dropout(context_layer))) * scaling

        # =================
        # Output. [sq, b, h]
        # =================

        # output = context_layer @ self.dense.weight.T + self.dense.bias
        return output, kv_cache, attention_probs


def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs


class MLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: GLMConfig, device=None):
        super(MLP, self).__init__()

        self.add_bias = config.add_bias_linear
        self.moe = config.moe
        self.mlp_lora = config.mlp_lora
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token # 2

        if self.moe is True and self.mlp_lora is True:
            raise NotImplementedError(f"moe and mlp_lora are both enabled")

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

        def swiglu(x):
           x = torch.chunk(x, 2, dim=-1)
           return x[0] * F.silu(x[1])

        def geglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return x[0] * F.gelu(x[1])

        if config.glu_activation == 'geglu':
            self.activation_func = geglu
        elif config.glu_activation == 'swiglu':
            self.activation_func = swiglu
        else:
            assert RuntimeError(f"Unsupported glu_activation: {config.glu_activation}")

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

        if self.moe:
            assert self.num_experts > 1
            del self.dense_h_to_4h
            del self.dense_4h_to_h
            self.router = nn.Linear(
                config.hidden_size,
                config.num_experts,
                bias=False,
                device=device,
                dtype=torch.float32
            )
            for i in range(0, self.num_experts):
                self.register_module(f"dense_h_to_4h_{i}", nn.Linear(
                    config.hidden_size,
                    config.ffn_hidden_size * 2,
                    bias=self.add_bias,
                    device=device,
                    **_config_to_kwargs(config)
                ))
                self.register_module(f"dense_4h_to_h_{i}", nn.Linear(
                    config.ffn_hidden_size,
                    config.hidden_size,
                    bias=self.add_bias,
                    device=device,
                    **_config_to_kwargs(config)
                ))

        if self.mlp_lora:
            self.lora_linear = torch.nn.ModuleDict()
            self.lora_dropout = torch.nn.Dropout(config.lora_dropout)
            self.lora_alpha = config.lora_alpha
            self.lora_r = config.lora_r
            for name in ('dense_h_to_4h', 'dense_4h_to_h'):
                if name == 'dense_h_to_4h':
                    self.lora_linear[f'{name}_A'] = torch.nn.Linear(config.hidden_size, config.lora_r, bias=False)
                    self.lora_linear[f'{name}_B'] = torch.nn.Linear(config.lora_r, config.ffn_hidden_size * 2, bias=False)
                elif name == 'dense_4h_to_h':
                    self.lora_linear[f'{name}_A'] = torch.nn.Linear(config.ffn_hidden_size, config.lora_r, bias=False)
                    self.lora_linear[f'{name}_B'] = torch.nn.Linear(config.lora_r, config.hidden_size, bias=False)
                torch.nn.init.kaiming_uniform_(self.lora_linear[f"{name}_A"].weight, a=math.sqrt(5))
                torch.nn.init.zeros_(self.lora_linear[f'{name}_B'].weight)

    def moe_forward(self, hidden_states, expert_idx):
        intermediate_parallel = getattr(self, f"dense_h_to_4h_{expert_idx}")(hidden_states) # torch.Size([503, 20480])
        intermediate_parallel = self.activation_func(intermediate_parallel) # torch.Size([503, 10240])
        output = getattr(self, f"dense_4h_to_h_{expert_idx}")(intermediate_parallel) # torch.Size([503, 1920])
        return output

    def forward(self, hidden_states):
        if self.moe:
            s, b, n = hidden_states.shape
            dtype = hidden_states.dtype
            hidden_states = hidden_states.view(-1, hidden_states.size(2)) # [s*b h]
            
            route = self.router(hidden_states).to(dtype)

            weights, selected_experts = torch.topk(route, self.experts_per_token)
            weights = F.softmax(weights, dim=1, dtype=torch.float).to(hidden_states.dtype)
            output = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            for expert_idx in range(self.num_experts):
                batch_idx, nth_expert = torch.where(selected_experts == expert_idx)
                if nth_expert.shape[0] == 0:
                    continue
                cur_out = self.moe_forward(hidden_states[batch_idx], expert_idx)
                output[batch_idx] += weights[batch_idx, nth_expert, None] * cur_out
            output = output.reshape(s, b, n)
        else:
            # [s, b, 4hp]
            #intermediate_parallel = hidden_states @ self.dense_h_to_4h.weight.T + self.dense_h_to_4h.bias
            intermediate_parallel = self.dense_h_to_4h(hidden_states)
            if self.mlp_lora:
                scaling = self.lora_alpha / self.lora_r
                intermediate_parallel = intermediate_parallel + ( self.lora_linear['dense_h_to_4h_B'](self.lora_linear['dense_h_to_4h_A'](self.lora_dropout(hidden_states))) * scaling )
                
            intermediate_parallel = self.activation_func(intermediate_parallel)
            # [s, b, h]
            output = self.dense_4h_to_h(intermediate_parallel)
            if self.mlp_lora:
                output = output + ( self.lora_linear['dense_4h_to_h_B'](self.lora_linear['dense_4h_to_h_A'](self.lora_dropout(intermediate_parallel))) * scaling )# .reshape(output.shape)

            #output = intermediate_parallel @ self.dense_4h_to_h.weight.T + self.dense_4h_to_h.bias # self.dense_4h_to_h(intermediate_parallel)
        return output

class GLMBlock(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: GLMConfig, layer_number, device=None):
        super(GLMBlock, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon)

        # Self attention.
        self.self_attention = SelfAttention(config, layer_number, device=device)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon)

        # MLP
        self.mlp = MLP(config, device=device)

        # output attention
        self.output_attentions = config.output_attentions

        self.deepnorm_coeff = get_deepnorm_coefficients(config) if config.deepnorm else None

    def forward(
            self, hidden_states, attention_mask, position_ids, kv_cache=None, use_cache=True,
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention. attention_probs: [b * np, sq, sk]
        attention_output, kv_cache, attention_probs = self.self_attention(
            layernorm_output,
            attention_mask,
            position_ids, # [batch_size, 2, seq_len, 32, 2]
            kv_cache=kv_cache,
            use_cache=use_cache
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)

        if self.deepnorm_coeff is not None: 
            layernorm_input = residual*self.deepnorm_coeff.alpha + layernorm_input
        else:
            layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        #print('====output====')
        #print(output)
        if self.deepnorm_coeff is not None: 
            #print(f"2 self.deepnorm_coeff is not None")
            output = residual*self.deepnorm_coeff.alpha + output
        else:
            #print(f"2 self.deepnorm_coeff is None")
            output = residual + output

        return output, kv_cache, attention_probs


class GLMTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, config: GLMConfig, device=None):
        super(GLMTransformer, self).__init__()

        self.config = config

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon)

        self.gradient_checkpointing = False
        # Introduce a gradient checkpointing for per num_checkpoint layers
        # For example:  num_checkpoint=1 will checkpoint all layers, num_checkpoint=2 will checkpoint half of layers
        self.num_checkpoint = 1 

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
            self, hidden_states, attention_mask, position_ids, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
    ):
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled() and index % self.num_checkpoint == 0:
                #### A trick to enable gradient to avoid gradient checkpointing error
                if hidden_states.requires_grad is False and deepspeed.checkpointing.is_configured() and (self.config.lora or self.config.mlp_lora):
                    # print(f"index={index}, set hidden_states.requires_grad = True")
                    hidden_states = hidden_states.clone()
                    hidden_states.requires_grad = True
                layer_ret = get_checkpoint_fn()(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    kv_caches[index],
                    use_cache
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )
            
            hidden_states, kv_cache, attention_probs = layer_ret
            # import pdb; pdb.set_trace();
            # print(hidden_states)
            if use_cache:
                presents = presents + (kv_cache,)
            if output_attentions:
                all_self_attentions = all_self_attentions + (attention_probs,)
                # attention_probs: [batch_size * num_heads, seq_len, seq_len]


        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions


class GLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = GLMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        return

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        batch_size, seq_length = input_ids.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[0]
        if past_length:
            full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length,
                                                        device=input_ids.device), full_attention_mask), dim=-1)
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    def get_position_ids(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        position_ids_1 = torch.zeros( seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1) # [batch_size, seq_len]
        position_ids_2 = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1) # [batch_size, seq_len]
        position_ids   = torch.stack([position_ids_1, position_ids_2], axis=1) # [batch_size, 2, seq_len]
        return position_ids

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GLMTransformer):
            module.gradient_checkpointing = value


class Embedding(torch.nn.Module):
    """Language model embeddings."""

    def __init__(self, config: GLMConfig, device=None):
        super(Embedding, self).__init__()

        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size,
            self.hidden_size,
            dtype=config.torch_dtype,
            device=device
        )
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class GLMModel(GLMPreTrainedModel):
    def __init__(self, config: GLMConfig, device=None, empty_init=True):
        super().__init__(config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device
        self.embedding = init_method(Embedding, config, **init_kwargs)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels

        # Rotary positional embeddings
        self.seq_length = config.seq_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )

        # self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, base=10000, precision=config.torch_dtype, learnable=False)
        self.encoder = init_method(GLMTransformer, config, **init_kwargs)

        self.glm_transform = ConvLayer(config.hidden_size, config.out_dim)
        self.pool  =  Pooling(pool_method=config.pool_method, hiddn_size=config.out_dim)
        self.species_pool  =  Pooling(pool_method=config.species_pool_method, hiddn_size=config.out_dim)

        self.pre_seq_len = config.pre_seq_len
        self.prefix_projection = config.prefix_projection
        if self.pre_seq_len is not None:
            for param in self.parameters():
                param.requires_grad = False
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(config)
            self.dropout = torch.nn.Dropout(0.1)

    def init_lora_modules(self):
        for name, param in self.named_parameters():
            if 'lora_linear' in name:
                if '_A' in name:
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif '_B' in name:
                    torch.nn.init.zeros_(param)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def get_prompt(self, batch_size, device, dtype=torch.half):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_layers * 2,
            self.multi_query_group_num,
            self.kv_channels
        )
        # seq_len, b, nh, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
        return past_key_values

    def forward(
            self,
            input_ids,
            position_ids: Optional[torch.Tensor] = None, # position_ids: [batch_size, 2, seq_len]
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)),
                                            attention_mask], dim=-1)

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, full_attention_mask, position_ids=position_ids,
            kv_caches=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def load_pretrain_ckpt(self, ckpt_path, strict:bool=True, is_moe=False):
        state_dict = torch.load(ckpt_path)
        self.load_state_dict(state_dict)


class GLMForConditionalGeneration(GLMPreTrainedModel):
    def __init__(self, config: GLMConfig, empty_init=True, device=None):
        super().__init__(config)

        self.max_sequence_length = config.max_length
        self.transformer = GLMModel(config, empty_init=empty_init, device=device)
        self.config = config

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:

        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        if 'full_attention_mask' in model_kwargs:
            raise NotImplementedError(f"full_attention_mask...")
            model_kwargs['full_attention_mask'] = F.pad(model_kwargs['full_attention_mask'], [0, 1, 0, 1])
            if self.config.is_causal:
                model_kwargs['full_attention_mask'][..., -1] = True

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone() # [batch_size, 2, 1]
            if self.config.rotary_embedding_2d:
                new_position_id[:, 1] += 1 # Only update the 2nd dimension
            else:
                new_position_id[:] += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            ) # [batch_size, 2, seq_len+1]

        model_kwargs["is_first_forward"] = False
        return model_kwargs

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            full_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids, # position_ids: [batch_size, 2, seq_len]
            attention_mask=attention_mask,
            full_attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[-1:]
            
        glm_emb = self.transformer.glm_transform(hidden_states)
        gene_emb = self.transformer.pool(glm_emb)
        lm_logits = self.transformer.species_pool(glm_emb)

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(1, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )
