import torch
import torch.nn as nn
from typing import Optional, Callable, List, Tuple, Sequence
from functools import reduce
from operator import mul
from flash_attn import flash_attn_varlen_func


def _flash_attention_block(
    # [*, Q, H, C_hidden]
    query: torch.Tensor,
    # [*, K, H, C_hidden]
    key: torch.Tensor,
    # [*, V, H, C_hidden]
    value: torch.Tensor,
    # [*, L]
    attn_mask: torch.Tensor,
    causal = False,
    window_size=(-1, -1),
    softmax_scale=None,
    dropout_prob=0.0,
):
    
    token_size = list(query.shape[:-2])
    embed_size = list(query.shape[-2:])
    query = query.reshape([-1]+embed_size)
    key   = key.reshape([-1]+embed_size)
    value = value.reshape([-1]+embed_size)

    cu_seqlens_q = torch.cumsum(torch.cat([torch.zeros([1]).to(attn_mask), attn_mask[torch.nonzero(attn_mask, as_tuple=True)]]), dim=0).int()
    cu_seqlens_k = cu_seqlens_q.clone()
    max_seqlen_q = attn_mask.max().item()
    max_seqlen_k = attn_mask.max().item()
    if cu_seqlens_q[-1] != query.shape[0]:
        print("attn_mask: ")
        print(attn_mask[:, :10])
    assert cu_seqlens_q[-1] == query.shape[0], f"cu_seqlens_q={cu_seqlens_q}, query.shape={query.shape}"

    outputs = flash_attn_varlen_func(query, key, value, 
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        window_size=window_size,
        softmax_scale=softmax_scale,
        dropout_p=dropout_prob,
        return_attn_probs=True,
    )
    
    o = outputs[0].reshape(token_size + embed_size)
    attn_probs = outputs[1].reshape(token_size + [embed_size[0]])

    return o, attn_probs