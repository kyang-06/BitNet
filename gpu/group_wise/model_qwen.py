# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from xformers.ops import RMSNorm, fmha, rope_padded
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
)

import ctypes
bitnet_lib = ctypes.CDLL('bitnet_kernels/libbitnet.so')

def bitnet_int8xint2_linear(input0, input1, s, ws, group_size):
    out_shape = list(input0.shape)
    out_shape[-1] = input1.shape[0]

    stream = torch.cuda.current_stream()

    M = input0.shape[0]
    if len(out_shape) == 3:
        M *= input0.shape[1]
    N = input1.shape[0]
    K = input1.shape[1] * 4

    ret = torch.zeros(*out_shape, dtype=torch.bfloat16, device=input0.device)

    bitnet_lib.bitlinear_int8xint2(*[ctypes.c_void_p(input0.data_ptr()), ctypes.c_void_p(input1.data_ptr()), ctypes.c_void_p(ret.data_ptr()), ctypes.c_void_p(s.data_ptr()), ctypes.c_void_p(ws.data_ptr()), ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(group_size), ctypes.c_void_p(stream.cuda_stream)])

    return ret


@dataclass
class ModelArgs:
    # Qwen3-4B defaults
    dim: int = 2560
    head_dim: int = 128
    n_layers: int = 36
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 151936
    ffn_dim: int = 9728
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    use_kernel: bool = False


LayerCache = Tuple[torch.Tensor, torch.Tensor]


class BitLinearKernel(nn.Module):
    in_features: int
    out_features: int
    weight: torch.Tensor
    weight_scale: torch.Tensor
    group_size: int

    def __init__(self, in_features: int, out_features: int, bias: bool = False, ws_num: int = 0, group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.ws_num = ws_num

        self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features // 4, dtype=torch.int8), requires_grad=False)

        self.tensor_wise_weight_scale = False
        if self.tensor_wise_weight_scale:
            self.weight_scale = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16), requires_grad=False)
        else:
            self.weight_scale = torch.nn.Parameter(torch.zeros([out_features, in_features // group_size, 1], dtype=torch.bfloat16), requires_grad=False)

    @torch.compile
    def quant_input(self, input):
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)     # per token quantization
        return (input * s).round().clamp(-128, 127).to(torch.int8), s

    def forward(self, input):
        input, s = self.quant_input(input)
        if self.tensor_wise_weight_scale:
            weight_scale = self.weight_scale[:self.ws_num].reshape(self.ws_num, 1).repeat([1, self.out_features // self.ws_num * self.in_features // self.group_size]).reshape(self.out_features, self.in_features // self.group_size).flatten().contiguous()
        else:
            weight_scale = self.weight_scale.squeeze(-1).flatten().contiguous()
        ret = bitnet_int8xint2_linear(input, self.weight, s, weight_scale, self.group_size)
        return ret


class BitLinear(nn.Linear):
    @torch.compile
    def quant_input(self, input):
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127) / s

    def forward(self, input):
        # input = self.quant_input(input)
        return F.linear(input, self.weight)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        norm_eps: float,
        use_kernel: bool,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_kv_heads

        # fused_qkv: (n_heads + 2 * n_kv_heads) * head_dim output features
        qkv_out = (n_heads + 2 * n_kv_heads) * head_dim
        if use_kernel:
            self.fused_qkv = BitLinearKernel(dim, qkv_out, bias=False, ws_num=1)
            self.o_proj    = BitLinearKernel(n_heads * head_dim, dim, bias=False, ws_num=1)
        else:
            self.fused_qkv = BitLinear(dim, qkv_out, bias=False)
            self.o_proj    = BitLinear(n_heads * head_dim, dim, bias=False)

        # Qwen3: per-head RMSNorm on Q and K before RoPE
        self.q_norm = RMSNorm(head_dim, norm_eps)
        self.k_norm = RMSNorm(head_dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        attn_bias: AttnBias,
    ) -> torch.Tensor:

        # --- fused QKV projection then split ---
        qkv = self.fused_qkv(x)   # [seq, (n_heads + 2*n_kv_heads) * head_dim]
        q_size = self.n_local_heads    * self.head_dim
        k_size = self.n_local_kv_heads * self.head_dim
        xq, xk, xv = qkv.split([q_size, k_size, k_size], dim=-1)

        output_shape = xq.shape
        heads_per_group = self.n_local_heads // self.n_local_kv_heads

        # Reshape to head layout for per-head norm and RoPE
        xq = xq.view(1, xq.shape[0], self.n_local_kv_heads, heads_per_group, self.head_dim).contiguous()
        xk = xk.view(1, xk.shape[0], self.n_local_kv_heads, 1, self.head_dim).contiguous()
        xv = xv.view(1, xv.shape[0], self.n_local_kv_heads, 1, self.head_dim).contiguous()

        # Qwen3: apply q_norm / k_norm per head before RoPE
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        cache_k, cache_v = cache

        xq = rope_padded(
            xq=xq,
            xk=xk,
            xv=xv,
            cache_k=cache_k,
            cache_v=cache_v,
            attn_bias=attn_bias,
            theta=self.rope_theta,
        )

        output = fmha.memory_efficient_attention_forward(
            xq, cache_k, cache_v, attn_bias, op=fmha.flash.FwOp
        )

        output = output.reshape(output_shape)
        output = self.o_proj(output)

        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        norm_eps: float,
        use_kernel: bool,
    ):
        super().__init__()

        if use_kernel:
            self.fused_gate_up = BitLinearKernel(dim, 2 * hidden_dim, bias=False, ws_num=1)
            self.down_proj     = BitLinearKernel(hidden_dim, dim, bias=False, ws_num=1)
        else:
            self.fused_gate_up = BitLinear(dim, 2 * hidden_dim, bias=False)
            self.down_proj     = BitLinear(hidden_dim, dim, bias=False)

        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.fused_gate_up(x).split(self.hidden_dim, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.dim % args.n_heads == 0
        head_dim = args.head_dim
        n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        assert args.n_heads % n_kv_heads == 0

        self.self_attn = Attention(
            dim=args.dim,
            head_dim=head_dim,
            n_heads=args.n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=args.rope_theta,
            norm_eps=args.norm_eps,
            use_kernel=args.use_kernel,
        )
        self.mlp = FeedForward(
            dim=args.dim,
            hidden_dim=args.ffn_dim,
            norm_eps=args.norm_eps,
            use_kernel=args.use_kernel,
        )
        self.input_layernorm          = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        attn_bias: AttnBias,
    ) -> torch.Tensor:
        h = x + self.self_attn.forward(
            self.input_layernorm(x),
            cache,
            attn_bias,
        )
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size > 0

        self.model = nn.ModuleDict({
            "embed_tokens": nn.Embedding(
                num_embeddings=args.vocab_size,
                embedding_dim=args.dim,
            ),
            "layers": nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)]),
            "norm": RMSNorm(args.dim, eps=args.norm_eps),
        })

        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

    @property
    def embed_tokens(self):
        return self.model["embed_tokens"]

    @property
    def layers(self):
        return self.model["layers"]

    @property
    def norm(self):
        return self.model["norm"]

    @torch.no_grad()
    def forward_with_attn_bias(
        self,
        token_values: torch.Tensor,
        attn_bias: AttnBias,
        cache: list[LayerCache],
    ) -> torch.Tensor:
        h = self.embed_tokens(token_values)

        for i, layer in enumerate(self.layers):
            h = layer(h, cache[i], attn_bias)

        logits = self.lm_head(self.norm(h))
        return logits.float()

    def forward(
        self,
        token_values: torch.Tensor,
        token_lengths: torch.Tensor,
        start_pos: torch.Tensor,
        cache: list[LayerCache],
        kv_padding: int,
    ) -> torch.Tensor:
        attn_bias = AttnBias.from_seqlens(
            q_seqlen=token_lengths.tolist(),
            kv_seqlen=(start_pos + token_lengths).tolist(),
            kv_padding=kv_padding,
        )
        return self.forward_with_attn_bias(token_values, attn_bias, cache)


def make_cache(
    args: ModelArgs,
    length: int,
    device: Optional[Union[str, torch.device]] = None,
    n_layers: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> list[LayerCache]:
    head_dim = args.head_dim
    n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads

    if n_layers is None:
        n_layers = args.n_layers

    shape = (1, length, n_kv_heads, 1, head_dim)
    heads_per_group = args.n_heads // n_kv_heads
    expansion = (-1, -1, -1, heads_per_group, -1)
    return [
        (
            torch.zeros(shape, device=device, dtype=dtype).expand(expansion),
            torch.zeros(shape, device=device, dtype=dtype).expand(expansion),
        )
        for _ in range(n_layers)
    ]


def cache_prefix(cache: list[LayerCache], length: int) -> list[LayerCache]:
    if len(cache) > 0:
        assert cache[0][0].shape[1] >= length
    return [(ck[:, :length], cv[:, :length]) for ck, cv in cache]