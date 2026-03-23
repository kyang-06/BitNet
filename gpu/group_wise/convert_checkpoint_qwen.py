import json
import os
import re
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import torch
from einops import rearrange
from safetensors.torch import save_file
import model
from pack_weight import convert_weight_int8_to_int2
#
# import pydevd_pycharm
# pydevd_pycharm.settrace('10.238.129.229', port=12312, stdout_to_server=True, stderr_to_server=True, suspend=False)

@torch.inference_mode()
def convert_ts_checkpoint(
    *,
    input_path: str = "",
    group_size: int,
    output_dir: str
) -> None:

    config = model.ModelArgs(dim=2560, n_layers=36, n_heads=32, ffn_dim=9728, vocab_size=151396, n_kv_heads=8, rope_theta=1000000)
    print(f"Model config {config.__dict__}")

    def quant_weight_int8(weight):
        s = 1.0 / weight.abs().reshape(weight.shape[0], weight.shape[1] // group_size, group_size).max(dim=-1, keepdims=True)[0].clamp_(min=1e-5)
        new_weight = (weight * s.repeat([1, 1, group_size]).reshape(weight.shape)).round().clamp(-1, 1).to(torch.int8)
        new_scale = (1.0 / s).to(torch.bfloat16)
        return new_weight, new_scale

    def quant_weight_fp16(weight):
        s = 1.0 / weight.abs().reshape(weight.shape[0], weight.shape[1] // group_size, group_size).max(dim=-1, keepdims=True)[0].clamp_(min=1e-5)
        new_weight = (weight * s.repeat([1, 1, group_size]).reshape(weight.shape)).round().clamp(-1, 1) / s.repeat([1, 1, group_size]).reshape(weight.shape)
        return new_weight

    def convert_int8_to_int2(weight):
        return convert_weight_int8_to_int2(weight)

    merged_result = torch.load(input_path, map_location="cpu", mmap=True)
    int2_result = {}
    fp16_result = {}

    for key, value in merged_result.items():
        # ------------------------------------------------------------------
        # Q / K / V projections  (Qwen3: self_attn.{q,k,v}_proj.weight)
        # ------------------------------------------------------------------
        if 'self_attn.fused_qkv.weight' in key:
            wqkv = value
            wqkv_weight, wqkv_scale = quant_weight_int8(wqkv)
            int2_result[key] = convert_int8_to_int2(wqkv_weight)
            int2_result[key.replace('weight', 'weight_scale')] = wqkv_scale
            fp16_result[key] = quant_weight_fp16(wqkv)

        # ------------------------------------------------------------------
        # Output projection  (Qwen3: self_attn.o_proj.weight)
        # ------------------------------------------------------------------
        elif 'self_attn.o_proj.weight' in key:
            weight, scale = quant_weight_int8(value)
            int2_result[key] = convert_int8_to_int2(weight)
            int2_result[key.replace('weight', 'weight_scale')] = scale
            fp16_result[key] = quant_weight_fp16(value)

        # ------------------------------------------------------------------
        # FFN gate / up projections  (Qwen3: mlp.gate_proj / mlp.up_proj)
        # ------------------------------------------------------------------
        elif 'mlp.fused_gate_up.weight' in key:
            w13 = value
            w13_weight, w13_scale = quant_weight_int8(w13)
            int2_result[key] = convert_int8_to_int2(w13_weight)
            int2_result[key.replace('weight', 'weight_scale')] = w13_scale
            fp16_result[key] = quant_weight_fp16(w13)

        # ------------------------------------------------------------------
        # FFN down projection  (Qwen3: mlp.down_proj.weight)
        # ------------------------------------------------------------------
        elif 'mlp.down_proj.weight' in key:
            weight, scale = quant_weight_int8(value)
            int2_result[key] = convert_int8_to_int2(weight)
            int2_result[key.replace('weight', 'weight_scale')] = scale
            fp16_result[key] = quant_weight_fp16(value)

        # ------------------------------------------------------------------
        # All other weights (norms, embeddings, lm_head …) – pass through
        # ------------------------------------------------------------------
        else:
            int2_result[key] = value.clone()
            fp16_result[key] = value.clone()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving checkpoint to {output_dir}/model_state_int2.pt")
    torch.save(int2_result, f"{output_dir}/model_state_int2.pt")

    print(f"Saving checkpoint to {output_dir}/model_state_fp16.pt")
    torch.save(fp16_result, f"{output_dir}/model_state_fp16.pt")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert TorchScale checkpoint.')
    parser.add_argument('--input', type=str)
    parser.add_argument('--group_size', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='./')

    args = parser.parse_args()
    convert_ts_checkpoint(
        input_path=args.input,
        group_size=args.group_size,
        output_dir=args.output_dir
    )