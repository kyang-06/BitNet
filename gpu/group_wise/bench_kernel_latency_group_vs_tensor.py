"""
Benchmark script for bitnet_int8xint2_linear kernel vs bf16 GEMV latency.

Usage:
    python bench_kernel.py                           # single shape, both kernels
    python bench_kernel.py --m 1 --n 2560 --k 2560  # custom shape, both kernels
    python bench_kernel.py --sweep                   # sweep Qwen3-4B shapes, side-by-side
    python bench_kernel.py --kernel bitnet           # only bitnet kernel
    python bench_kernel.py --kernel bf16             # only bf16 gemv
"""

import ctypes
import statistics
import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np
# --------------------------------------------------------------------------- #
# Load bitnet kernel
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Tensor allocation helpers
# --------------------------------------------------------------------------- #
def make_inputs_group_wise(M: int, N: int, K: int, group_size: int, device="cuda"):
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert K % 4 == 0,         f"K={K} must be divisible by 4 (int2 packing)"

    x  = torch.randint(-128, 127, (M, K),       dtype=torch.int8,     device=device)
    s  = torch.rand(M, 1,                        dtype=torch.bfloat16, device=device) * 0.01 + 1e-5
    w  = torch.randint(-128, 127, (N, K // 4),  dtype=torch.int8,     device=device)
    ws = torch.rand(N * (K // group_size),       dtype=torch.bfloat16, device=device)
    return x, w, s, ws

def make_inputs_channel_wise(M: int, N: int, K: int, device="cuda"):
    assert K % 4 == 0,         f"K={K} must be divisible by 4 (int2 packing)"

    x  = torch.randint(-128, 127, (M, K),       dtype=torch.int8,     device=device)
    s  = torch.rand(M, 1,                        dtype=torch.bfloat16, device=device) * 0.01 + 1e-5
    w  = torch.randint(-128, 127, (N, K // 4),  dtype=torch.int8,     device=device)
    ws = torch.rand(N,       dtype=torch.bfloat16, device=device)
    return x, w, s, ws

def make_inputs_tensor_wise(M: int, N: int, K: int, device="cuda"):
    assert K % 4 == 0,         f"K={K} must be divisible by 4 (int2 packing)"

    x  = torch.randint(-128, 127, (M, K),       dtype=torch.int8,     device=device)
    s  = torch.rand(M, 1,                        dtype=torch.bfloat16, device=device) * 0.01 + 1e-5
    w  = torch.randint(-128, 127, (N, K // 4),  dtype=torch.int8,     device=device)
    ws = torch.rand(2 if N in [6144, 19456] else 1,       dtype=torch.bfloat16, device=device)
    return x, w, s, ws


# --------------------------------------------------------------------------- #
# Core timing helper
# --------------------------------------------------------------------------- #
def _timed(fn, n_warmup: int, n_repeat: int) -> tuple[float, float]:
    """Return (mean_ms, std_ms) for fn(), using per-iteration synchronize."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    mean_ms = statistics.mean(times)
    std_ms  = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean_ms, std_ms


# --------------------------------------------------------------------------- #
# Benchmark functions
# --------------------------------------------------------------------------- #
def bench_bitnet_group_wise(
    M: int, N: int, K: int,
    group_size: int = 128,
    n_warmup: int = 50,
    n_repeat: int = 200,
    n_request: int = 1,
    device: str = "cuda",
) -> dict:
    mean_ms_reqs = []
    std_ms_reqs = []
    for _ in range(n_request):
        x, w, s, ws = make_inputs_group_wise(M, N, K, group_size, device)

        out_shape = list(x.shape)
        out_shape[-1] = w.shape[0]

        stream = torch.cuda.current_stream()

        M = x.shape[0]
        if len(out_shape) == 3:
            M *= x.shape[1]
        N = w.shape[0]
        K = w.shape[1] * 4  # int2 packed: 4 values per byte

        ret = torch.zeros(*out_shape, dtype=torch.bfloat16, device=x.device)
        bitnet_lib = ctypes.CDLL("/home/kyyx/github/LLM/quantization/CATQ/demo/BitNet-2026-03-05/gpu/group_wise/bitnet_kernels/libbitnet.so")
        mean_ms, std_ms = _timed(
            lambda: bitnet_lib.bitlinear_int8xint2(
                        ctypes.c_void_p(x.data_ptr()),
                        ctypes.c_void_p(w.data_ptr()),
                        ctypes.c_void_p(ret.data_ptr()),
                        ctypes.c_void_p(s.data_ptr()),
                        ctypes.c_void_p(ws.data_ptr()),
                        ctypes.c_int(M),
                        ctypes.c_int(N),
                        ctypes.c_int(K),
                        ctypes.c_int(group_size),
                        ctypes.c_void_p(stream.cuda_stream),
                    ),
            n_warmup, n_repeat,
        )
        mean_ms_reqs.append(mean_ms)
        std_ms_reqs.append(std_ms)

    mean_ms = np.mean(mean_ms_reqs)
    std_ms = np.mean(std_ms_reqs)
    flops        = 2 * M * N * K
    total_bytes  = (N * K // 4              # weight int2 packed
                  + M * K                   # activation int8
                  + N * (K // group_size) * 2  # weight scale bf16
                  + M * 2                   # activation scale bf16
                  + M * N * 2)              # output bf16

    return dict(M=M, N=N, K=K,
                mean_ms=mean_ms, std_ms=std_ms,
                tflops=flops / (mean_ms * 1e-3) / 1e12,
                achieved_bw_GBs=total_bytes / (mean_ms * 1e-3) / 1e9,
                arithmetic_intensity=flops / total_bytes,
                total_bytes_mb=total_bytes / 1e6)

def bench_bitnet_channel_wise(
    M: int, N: int, K: int,
    n_warmup: int = 50,
    n_repeat: int = 200,
    n_request: int = 1,
    device: str = "cuda",
) -> dict:
    mean_ms_reqs = []
    std_ms_reqs = []
    for _ in range(n_request):
        x, w, s, ws = make_inputs_channel_wise(M, N, K, device)

        out_shape = list(x.shape)
        out_shape[-1] = w.shape[0]

        stream = torch.cuda.current_stream()

        M = x.shape[0]
        if len(out_shape) == 3:
            M *= x.shape[1]
        N = w.shape[0]
        K = w.shape[1] * 4  # int2 packed: 4 values per byte

        ret = torch.zeros(*out_shape, dtype=torch.bfloat16, device=x.device)
        bitnet_lib = ctypes.CDLL("/home/kyyx/github/LLM/quantization/CATQ/demo/BitNet-2026-03-05/gpu/channel_wise/bitnet_kernels/libbitnet.so")
        mean_ms, std_ms = _timed(
            lambda: bitnet_lib.bitlinear_int8xint2(
                        ctypes.c_void_p(x.data_ptr()),
                        ctypes.c_void_p(w.data_ptr()),
                        ctypes.c_void_p(ret.data_ptr()),
                        ctypes.c_void_p(s.data_ptr()),
                        ctypes.c_void_p(ws.data_ptr()),
                        ctypes.c_int(M),
                        ctypes.c_int(N),
                        ctypes.c_int(K),
                        ctypes.c_void_p(stream.cuda_stream),
                    ),
            n_warmup, n_repeat,
        )
        mean_ms_reqs.append(mean_ms)
        std_ms_reqs.append(std_ms)

    mean_ms = np.mean(mean_ms_reqs)
    std_ms = np.mean(std_ms_reqs)
    flops        = 2 * M * N * K
    total_bytes  = (N * K // 4              # weight int2 packed
                  + M * K                   # activation int8
                  + N * 2  # weight scale bf16
                  + M * 2                   # activation scale bf16
                  + M * N * 2)              # output bf16

    return dict(M=M, N=N, K=K,
                mean_ms=mean_ms, std_ms=std_ms,
                tflops=flops / (mean_ms * 1e-3) / 1e12,
                achieved_bw_GBs=total_bytes / (mean_ms * 1e-3) / 1e9,
                arithmetic_intensity=flops / total_bytes,
                total_bytes_mb=total_bytes / 1e6)

def bench_bitnet_tensor_wise(
    M: int, N: int, K: int,
    n_warmup: int = 50,
    n_repeat: int = 200,
    n_request: int = 1,
    device: str = "cuda",
) -> dict:
    mean_ms_reqs = []
    std_ms_reqs = []
    for _ in range(n_request):
        x, w, s, ws = make_inputs_tensor_wise(M, N, K, device)

        out_shape = list(x.shape)
        out_shape[-1] = w.shape[0]

        stream = torch.cuda.current_stream()

        M = x.shape[0]
        if len(out_shape) == 3:
            M *= x.shape[1]
        N = w.shape[0]
        K = w.shape[1] * 4  # int2 packed: 4 values per byte

        ret = torch.zeros(*out_shape, dtype=torch.bfloat16, device=x.device)
        bitnet_lib = ctypes.CDLL("/home/kyyx/github/LLM/quantization/CATQ/demo/BitNet-2026-03-05/gpu/tensor_wise/bitnet_kernels/libbitnet.so")
        mean_ms, std_ms = _timed(
            lambda: bitnet_lib.bitlinear_int8xint2(
                        ctypes.c_void_p(x.data_ptr()),
                        ctypes.c_void_p(w.data_ptr()),
                        ctypes.c_void_p(ret.data_ptr()),
                        ctypes.c_void_p(s.data_ptr()),
                        ctypes.c_void_p(ws.data_ptr()),
                        ctypes.c_int(M),
                        ctypes.c_int(N),
                        ctypes.c_int(K),
                        ctypes.c_void_p(stream.cuda_stream),
                    ),
            n_warmup, n_repeat,
        )
        mean_ms_reqs.append(mean_ms)
        std_ms_reqs.append(std_ms)

    mean_ms = np.mean(mean_ms_reqs)
    std_ms = np.mean(std_ms_reqs)
    flops        = 2 * M * N * K
    total_bytes  = (N * K // 4              # weight int2 packed
                  + M * K                   # activation int8
                  + 2  # weight scale bf16
                  + M * 2                   # activation scale bf16
                  + M * N * 2)              # output bf16

    return dict(M=M, N=N, K=K,
                mean_ms=mean_ms, std_ms=std_ms,
                tflops=flops / (mean_ms * 1e-3) / 1e12,
                achieved_bw_GBs=total_bytes / (mean_ms * 1e-3) / 1e9,
                arithmetic_intensity=flops / total_bytes,
                total_bytes_mb=total_bytes / 1e6)


def print_single(r: dict, kernel: str, label: str = ""):
    tag = f"[{label}]" if label else f"[M={r['M']} N={r['N']} K={r['K']}]"
    print(f"{tag:<32}  [{kernel:<6}]  {_fmt(r)}")

PEAK_BW_GBS = 768.0  # A6000 峰值带宽

def _fmt(r: dict) -> str:
    bw_util = r['achieved_bw_GBs'] / PEAK_BW_GBS * 100
    return (f"{r['mean_ms']*1e3:.1f} ± {r['std_ms']*1e3:.1f} us  "
            f"AI={r['arithmetic_intensity']:.2f} F/B  "
            f"BW={r['achieved_bw_GBs']:.1f} GB/s ({bw_util:.1f}%峰值)  "
            f"{r['tflops']:.4f} TFLOPS")

def print_comparison(r_groupwise, r_channelwise, r_tensorwise, label=""):
    speedup_gw_vs_tw = r_tensorwise["mean_ms"] / r_groupwise["mean_ms"]
    speedup_cw_vs_tw = r_tensorwise["mean_ms"] / r_channelwise["mean_ms"]
    theoretical_gw = r_tensorwise["total_bytes_mb"] / r_groupwise["total_bytes_mb"]
    theoretical_cw = r_tensorwise["total_bytes_mb"] / r_channelwise["total_bytes_mb"]
    tag = f"[{label}]" if label else f"[M={r_groupwise['M']} N={r_groupwise['N']} K={r_groupwise['K']}]"
    print(f"\n{tag}")
    print(f"  group_wise   : {_fmt(r_groupwise)}")
    print(f"  channel_wise : {_fmt(r_channelwise)}")
    print(f"  tensor_wise  : {_fmt(r_tensorwise)}")
    print(f"  group   vs tensor: 实测 {speedup_gw_vs_tw:.2f}x | 理论 {theoretical_gw:.2f}x | 效率 {speedup_gw_vs_tw/theoretical_gw*100:.1f}%")
    print(f"  channel vs tensor: 实测 {speedup_cw_vs_tw:.2f}x | 理论 {theoretical_cw:.2f}x | 效率 {speedup_cw_vs_tw/theoretical_cw*100:.1f}%")


# --------------------------------------------------------------------------- #
# Typical Qwen3-4B projection shapes
# --------------------------------------------------------------------------- #
QWEN3_4B_SHAPES = [
    # (label,           M,   N,     K   )
    # ("q_proj    bs1",   1,  4096,  2560),
    # ("k_proj    bs1",   1,  1024,  2560),
    # ("v_proj    bs1",   1,  1024,  2560),
    ("qkv_proj    bs1",   1,  6144, 2560),
    ("o_proj    bs1",   1,  2560,  4096),
    # ("gate_proj bs1",   1,  9728,  2560),
    # ("up_proj   bs1",   1,  9728,  2560),
    ("gate_up_proj   bs1",   1,  19456,  2560),
    ("down_proj bs1",   1,  2560,  9728),
    # ("q_proj    bs8",   8,  4096,  2560),
    # ("gate_proj bs8",   8,  9728,  2560),
    # ("down_proj bs8",   8,  2560,  9728),
]

QWEN3_NEXT_SHAPES = [
    # (label,           M,   N,     K   )
    ("qkvz_proj    bs1",   1,  12288,  2048),
    ("qkvg_proj    bs1",   1,  9216,  2048),
    ("o_proj    bs1",   1,  2048,  4096),
    ("gate_up_proj bs1",   1,  512*10,  2048),
    ("down_proj bs1",   1,  2048,  512),
]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",          type=int,   default=1)
    parser.add_argument("--n",          type=int,   default=4096)
    parser.add_argument("--k",          type=int,   default=2560)
    parser.add_argument("--group_size", type=int,   default=128)
    parser.add_argument("--n_warmup",   type=int,   default=50)
    parser.add_argument("--n_repeat",   type=int,   default=200)
    parser.add_argument("--n_request",   type=int,   default=100)
    parser.add_argument("--sweep_name",  type=str, default='Qwen3-4B')
    parser.add_argument("--kernel",     type=str,   default="all",
                        choices=["all", "group_wise", "channel_wise", "tensor_wise",],
                        help="Which kernel(s) to benchmark")
    args = parser.parse_args()

    device = "cuda"
    print(f"Device     : {torch.cuda.get_device_name(device)}")
    print(f"Warmup     : {args.n_warmup}   Repeat : {args.n_repeat}")
    print(f"group_size : {args.group_size}   kernel : {args.kernel}")
    print("=" * 110)

    if args.sweep_name == 'Qwen3-4B':
        shapes = QWEN3_4B_SHAPES
    elif args.sweep_name == 'Qwen3-NEXT':
        shapes = QWEN3_NEXT_SHAPES
    else:
        shapes = [("custom", args.m, args.n, args.k)]

    for label, M, N, K in shapes:
        if args.kernel == "group_wise":
            r = bench_bitnet_group_wise(M, N, K, args.group_size, args.n_warmup, args.n_repeat, args.n_request, device)
            print_single(r, "group_wise", label)
        elif args.kernel == "channel_wise":
            r = bench_bitnet_channel_wise(M, N, K, args.n_warmup, args.n_repeat, args.n_request, device)
            print_single(r, "channel_wise", label)
        elif args.kernel == "tensor_wise":
            r = bench_bitnet_tensor_wise(M, N, K, args.n_warmup, args.n_repeat, args.n_request, device)
            print_single(r, "bf16", label)


        else:  # all — side-by-side with speedup

            r_g = bench_bitnet_group_wise(M, N, K, args.group_size, args.n_warmup, args.n_repeat, args.n_request,
                                          device)

            r_c = bench_bitnet_channel_wise(M, N, K, args.n_warmup, args.n_repeat, args.n_request, device)

            r_t = bench_bitnet_tensor_wise(M, N, K, args.n_warmup, args.n_repeat, args.n_request, device)

            print_comparison(r_g, r_c, r_t, label)

    print("=" * 110)


if __name__ == "__main__":
    main()