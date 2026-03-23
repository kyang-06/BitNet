"""
Profile script for ncu to analyze bitnet_int8xint2 vs bf16 GEMV kernels.

Usage:
    # Profile bf16 kernel
    sudo ncu --kernel-name regex:gemv --section SpeedOfLight --section MemoryWorkloadAnalysis \\
        python profile_kernel.py --kernel bf16 --m 1 --n 4096 --k 2560

    # Profile bitnet kernel
    sudo ncu --kernel-name regex:bitlinear --section SpeedOfLight --section MemoryWorkloadAnalysis \\
        python profile_kernel.py --kernel bitnet --m 1 --n 4096 --k 2560

    # Full metrics (occupancy + warp stall + memory)
    sudo ncu --set full --kernel-name regex:bitlinear -o bitnet_report \\
        python profile_kernel.py --kernel bitnet --m 1 --n 4096 --k 2560

    # Sweep all Qwen3-4B shapes (ncu will capture all kernel launches)
    sudo ncu --set full -o sweep_report \\
        python profile_kernel.py --kernel bitnet --sweep
"""

import ctypes
import argparse
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Load bitnet kernel
# --------------------------------------------------------------------------- #
bitnet_lib = ctypes.CDLL("bitnet_kernels/libbitnet.so")


def bitnet_int8xint2_linear(input0, input1, s, ws, group_size):
    out_shape = list(input0.shape)
    out_shape[-1] = input1.shape[0]

    stream = torch.cuda.current_stream()

    M = input0.shape[0]
    if len(out_shape) == 3:
        M *= input0.shape[1]
    N = input1.shape[0]
    K = input1.shape[1] * 4  # int2 packed: 4 values per byte

    ret = torch.zeros(*out_shape, dtype=torch.bfloat16, device=input0.device)

    bitnet_lib.bitlinear_int8xint2(
        ctypes.c_void_p(input0.data_ptr()),
        ctypes.c_void_p(input1.data_ptr()),
        ctypes.c_void_p(ret.data_ptr()),
        ctypes.c_void_p(s.data_ptr()),
        ctypes.c_void_p(ws.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
        ctypes.c_int(group_size),
        ctypes.c_void_p(stream.cuda_stream),
    )
    return ret


# --------------------------------------------------------------------------- #
# Tensor allocation helpers
# --------------------------------------------------------------------------- #
def make_inputs_bitnet(M, N, K, group_size, device="cuda"):
    assert K % group_size == 0
    assert K % 4 == 0
    x  = torch.randint(-128, 127, (M, K),      dtype=torch.int8,     device=device)
    s  = torch.rand(M, 1,                       dtype=torch.bfloat16, device=device) * 0.01 + 1e-5
    w  = torch.randint(-128, 127, (N, K // 4), dtype=torch.int8,     device=device)
    ws = torch.rand(N * (K // group_size),      dtype=torch.bfloat16, device=device)
    return x, w, s, ws


def make_inputs_bf16(M, N, K, device="cuda"):
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    return x, w


# --------------------------------------------------------------------------- #
# NVTX range helper  (让 ncu / Nsight Systems 可以按名字过滤)
# --------------------------------------------------------------------------- #
try:
    import nvtx
    def nvtx_range(name):
        return nvtx.annotate(name)
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def nvtx_range(name):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()


# --------------------------------------------------------------------------- #
# Profile runners
# --------------------------------------------------------------------------- #
def profile_bitnet(M, N, K, group_size=128, n_warmup=20, n_profile=10, device="cuda"):
    """
    Warmup first (ncu skips these via --launch-skip if needed),
    then run n_profile iterations inside NVTX range for easy filtering.
    """
    x, w, s, ws = make_inputs_bitnet(M, N, K, group_size, device)

    # Warmup — ncu 默认会 replay，warmup 只是为了让 CUDA context 稳定
    print(f"[bitnet] warmup M={M} N={N} K={K} ...", flush=True)
    for _ in range(n_warmup):
        bitnet_int8xint2_linear(x, w, s, ws, group_size)
    torch.cuda.synchronize()

    # Profile region
    print(f"[bitnet] profiling M={M} N={N} K={K} ({n_profile} iters) ...", flush=True)
    with nvtx_range(f"bitnet_M{M}_N{N}_K{K}"):
        for _ in range(n_profile):
            bitnet_int8xint2_linear(x, w, s, ws, group_size)
    torch.cuda.synchronize()
    print(f"[bitnet] done.", flush=True)


def profile_bf16(M, N, K, n_warmup=20, n_profile=10, device="cuda"):
    x, w = make_inputs_bf16(M, N, K, device)

    print(f"[bf16]   warmup M={M} N={N} K={K} ...", flush=True)
    for _ in range(n_warmup):
        F.linear(x, w)
    torch.cuda.synchronize()

    print(f"[bf16]   profiling M={M} N={N} K={K} ({n_profile} iters) ...", flush=True)
    with nvtx_range(f"bf16_M{M}_N{N}_K{K}"):
        for _ in range(n_profile):
            F.linear(x, w)
    torch.cuda.synchronize()
    print(f"[bf16]   done.", flush=True)


# --------------------------------------------------------------------------- #
# Qwen3-4B shapes
# --------------------------------------------------------------------------- #

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
    parser.add_argument("--kernel",     type=str, default="bitnet",
                        choices=["bitnet", "bf16", "all"])
    parser.add_argument("--m",          type=int, default=1)
    parser.add_argument("--n",          type=int, default=4096)
    parser.add_argument("--k",          type=int, default=2560)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--n_warmup",   type=int, default=20)
    parser.add_argument("--n_profile",  type=int, default=10)
    parser.add_argument("--sweep",      action="store_true",
                        help="Sweep all Qwen3-4B shapes")
    args = parser.parse_args()

    print(f"Device  : {torch.cuda.get_device_name('cuda')}")
    print(f"Kernel  : {args.kernel}  Warmup: {args.n_warmup}  Profile iters: {args.n_profile}")
    print("=" * 60)

    if args.sweep:
        shapes = QWEN3_NEXT_SHAPES
    else:
        shapes = [(f"M{args.m}_N{args.n}_K{args.k}", args.m, args.n, args.k)]

    for label, M, N, K in shapes:
        print(f"\n>>> {label}")
        if args.kernel in ("bitnet", "all"):
            profile_bitnet(M, N, K, args.group_size, args.n_warmup, args.n_profile)
        if args.kernel in ("bf16", "all"):
            profile_bf16(M, N, K, args.n_warmup, args.n_profile)

    print("\n[profile_kernel.py] finished.")


if __name__ == "__main__":
    main()