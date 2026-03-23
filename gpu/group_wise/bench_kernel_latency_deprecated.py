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
def make_inputs_bitnet(M: int, N: int, K: int, group_size: int, device="cuda"):
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert K % 4 == 0,         f"K={K} must be divisible by 4 (int2 packing)"

    x  = torch.randint(-128, 127, (M, K),       dtype=torch.int8,     device=device)
    s  = torch.rand(M, 1,                        dtype=torch.bfloat16, device=device) * 0.01 + 1e-5
    w  = torch.randint(-128, 127, (N, K // 4),  dtype=torch.int8,     device=device)
    ws = torch.rand(N * (K // group_size),       dtype=torch.bfloat16, device=device)
    return x, w, s, ws


def make_inputs_bf16(M: int, N: int, K: int, device="cuda"):
    # x: (M, K)  w: (N, K)  — F.linear computes x @ w.T
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    return x, w


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
def bench_bitnet(
    M: int, N: int, K: int,
    group_size: int = 128,
    n_warmup: int = 50,
    n_repeat: int = 200,
    device: str = "cuda",
) -> dict:
    x, w, s, ws = make_inputs_bitnet(M, N, K, group_size, device)

    mean_ms, std_ms = _timed(
        lambda: bitnet_int8xint2_linear(x, w, s, ws, group_size),
        n_warmup, n_repeat,
    )

    flops        = 2 * M * N * K
    total_bytes  = (N * K // 4              # weight int2 packed
                  + M * K                   # activation int8
                  + N * (K // group_size) * 2  # weight scale bf16
                  + M * 2                   # activation scale bf16
                  + M * N * 2)              # output bf16

    return dict(M=M, N=N, K=K,
                mean_ms=mean_ms, std_ms=std_ms,
                tflops=flops / (mean_ms * 1e-3) / 1e12,
                total_bytes_mb=total_bytes / 1e6)

def make_inputs_w4a8(M: int, N: int, K: int, group_size: int, device="cuda"):
    """
    W4A8 inputs:
      - activation: int8 + per-row scale (M, 1)
      - weight: int4 values stored in int8 tensor (N, K), each in [-8, 7]
      - weight scale: per-group bf16 scale (N, K // group_size)
    """
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"

    x_q = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=device)
    x_s = torch.rand(M, 1, dtype=torch.bfloat16, device=device) * 0.01 + 1e-5

    # int4 range carried in int8 container for easy arithmetic
    w_q = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=device)
    w_s = torch.rand(N, K // group_size, dtype=torch.bfloat16, device=device) * 0.02 + 1e-5
    return x_q, w_q, x_s, w_s


def w4a8_int8_gemv_kernel(x_i8, w_i8):
    # Preferred: custom op (if loaded)
    if hasattr(torch.ops, "w4a8") and hasattr(torch.ops.w4a8, "int8_gemv"):
        return torch.ops.w4a8.int8_gemv(x_i8, w_i8)

    # Fallback: int8 GEMM/GEMV path (often cublasLt-backed via PyTorch internal op)
    if hasattr(torch, "_int_mm"):
        # _int_mm: (M,K) x (K,N) -> (M,N), int32 accumulate
        return torch._int_mm(x_i8, w_i8.t().contiguous())


def bench_w4a8_gemv(
    M: int, N: int, K: int,
    group_size: int = 128,
    n_warmup: int = 50,
    n_repeat: int = 200,
    device: str = "cuda",
) -> dict:
    x_q, w_q, x_s, w_s = make_inputs_w4a8(M, N, K, group_size, device)

    # 关键：这里不要再转 bf16 + F.linear
    # 若 w_q 是 int4-packed，先解包到 int8；建议放计时外
    w_i8 = w_q.contiguous()   # 占位：实际应是 unpack_int4_to_int8(...)
    x_i8 = x_q.contiguous()

    mean_ms, std_ms = _timed(
        lambda: w4a8_int8_gemv_kernel(x_i8, w_i8),
        n_warmup, n_repeat,
    )

    flops = 2 * M * N * K
    return dict(
        M=M, N=N, K=K,
        mean_ms=mean_ms, std_ms=std_ms,
        tflops=flops / (mean_ms * 1e-3) / 1e12,
        total_bytes_mb=(N*K//2 + M*K + M*N*2) / 1e6,
    )

def bench_bf16_gemv(
    M: int, N: int, K: int,
    n_warmup: int = 50,
    n_repeat: int = 200,
    device: str = "cuda",
) -> dict:
    """
    bf16 GEMV via torch.nn.functional.linear (cuBLAS).
    For decode (M=1 or small), this is a pure GEMV.
    """
    x, w = make_inputs_bf16(M, N, K, device)

    mean_ms, std_ms = _timed(
        lambda: F.linear(x, w),
        n_warmup, n_repeat,
    )

    flops       = 2 * M * N * K
    total_bytes = (M * K * 2    # activation bf16
                 + N * K * 2    # weight bf16
                 + M * N * 2)   # output bf16

    return dict(M=M, N=N, K=K,
                mean_ms=mean_ms, std_ms=std_ms,
                tflops=flops / (mean_ms * 1e-3) / 1e12,
                total_bytes_mb=total_bytes / 1e6)


# --------------------------------------------------------------------------- #
# Pretty printers
# --------------------------------------------------------------------------- #
def _fmt(r: dict) -> str:
    return (f"{r['mean_ms']*1e3:.3f} ± {r['std_ms']*1e3:.3f} us  "
            f"({r['tflops']:.3f} TFLOPS  {r['total_bytes_mb']:.1f} MB)")


def print_single(r: dict, kernel: str, label: str = ""):
    tag = f"[{label}]" if label else f"[M={r['M']} N={r['N']} K={r['K']}]"
    print(f"{tag:<32}  [{kernel:<6}]  {_fmt(r)}")


def print_comparison_int4(r_bitnet: dict, r_int4: dict, label: str = ""):
    M, N, K = r_bitnet["M"], r_bitnet["N"], r_bitnet["K"]
    tag = f"[{label}]" if label else f"[M={M} N={N} K={K}]"
    speedup = r_int4["mean_ms"] / r_bitnet["mean_ms"]
    print(f"{tag:<32}  "
          f"bitnet: {_fmt(r_bitnet)}  |  "
          f"int4:   {_fmt(r_int4)}  |  "
          f"speedup: {speedup:.2f}x")

def print_comparison_bf16(r_bitnet: dict, r_bf16: dict, label: str = ""):
    M, N, K = r_bitnet["M"], r_bitnet["N"], r_bitnet["K"]
    tag = f"[{label}]" if label else f"[M={M} N={N} K={K}]"
    speedup = r_bf16["mean_ms"] / r_bitnet["mean_ms"]
    print(f"{tag:<32}  "
          f"bitnet: {_fmt(r_bitnet)}  |  "
          f"bf16:   {_fmt(r_bf16)}  |  "
          f"speedup: {speedup:.2f}x")


# --------------------------------------------------------------------------- #
# Typical Qwen3-4B projection shapes
# --------------------------------------------------------------------------- #
QWEN3_4B_SHAPES = [
    # (label,           M,   N,     K   )
    ("q_proj    bs1",   1,  4096,  2560),
    ("k_proj    bs1",   1,  1024,  2560),
    ("v_proj    bs1",   1,  1024,  2560),
    ("o_proj    bs1",   1,  2560,  4096),
    ("gate_proj bs1",   1,  9728,  2560),
    ("up_proj   bs1",   1,  9728,  2560),
    ("down_proj bs1",   1,  2560,  9728),
    ("q_proj    bs8",   8,  4096,  2560),
    ("gate_proj bs8",   8,  9728,  2560),
    ("down_proj bs8",   8,  2560,  9728),
]

QWEN3_NEXT_SHAPES = [
    # (label,           M,   N,     K   )
    ("qkvz_proj    bs1",   1,  12288,  2048),
    ("qkvg_proj    bs1",   1,  9216,  2048),
    ("o_proj    bs1",   1,  2048,  4096),
    ("gate_up_proj bs1",   1,  512*11,  2048),
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
    parser.add_argument("--group_size", type=int,   default=256)
    parser.add_argument("--n_warmup",   type=int,   default=50)
    parser.add_argument("--n_repeat",   type=int,   default=200)
    parser.add_argument("--sweep_name",  type=str, default='Qwen3-4B')
    parser.add_argument("--kernel",     type=str,   default="all",
                        choices=["all", "bitnet", "w4a8", "bf16"],
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
        raise ValueError(f"Unknown sweep_name: {args.sweep_name}")

    for label, M, N, K in shapes:
        if args.kernel == "bitnet":
            r = bench_bitnet(M, N, K, args.group_size, args.n_warmup, args.n_repeat, device)
            print_single(r, "bitnet", label)

        elif args.kernel == 'w4a8':
            r = bench_w4a8_gemv(M, N, K, args.group_size, args.n_warmup, args.n_repeat, device)
            print_single(r, "w4a8", label)

        elif args.kernel == "bf16":
            r = bench_bf16_gemv(M, N, K, args.n_warmup, args.n_repeat, device)
            print_single(r, "bf16", label)

        else:  # both — side-by-side with speedup
            r_b = bench_bitnet(M, N, K, args.group_size, args.n_warmup, args.n_repeat, device)
            r_i = bench_w4a8_gemv(M, N, K, args.group_size, args.n_warmup, args.n_repeat, device)
            r_f = bench_bf16_gemv(M, N, K, args.n_warmup, args.n_repeat, device)
            print_comparison_int4(r_b, r_i, label)
            print_comparison_bf16(r_b, r_f, label)

    print("=" * 110)


if __name__ == "__main__":
    main()