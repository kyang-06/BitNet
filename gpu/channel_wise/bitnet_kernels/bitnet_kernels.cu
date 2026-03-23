#include "bitnet_kernels.h"

extern "C" void bitlinear_int8xint2(int8_t* input0, int8_t* input1, __nv_bfloat16* output0, __nv_bfloat16* s, __nv_bfloat16* ws, int M, int N, int K, cudaStream_t stream){
    // Qwen3 (ws_num=N)
    if (M == 1 && N == 4096 && K == 2560 ){
        ladder_int8xint2_kernel<1, 4096, 2560, 4096, 8, 16><<<dim3(256, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 1024 && K == 2560 ){
        ladder_int8xint2_kernel<1, 1024, 2560, 1024, 8, 16><<<dim3(64, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 6144 && K == 2560 ){
        ladder_int8xint2_kernel<1, 6144, 2560, 6144, 8, 16><<<dim3(384, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 9728 && K == 2560 ){
        ladder_int8xint2_kernel<1, 9728, 2560, 9728, 8, 16><<<dim3(608, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 19456 && K == 2560 ){
        ladder_int8xint2_kernel<1, 19456, 2560, 19456, 8, 16><<<dim3(1216, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 2560 && K == 4096 ){
        ladder_int8xint2_kernel<1, 2560, 4096, 2560, 8, 16><<<dim3(160, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 2560 && K == 9728 ){
        ladder_int8xint2_kernel<1, 2560, 9728, 2560, 8, 16><<<dim3(160, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else{
        std::cout << "required ladder gemm kernel: M " << M << ", N " << N << ", K " << K << std::endl;
    }
}