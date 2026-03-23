# 先 profile bf16
sudo /usr/local/cuda/bin/ncu --kernel-name Kernel2 \
  --metrics "dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,sm__throughput.avg.pct_of_peak_sustained_elapsed" \
  --section SpeedOfLight \
  -o /home/kyyx/github/LLM/quantization/qwen3-coder/performance_analysis/ncu_2024/report_bf16_kernel2.ncu-rep \
  /home/kyyx/anaconda3/envs/qwen3-coder/bin/python bench_kernel_latency_precise.py --kernel bf16 --m 1 --n 9216 --k 2048 --n_warmup 2 --n_repeat 1 --sweep_name None

# 再 profile bitnet
sudo /usr/local/cuda/bin/ncu --kernel-name ladder_int8xint2_kernel \
  --metrics "dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,sm__throughput.avg.pct_of_peak_sustained_elapsed" \
  --section SpeedOfLight \
  -o /home/kyyx/github/LLM/quantization/qwen3-coder/performance_analysis/ncu_2024/report_i2s_kernel2.ncu-rep \
  /home/kyyx/anaconda3/envs/qwen3-coder/bin/python bench_kernel_latency_precise.py --kernel bitnet --m 1 --n 9216 --k 2048 --n_warmup 2 --n_repeat 1 --sweep_name None
